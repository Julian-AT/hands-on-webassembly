import {createHash} from 'node:crypto';
import {createReadStream, createWriteStream} from 'node:fs';
import {mkdir, readFile, writeFile, rename, stat, rm, open} from 'node:fs/promises';
import {resolve, posix} from 'node:path';
import {createGunzip} from 'node:zlib';
import {Readable} from 'node:stream';
import {pipeline} from 'node:stream/promises';
export function safePath(name) {
  if(!name || name.includes('\\') || name.includes('\0') || name.startsWith('/') || name.split('/').some(x=>!x || x==='.' || x==='..') || posix.normalize(name)!==name) throw new Error(`Unsafe archive path: ${name}`);
  return name;
}
export async function hashFile(path) { const hash=createHash('sha256'); for await(const chunk of createReadStream(path)) hash.update(chunk); return hash.digest('hex'); }
export async function extractArchive(archive, staging, manifest) {
  await mkdir(staging,{recursive:false});
  const input=createReadStream(archive), gunzip=createGunzip(); input.on('error',e=>gunzip.destroy(e)); input.pipe(gunzip);
  const iterator=gunzip[Symbol.asyncIterator](); let buffer=Buffer.alloc(0), ended=false;
  async function take(count) {
    const pieces=[]; let left=count;
    while(left) {
      if(!buffer.length) { const next=await iterator.next(); if(next.done) throw new Error('Truncated asset archive'); buffer=next.value; }
      const n=Math.min(left,buffer.length); pieces.push(buffer.subarray(0,n)); buffer=buffer.subarray(n); left-=n;
    }
    return pieces.length===1?pieces[0]:Buffer.concat(pieces,count);
  }
  const seen=new Set();
  try {
    while(!ended) {
      const header=await take(512);
      if(header.every(x=>x===0)) { const second=await take(512); if(!second.every(x=>x===0)) throw new Error('Invalid tar ending'); ended=true; break; }
      const field=(start,length)=>header.subarray(start,start+length).toString('utf8').split('\0')[0];
      const name=safePath((field(345,155)?field(345,155)+'/':'')+field(0,100));
      const type=header[156]; if(type!==0 && type!==48) throw new Error(`Archive entry is not a regular file: ${name}`);
      if(seen.has(name) || !Object.hasOwn(manifest.files,name)) throw new Error(`Duplicate or unexpected entry: ${name}`);
      const sizeText=field(124,12).trim(); if(!/^[0-7]+$/.test(sizeText)) throw new Error('Invalid tar size');
      const size=parseInt(sizeText,8), expected=manifest.files[name]; if(size!==expected.bytes) throw new Error(`Wrong size: ${name}`);
      const checksum=parseInt(field(148,8).trim(),8); const actual=header.reduce((sum,x,i)=>sum+(i>=148&&i<156?32:x),0);
      if(!Number.isInteger(checksum) || actual!==checksum) throw new Error('Invalid tar header checksum');
      const target=resolve(staging,name); await mkdir(resolve(target,'..'),{recursive:true});
      const file=await open(target,'wx',0o644), hash=createHash('sha256');
      try { for(let left=size;left;) { const chunk=await take(Math.min(left,65536)); hash.update(chunk); let offset=0; while(offset<chunk.length) { const {bytesWritten}=await file.write(chunk,offset,chunk.length-offset); if(!bytesWritten) throw new Error('Short asset write'); offset+=bytesWritten; } left-=chunk.length; } } finally { await file.close(); }
      if(hash.digest('hex')!==expected.sha256) throw new Error(`Asset checksum rejected: ${name}`);
      const padding=await take((512-size%512)%512); if(!padding.every(x=>x===0)) throw new Error('Invalid tar padding');
      seen.add(name);
    }
    if(buffer.some(x=>x!==0)) throw new Error('Unexpected trailing archive data');
    for await(const chunk of {[Symbol.asyncIterator]:()=>iterator}) if(chunk.some(x=>x!==0)) throw new Error('Unexpected trailing archive data');
    if(seen.size!==Object.keys(manifest.files).length) throw new Error('Incomplete asset archive');
    for(const [name,expected] of Object.entries(manifest.files)) if((await stat(resolve(staging,name))).size!==expected.bytes || await hashFile(resolve(staging,name))!==expected.sha256) throw new Error(`Staging verification rejected: ${name}`);
    return seen.size;
  } catch(error) { input.destroy(); gunzip.destroy(); throw error; }
}
export async function prepareAssets(root) {
  const manifest=JSON.parse(await readFile(resolve(root,'manifests/application-files.json'),'utf8'));
  const reference=JSON.parse(await readFile(resolve(root,'manifests/assets-release.json'),'utf8'));
  if(reference.manifest_sha256!==manifest.manifest_sha256) throw new Error('Asset manifest reference mismatch');
  const hashesPath=resolve(root,'manifests/application-hashes.json');
  if(await hashFile(hashesPath)!==manifest.manifest_sha256) throw new Error('Invalid locked asset manifest');
  const hashes=JSON.parse(await readFile(hashesPath,'utf8'));
  if(Object.keys(hashes).length!==manifest.file_count || manifest.file_count!==Object.keys(manifest.files).length || Object.entries(manifest.files).some(([name,info])=>hashes[name]!==info.sha256)) throw new Error('Inconsistent asset manifests');
  const cache=resolve(root,'.tools'); await mkdir(cache,{recursive:true}); const archive=resolve(cache,`assets-${reference.sha256}.tar.gz`);
  try { await stat(archive); } catch {
    const local=process.env.COURSE_ASSET_ARCHIVE;
    if(local) await pipeline(createReadStream(local),createWriteStream(archive,{flags:'wx'}));
    else {
      if(!process.env.COURSE_ASSET_TOKEN) throw new Error('Missing build-only COURSE_ASSET_TOKEN');
      const response=await fetch(reference.api_url,{headers:{Accept:'application/octet-stream',Authorization:`Bearer ${process.env.COURSE_ASSET_TOKEN}`,'X-GitHub-Api-Version':'2022-11-28'}});
      if(!response.ok) throw new Error(`Release asset download HTTP ${response.status}`);
      await pipeline(Readable.fromWeb(response.body),createWriteStream(archive,{flags:'wx'}));
    }
  }
  if((await stat(archive)).size!==reference.bytes || await hashFile(archive)!==reference.sha256) throw new Error('Release archive checksum rejected');
  const staging=resolve(root,`.asset-staging-${process.pid}`);
  const count=await extractArchive(archive,staging,manifest);
  // Apply only the locked presentation change after verifying the original archive.
  const overrides=JSON.parse(await readFile(resolve(root,'manifests/startup-ui.json'),'utf8')).files;
  const allowed=Array.from({length:7},(_,i)=>`unit${i+1}/index.html`);
  if(!overrides || Object.keys(overrides).length!==7 || allowed.some(name=>!Object.hasOwn(overrides,name))) throw new Error('Incomplete startup UI overrides');
  for(const name of allowed) {
    const override=overrides[name], original=manifest.files[name];
    if(!original || override.original_sha256!==original.sha256) throw new Error(`Startup UI input binding rejected: ${name}`);
    const target=resolve(staging,name), source=await readFile(target,'utf8');
    const marker='<section id="course-startup" role="status"';
    if(source.split(marker).length!==2) throw new Error(`Startup panel contract changed: ${name}`);
    const patched=source.replace(marker,'<section id="course-startup" hidden role="status"');
    if(Buffer.byteLength(patched)!==override.bytes || createHash('sha256').update(patched).digest('hex')!==override.sha256) throw new Error(`Startup UI output binding rejected: ${name}`);
    await writeFile(target,patched);
    if(await hashFile(target)!==override.sha256) throw new Error(`Startup UI staging rejected: ${name}`);
  }
  // Publish only after all original assets and approved UI changes pass.
  await rm(resolve(root,'public'),{recursive:true,force:true}); await rename(staging,resolve(root,'public'));
  console.log(`Verified and published ${count} application files`);
}
