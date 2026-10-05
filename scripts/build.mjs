import {createHash} from 'node:crypto';
import {readFile, readdir, stat, writeFile, rm} from 'node:fs/promises';
import {resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {prepareAssets, hashFile} from './assets.mjs';
const root=process.cwd();
const inputs=['app','lib','scripts','manifests','package.json','package-lock.json','next.config.mjs','vercel.json'];
const locked={};
async function visit(path) { const file=resolve(root,path); if((await stat(file)).isDirectory()) { for(const name of (await readdir(file)).sort()) await visit(`${path}/${name}`); } else locked[path]=await hashFile(file); }
for(const path of inputs) await visit(path);
const identity=createHash('sha256').update(JSON.stringify(Object.fromEntries(Object.entries(locked).sort(([a],[b])=>a<b?-1:a>b?1:0)))).digest('hex');
await prepareAssets(root);
await rm(resolve(root,'.next'),{recursive:true,force:true}); await rm(resolve(root,'out'),{recursive:true,force:true});
const result=spawnSync(process.execPath,['node_modules/next/dist/bin/next','build','--webpack'],{stdio:'inherit',env:{...process.env,COURSE_BUILD_ID:identity,NEXT_TELEMETRY_DISABLED:'1'}});
if(result.error) throw result.error; if(result.status!==0) process.exit(result.status||1);
const expected=JSON.parse(await readFile(resolve(root,'manifests/application-files.json'),'utf8'));
for(const [name,info] of Object.entries(expected.files)) if((await stat(resolve(root,'out',name))).size!==info.bytes || await hashFile(resolve(root,'out',name))!==info.sha256) throw new Error(`Export changed application asset ${name}`);
const files={};
async function output(path='') { for(const name of (await readdir(resolve(root,'out',path))).sort()) { const key=path?`${path}/${name}`:name; const file=resolve(root,'out',key); const info=await stat(file); if(info.isDirectory()) await output(key); else files[key]={bytes:info.size,sha256:await hashFile(file)}; } }
await output();
await writeFile(resolve(root,'out/release-manifest.json'),JSON.stringify({build_id:identity,application_build_id:JSON.parse(await readFile(resolve(root,'lib/assignment-sessions.json'),'utf8')).build_id,files},null,2)+'\n');
console.log(`Static export ${identity}: ${Object.keys(files).length} files; no runtime functions configured`);
