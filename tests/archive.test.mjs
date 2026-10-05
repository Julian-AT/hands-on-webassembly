import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {safePath,extractArchive} from '../scripts/assets.mjs';
function tar(entries){ const blocks=[];for(const {name,data='test',type='0'} of entries){const bytes=Buffer.from(data),header=Buffer.alloc(512);header.write(name,0,100);header.write('0000644\0',100);header.write('0000000\0',108);header.write('0000000\0',116);header.write(bytes.length.toString(8).padStart(11,'0')+'\0',124);header.write('00000000000\0',136);header.fill(32,148,156);header.write(type,156);header.write('ustar\0',257);header.write(header.reduce((s,x)=>s+x,0).toString(8).padStart(6,'0')+'\0 ',148);blocks.push(header,bytes,Buffer.alloc((512-bytes.length%512)%512));}blocks.push(Buffer.alloc(1024));return gzipSync(Buffer.concat(blocks));}
const manifest={files:{'assets/data.txt':{bytes:4,sha256:createHash('sha256').update('test').digest('hex')}}};
test('safe paths reject traversal and noncanonical entries',()=>{for(const name of ['../escape','/escape','a/../escape','a\\b','a//b','./a','a/','a\0b'])assert.throws(()=>safePath(name));assert.equal(safePath('assets/data.txt'),'assets/data.txt');});
test('complete verified archive extracts exact file values',async()=>{const dir=await mkdtemp(join(tmpdir(),'course-archive-'));try{const file=join(dir,'asset.tgz');await writeFile(file,tar([{name:'assets/data.txt'}]));assert.equal(await extractArchive(file,join(dir,'stage'),manifest),1);assert.equal(await readFile(join(dir,'stage/assets/data.txt'),'utf8'),'test');}finally{await rm(dir,{recursive:true,force:true});}});
test('rejects missing, duplicate, unexpected, traversal, link, corrupt and wrong-size entries',async()=>{
 const fixtures=[[],[{name:'assets/data.txt'},{name:'assets/data.txt'}],[{name:'extra.txt'}],[{name:'../escape'}],[{name:'assets/data.txt',type:'2'}],[{name:'assets/data.txt',data:'fail'}],[{name:'assets/data.txt',data:'wrong size'}]];
 for(const entries of fixtures){const dir=await mkdtemp(join(tmpdir(),'course-archive-'));try{const file=join(dir,'asset.tgz');await writeFile(file,tar(entries));await assert.rejects(extractArchive(file,join(dir,'stage'),manifest));}finally{await rm(dir,{recursive:true,force:true});}}
});
