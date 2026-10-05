"""Streaming HTML must preserve bytes and carry valid representation metadata."""
import subprocess
import unittest
from pathlib import Path


class HtmlProxyTests(unittest.TestCase):
    def test_actual_proxy_channel_preserves_streams_and_metadata(self):
        script = r'''
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const {MessageChannel}=require('node:worker_threads');
const pending=new Map();
const ctx={TextEncoder,Uint8Array,Headers,Request,Response,ReadableStream,DOMException,MessageChannel,
 setTimeout,clearTimeout,setInterval,clearInterval,courseRequests:pending,courseAppOwners:{},apps:{},
 self:{location:{pathname:'/nested/shinylive-sw.js'}},dirname:()=>'/nested',reqToASGI:()=>({}),
 asgiToRes:(msg,body)=>new Response(body,{status:msg.status,headers:msg.headers})};
for(const file of ['service-worker-http.js','service-worker-html.js'])
 vm.runInNewContext(fs.readFileSync('proof/runtime/'+file,'utf8'),ctx);
const encoder=new TextEncoder(),insert='<script src="/nested/shinylive-inject-socket.js" type="module"></script>\n';
async function exchange(input,chunks,html=true,finalEmpty=true){
 const client={postMessage:(_,ports)=>{
  const port=ports[0];
  port.onmessage=e=>{if(e.data.type!=='http.request'||e.data.more_body)return;
   port.postMessage({type:'http.response.start',status:200,headers:[['content-type',html?'text/html; charset=utf-8':'application/octet-stream'],
    ['content-length',String(input.length)],['content-range','bytes 0-1/2'],['accept-ranges','bytes'],
    ['etag','original'],['content-md5','original'],['digest','original'],['content-digest','original'],['repr-digest','original'],['x-app','preserved']]});
   chunks.forEach((body,i)=>port.postMessage({type:'http.response.body',body,more_body:finalEmpty||i<chunks.length-1}));
   if(finalEmpty)port.postMessage({type:'http.response.body',body:new Uint8Array(),more_body:false});
   port.close();
  };port.start();
 }};
 const response=await ctx.fetchASGI(client,new Request('http://localhost/app/'),undefined,ctx.createInjectSocketFilter(),'app/');
 const received=Buffer.from(await response.arrayBuffer());
 const expected=html?Buffer.from(new TextDecoder().decode(input).replace('</head>',insert+'</head>')):Buffer.from(input);
 assert.deepEqual(received,expected);
 assert.equal(response.headers.get('x-app'),'preserved');
 for(const header of ['content-length','content-range','accept-ranges','etag','content-md5','digest','content-digest','repr-digest'])
  assert.equal(response.headers.has(header),!html);
}
(async()=>{
 const input=encoder.encode('<head>Grüße 🐍 中文</head><body>é</body>');
 for(let split=0;split<=input.length;split++)
  await exchange(input,[input.slice(0,split),input.slice(split)]);
 await exchange(input,[...input].map(b=>new Uint8Array([b])),true,false);
 await Promise.all(Array.from({length:16},(_,i)=>exchange(input,[input.slice(0,i),input.slice(i)],true,i%2===0)));
 for(const text of ['', '<head>missing closing tag', '</hea', '</head></head>']){
  const bytes=encoder.encode(text);await exchange(bytes,[...bytes].map(b=>new Uint8Array([b])));
 }
 const binary=new Uint8Array([0,255,128,13,10,60,47,104,101,97,100,62]);
 await exchange(binary,[binary.slice(0,6),binary.slice(6)],false);
 assert.equal(pending.size,0);
})().catch(error=>{console.error(error);process.exit(1)});
'''
        subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[2],
                       capture_output=True, text=True, check=True, timeout=15)

    def test_every_split_utf8_final_empty_chunk_headers_and_response_isolation(self):
        script = r'''
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const ctx={TextEncoder,Uint8Array,Headers,self:{location:{pathname:'/nested/shinylive-sw.js'}},dirname:()=>'/nested'};
vm.runInNewContext(fs.readFileSync('proof/runtime/service-worker-html.js','utf8'),ctx);
const encoder=new TextEncoder();
const original=encoder.encode('<!doctype html><head><title>Grüße 🐍 中文</title></head><body>é</body>');
const expected=encoder.encode('<!doctype html><head><title>Grüße 🐍 中文</title><script src="/nested/shinylive-inject-socket.js" type="module"></script>\n</head><body>é</body>');
const html=new Response(null,{headers:{'content-type':'text/html; charset=utf-8'}});
function join(chunks){return Buffer.concat(chunks.map(chunk=>Buffer.from(chunk)))}
for(let split=0;split<=original.length;split++){
 const filter=ctx.createInjectSocketFilter();
 assert.deepEqual(join([filter(original.slice(0,split),html),filter(original.slice(split),html),filter(new Uint8Array(),html,true)]),Buffer.from(expected));
}
const single=ctx.createInjectSocketFilter();
assert.deepEqual(join([...original].map(byte=>single(new Uint8Array([byte]),html)).concat([single(new Uint8Array(),html,true)])),Buffer.from(expected));
const message={status:200,headers:[['content-type','text/html'],['content-length','62920'],['etag','old'],['content-range','bytes 0-9/10'],['content-digest','old'],['x-app','kept']]};
const transformed=ctx.createInjectSocketFilter().transformResponseStart(message);
const headers=new Headers(transformed.headers);
for(const name of ['content-length','etag','content-range','content-digest'])assert.equal(headers.has(name),false);
assert.equal(headers.get('x-app'),'kept');assert.equal(message.headers.length,6);
const plain=ctx.createInjectSocketFilter(),nonHtml=new Response(null,{headers:{'content-type':'application/octet-stream'}});
assert.equal(plain(original,nonHtml,true),original);
assert.equal(new Headers(plain.transformResponseStart({status:200,headers:[['content-type','image/png'],['content-length','10']]}).headers).get('content-length'),'10');
for(const text of ['','<head>no closing tag','</hea','</head></head>']){
 const input=encoder.encode(text), filter=ctx.createInjectSocketFilter();
 const output=join([...input].map(b=>filter(new Uint8Array([b]),html)).concat([filter(new Uint8Array(),html,true)]));
 assert.equal(output.toString(),text.replace('</head>','<script src="/nested/shinylive-inject-socket.js" type="module"></script>\n</head>'));
}
// Interleaving distinct response bodies must never share buffered bytes.
const a=ctx.createInjectSocketFilter(),b=ctx.createInjectSocketFilter();
const prefix=a(encoder.encode('A</hea'),html),other=b(encoder.encode('B</head>'),html,true),suffix=a(encoder.encode('d>'),html,true);
assert.equal(join([prefix,suffix]).toString(),'A<script src="/nested/shinylive-inject-socket.js" type="module"></script>\n</head>');
assert.equal(Buffer.from(other).toString(),'B<script src="/nested/shinylive-inject-socket.js" type="module"></script>\n</head>');
assert.throws(()=>ctx.createInjectSocketFilter().transformResponseStart({headers:[['content-type','text/html'],['content-encoding','gzip']]}),/encoded/);
'''
        subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[2],
                       capture_output=True, text=True, check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
