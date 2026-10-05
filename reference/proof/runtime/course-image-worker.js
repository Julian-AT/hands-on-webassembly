// Responsive owner supervises transport and a disposable numerical child.
let owner,abort,decoder;
const fields=['protocol','unit','operation','build_id','source_id','session_id','task_id','dataset_generation','model_generation'];
const matches=(a,b)=>fields.every(key=>a?.[key]===b?.[key]);
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
  value=>value.toString(16).padStart(2,'0')).join('');
self.onmessage=async ({data})=>{
  if(data.kind==='cancel'){
    if(matches(owner,data.tags)){
      abort?.abort();decoder?.terminate();
      self.postMessage({kind:'cancelled',tags:owner});self.close();
    }
    return;
  }
  const tags=data.tags;
  if(owner||data.kind!=='prepare'||tags?.protocol!==1||![5,6,7].includes(tags.unit)||
     tags.operation!=='image-loading'||!tags.build_id||!tags.source_id||!tags.session_id||
     !Number.isInteger(tags.task_id)||!Number.isInteger(tags.dataset_generation)||!Number.isInteger(tags.model_generation))return;
  owner=tags;abort=new AbortController();
  try{
    if(!/^[a-f0-9]{64}$/.test(data.manifest_sha256)||
       await digest(new TextEncoder().encode(data.unpack_source))!==data.unpack_sha256)
      throw new Error('Image preparation code or manifest identity changed. Reload and retry.');
    const root=new URL('../assets/v1/images/',self.location.href);
    async function bytes(name){
      if(!/^[A-Za-z0-9_-]+\.(json|npz)$/.test(name))throw new Error('Unexpected image asset path.');
      const response=await fetch(new URL(name,root),{signal:abort.signal,cache:'no-cache'});
      if(!response.ok)throw new Error(`Dataset asset ${name}: HTTP ${response.status}. Retry loading.`);
      return new Uint8Array(await response.arrayBuffer());
    }
    const rawManifest=await bytes('manifest.json');
    if(await digest(rawManifest)!==data.manifest_sha256)throw new Error('Dataset manifest belongs to another build. Reload and retry.');
    const manifest=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(rawManifest));
    const keys=['train','test'].map(split=>`${data.dataset}/${split}`);
    const entries=keys.map(key=>{
      const entry=manifest.datasets?.[key];
      if(!entry)throw new Error(`The complete ${key} dataset has not been packaged.`);
      return entry;
    });
    const archives=[];
    // Sequential reads bound memory for SVHN and CIFAR10. No partial commit.
    for(let i=0;i<entries.length;i++){
      const entry=entries[i],raw=await bytes(entry.file);
      if(raw.byteLength!==entry.bytes||await digest(raw)!==entry.sha256)
        throw new Error(`Dataset checksum or length mismatch: ${keys[i]}. Retry loading.`);
      archives.push(raw);
      self.postMessage({kind:'progress',tags,payload:{dataset:data.dataset,split:keys[i].split('/')[1],status:'verified'}});
    }
    decoder=new Worker(new URL('course-image-decode-worker.js',self.location.href));
    const result=await new Promise((resolve,reject)=>{
      decoder.onmessage=({data:message})=>{
        if(!matches(tags,message.tags))return;
        message.kind==='error'?reject(new Error(message.error)):resolve(message);
      };
      decoder.onerror=event=>reject(new Error(event.message));
      decoder.postMessage({tags,keys,entries,archives,unpack_source:data.unpack_source},archives.map(a=>a.buffer));
    });
    if(abort.signal.aborted)return;
    self.postMessage(result,result.splits.flatMap(split=>[split.images.buffer,split.labels.buffer]));
  }catch(error){
    if(!abort.signal.aborted)self.postMessage({kind:'error',tags,error:String(error)});
  }finally{abort.abort();decoder?.terminate();self.close();}
};
