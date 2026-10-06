// Full immutable model preparation belongs to a disposable context.
let owner, abort;
const fields=['protocol','unit','operation','build_id','source_id','session_id','task_id','dataset_generation','model_generation'];
const matches=(a,b)=>fields.every(key=>a?.[key]===b?.[key]);
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
  value=>value.toString(16).padStart(2,'0')).join('');
self.onmessage=async ({data})=>{
  if(data.kind==='cancel'){
    if(matches(data.tags,owner)){abort?.abort();self.postMessage({kind:'cancelled',tags:owner});self.close();}
    return;
  }
  if(owner)return;
  const tags=data.tags;
  if(tags?.protocol!==1 || tags.unit!==2 || tags.operation!=='embedding-loading' ||
     !tags.build_id || !tags.source_id || !tags.session_id || !Number.isInteger(tags.task_id) ||
     !Number.isInteger(tags.dataset_generation) || !Number.isInteger(tags.model_generation))return;
  owner=tags;abort=new AbortController();
  try{
    const root=new URL('../assets/v1/embedding-runtime/',self.location.href);
    async function bytes(name){
      const response=await fetch(new URL(name,root),{signal:abort.signal,cache:'no-cache'});
      if(!response.ok)throw new Error(`Embedding asset ${name}: HTTP ${response.status}. Retry loading.`);
      return new Uint8Array(await response.arrayBuffer());
    }
    const manifestBytes=await bytes('manifest.json');
    if(await digest(manifestBytes)!==data.manifest_sha256)throw new Error('Embedding manifest changed. Reload and retry.');
    const decoder=new TextDecoder('utf-8',{fatal:true});
    const manifest=JSON.parse(decoder.decode(manifestBytes));
    const names=['tokenizer.json','word-rows.json.gz','vectors.npy'];
    if(JSON.stringify(Object.keys(manifest.files).sort())!==JSON.stringify([...names].sort()) ||
       manifest.keys!==514157 || JSON.stringify(manifest.vector_shape)!=='[20000,300]')
      throw new Error('The complete pinned embedding model is required.');
    const assets=await Promise.all(names.map(async name=>{
      const raw=await bytes(name);
      if(await digest(raw)!==manifest.files[name])throw new Error(`Embedding checksum mismatch: ${name}. Retry loading.`);
      self.postMessage({kind:'progress',tags,payload:{asset:name,status:'verified'}});
      return raw;
    }));
    const tokenizer=decoder.decode(assets[0]);JSON.parse(tokenizer);
    const decompressed=new Response(new Blob([assets[1]]).stream().pipeThrough(new DecompressionStream('gzip')));
    const pairs=JSON.parse(await decompressed.text());
    const rows=Object.create(null);
    for(const pair of pairs){
      if(!Array.isArray(pair)||pair.length!==2||typeof pair[0]!=='string'||
         !Number.isInteger(pair[1])||pair[1]<0||pair[1]>=20000||Object.hasOwn(rows,pair[0]))
        throw new Error('Embedding vocabulary contains invalid or duplicate entries.');
      rows[pair[0]]=pair[1];
    }
    if(Object.keys(rows).length!==514157)throw new Error('Embedding vocabulary is incomplete.');
    const npy=assets[2],view=new DataView(npy.buffer,npy.byteOffset,npy.byteLength);
    if(npy.length<10||npy[0]!==147||decoder.decode(npy.subarray(1,6))!=='NUMPY'||npy[6]!==1||npy[7]!==0)
      throw new Error('Embedding vector format differs from the pinned model.');
    const offset=10+view.getUint16(8,true),header=decoder.decode(npy.subarray(10,offset));
    if(!/'descr':\s*'<f4'/.test(header)||!/'fortran_order':\s*False/.test(header)||
       !/'shape':\s*\(20000,\s*300\s*\)/.test(header)||npy.byteLength-offset!==20000*300*4)
      throw new Error('Embedding vector dimensions or dtype are invalid.');
    const vectors=npy.slice(offset);
    self.postMessage({kind:'result',tags,tokenizer,rows,vectors,shape:[20000,300]},[vectors.buffer]);
  }catch(error){
    if(!abort.signal.aborted)self.postMessage({kind:'error',tags,error:String(error)});
  }finally{abort.abort();self.close();}
};
