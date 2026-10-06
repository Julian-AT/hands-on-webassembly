"""Shared browser observations retained verbatim from the historical harness."""

DIAGNOSTICS = """async()=>{
 const registrations=await navigator.serviceWorker.getRegistrations();
 const query=worker=>new Promise(resolve=>{
  if(!worker)return resolve(null);
  const channel=new MessageChannel(),timeout=setTimeout(()=>{channel.port1.close();resolve({error:'diagnostics timeout'})},2000);
  channel.port1.onmessage=e=>{clearTimeout(timeout);channel.port1.close();resolve(e.data)};
  try{worker.postMessage({type:'course:diagnostics'},[channel.port2])}catch(error){clearTimeout(timeout);channel.port1.close();resolve({error:String(error)})}
 });
 return {controller:navigator.serviceWorker.controller?.scriptURL,
  registrations:await Promise.all(registrations.map(async registration=>({
   active:registration.active?.scriptURL,waiting:registration.waiting?.scriptURL,
   active_diagnostics:await query(registration.active),waiting_diagnostics:await query(registration.waiting)})))};
}"""
