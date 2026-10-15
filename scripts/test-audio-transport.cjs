const assert=require('node:assert/strict');
const {AudioChunkQueue}=require('../web/static/audio-transport.js');

(async()=>{
  const sent=[];
  const queue=new AudioChunkQueue(chunk=>sent.push(chunk),8);
  for(let i=0;i<8;i++)queue.push({timestamp_ms:i*500,pcm:`chunk-${i}`});
  assert.equal(sent.length,1);
  let drained=false;
  const finished=queue.drain().then(()=>{drained=true;});
  for(let i=0;i<8;i++)queue.acknowledge();
  await finished;
  assert(drained);
  assert.deepEqual(sent.map(x=>x.timestamp_ms),[0,500,1000,1500,2000,2500,3000,3500]);
  assert.deepEqual(sent.map(x=>x.pcm),Array.from({length:8},(_,i)=>`chunk-${i}`));
  const blocked=new AudioChunkQueue(()=>{},2);
  blocked.push({});blocked.push({});blocked.push({});
  assert.throws(()=>blocked.push({}),/积压/);
  blocked.close();
  assert.equal(blocked.pending.length,0);
  assert.throws(()=>blocked.push({}),/停止/);
  console.log('Audio ordering, backlog limit and disposal passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
