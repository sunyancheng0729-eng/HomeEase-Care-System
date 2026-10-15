class HomeEasePCM extends AudioWorkletProcessor {
  constructor(){super();this.buffer=new Float32Array(8000);this.offset=0;this.startFrame=0;}
  process(inputs){const input=inputs[0]?.[0];if(!input)return true;for(let i=0;i<input.length;i++){if(this.offset===0)this.startFrame=currentFrame+i;this.buffer[this.offset++]=input[i];if(this.offset===this.buffer.length){const samples=this.buffer;this.port.postMessage({samples,timestamp_ms:this.startFrame/sampleRate*1000},[samples.buffer]);this.buffer=new Float32Array(8000);this.offset=0;}}return true;}
}
registerProcessor("homeease-pcm",HomeEasePCM);
