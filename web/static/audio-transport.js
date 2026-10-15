"use strict";

class AudioChunkQueue {
  constructor(send, maxPending=8) {
    this.send=send;this.maxPending=maxPending;this.pending=[];this.busy=false;this.waiters=[];this.closed=false;
  }
  push(message) {
    if(this.closed)throw new Error("音频连接已停止");
    if(this.pending.length>=this.maxPending)throw new Error("声音处理积压超过 4 秒，监听已停止，请检查设备负载后重新启动");
    this.pending.push(message);this.next();
  }
  next() {
    if(this.busy||this.closed)return;
    if(this.pending.length){this.busy=true;this.send(this.pending.shift());}
    else this.waiters.splice(0).forEach(resolve=>resolve());
  }
  acknowledge(){this.busy=false;this.next();}
  drain(){return this.busy||this.pending.length?new Promise(resolve=>this.waiters.push(resolve)):Promise.resolve();}
  close(){this.closed=true;this.pending.length=0;this.busy=false;this.waiters.splice(0).forEach(resolve=>resolve());}
}

if(typeof module!=="undefined")module.exports={AudioChunkQueue};
