"use strict";
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let currentEvent=null,settings=null,recording=false,stream=null,recorder=null,timer=null;
let voiceEnabled=false;
const promptedEvents=new Set();
document.addEventListener("DOMContentLoaded",()=>{
  document.querySelectorAll('[data-lucide="house-heart"]').forEach(el=>el.dataset.lucide="house");
  const control=document.createElement("label");
  control.className="toggle-row elder-voice-control";
  control.innerHTML='<span><strong>接收语音问询</strong><small id="voiceReadyState">未开启</small></span><input id="enableVoiceInquiry" type="checkbox" class="switch" role="switch">';
  $("elderStatus").after(control);
  $("enableVoiceInquiry").onchange=async()=>{
    const toggle=$("enableVoiceInquiry");
    if(!toggle.checked){voiceEnabled=false;$("voiceReadyState").textContent="未开启";return;}
    try{
      if(!settings?.audio_consent)throw new Error("请先由照护者记录音频授权");
      if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
      const permissionStream=await navigator.mediaDevices.getUserMedia({audio:true});
      permissionStream.getTracks().forEach(t=>t.stop());
      const ready=new SpeechSynthesisUtterance("语音问询已开启");ready.lang="zh-CN";speechSynthesis.speak(ready);
      voiceEnabled=true;$("voiceReadyState").textContent="已开启";
      await refresh();
    }catch(error){toggle.checked=false;voiceEnabled=false;toast(error.message);}
  };
});
async function api(url,options={}){if(options.body&&!(options.body instanceof FormData)){options.headers={"Content-Type":"application/json"};options.body=JSON.stringify(options.body);}const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(typeof(data.error||data.detail)==="string"?(data.error||data.detail):"操作失败，请重试");return data;}
function toast(message){$("toast").textContent=message;$("toast").hidden=false;setTimeout(()=>$("toast").hidden=true,6500);}
async function refresh(){try{const [status,profile,events,sessions]=await Promise.all([api("/api/status"),api("/api/settings"),api("/api/events?state=open"),api("/api/sessions?mode=live")]);settings=profile;$("elderGreeting").textContent=profile.resident.name?`${profile.resident.name}，您好`:"您好";const active=status.coverage_state==="active";$("elderStatus").classList.toggle("active",active);$("elderStatus").innerHTML=`<i data-lucide="${active?"shield-check":"shield-alert"}"></i><div><strong>${active?"居家监测已开启":status.coverage_state==="reduced"?"部分监测暂不可用":"居家监测尚未开启"}</strong><p>${active?`${esc(profile.resident.room)} · 输入有效`:status.coverage.filter(x=>x.channel!=="audio"&&x.state!=="active").map(x=>esc(x.reason)).join("；")}</p></div>`;const query=new URLSearchParams(location.search).get("event");const e=query?events.find(e=>e.id===query):events.find(e=>e.mode==="live");if(!recording){currentEvent=e||null;$("elderEvent").hidden=!e;if(e){$("elderEventMode").textContent=e.mode==="live"?"实时事件":"样例 / 回放";$("elderEventText").textContent=`${e.room}有一条${e.level==="EMERGENCY"?"紧急求助":"待核实的跌倒"}记录。${e.acknowledged_at?"照护者已接手处理。":"等待照护者核实。"}`;$("elderSafe").disabled=e.inquiry.status==="button_safe";$("elderInquiry").disabled=!profile.audio_consent||e.inquiry.status==="listening"||e.inquiry.status==="processing";}}const today=new Date().toLocaleDateString("en-CA");$("elderExerciseCount").textContent=`${sessions.filter(s=>s.status==="ended"&&new Date(s.created_at).toLocaleDateString("en-CA")===today).length} 次训练`;$("elderContacts").innerHTML=profile.contacts.map(c=>`<div class="elder-contact"><strong>${esc(c.name)}</strong><small>${esc(c.relation)}</small><a class="button secondary" href="tel:${esc(c.phone.replace(/[^+0-9]/g,""))}"><i data-lucide="phone"></i>${esc(c.phone)}</a></div>`).join("");window.lucide?.createIcons();}catch(e){$("elderStatus").innerHTML='<i data-lucide="wifi-off"></i><div><strong>照护系统连接中断</strong><p>请直接联系家人</p></div>';window.lucide?.createIcons();}}
const refreshStatus=refresh;
refresh=async function(){
  await refreshStatus();
  if(settings&&!settings.audio_consent){
    voiceEnabled=false;
    const toggle=$("enableVoiceInquiry");if(toggle)toggle.checked=false;
    if(recording){clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());if(recorder?.state==="recording")recorder.stop();}
  }
  if(voiceEnabled&&!recording&&currentEvent?.inquiry.status==="not_started"&&!promptedEvents.has(currentEvent.id)){
    promptedEvents.add(currentEvent.id);
    inquiry();
  }
};
function speak(){return new Promise((resolve,reject)=>{if(!("speechSynthesis" in window))return reject(new Error("当前浏览器无法播放询问"));speechSynthesis.cancel();const phrase=new SpeechSynthesisUtterance("请问你怎么啦？你需要帮助吗？");phrase.lang="zh-CN";phrase.rate=.85;phrase.onend=resolve;phrase.onerror=()=>reject(new Error("询问语音没有成功播放"));speechSynthesis.speak(phrase);});}
async function unavailable(id,reason){try{await api(`/api/events/${id}/inquiry/unavailable`,{method:"POST",body:{reason}});}catch{}}
async function inquiry(){if(!currentEvent||recording)return;if(!settings.audio_consent){toast("尚未获得音频处理授权");return;}const id=currentEvent.id,mode=currentEvent.mode;recording=true;$("elderInquiry").disabled=true;$("inquiryProgress").hidden=false;$("elderInquiryResult").hidden=true;$("inquiryStatus").textContent="准备麦克风";let started=false;try{if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");stream=await navigator.mediaDevices.getUserMedia({audio:true});$("inquiryStatus").textContent="请听语音询问";await speak();await api(`/api/events/${id}/inquiry`,{method:"POST",body:{prompt_played:true}});started=true;const parts=[];recorder=new MediaRecorder(stream);recorder.ondataavailable=e=>{if(e.data.size)parts.push(e.data);};const audio=await new Promise((resolve,reject)=>{recorder.onerror=()=>reject(new Error("麦克风录音失败"));recorder.onstop=()=>resolve(new Blob(parts,{type:recorder.mimeType}));recorder.start();let seconds=0;$("inquiryStatus").textContent="正在听您的回答";$("inquiryCountdown").textContent="10 秒";$("inquiryBar").value=0;timer=setInterval(()=>{seconds++;$("inquiryCountdown").textContent=`${Math.max(0,10-seconds)} 秒`;$("inquiryBar").value=seconds;if(seconds>=10){clearInterval(timer);recorder.stop();}},1000);});stream.getTracks().forEach(t=>t.stop());stream=null;$("inquiryStatus").textContent="正在识别回答";$("inquiryCountdown").textContent="";await api(`/api/events/${id}/inquiry/processing`,{method:"POST"});const form=new FormData();form.append("file",audio,"reply.webm");form.append("task","reply");if(!settings.audio_consent)throw new Error("音频授权已撤回");form.append("mode",mode);form.append("event_id",id);const result=await api("/api/audio/analyze",{method:"POST",body:form});const outcome=result.event.inquiry.result;$("elderInquiryResult").textContent=outcome.escalate?"求助已升级，等待照护者响应。":outcome.cancelled?"已记录您的回应，等待照护者核实。":outcome.state==="audio_unavailable"?"语音识别暂不可用，请使用求助按钮。":"已记录您的回应，等待照护者响应。";$("elderInquiryResult").hidden=false;}catch(e){toast(e.message);if(started)await unavailable(id,e.message);else{try{await api(`/api/events/${id}/inquiry`,{method:"POST",body:{prompt_played:false,reason:e.message}});}catch{}}$("elderInquiryResult").textContent="语音问询不可用，请使用下方求助按钮。";$("elderInquiryResult").hidden=false;}finally{clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());stream=null;recording=false;$("inquiryProgress").hidden=true;$("elderInquiry").disabled=false;await refresh();}}
document.addEventListener("DOMContentLoaded",()=>{$("elderDate").textContent=new Date().toLocaleDateString("zh-CN",{month:"long",day:"numeric",weekday:"long"});$("elderHelp").onclick=async()=>{$("elderHelp").disabled=true;try{const event=await api("/api/help",{method:"POST"});$("helpConfirm").hidden=false;$("elderHelpSub").textContent="求助已发出";await refresh();}catch(e){toast(e.message);}finally{$("elderHelp").disabled=false;}};$("elderSafe").onclick=async()=>{if(!currentEvent)return;try{await api(`/api/events/${currentEvent.id}/action`,{method:"POST",body:{action:"safe"}});$("elderInquiryResult").textContent="已记录您表示安全，等待照护者核实。";$("elderInquiryResult").hidden=false;await refresh();}catch(e){toast(e.message);}};$("elderInquiry").onclick=inquiry;refresh();setInterval(refresh,2500);window.lucide?.createIcons();window.addEventListener("pagehide",()=>{clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());if(recording&&currentEvent)fetch(`/api/events/${currentEvent.id}/inquiry/unavailable`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reason:"长者端已关闭"}),keepalive:true});speechSynthesis?.cancel();});});
