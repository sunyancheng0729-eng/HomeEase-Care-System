"use strict";
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<i data-lucide="${name}"></i>`;
const icons = () => window.lucide?.createIcons();
const names = {chair_sit_to_stand:"坐站训练",squat:"深蹲训练",standing_leg_raise:"站立抬腿",hurdle_step:"跨步训练"};
const phases = {seated:"坐姿",sitting:"坐姿",rising:"起身",standing:"站立",lowering:"下落",bottom:"最低点",descending:"下蹲",ascending:"起立",baseline:"起始姿态",lifting:"抬腿",raised:"抬起",stepping:"跨步",returning:"回落",other:"等待动作",up:"抬起",down:"落下",start:"起始",peak:"顶点"};
const qualityNames = {good:"观察有效",waiting_for_input:"等待画面",insufficient_pose:"关键点不足，计数暂不可用",multiple_people:"画面中有多人，计数暂不可用",no_person:"未检测到人体",timestamp_gap:"画面间隔过大，等待完整动作",person_changed:"目标变化，等待完整动作",degenerate_pose:"关键点异常，计数暂不可用",paused:"已暂停",module_error:"模块异常，训练中断",service_restarted:"服务重启，训练中断"};
const channels = {vision:"视频",accel:"加速度",audio:"音频",manual:"主动求助"};
const outcomes = {confirmed_fall:"确认跌倒",false_alarm:"误报",safe:"核实安全",unresolved:"结果未确定"};
const stateNames = {ready:"已就绪",loading:"加载中",not_initialized:"未初始化",unavailable:"不可用"};
const moduleNames = {video:"算法一 · 视频跌倒",signal:"算法一 · 加速度跌倒",exercise:"算法二 · 四动作计数",audio:"算法三 · 声音与语音"};
const modelNames = {video:"YOLOv8n + P1 + A4",signal:"A2 · RF + XGB",exercise:"opt_tcn_b4_w48",audio:"Silero / YAMNet + LR / Whisper tiny / rule_v2"};
const soundNames = {HELP:"求助相关声音",IMPACT:"撞击声",ABNORMAL:"异常声音",DAILY:"日常声音",UNKNOWN:"未确定"};
const dateTime = (v) => v ? new Date(v).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}) : "--";
const timeOnly = (v) => v ? new Date(v).toLocaleTimeString("zh-CN",{hour12:false}) : "--";
const duration = (v) => v >= 3600 ? `${(v/3600).toFixed(1)} 小时` : v >= 60 ? `${Math.floor(v/60)} 分 ${Math.round(v%60)} 秒` : `${Math.round(v || 0)} 秒`;
const modeBadge = (v) => `<span class="badge ${v === "live" ? "green" : "blue"}">${v === "live" ? "实时" : "样例 / 回放"}</span>`;
const levelBadge = (e) => `<span class="badge ${e.state === "resolved" ? "green" : e.level === "EMERGENCY" ? "red" : "amber"}">${e.state === "resolved" ? "已处置" : e.level === "EMERGENCY" ? "紧急求助" : "待核实"}</span>`;
const empty = (title, sym="inbox", subtitle="", compact=false) => `<div class="empty-state ${compact?"compact":""}">${icon(sym)}<strong>${esc(title)}</strong>${subtitle?`<p>${esc(subtitle)}</p>`:""}</div>`;
const app = {page:"overview",settings:null,status:null,events:[],sessions:[],notices:[],sounds:[],report:null,eventFilter:"open",eventMode:"live",reportDays:7,reportMode:"live",selectedExercise:"chair_sit_to_stand",sid:null,exercise:null,camera:null,capturing:false,exerciseSource:null,audioStream:null,audioSocket:null,audioContext:null,audioRecorder:null,recordedBlob:null,videoHistory:[],selectedEvent:null,noticeIds:new Set(),ready:false};
let toastTimer, refreshBusy=false, exercisePollBusy=false;

async function api(path, options={}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeout || 180000);
  const settings = {...options, signal:controller.signal};
  delete settings.timeout;
  if (settings.body && !(settings.body instanceof FormData)) {
    settings.headers = {...settings.headers,"Content-Type":"application/json"};
    settings.body = JSON.stringify(settings.body);
  }
  try {
    const response = await fetch(path,settings);
    const result = await response.json();
    if (!response.ok) {
      let message = result.error || result.detail || `请求失败 (${response.status})`;
      if (Array.isArray(message)) message = message.map(x=>`${x.loc?.slice(1).join(".") || "参数"}: ${x.msg}`).join("；");
      if (typeof message !== "string") message = JSON.stringify(message);
      throw new Error(message);
    }
    return result;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("请求超时，请检查模块状态后重试");
    throw e;
  } finally {clearTimeout(timer);}
}
function toast(message,error=false){$("toast").textContent=message;$("toast").className=`toast ${error?"error":""}`;$("toast").hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").hidden=true,error?7500:4000);}
async function busy(button, fn){if(button.disabled)return;const wasDisabled=button.disabled;button.disabled=true;button.classList.add("loading");try{return await fn();}catch(e){toast(e.message,true);}finally{button.disabled=wasDisabled;button.classList.remove("loading");}}
function metric(title,value,unit,sym,note=""){return `<div class="metric"><div class="metric-header">${icon(sym)}${esc(title)}</div><div class="metric-value">${esc(value)}<small>${esc(unit)}</small></div><div class="metric-note">${esc(note)}</div></div>`;}
function channelRow(row){const active=row?.state==="active";return `<div class="channel-item">${icon({vision:"video",accel:"activity",audio:"mic"}[row.channel])}<div><strong>${channels[row.channel]}</strong><small><span class="dot ${active?"green":"amber"}"></span>${active?"输入有效":row.state==="reduced"?"覆盖受限":"未连接"}</small></div></div>`;}

function navigate(){const requested=location.hash.slice(1).split("?")[0]||"overview";app.page=$("page-"+requested)?requested:"overview";document.querySelectorAll(".page").forEach(x=>x.hidden=x.id!==`page-${app.page}`);document.querySelectorAll("nav [data-page]").forEach(x=>x.classList.toggle("active",x.dataset.page===app.page));const label=document.querySelector(`nav [data-page="${app.page}"]`)?.textContent.replace(/\d+$/,"")||"照护总览";$("pageCrumb").textContent=label;document.title=`${label} · HomeEase`;$("sidebar").classList.remove("open");$("navBackdrop").hidden=true;renderCurrent();if(app.page==="reports")loadReport();if(app.page==="settings")loadDiagnostics();window.scrollTo(0,0);}

async function refresh(){if(refreshBusy)return;refreshBusy=true;try{const results=await Promise.allSettled([api("/api/status",{timeout:12000}),api("/api/events"),api("/api/sessions"),api("/api/notifications"),app.page==="audio"?api("/api/sounds"):Promise.resolve(app.sounds),api("/api/settings")]);if(results[0].status==="rejected")throw results[0].reason;app.status=results[0].value;if(results[5].status==="fulfilled"){app.settings=results[5].value;if(!app.settings.audio_consent){if(app.audioStream)stopAudioLive();if(app.audioRecorder?.state==="recording")app.audioRecorder.stop();app.recordingStream?.getTracks().forEach(t=>t.stop());app.recordedBlob=null;}}if(results[1].status==="fulfilled")app.events=results[1].value;if(results[2].status==="fulfilled")app.sessions=results[2].value;if(results[3].status==="fulfilled")app.notices=results[3].value;if(results[4].status==="fulfilled")app.sounds=results[4].value;$("connectionState").innerHTML='<span class="dot green"></span>系统已连接';const n=app.events.filter(e=>e.state==="open"&&e.mode==="live").length;$("navEventCount").textContent=n;$("navEventCount").hidden=!n;$("noticeDot").hidden=!app.notices.some(n=>n.status!=="seen"&&n.kind==="inapp");for(const notice of app.notices){if(app.ready&&!app.noticeIds.has(notice.id)&&notice.kind==="inapp"&&notice.mode==="live"){toast(`${notice.title} · ${notice.message}`);if("Notification" in window&&Notification.permission==="granted")new Notification(notice.title,{body:notice.message,tag:notice.event_id});}app.noticeIds.add(notice.id);}app.ready=true;renderCurrent();if(app.selectedEvent&&$("eventDialog").open){const latest=app.events.find(e=>e.id===app.selectedEvent.id);if(latest&&latest.revision!==app.selectedEvent.revision&&!$("eventDetail").contains(document.activeElement)){renderEventDetail(latest);}}}catch(e){$("connectionState").innerHTML='<span class="dot red"></span>系统连接中断';}finally{refreshBusy=false;}}
function renderCurrent(){if(!app.status)return;if(!["overview","monitor"].includes(app.page)){for(const id of ["overviewFrame","monitorFrame"]){const image=$(id);if(image.dataset.streaming){image.removeAttribute("src");delete image.dataset.streaming;}}}renderModuleIndicators();if(app.page==="overview")renderOverview();if(app.page==="monitor")renderMonitor();if(app.page==="exercise")renderExerciseHistory();if(app.page==="audio")renderAudioHistory();if(app.page==="events")renderEvents();icons();}
function renderModuleIndicators(){for(const m of app.status.modules){if(m.id==="exercise"){$("exerciseModelState").textContent=`运动模块${stateNames[m.state]||m.state}`;$("exerciseModelState").className=`badge ${m.state==="ready"?"green":"amber"}`;}}const consent=app.settings?.audio_consent;$("audioConsentBadge").textContent=consent?"已授权本地音频":"未授权音频";$("audioConsentBadge").className=`badge ${consent?"green":"neutral"}`;$("audioConsentBanner").hidden=!!consent;if(app.page==="settings")renderModules();}
function renderOverview(){const s=app.status;const coverage={active:["监测已开启","视频与加速度输入有效"],reduced:["监测覆盖受限",s.coverage.filter(x=>x.channel!=="audio"&&x.state!=="active").map(x=>x.reason).join("；")],unavailable:["监测尚未开启","视频与加速度输入未连接"]}[s.coverage_state];$("coverageTitle").textContent=coverage[0];$("coverageDescription").textContent=coverage[1];$("coverageBanner").classList.toggle("is-active",s.coverage_state==="active");const today=new Date().toLocaleDateString("en-CA");const live=app.events.filter(e=>e.mode==="live");const dayEvents=live.filter(e=>new Date(e.created_at).toLocaleDateString("en-CA")===today);const todaySessions=app.sessions.filter(s=>s.mode==="live"&&s.status==="ended"&&new Date(s.created_at).toLocaleDateString("en-CA")===today);const open=live.filter(e=>e.state==="open");$("dashboardMetrics").innerHTML=metric("待处理事件",open.length,"条","bell-ring",open.some(e=>e.level==="EMERGENCY")?"有紧急求助待响应":"当前实时记录")+metric("今日确认跌倒",dayEvents.filter(e=>e.outcome==="confirmed_fall").length,"次","shield-check","以照护者处置结果为准")+metric("今日运动",todaySessions.reduce((sum,s)=>sum+s.count,0),"次","activity",`${todaySessions.length} 次已完成训练`)+metric("有效监测通道",s.coverage.filter(x=>x.state==="active"&&x.channel!=="audio").length,"/ 2","radio","视频 / 加速度");$("dashboardChannels").innerHTML=s.coverage.sort((a,b)=>["vision","accel","audio"].indexOf(a.channel)-["vision","accel","audio"].indexOf(b.channel)).map(channelRow).join("");$("openEventCount").textContent=open.length;$("attentionList").innerHTML=open.length?open.slice(0,4).map(e=>`<div class="event-item" role="button" tabindex="0" data-event="${e.id}"><span class="event-icon ${e.level==="EMERGENCY"?"red":""}">${icon(e.level==="EMERGENCY"?"bell-ring":"triangle-alert")}</span><div class="event-body"><div class="event-title"><strong>${esc(e.room)} · ${e.level==="EMERGENCY"?"紧急求助":"疑似跌倒"}</strong>${levelBadge(e)}</div><p>${channels[e.channel]}${e.channel==="manual"?"":"检测"} · ${esc(e.resident)}</p><div class="event-meta"><span>${dateTime(e.occurred_at)}</span><span>#${e.id.slice(0,6)}</span></div></div>${icon("chevron-right")}</div>`).join(""):empty("暂无待处理事件","circle-check","实时事件将显示在这里");$("recentSessions").innerHTML=app.sessions.filter(s=>s.mode==="live").slice(0,3).map(s=>`<div class="session-row"><span class="session-icon">${icon("activity")}</span><div>${names[s.exercise]}<small>${dateTime(s.created_at)} · ${s.status==="ended"?"已完成":s.status==="interrupted"?"已中断":"进行中"}</small></div><span class="session-count">${s.count}<small> 次</small></span></div>`).join("")||empty("今日还没有训练记录","activity","",true);const running=s.video.state==="running";$("overviewVideoMode").textContent=running?(s.video_mode==="live"?"实时监测":"样例 / 回放"):"未开启";$("overviewVideoMode").className=`badge ${running?"green":"neutral"}`;$("cameraSourceLabel").textContent=running?(s.video_mode==="live"?"实时画面":"样例 / 回放"):"样例预览";$("cameraIdle").hidden=running;loadFrame("overviewFrame");if(app.report)drawBars($("overviewChart"),app.report.series,"reps","#579579");else loadReport(true);}
function renderMonitor(){
  const v=app.status.video,running=v.state==="running",sensor=app.status.sensor;
  const coverage=app.status.coverage.filter(c=>["vision","accel"].includes(c.channel));
  const active=coverage.filter(c=>c.state==="active").length;
  $("monitorChannels").innerHTML=coverage.map(channelRow).join("");
  $("monitorMode").textContent=active===2?"双通道监测中":active===1?"1 / 2 通道有效":"等待实时输入";
  $("monitorMode").className=`badge ${active===2?"green":active===1?"amber":"neutral"}`;
  $("videoStateLabel").textContent=({idle:"尚未启动",running:app.status.video_mode==="live"?"实时视频检测":"视频回放",finished:"本次检测已结束",error:"检测异常"})[v.state]||v.state;
  $("videoFps").textContent=`推理 ${v.fps||0} FPS`;
  $("videoProbability").textContent=v.frames?`${(100*(v.prob||0)).toFixed(1)}%`:"--";
  $("videoFrames").textContent=v.frames||0;$("videoFalls").textContent=v.n_falls||0;
  $("videoVisibility").textContent=v.frames?(v.person_visible?"人体可见":"未检测到人体"):"未检测";
  $("startVideo").disabled=running||app.status.modules.find(m=>m.id==="video").state!=="ready";
  $("stopVideo").disabled=!running;
  $("videoModuleInfo").innerHTML=`<div><span>检测方案</span><strong>A4</strong></div><div><span>原始阈值</span><strong>0.60</strong></div><div><span>确认次数</span><strong>2</strong></div><div><span>目标采样帧率</span><strong>15 FPS</strong></div>${v.error?`<div class="error-inline">${esc(v.error)}</div>`:""}`;
  if(v.frames){const last=app.videoHistory.at(-1);if(!last||last.frame!==v.frames)app.videoHistory.push({frame:v.frames,value:v.prob});app.videoHistory=app.videoHistory.slice(-120);}
  drawLines($("videoChart"),[{values:app.videoHistory.map(x=>x.value),color:"#47836a"}],{min:0,max:1,threshold:.6});
  const fresh=sensor.state==="connected"&&Date.now()/1000-(sensor.last_received||0)<5;
  $("sensorStatus").textContent=sensor.state==="connected"?(fresh?`实时数据已连接 · ${sensor.windows||0} 个窗口`:"传感器数据中断"):"未连接";
  if(sensor.state==="connected"){
    $("signalState").textContent=fresh?"实时监测中":"数据中断";
    $("signalState").className=`badge ${fresh?"green":"amber"}`;
    const samples=sensor.samples||[],recent=sensor.recent||[];
    drawLines($("signalChart"),[1,2,3].map((axis,i)=>({values:samples.map(s=>s[axis]),color:["#547bb4","#5a9877","#bf9459"][i]})));
    drawLines($("signalProbabilityChart"),[{values:recent.map(s=>s.p),color:"#52886d"}],{min:0,max:1,threshold:.5});
    $("signalResult").textContent=`实时窗口跌倒概率：${((sensor.prob||0)*100).toFixed(1)}%`;
    $("signalResult").dataset.live="true";
  }else if($("signalResult").dataset.live){
    $("signalState").textContent="传感器已断开";$("signalState").className="badge neutral";
  }else if(!$("signalChart").dataset.drawn){
    drawLines($("signalChart"),[],{min:-2,max:2});drawLines($("signalProbabilityChart"),[],{min:0,max:1,threshold:.5});
    $("signalChart").dataset.drawn="true";
  }
  loadFrame("monitorFrame");
}
function loadFrame(id){
  const image=$(id), visible=(id==="overviewFrame"&&app.page==="overview")||(id==="monitorFrame"&&app.page==="monitor");
  if(!visible)return;
  const other=$(id==="overviewFrame"?"monitorFrame":"overviewFrame");
  if(other.dataset.streaming){other.removeAttribute("src");delete other.dataset.streaming;}
  if(app.status.video.state==="running"){
    const key=String(app.status.video.session);
    if(image.dataset.streaming!==key){image.src=`/api/video/stream?session=${key}`;image.dataset.streaming=key;}
  }else if(image.dataset.streaming||!image.getAttribute("src")){
    image.src=app.status.video.frames?`/api/video/frame?t=${Date.now()}`:"/static/media/room-preview.jpg";
    delete image.dataset.streaming;
  }
}

function renderEvents(){let list=app.events.filter(e=>(app.eventFilter==="all"||e.state===app.eventFilter)&&(app.eventMode==="all"||e.mode===app.eventMode));const q=$("eventSearch").value.trim().toLowerCase();if(q)list=list.filter(e=>`${e.id} ${e.room} ${e.resident}`.toLowerCase().includes(q));$("eventsTotal").textContent=`${list.length} 条记录`;$("eventList").innerHTML=list.length?list.map(e=>`<article class="event-record ${e.level==="EMERGENCY"?"emergency":""} ${e.state==="resolved"?"resolved":""}" data-event="${e.id}" role="button" tabindex="0"><div><h3>${esc(e.room)} · ${e.channel==="manual"?"主动求助":e.channel==="audio"?"语音求救":"疑似跌倒"}${levelBadge(e)}${modeBadge(e.mode)}</h3><p>${channels[e.channel]}${e.channel==="manual"?"":"检测"} · ${esc(e.resident)}${e.outcome?` · ${outcomes[e.outcome]}`:""}</p><footer><span>${dateTime(e.occurred_at)}</span><span>事件 #${e.id}</span><span>${e.acknowledged_at?"照护者已接手":""}</span></footer></div><span class="badge neutral">${e.tag?({single_channel:"单通道",dual_channel:"双通道",triple_channel:"三通道"})[e.tag]:"主动求助"}</span>${icon("chevron-right")}</article>`).join(""):empty("暂无符合条件的事件","inbox");icons();}
async function openEvent(id){try{renderEventDetail(await api(`/api/events/${id}`));if(!$("eventDialog").open)$("eventDialog").showModal();}catch(e){toast(e.message,true);}}
function evidenceText(e){const v=e.evidence;if(e.channel==="vision")return `视频原算法检测到疑似跌倒。触发概率 ${((v.prob||0)*100).toFixed(1)}%，判定阈值 ${v.threshold??.6}${v.peak?`，峰值概率 ${(v.peak*100).toFixed(1)}%`:""}。`;if(e.channel==="accel")return `加速度原算法检测到疑似跌倒。触发概率 ${((v.score||0)*100).toFixed(1)}%，判定阈值 ${v.threshold??.5}${v.n_windows?`，命中 ${v.n_windows} 个窗口`:""}。`;if(e.channel==="audio")return "原语音求救监听器识别到明确求助。";return "长者通过求助按钮主动请求协助。";}
function renderEventDetail(e){app.selectedEvent=e;const settings=app.settings;const inquiry=e.inquiry;const inquiryText={not_started:"尚未发起",listening:"等待回复",audio_unavailable:"音频不可用",completed:"已完成",button_safe:"长者已表示安全"}[inquiry.status]||inquiry.status;$("eventDetail").innerHTML=`<div class="event-detail-body"><div class="event-detail-header"><h3>${esc(e.room)} · ${esc(e.resident)}</h3><div>${levelBadge(e)} ${modeBadge(e.mode)}</div></div><div class="event-detail-meta"><div><span>事件编号</span>#${e.id}</div><div><span>发生时间</span>${dateTime(e.occurred_at)}</div><div><span>检测通道</span>${channels[e.channel]}</div><div><span>语音询问</span>${esc(inquiryText)}</div><div><span>检测时间</span>${dateTime(e.detected_at)}</div><div><span>输入来源</span>${esc(e.source)}</div></div><div class="evidence-summary">${evidenceText(e)}${e.mode==="replay"?'<br><span class="badge blue">样例 / 回放记录</span>':""}</div>${e.evidence.clip?`<video class="event-video" controls preload="metadata" src="/api/video/clip/${e.id}"></video>`:""}${e.evidence.snippet?'<div class="section-heading"><h3>报警前后加速度</h3><span class="muted small">−2 秒至 +1 秒 · g</span></div><div class="chart-wrap"><canvas id="eventSignalChart"></canvas></div>':""}${e.state!=="resolved"?`<div class="event-actions"><button class="button primary" data-action="acknowledge" ${e.acknowledged_at?"disabled":""}>${icon("user-check")}${e.acknowledged_at?"已接手":"接手处理"}</button><button class="button danger-outline" data-action="escalate">${icon("arrow-up-right")}升级响应</button><a class="button secondary" href="/elder?event=${e.id}" target="_blank">${icon("audio-lines")}长者问询</a></div>`:""}<div>${(settings?.contacts||[]).map((c,i)=>`<div class="event-contact-row"><span class="badge neutral">${i+1}</span><strong>${esc(c.name)}</strong><span class="muted">${esc(c.relation)}</span><a class="button secondary" href="tel:${esc(c.phone.replace(/[^+0-9]/g,""))}" data-contact-call="true">${icon("phone")}${esc(c.phone)}</a></div>`).join("")||'<p class="muted small">暂无紧急联系人</p>'}</div><div class="section-heading spaced"><h3>响应时间线</h3><a class="text-action" href="/reports/events/${e.id}" target="_blank">${icon("printer")}事件报告</a></div><ol class="timeline">${e.timeline.map(t=>`<li><time>${dateTime(t.at)}</time><div>${esc(t.note)}<small>${esc(t.actor)}</small></div></li>`).join("")}</ol>${e.state==="resolved"?`<div class="completion-banner">${icon("circle-check")}<strong>${outcomes[e.outcome]}</strong><span>${e.response_s!==null?`响应时长 ${duration(e.response_s)}`:""}</span></div>${e.notes?`<p class="muted small" style="margin-top:15px">${esc(e.notes)}</p>`:""}`:`<form id="resolveForm" class="resolution-form"><h3>处置结论</h3><label>最终结果<select id="resolveOutcome" required><option value="">选择处置结果</option><option value="confirmed_fall">确认跌倒</option><option value="false_alarm">误报</option><option value="safe">核实安全</option><option value="unresolved">结果未确定</option></select></label><label>处理备注<textarea id="resolveNote" rows="3" maxlength="2000" placeholder="联系情况、采取的措施等"></textarea></label><button class="button primary" type="submit">${icon("check")}记录结果并关闭事件</button></form>`}</div>`;icons();if(e.evidence.snippet){requestAnimationFrame(()=>{const s=e.evidence.snippet;drawLines($("eventSignalChart"),[{values:s.ax,color:"#547bb4"},{values:s.ay,color:"#5a9877"},{values:s.az,color:"#bf9459"}],{labels:s.t?.map(t=>`${t.toFixed(1)}s`)});});}$("resolveForm")?.addEventListener("submit",ev=>{ev.preventDefault();busy(ev.submitter,()=>eventAction("resolve",{outcome:$("resolveOutcome").value,note:$("resolveNote").value}));});}
async function eventAction(action,extra={}){const e=app.selectedEvent;const result=await api(`/api/events/${e.id}/action`,{method:"POST",body:{action,revision:e.revision,...extra}});renderEventDetail(result);await refresh();toast(action==="resolve"?"处置结果已记录":"操作已记录");}

function renderExerciseHistory(){const sessions=app.sessions;$("exerciseHistory").innerHTML=sessions.length?`<table><thead><tr><th>训练时间</th><th>动作</th><th>次数 / 目标</th><th>有效观察</th><th>来源</th><th>状态</th><th>报告</th></tr></thead><tbody>${sessions.slice(0,30).map(s=>`<tr><td>${dateTime(s.created_at)}</td><td>${names[s.exercise]}</td><td>${s.count} / ${s.target_reps}</td><td>${duration(s.valid_duration_s)}</td><td>${modeBadge(s.mode)}</td><td>${({running:"进行中",paused:"已暂停",ended:"已结束",interrupted:"已中断"})[s.status]||s.status}</td><td><a class="icon-button" title="下载训练报告" href="/api/sessions/${s.session_id}/export.json" aria-label="下载训练报告">${icon("download")}</a></td></tr>`).join("")}</tbody></table>`:empty("暂无训练记录","activity","",true);icons();}
function chooseExercise(key){app.selectedExercise=key;document.querySelectorAll(".exercise-option").forEach(x=>{const selected=x.dataset.exercise===key;x.classList.toggle("selected",selected);x.setAttribute("aria-pressed",selected);});}
async function startExercise(ev){
  ev.preventDefault();
  await busy($("startExercise"),async()=>{
    const source=$("exerciseSource").value;
    let uploadId=null;
    if(source==="upload"){
      const file=$("exerciseUpload").files[0];if(!file)throw new Error("请选择运动视频");
      const form=new FormData();form.append("file",file);
      uploadId=(await api("/api/uploads/video",{method:"POST",body:form})).upload_id;
    }
    const result=await api("/api/sessions",{method:"POST",body:{exercise:app.selectedExercise,target_reps:Number($("targetReps").value),source,upload_id:uploadId,camera_index:Number($("exerciseCameraIndex").value),capture:"shared"}});
    app.sid=result.session_id;app.exerciseSource=source;
    $("exerciseChooser").hidden=true;$("exerciseActive").hidden=false;$("exerciseCompleted").hidden=true;
    $("pauseExercise").disabled=false;$("endExercise").disabled=false;
    if(source==="camera"){
      $("exerciseCamera").hidden=false;$("exerciseCamera").src=`/api/sessions/${app.sid}/camera/stream`;
    }else $("exerciseCamera").hidden=true;
    renderExerciseState(result);await refresh();
  });
}
async function controlExercise(action){
  if(!app.sid||app.exerciseControlling)return;
  app.exerciseControlling=true;
  $("pauseExercise").disabled=true;$("endExercise").disabled=true;
  try{
    renderExerciseState(await api(`/api/sessions/${app.sid}/control`,{method:"POST",body:{action}}));
    if(action==="end")await refresh();
  }finally{
    app.exerciseControlling=false;
    syncExerciseControls();
  }
}
function syncExerciseControls(){const finished=["ended","interrupted"].includes(app.exercise?.status);$("pauseExercise").disabled=finished||!!app.exerciseControlling;$("endExercise").disabled=finished||!!app.exerciseControlling;}
async function stopCamera(){ $("exerciseCamera").removeAttribute("src");$("exerciseCamera").hidden=true; }
function renderExerciseState(s){
  app.exercise=s;$("activeExerciseName").textContent=names[s.exercise];$("exerciseMode").textContent=s.mode==="live"?"实时训练":"样例 / 回放";
  $("repCount").textContent=s.count;$("repTarget").textContent=s.target_reps;$("repProgress").value=s.progress;
  $("exercisePhase").textContent=phases[s.phase]||s.phase;$("exerciseDuration").textContent=duration(s.valid_duration_s);
  const q=["multiple_people","no_person"].includes(s.person_id)?s.person_id:s.quality;
  const diagnostic=s.capture_diagnostics;
  const jointNames={left_shoulder:"左肩",right_shoulder:"右肩",left_hip:"左髋",right_hip:"右髋",left_knee:"左膝",right_knee:"右膝",left_ankle:"左踝",right_ankle:"右踝"};
  let quality=qualityNames[q]||q;
  if(diagnostic?.missing_joints.length&&q==="insufficient_pose")quality+=`：${diagnostic.missing_joints.map(n=>jointNames[n]||n).join("、")}`;
  if(diagnostic?.gap_ms>250&&q==="good")quality="画面曾中断，当前动作重新观察";
  $("exerciseQuality").textContent=quality;
  $("exerciseQuality").title=diagnostic?`有效采样 ${diagnostic.observation_fps||0} FPS · 最近间隔 ${diagnostic.gap_ms||0} ms`:"";
  $("exerciseQuality").classList.toggle("bad",!["good","waiting_for_input"].includes(q)||diagnostic?.gap_ms>250);
  $("pauseExercise").innerHTML=icon(s.status==="paused"?"play":"pause")+`<span>${s.status==="paused"?"继续":"暂停"}</span>`;
  if(["ended","interrupted"].includes(s.status)){
    $("exerciseCompleted").hidden=false;$("exerciseCompleted").querySelector("strong").textContent=s.status==="ended"?`训练已结束 · ${s.count} 次`:"训练已中断";
    $("pauseExercise").disabled=true;$("endExercise").disabled=true;stopCamera();
  }
  drawPose($("poseCanvas"),s.keypoints,s.mode!=="live");icons();
}
async function pollExercise(){if(!app.sid||exercisePollBusy)return;const sid=app.sid;exercisePollBusy=true;try{const s=await api(`/api/sessions/${sid}`,{timeout:15000});if(app.sid===sid)renderExerciseState(s);}catch(e){}finally{exercisePollBusy=false;}}
function renderAudioHistory(){
  $("soundHistory").innerHTML=app.sounds.length?`<table><thead><tr><th>时间</th><th>任务</th><th>结果</th><th>来源</th></tr></thead><tbody>${app.sounds.map(s=>{
    const labels=(s.result.sounds?.events||s.result.events||[]).map(e=>soundNames[e.category]||e.category).join("、")||"无已确认声音事件";
    const help=(s.result.help_result||s.result).help?"检测到求救":"未检测到明确求救";
    return `<tr><td>${dateTime(s.created_at)}</td><td>${s.task==="monitor"?"联合分析":s.task==="help"?"求救监听":"声音标签"}</td><td>${esc(s.task==="monitor"?`${labels} · ${help}`:s.task==="help"?help:labels)}</td><td>${modeBadge(s.mode)}</td></tr>`;
  }).join("")}</tbody></table>`:empty("暂无声音记录","audio-lines","",true);
  const inquiries=app.events.filter(e=>e.mode==="live"&&e.state==="open").slice(0,8);
  $("inquiryQueue").innerHTML=inquiries.map(e=>`<div class="inquiry-row"><span>${esc(e.room)} · #${e.id.slice(0,6)}</span><span class="badge neutral">${({not_started:"待问询",listening:"等待语音回复",processing:"正在识别回复",completed:"问询已完成",audio_unavailable:"问询音频不可用",button_safe:"已按安全按钮"})[e.inquiry?.status]||"待问询"}</span><button class="text-action" data-event="${e.id}">查看事件 ${icon("arrow-right")}</button></div>`).join("")||empty("暂无待问询事件","circle-check","",true);
}
function renderJointAudioResult(result){
  renderAudioResult(result.sounds,"sounds");const sounds=$("audioResult").innerHTML;
  renderAudioResult(result.help_result,"help");
  $("audioResult").innerHTML=`<h3>环境声音</h3>${sounds}<div class="separator"></div><h3>语音求助</h3>${$("audioResult").innerHTML}`;
  $("audioResultDuration").textContent=`${Number(result.duration_s||0).toFixed(1)} 秒`;icons();
}
function renderLiveAudio(result){
  const degraded=result.pipeline?.status?.startsWith("degraded")||String(result.listener?.status||"").startsWith("degraded");
  $("audioLiveStatus").textContent=degraded?"监听覆盖受限，请检查运行状态":"麦克风已连接 · 声音与求助同时监听";
  for(const id of ["soundLiveState","helpLiveState"]){$(id).textContent=degraded?"覆盖受限":"监听中";$(id).className=`badge ${degraded?"amber":"green"}`;}
  $("liveSoundResult").innerHTML=(result.events||[]).slice().reverse().map(e=>`<div class="sound-event"><span class="sound-type">${icon("audio-lines")}</span><div><h3>${soundNames[e.category]||esc(e.category)}</h3><small>${dateTime(e.t_start_utc)} · ${e.n_windows||e.windows||0} 个窗口</small></div><span class="badge blue">${(e.confidence*100).toFixed(1)}%</span></div>`).join("")||empty("暂无已确认声音事件","volume-1","",true);
  app.liveCalls=[...(app.liveCalls||[]),...(result.calls||[])].slice(-20);
  $("liveHelpResult").innerHTML=app.liveCalls.slice().reverse().map(c=>`<div class="sound-event"><span class="sound-type">${icon("bell-ring")}</span><div><h3>检测到语音求助</h3><small>${dateTime(c.t_utc||new Date().toISOString())}</small></div><span class="badge red">已生成事件</span></div>`).join("")||empty("暂无语音求助","speech","",true);
  icons();
}
function renderAudioResult(result,task){$("audioResultDuration").textContent=result.duration_s?`${result.duration_s} 秒`:"";if(task==="sounds"){$("audioResult").innerHTML=(result.events||[]).map(e=>`<div class="sound-event"><span class="sound-type">${icon("audio-lines")}</span><div><h3>${soundNames[e.category]||esc(e.category)}</h3><small>${e.start_s??dateTime(e.t_start_utc)}${e.end_s!==undefined?` – ${e.end_s} 秒`:""} · ${e.windows||e.n_windows||0} 个窗口</small></div><span class="badge blue">${(e.confidence*100).toFixed(1)}%</span></div>`).join("")||empty("未确认声音事件","volume-1");}else{$("audioResult").innerHTML=`<div class="help-result">${icon(result.help?"bell-ring":"circle-check")}<h3>${result.help?"检测到明确语音求助":"未检测到明确求助"}</h3><p>${(result.segments||[]).length} 个语音片段</p>${(result.segments||[]).map(s=>`<div class="sound-event"><div><h3>${({help:"明确求助",not_help:"非求助意图",not_a_call:"非明确求助话语",cooldown:"处于冷却时间",too_short:"语音过短",too_long:"语音过长",low_confidence:"识别置信度不足",no_text:"未识别到文字",too_many_words:"语句过长"})[s.verdict]||esc(s.verdict)}</h3><small>${s.start_s} 秒 · ${s.duration_s?.toFixed(2)} 秒</small></div></div>`).join("")}</div>`;}icons();}
async function analyzeAudio(ev){ev.preventDefault();await busy($("analyzeAudio"),async()=>{if(!app.settings.audio_consent)throw new Error("请先取得音频处理授权");const task="monitor",source=$("audioSource").value;let response;if(source==="sample"){response=await api("/api/audio/sample",{method:"POST",body:{sample:$("audioSampleSelect").value,task}});}else{const file=source==="record"?app.recordedBlob:$("audioFile").files[0];if(!file)throw new Error(source==="record"?"请先完成录音":"请选择音频文件");const form=new FormData();form.append("file",file,"audio.webm");form.append("task",task);form.append("mode",source==="record"?"live":"replay");response=await api("/api/audio/analyze",{method:"POST",body:form});}renderJointAudioResult(response.result);await refresh();});}
async function recordAudio(){await busy($("recordAudio"),async()=>{
  if(!app.settings.audio_consent)throw new Error("请先取得音频处理授权");
  if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
  const stream=await navigator.mediaDevices.getUserMedia({audio:true});let interval;
  app.recordingStream=stream;
  try{
    const recorder=new MediaRecorder(stream),chunks=[];app.audioRecorder=recorder;
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    await new Promise((resolve,reject)=>{
      recorder.onerror=()=>reject(new Error("录音失败"));
      recorder.onstop=()=>{
        clearInterval(interval);
        if(app.settings.audio_consent){app.recordedBlob=new Blob(chunks,{type:recorder.mimeType});$("audioPreview").src=URL.createObjectURL(app.recordedBlob);}
        else app.recordedBlob=null;
        resolve();
      };
      recorder.start();let remaining=10;$("recordStatus").textContent=`录音中 · ${remaining} 秒`;
      interval=setInterval(()=>{remaining--;$("recordStatus").textContent=`录音中 · ${remaining} 秒`;if(remaining<=0||!app.settings.audio_consent){clearInterval(interval);if(recorder.state!=="inactive")recorder.stop();}},1000);
    });
    $("recordStatus").textContent=app.settings.audio_consent?"录音已就绪":"音频授权已撤回";
  }finally{clearInterval(interval);stream.getTracks().forEach(t=>t.stop());app.audioRecorder=null;app.recordingStream=null;}
});}
async function startAudioLive(){
  await busy($("startAudioLive"),async()=>{
    if(!app.settings.audio_consent)throw new Error("请先取得音频处理授权");
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
    const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    const context=new AudioContext({sampleRate:16000});
    try{
      if(context.sampleRate!==16000)throw new Error("浏览器不支持 16 kHz 音频输入");
      await context.audioWorklet.addModule("/static/pcm-worklet.js");
      const socket=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/audio`);
      app.audioStream=stream;app.audioContext=context;app.audioSocket=socket;
      await new Promise((resolve,reject)=>{
        socket.onmessage=event=>{const r=JSON.parse(event.data);if(r.error)reject(new Error(r.error));else if(r.state==="listening")resolve();};
        socket.onerror=()=>reject(new Error("声音监听连接失败"));
        socket.onclose=()=>reject(new Error("声音监听未连接，请检查授权和模块状态"));
      });
      app.liveCalls=[];
      const queue=new AudioChunkQueue(message=>socket.send(JSON.stringify(message)));app.audioQueue=queue;
      const source=context.createMediaStreamSource(stream),processor=new AudioWorkletNode(context,"homeease-pcm");source.connect(processor);
      const silent=context.createGain();silent.gain.value=0;processor.connect(silent).connect(context.destination);
      processor.port.onmessage=event=>{
        const {samples,timestamp_ms}=event.data;
        const rms=Math.sqrt(samples.reduce((sum,v)=>sum+v*v,0)/samples.length);$("audioLevel").style.width=`${Math.min(100,rms*350)}%`;
        if(!app.settings.audio_consent||socket.readyState!==WebSocket.OPEN||app.audioStopping)return;
        const bytes=new Uint8Array(samples.buffer);let binary="";for(let i=0;i<bytes.length;i++)binary+=String.fromCharCode(bytes[i]);
        try{queue.push({pcm:btoa(binary),timestamp_ms});}catch(e){toast(e.message,true);queue.close();socket.close();stopAudioLive();}
      };
      socket.onmessage=event=>{
        const result=JSON.parse(event.data);
        if(result.error){toast(result.error,true);queue.close();stopAudioLive();return;}renderLiveAudio(result);
        queue.acknowledge();
      };
      socket.onclose=()=>{queue.close();stopAudioLive();};socket.onerror=()=>toast("麦克风连接中断",true);
      $("startAudioLive").disabled=true;$("stopAudioLive").disabled=false;renderLiveAudio({});await context.resume();
    }catch(e){stream.getTracks().forEach(t=>t.stop());await context.close();app.audioSocket?.close();app.audioSocket=null;app.audioStream=null;app.audioContext=null;throw e;}
  });
  if(app.audioStream)$("startAudioLive").disabled=true;
}
async function stopAudioLive(){
  if(app.audioStopping)return app.audioStopping;
  app.audioStopping=(async()=>{
    const socket=app.audioSocket;app.audioSocket=null;
    $("startAudioLive").disabled=true;$("stopAudioLive").disabled=true;
    app.audioStream?.getTracks().forEach(t=>t.stop());app.audioStream=null;
    if(app.audioContext&&app.audioContext.state!=="closed")await app.audioContext.close();
    app.audioContext=null;
    if(socket?.readyState===WebSocket.OPEN){
      $("audioLiveStatus").textContent="正在结束监听";
      let timer;
      await Promise.race([app.audioQueue?.drain(),new Promise(resolve=>{timer=setTimeout(()=>{socket.close();resolve();},10000);})]);
      clearTimeout(timer);
    }
    if(socket?.readyState===WebSocket.OPEN){
      await new Promise(resolve=>{
        const timer=setTimeout(()=>{socket.close();resolve();},18000);
        const done=()=>{clearTimeout(timer);resolve();};
        socket.onclose=done;socket.onmessage=event=>{if(JSON.parse(event.data).state==="stopped"){socket.close();done();}};
        socket.send(JSON.stringify({type:"stop"}));
      });
    }else socket?.close();
    app.audioQueue?.close();app.audioQueue=null;$("audioLevel").style.width="0%";
    $("audioLiveStatus").textContent="麦克风未连接";$("startAudioLive").disabled=false;
    for(const id of ["soundLiveState","helpLiveState"]){$(id).textContent="未监听";$(id).className="badge neutral";}
  })();
  try{await app.audioStopping;}finally{app.audioStopping=null;}
}

async function loadReport(overview=false){try{const report=await api(`/api/reports?days=${overview?7:app.reportDays}&mode=${overview?"live":app.reportMode}`);if(overview){app.report=report;if(app.page==="overview")drawBars($("overviewChart"),report.series,"reps","#579579");return;}app.report=report;renderReport();}catch(e){toast(e.message,true);}}
function renderReport(){const r=app.report;if(!r)return;const s=r.summary;$("reportRange").textContent=`${r.series[0].date} 至 ${r.series.at(-1).date} · ${r.mode==="live"?"实时记录":"样例 / 回放"}`;$("reportMetrics").innerHTML=metric("确认跌倒",s.confirmed_falls,"次","shield-alert",`${s.false_alarms} 次误报`)+metric("未处置事件",s.open_events,"条","bell-ring",`${s.unresolved_outcomes} 条结果未确定`)+metric("已完成训练",s.sessions,"次","activity",`${s.participation_days} 天参与`)+metric("有效运动次数",s.reps,"次","check-check","已结束训练累计");drawBars($("safetyChart"),r.series,"events","#bc8d58");drawBars($("exerciseReportChart"),r.series,"reps","#5d83bb");$("coverageSummary").innerHTML=r.mode==="live"?Object.entries(s.coverage_unavailable_s).map(([ch,n])=>`<div><span class="muted small">${channels[ch]}受限 / 不可用时长</span><strong>${duration(n)}</strong></div>`).join(""):"";$("coverageHistory").innerHTML=r.coverage.length?`<table><thead><tr><th>通道</th><th>开始</th><th>结束</th><th>状态</th><th>原因</th></tr></thead><tbody>${r.coverage.slice(0,30).map(c=>`<tr><td>${channels[c.channel]}</td><td>${dateTime(c.started_at)}</td><td>${c.ended_at?dateTime(c.ended_at):"持续中"}</td><td><span class="badge ${c.state==="active"?"green":"amber"}">${c.state==="active"?"有效":c.state==="reduced"?"受限":"不可用"}</span></td><td>${esc(c.reason)}</td></tr>`).join("")}</tbody></table>`:empty(r.mode==="replay"?"回放不计入实时监测覆盖":"暂无覆盖记录","radio","",true);$("exerciseSummary").innerHTML=`<div class="exercise-summary-grid">${Object.entries(s.reps_by_exercise).map(([key,n])=>`<div>${names[key]}<strong>${n}<small>次</small></strong></div>`).join("")}</div>`;$("exportCsv").href=`/api/reports/export.csv?days=${app.reportDays}&mode=${app.reportMode}`;$("printReport").href=`/reports/print?days=${app.reportDays}&mode=${app.reportMode}`;icons();}
function contactEditor(c={name:"",relation:"家属",phone:""}){const div=document.createElement("div");div.className="contact-editor";div.innerHTML=`<label>姓名<input class="contact-name" value="${esc(c.name)}" maxlength="60" required></label><label>关系<input class="contact-relation" value="${esc(c.relation)}" maxlength="30"></label><label>联系电话<input class="contact-phone" type="tel" value="${esc(c.phone)}" maxlength="40" pattern="[+0-9 ()-]+" required></label><button class="icon-button remove-contact" type="button" title="移除联系人" aria-label="移除联系人">${icon("trash-2")}</button>`;$("contactsEditor").append(div);icons();}
function fillSettings(s){app.settings=s;$("residentName").value=s.resident.name;$("residentRoom").value=s.resident.room;$("residentNotes").value=s.resident.notes;$("audioConsent").checked=s.audio_consent;$("escalationSeconds").value=s.escalation_seconds;$("defaultTarget").value=s.exercise_target;$("targetReps").value=s.exercise_target;$("webhookUrl").value=s.webhook_url;$("contactsEditor").innerHTML="";s.contacts.forEach(contactEditor);$("consentStatusText").textContent=s.audio_consent?"已授权":"未授权";$("notificationTestStatus").textContent=s.notification_test_at?`最近测试 ${dateTime(s.notification_test_at)}`:"尚未测试";$("setupBanner").hidden=!!(s.resident.name&&s.contacts.length);$("overviewSubtitle").textContent=s.resident.name?`${s.resident.name} · 今天的安全与活动记录`:"今天的安全与活动记录";$("overviewRoom").textContent=s.resident.room.toUpperCase()+" / HOME CARE";$("cameraRoomLabel").textContent=s.resident.room;}
async function saveSettings(ev){ev.preventDefault();await busy($("saveSettings"),async()=>{const contacts=Array.from(document.querySelectorAll(".contact-editor")).map(el=>({name:el.querySelector(".contact-name").value.trim(),relation:el.querySelector(".contact-relation").value.trim(),phone:el.querySelector(".contact-phone").value.trim()}));const value={resident:{name:$("residentName").value.trim(),room:$("residentRoom").value.trim(),notes:$("residentNotes").value.trim()},contacts,audio_consent:$("audioConsent").checked,exercise_target:Number($("defaultTarget").value),escalation_seconds:Number($("escalationSeconds").value),webhook_url:$("webhookUrl").value.trim()};if(!value.audio_consent)await stopAudioLive();const result=await api("/api/settings",{method:"PUT",body:value});fillSettings(result);toast("设置已保存");await refresh();});}
function renderModules(){$("moduleList").innerHTML=app.status.modules.map(m=>`<div class="module-row"><div><h3>${moduleNames[m.id]}</h3><small>${modelNames[m.id]}</small>${m.error?`<small style="color:var(--red)">${esc(m.error)}</small>`:""}</div><span class="badge ${m.state==="ready"?"green":m.state==="unavailable"?"red":"amber"}">${stateNames[m.state]||m.state}</span><button class="icon-button" type="button" data-init="${m.id}" title="重新初始化模块" aria-label="重新初始化${moduleNames[m.id]}" ${m.state==="loading"?"disabled":""}>${icon("rotate-cw")}</button></div>`).join("");icons();}
async function loadDiagnostics(){try{const d=await api("/api/diagnostics");$("diagnosticsList").innerHTML=d.logs.length?d.logs.slice(0,30).map(x=>`<div class="diagnostic-item"><time>${dateTime(x.at)}</time><span>${esc(x.module)} · ${esc(x.action)}${x.payload.error?`<br>${esc(x.payload.error)}`:""}</span></div>`).join(""):empty("暂无运行记录","list", "",true);}catch(e){toast(e.message,true);}}

function canvasContext(canvas){if(!canvas)return null;const rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return null;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height};}
function chartBase(canvas,max,labels=[]){const setup=canvasContext(canvas);if(!setup)return null;const {ctx,w,h}=setup;const area={l:35,r:w-10,t:13,b:h-29};ctx.clearRect(0,0,w,h);ctx.font='10px "Segoe UI","Microsoft YaHei",sans-serif';ctx.textBaseline="middle";for(let i=0;i<4;i++){const y=area.t+(area.b-area.t)*i/3;ctx.strokeStyle="#e4ebe6";ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(area.l,y);ctx.lineTo(area.r,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#9ba99f";ctx.textAlign="right";const val=max*(1-i/3);ctx.fillText(max<=1?val.toFixed(1):String(Math.round(val)),area.l-9,y);}ctx.textAlign="center";if(labels.length){const step=Math.max(1,Math.ceil(labels.length/7));labels.forEach((label,i)=>{if(i%step===0||i===labels.length-1){const x=area.l+(area.r-area.l)*(i+.5)/labels.length;ctx.fillStyle="#98a89c";ctx.fillText(label,x,h-11);}});}return {...setup,area};}
function drawBars(canvas,series,key,color){const max=Math.max(3,...series.map(x=>x[key]||0));const setup=chartBase(canvas,max,series.map(x=>x.date.slice(5).replace("-","/")));if(!setup)return;const {ctx,area}=setup;const slot=(area.r-area.l)/series.length;series.forEach((s,i)=>{const n=s[key]||0;const bar=Math.max(2,n/max*(area.b-area.t));const width=Math.min(27,slot*.44);const x=area.l+slot*(i+.5)-width/2;ctx.fillStyle=n?color:"#e6ece8";ctx.beginPath();ctx.roundRect(x,area.b-bar,width,bar,[3,3,0,0]);ctx.fill();if(n){ctx.fillStyle=color;ctx.font="10px Segoe UI";ctx.fillText(String(n),x+width/2,area.b-bar-9);}});}
function drawLines(canvas,series,options={}){const all=series.flatMap(s=>s.values||[]).filter(Number.isFinite);let min=options.min??Math.min(0,...all),max=options.max??Math.max(1,...all);if(max===min)max=min+1;const setup=chartBase(canvas,max,options.labels||[]);if(!setup)return;const {ctx,area}=setup;const y=v=>area.b-(v-min)/(max-min)*(area.b-area.t);if(options.threshold!==undefined){ctx.strokeStyle="#c39c68";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(area.l,y(options.threshold));ctx.lineTo(area.r,y(options.threshold));ctx.stroke();ctx.setLineDash([]);}for(const s of series){if(!s.values?.length)continue;ctx.strokeStyle=s.color;ctx.lineWidth=1.7;ctx.beginPath();s.values.forEach((v,i)=>{const x=area.l+(area.r-area.l)*i/Math.max(1,s.values.length-1);if(i===0)ctx.moveTo(x,y(v));else ctx.lineTo(x,y(v));});ctx.stroke();}if(!all.length){ctx.textAlign="center";ctx.fillStyle="#9aaba0";ctx.font='11px "Microsoft YaHei"';ctx.fillText("等待数据",(area.l+area.r)/2,(area.t+area.b)/2);}}
function drawPose(canvas,points,background=true){const setup=canvasContext(canvas);if(!setup)return;const {ctx,w,h}=setup;ctx.clearRect(0,0,w,h);if(background){ctx.fillStyle="#eaf1ed";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#dbe6de";ctx.lineWidth=1;for(let x=0;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}}if(!points)return;const visible=points.filter(p=>p[2]>=.3);if(!visible.length)return;let minX,maxX,minY,maxY;if(background){minX=Math.min(...visible.map(p=>p[0]))-80;maxX=Math.max(...visible.map(p=>p[0]))+80;minY=Math.min(...visible.map(p=>p[1]))-60;maxY=Math.max(...visible.map(p=>p[1]))+60;}else{minX=0;maxX=app.exercise?.frame_shape?.[1]||640;minY=0;maxY=app.exercise?.frame_shape?.[0]||480;}const scale=Math.min(w/(maxX-minX),h/(maxY-minY));const dx=(w-(maxX-minX)*scale)/2-minX*scale,dy=(h-(maxY-minY)*scale)/2-minY*scale;const point=p=>[p[0]*scale+dx,p[1]*scale+dy];const edges=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];ctx.lineWidth=background?5:3;ctx.lineCap="round";for(const [a,b] of edges){if(points[a]?.[2]<.3||points[b]?.[2]<.3)continue;ctx.strokeStyle=a%2?"#338567":"#5a84ba";ctx.beginPath();ctx.moveTo(...point(points[a]));ctx.lineTo(...point(points[b]));ctx.stroke();}for(const p of visible){ctx.fillStyle="#f9fcfa";ctx.strokeStyle="#4b876f";ctx.lineWidth=2;ctx.beginPath();ctx.arc(...point(p),background?5:3.5,0,Math.PI*2);ctx.fill();ctx.stroke();}}

document.addEventListener("DOMContentLoaded",async()=>{
  icons();$("headerDate").textContent=new Date().toLocaleDateString("zh-CN",{month:"long",day:"numeric",weekday:"long"});
  $("menuButton").onclick=()=>{$("sidebar").classList.toggle("open");$("navBackdrop").hidden=!$("sidebar").classList.contains("open");};$("navBackdrop").onclick=()=>{$("sidebar").classList.remove("open");$("navBackdrop").hidden=true;};
  window.addEventListener("hashchange",navigate);document.querySelectorAll("[data-close]").forEach(b=>b.onclick=()=>$(b.dataset.close).close());document.querySelectorAll("dialog").forEach(d=>d.addEventListener("click",e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
  document.addEventListener("click",async ev=>{const eventEl=ev.target.closest("[data-event]");if(eventEl)openEvent(eventEl.dataset.event);const action=ev.target.closest("[data-action]");if(action)busy(action,()=>eventAction(action.dataset.action));const contact=ev.target.closest("[data-contact-call]");if(contact&&app.selectedEvent?.state!=="resolved")api(`/api/events/${app.selectedEvent.id}/action`,{method:"POST",body:{action:"contact",note:"已发起电话联系"}}).catch(e=>toast(e.message,true));const init=ev.target.closest("[data-init]");if(init)busy(init,async()=>{await api(`/api/modules/${init.dataset.init}/initialize`,{method:"POST"});toast("模块正在重新初始化");await refresh();});const remove=ev.target.closest(".remove-contact");if(remove)remove.closest(".contact-editor").remove();});document.addEventListener("keydown",ev=>{if(["Enter"," "].includes(ev.key)&&ev.target.matches("[data-event]")){ev.preventDefault();openEvent(ev.target.dataset.event);}});
  $("notificationButton").onclick=async()=>{await refresh();$("notificationList").innerHTML=app.notices.length?app.notices.slice(0,30).map(n=>`<div class="notification-row" data-notice="${n.id}" ${n.event_id?`data-event="${n.event_id}"`:""}><strong>${esc(n.title)}</strong><p>${esc(n.message)}</p><footer><span>${dateTime(n.created_at)}</span><span>${n.kind==="webhook"?"Webhook":"站内"} · ${({queued:"排队中",delivered:"已送达",seen:"已查看",retry:"重试中",failed:"失败"})[n.status]||n.status}</span></footer></div>`).join(""):empty("暂无通知","bell");$("notificationsDialog").showModal();$("notificationList").querySelectorAll("[data-notice]").forEach(el=>el.addEventListener("click",()=>{api(`/api/notifications/${el.dataset.notice}/seen`,{method:"POST"});$("notificationsDialog").close();}));icons();};
  $("videoSource").onchange=()=>{$("cameraIndexField").hidden=$("videoSource").value!=="camera";$("videoUploadField").hidden=$("videoSource").value!=="upload";};$("videoForm").onsubmit=ev=>{ev.preventDefault();busy($("startVideo"),async()=>{let upload_id=null;if($("videoSource").value==="upload"){const file=$("videoUpload").files[0];if(!file)throw new Error("请选择视频文件");const form=new FormData();form.append("file",file);$("videoUploadStatus").textContent="正在上传视频…";try{const result=await api("/api/uploads/video",{method:"POST",body:form});upload_id=result.upload_id;$("videoUploadStatus").textContent=result.filename;}catch(e){$("videoUploadStatus").textContent="上传失败";throw e;}}app.videoHistory=[];await api("/api/video/start",{method:"POST",body:{source:$("videoSource").value,camera_index:Number($("videoForm").elements.camera_index.value),upload_id}});await refresh();toast("检测已启动");});};$("stopVideo").onclick=()=>busy($("stopVideo"),async()=>{await api("/api/video/stop",{method:"POST"});await refresh();});
  $("signalSource").onchange=()=>$("signalUploadField").hidden=$("signalSource").value!=="upload";$("analyzeSignal").onclick=()=>busy($("analyzeSignal"),async()=>{let result;if($("signalSource").value==="sample")result=await api("/api/signal/sample",{method:"POST"});else{const file=$("signalFile").files[0];if(!file)throw new Error("请选择 CSV 文件");const form=new FormData();form.append("file",file);result=await api("/api/signal/analyze",{method:"POST",body:form});}drawLines($("signalChart"),[{values:result.waveform.ax,color:"#547bb4"},{values:result.waveform.ay,color:"#5a9877"},{values:result.waveform.az,color:"#bf9459"}],{labels:result.waveform.t.map(t=>`${t.toFixed(1)}s`)});drawLines($("signalProbabilityChart"),[{values:result.info.probs,color:"#52886d"}],{min:0,max:1,threshold:.5,labels:result.info.t_end.map(t=>`${t.toFixed(1)}s`)});$("signalState").textContent="回放分析完成";$("signalResult").innerHTML=`${result.info.n_windows} 个窗口 · ${result.records.length} 个疑似跌倒事件 ${result.records.map(e=>`<button class="text-action" data-event="${e.id}">查看 #${e.id.slice(0,6)} ${icon("arrow-right")}</button>`).join(" ")}`;await refresh();icons();});
  $("exerciseOptions").innerHTML=Object.entries(names).map(([key,name],index)=>`<button type="button" class="exercise-option ${index===0?"selected":""}" data-exercise="${key}" aria-pressed="${index===0}"><img src="/static/media/${key}.png" alt="${name}骨架动作"><span class="selection-mark">${icon("check")}</span><div><strong>${name}</strong><small>${["坐姿 / 站立完整周期","下蹲 / 起立完整周期","抬腿 / 回落完整周期","跨步 / 复位完整周期"][index]}</small></div></button>`).join("");document.querySelectorAll("[data-exercise]").forEach(b=>b.onclick=()=>chooseExercise(b.dataset.exercise));$("exerciseSource").onchange=()=>{$("exerciseUploadField").hidden=$("exerciseSource").value!=="upload";$("exerciseCameraField").hidden=$("exerciseSource").value!=="camera";};$("exerciseForm").onsubmit=startExercise;
  $("pauseExercise").onclick=async()=>{await busy($("pauseExercise"),()=>controlExercise(app.exercise?.status==="paused"?"resume":"pause"));syncExerciseControls();};$("endExercise").onclick=async()=>{await busy($("endExercise"),()=>controlExercise("end"));syncExerciseControls();};$("newExercise").onclick=()=>{app.sid=null;app.exercise=null;$("exerciseActive").hidden=true;$("exerciseChooser").hidden=false;};
  $("audioSource").onchange=()=>{const source=$("audioSource").value;$("audioSampleField").hidden=source!=="sample";$("audioFileField").hidden=source!=="upload";$("audioRecordField").hidden=source!=="record";$("audioPreview").removeAttribute("src");if(source==="sample")$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;};$("audioSampleSelect").onchange=()=>$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;$("audioFile").onchange=()=>{const file=$("audioFile").files[0];if(file)$("audioPreview").src=URL.createObjectURL(file);};$("audioForm").onsubmit=analyzeAudio;$("recordAudio").onclick=recordAudio;$("startAudioLive").onclick=startAudioLive;$("stopAudioLive").onclick=stopAudioLive;
  $("eventStateFilter").querySelectorAll("button").forEach(b=>b.onclick=()=>{app.eventFilter=b.dataset.value;$("eventStateFilter").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));renderEvents();});$("eventModeFilter").onchange=()=>{app.eventMode=$("eventModeFilter").value;renderEvents();};$("eventSearch").oninput=renderEvents;$("reportDays").querySelectorAll("button").forEach(b=>b.onclick=()=>{app.reportDays=Number(b.dataset.value);$("reportDays").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));loadReport();});$("reportMode").onchange=()=>{app.reportMode=$("reportMode").value;loadReport();};
  $("settingsForm").onsubmit=saveSettings;$("addContact").onclick=()=>{if(document.querySelectorAll(".contact-editor").length>=10){toast("最多添加 10 位联系人",true);return;}contactEditor();};$("audioConsent").onchange=()=>$("consentStatusText").textContent=$("audioConsent").checked?"待保存授权":"待保存撤回";$("testNotification").onclick=()=>busy($("testNotification"),async()=>{await api("/api/notifications/test",{method:"POST"});if("Notification" in window&&Notification.permission==="default")await Notification.requestPermission();toast("测试通知已加入通知中心");app.settings=await api("/api/settings");$("notificationTestStatus").textContent=`最近测试 ${dateTime(app.settings.notification_test_at)}`;await refresh();});
  try{fillSettings(await api("/api/settings"));const samples=await api("/api/audio/samples");$("audioSampleSelect").innerHTML=[...(samples.scenes||[]),...(samples.replies||[])].map(s=>`<option value="${s.id}">${esc(s.title)}</option>`).join("");$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;}catch(e){toast(e.message,true);}navigate();await refresh();icons();setInterval(refresh,2500);setInterval(pollExercise,600);window.addEventListener("resize",()=>{if(app.page==="overview"&&app.report)drawBars($("overviewChart"),app.report.series,"reps","#579579");if(app.page==="monitor")renderMonitor();if(app.page==="reports")renderReport();if(app.exercise)drawPose($("poseCanvas"),app.exercise.keypoints,app.exercise.mode!=="live");});
  window.addEventListener("pagehide",()=>{app.camera?.getTracks().forEach(t=>t.stop());app.audioStream?.getTracks().forEach(t=>t.stop());app.recordingStream?.getTracks().forEach(t=>t.stop());if(app.sid&&["running","paused"].includes(app.exercise?.status)&&app.exerciseSource==="camera")fetch(`/api/sessions/${app.sid}/control`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:"end"}),keepalive:true});});
});
