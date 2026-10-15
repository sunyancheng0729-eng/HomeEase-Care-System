const {chromium}=require('playwright');
const fs=require('fs');
const path=require('path');
const assert=require('assert');

const base=process.env.HOMEEASE_TEST_URL||'http://127.0.0.1:8011';
const out=path.resolve(__dirname,'../artifacts/browser');
fs.mkdirSync(out,{recursive:true});

async function waitJson(page,endpoint,predicate,timeout=120000){
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline){
    const response=await page.request.get(`${base}${endpoint}`);
    assert(response.ok(),`${endpoint}: ${response.status()}`);
    const data=await response.json();
    if(predicate(data))return data;
    await page.waitForTimeout(200);
  }
  throw new Error(`Timed out waiting for ${endpoint}`);
}

async function checkLayout(page,label){
  const result=await page.evaluate(()=>{
    const overflow=document.documentElement.scrollWidth>innerWidth+1;
    const badImages=Array.from(document.images).filter(x=>x.offsetWidth&&(!x.complete||x.naturalWidth===0)).map(x=>x.src);
    const clippedText=Array.from(document.querySelectorAll('button,h1,h2,h3,.badge,.metric,.brand')).filter(el=>{
      const r=el.getBoundingClientRect();
      return r.width>0&&r.height>0&&el.scrollWidth>el.clientWidth+3;
    }).map(x=>({tag:x.tagName,text:x.textContent.slice(0,70),w:x.clientWidth,sw:x.scrollWidth}));
    return {overflow,badImages,clippedText};
  });
  assert(!result.overflow,`${label}: horizontal overflow`);
  assert.deepStrictEqual(result.badImages,[],`${label}: missing images`);
  assert.deepStrictEqual(result.clippedText,[],`${label}: clipped text`);
  return result;
}

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
  const context=await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1,permissions:['camera','microphone']});
  const page=await context.newPage();
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=500)errors.push(`${r.status()} ${r.url()}`);});
  const results={pages:[],workflows:{},errors};
  try{
    await page.goto(base,{waitUntil:'networkidle'});
    await waitJson(page,'/api/health',h=>h.modules.every(m=>m.state==='ready'));
    await page.waitForFunction(()=>document.getElementById('connectionState').textContent.includes('已连接'));
    for(const view of ['overview','monitor','exercise','audio','events','reports','settings']){
      await page.goto(`${base}/#${view}`);
      await page.locator(`#page-${view}`).waitFor({state:'visible'});
      await page.waitForTimeout(350);
      results.pages.push({view,viewport:'desktop',...await checkLayout(page,`desktop-${view}`)});
      await page.screenshot({path:path.join(out,`desktop-${view}.png`),fullPage:true});
    }
    await page.setViewportSize({width:1280,height:800});
    for(const view of ['overview','monitor','exercise','audio','events','reports','settings']){
      await page.goto(`${base}/#${view}`);
      await page.locator(`#page-${view}`).waitFor({state:'visible'});
      await page.waitForTimeout(250);
      results.pages.push({view,viewport:'compact-desktop',...await checkLayout(page,`compact-desktop-${view}`)});
      await page.screenshot({path:path.join(out,`compact-desktop-${view}.png`),fullPage:true});
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.goto(`${base}/#exercise`);
    await page.selectOption('#exerciseSource','replay');
    await page.click('#startExercise');
    await page.locator('#exerciseActive').waitFor({state:'visible',timeout:60000});
    await page.waitForFunction(()=>!document.getElementById('exerciseCompleted').hidden,null,{timeout:120000});
    const reps=Number(await page.locator('#repCount').textContent());
    assert(reps>0,'Original TCN replay produced no repetitions');
    const canvas=await page.evaluate(()=>{const c=document.getElementById('poseCanvas');const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let nonzero=0;for(let i=3;i<d.length;i+=4)if(d[i])nonzero++;return {w:c.width,h:c.height,nonzero};});
    assert(canvas.nonzero>10000,'Skeleton canvas is blank');
    results.workflows.exercise={reps,canvas};
    await page.screenshot({path:path.join(out,'exercise-replay-result.png'),fullPage:true});
    await page.goto(`${base}/#monitor`);
    assert(await page.locator('#videoTab').isVisible());
    assert(await page.locator('#signalTab').isVisible());
    await page.locator('#signalTab details summary').click();
    await page.selectOption('#signalSource','sample');
    await page.click('#analyzeSignal');
    await page.waitForFunction(()=>document.getElementById('signalState').textContent.includes('完成'),null,{timeout:60000});
    results.workflows.signal=await page.locator('#signalResult').innerText();
    await page.screenshot({path:path.join(out,'signal-result.png'),fullPage:true});
    await page.locator('#signalResult [data-event]').first().click();
    await page.locator('#eventDialog').waitFor({state:'visible'});
    await page.selectOption('#resolveOutcome','false_alarm');
    await page.fill('#resolveNote','浏览器集成测试：样例回放');
    await page.locator('#resolveForm button[type=submit]').click();
    await page.waitForFunction(()=>document.querySelector('#eventDetail .completion-banner'));
    results.workflows.eventResolution=true;
    await page.screenshot({path:path.join(out,'event-detail.png'),fullPage:true});
    await page.locator('[data-close="eventDialog"]').click();
    await page.goto(`${base}/#monitor`);
    assert(await page.locator('#signalTab').isVisible());
    await page.selectOption('#videoSource','sample');
    const videoStarted=page.waitForResponse(response=>response.request().method()==='POST'&&response.url().endsWith('/api/video/start'));
    await page.click('#startVideo');
    const startedResponse=await videoStarted;
    assert(startedResponse.ok(),'Video start failed');
    const startedSession=(await startedResponse.json()).session;
    await waitJson(page,'/api/status',s=>s.video.session===startedSession&&s.video.state==='finished'&&!s.video.processing);
    const videoEvents=await waitJson(page,'/api/events?mode=replay',e=>e.some(x=>x.channel==='vision'&&x.evidence.clip),20000);
    const videoEvent=videoEvents.find(x=>x.channel==='vision'&&x.evidence.clip);
    assert(videoEvent,'Video replay produced no evidence');
    await page.goto(`${base}/#events`);
    await page.selectOption('#eventModeFilter','replay');
    await page.locator(`[data-event="${videoEvent.id}"]`).click();
    await page.locator('.event-video').waitFor();
    await page.waitForFunction(()=>document.querySelector('.event-video')?.readyState>=2,null,{timeout:30000});
    const playback=await page.locator('.event-video').evaluate(async v=>{await v.play();await new Promise(r=>setTimeout(r,500));v.pause();return {duration:v.duration,time:v.currentTime,width:v.videoWidth};});
    assert(playback.time>0&&playback.width>0,'Evidence video did not play');
    results.workflows.videoPlayback=playback;
    await page.screenshot({path:path.join(out,'video-evidence.png'),fullPage:true});
    await page.locator('[data-close="eventDialog"]').click();
    await page.evaluate(()=>fetch('/api/video/stop',{method:'POST'}));
    await page.goto(`${base}/#exercise`);
    await page.click('#newExercise');
    await page.selectOption('#exerciseSource','camera');
    await page.click('#startExercise');
    await page.waitForFunction(()=>document.getElementById('exerciseCamera').naturalWidth>0,null,{timeout:60000});
    await page.evaluate(async()=>{
      const response=await fetch('/api/video/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:'camera',camera_index:0})});
      if(!response.ok)throw new Error(await response.text());
    });
    await page.locator('nav [data-page="monitor"]').click();
    await page.waitForFunction(()=>document.getElementById('monitorFrame').naturalWidth>0&&document.getElementById('monitorFrame').src.includes('/stream'),null,{timeout:30000});
    const preview=await page.evaluate(async()=>{
      const image=document.getElementById('monitorFrame'),canvas=document.createElement('canvas');
      canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
      const ctx=canvas.getContext('2d'),hashes=[],start=performance.now();let nonzero=0;
      for(let frame=0;frame<15;frame++){
        ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
        let hash=0;nonzero=0;
        for(let i=0;i<pixels.length;i+=16){hash=(Math.imul(hash,31)+pixels[i])|0;if(pixels[i]>0)nonzero++;}
        hashes.push(hash);await new Promise(resolve=>setTimeout(resolve,200));
      }
      return {uniqueFrames:new Set(hashes).size,sampleSeconds:(performance.now()-start)/1000,nonzero,width:canvas.width,height:canvas.height};
    });
    assert(preview.uniqueFrames>=7,'Video preview still updates too slowly');
    assert(preview.nonzero>1000,'Live video preview is blank');
    results.workflows.concurrentLivePreview=preview;
    await page.screenshot({path:path.join(out,'concurrent-live-monitor.png'),fullPage:true});
    await page.evaluate(()=>fetch('/api/video/stop',{method:'POST'}));
    await page.locator('nav [data-page="exercise"]').click();
    await page.waitForTimeout(1800);
    await page.click('#pauseExercise');
    await page.waitForFunction(()=>document.getElementById('pauseExercise').textContent.includes('继续'));
    await page.click('#pauseExercise');
    await page.waitForFunction(()=>document.getElementById('pauseExercise').textContent.includes('暂停'));
    await page.waitForTimeout(1800);
    await page.click('#endExercise');
    await page.waitForFunction(()=>!document.getElementById('exerciseCompleted').hidden);
    assert(await page.locator('#pauseExercise').isDisabled());
    assert(await page.evaluate(()=>document.getElementById('exerciseCamera').hidden&&!document.getElementById('exerciseCamera').getAttribute('src')));
    results.workflows.serverCameraLifecycle=true;
    const profile=await page.evaluate(()=>fetch('/api/settings').then(r=>r.json()));
    await page.evaluate(async p=>{await fetch('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({...p,audio_consent:true})});},profile);
    await page.goto(`${base}/#audio`);
    await page.reload({waitUntil:'networkidle'});
    await page.locator('.audio-replay summary').click();
    await page.selectOption('#audioSource','sample');
    await page.selectOption('#audioSampleSelect','scene_quiet_home');
    await page.click('#analyzeAudio');
    await page.waitForFunction(()=>document.querySelector('#audioResult .sound-event'),null,{timeout:60000});
    await page.click('#startAudioLive');
    await page.waitForFunction(()=>document.getElementById('soundLiveState').textContent==='监听中'&&document.getElementById('helpLiveState').textContent==='监听中',null,{timeout:30000});
    await page.waitForTimeout(2500);
    await page.click('#stopAudioLive');
    await page.waitForFunction(()=>document.getElementById('audioLiveStatus').textContent==='麦克风未连接',null,{timeout:25000});
    assert(await page.locator('#startAudioLive').isEnabled());
    await page.click('#startAudioLive');
    await page.waitForFunction(()=>document.getElementById('soundLiveState').textContent==='监听中'&&document.getElementById('helpLiveState').textContent==='监听中',null,{timeout:30000});
    await page.click('#stopAudioLive');
    await page.waitForFunction(()=>document.getElementById('audioLiveStatus').textContent==='麦克风未连接',null,{timeout:25000});
    results.workflows.fakeMicrophoneAndRestart=true;
    await page.addInitScript(()=>{Object.defineProperty(window,'speechSynthesis',{value:{speak(utterance){setTimeout(()=>utterance.onend?.(),100);},cancel(){}}});});
    await page.goto(`${base}/elder`);
    await page.waitForTimeout(400);
    await page.click('#elderHelp');
    await page.locator('#helpConfirm').waitFor({state:'visible'});
    await page.locator('#elderEvent').waitFor({state:'visible'});
    await page.click('#elderSafe');
    await page.waitForFunction(()=>document.getElementById('elderInquiryResult').textContent.includes('等待照护者核实'));
    await page.click('#elderInquiry');
    await page.waitForFunction(()=>!document.getElementById('inquiryProgress').hidden);
    await page.waitForFunction(()=>document.getElementById('inquiryProgress').hidden&&!document.getElementById('elderInquiryResult').hidden,null,{timeout:90000});
    results.workflows.elderHelpSafeAndVoice=true;
    results.pages.push({view:'elder',viewport:'desktop',...await checkLayout(page,'elder-desktop')});
    await page.setViewportSize({width:1280,height:800});
    results.pages.push({view:'elder',viewport:'compact-desktop',...await checkLayout(page,'elder-compact-desktop')});
    await page.screenshot({path:path.join(out,'elder-compact-desktop.png'),fullPage:true});
    await page.goto(`${base}/sensor`);
    await page.fill('#sensorSamples',JSON.stringify(Array.from({length:60},(_,i)=>[i*.02,0,0,1])));
    await page.click('#sendSamples');
    await page.waitForFunction(()=>document.getElementById('sensorConnection').textContent.includes('已连接'));
    await page.click('#disconnectSensor');
    await page.waitForFunction(()=>document.getElementById('sensorConnection').textContent==='未连接');
    results.workflows.sensorPageTransport=true;
    results.pages.push({view:'sensor',viewport:'compact-desktop',...await checkLayout(page,'sensor-desktop')});
    await page.screenshot({path:path.join(out,'sensor-desktop.png'),fullPage:true});
    await page.goto(`${base}/reports/print?mode=replay`);
    assert(await page.locator('h1').textContent(),'Report not rendered');
    await page.pdf({path:path.join(out,'sample-report.pdf'),format:'A4',printBackground:true});
    results.workflows.pdf=true;
    await page.goto(`${base}/#reports`);
    await page.selectOption('#reportMode','replay');
    const [download]=await Promise.all([page.waitForEvent('download'),page.click('#exportCsv')]);
    await download.saveAs(path.join(out,'sample-report.csv'));
    results.workflows.csv=true;
    await page.evaluate(async p=>{await fetch('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});},profile);
    assert.deepStrictEqual(errors,[],'Browser errors');
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(results,null,2));
    console.log(JSON.stringify(results,null,2));
  }finally{
    try{
      const sessions=await (await context.request.get(`${base}/api/sessions`)).json();
      for(const session of sessions.filter(s=>['running','paused'].includes(s.status)))await context.request.post(`${base}/api/sessions/${session.session_id}/control`,{data:{action:'end'}});
    }catch{}
    try{await context.request.post(`${base}/api/video/stop`);}catch{}
    await context.close();await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
