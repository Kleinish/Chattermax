let rejects=[];
let reviewPos=0;
let currentDetail=null;
let logSource=null;
let referenceReady=false;
let uploadingReference=false;
let jobRunning=false;
let cleaningReference=false;

const $=id=>document.getElementById(id);

async function jfetch(url,opts={}){
  let r=await fetch(url,{headers:{'Content-Type':'application/json',...(opts.headers||{})},...opts});
  if(!r.ok)throw new Error(await r.text());
  return r.json();
}

function updateGenerationButton(){
  const btn=$('startBtn');
  if(!btn)return;
  btn.disabled=jobRunning || uploadingReference || cleaningReference || !referenceReady;
  if(uploadingReference)btn.textContent='Reference Uploading…';
  else if(cleaningReference)btn.textContent='Reference Cleaning…';
  else if(!referenceReady)btn.textContent='Upload Reference First';
  else btn.textContent='Start / Resume All';
}

function updateCleanerButtons(cleaner={}){
  const cleanBtn=$('cleanReferenceBtn');
  const originalBtn=$('useOriginalBtn');
  const cleanedBtn=$('useCleanedBtn');
  if(cleanBtn){
    cleanBtn.disabled=jobRunning || uploadingReference || cleaningReference || !cleaner.original_exists;
    cleanBtn.textContent=cleaningReference?'Cleaning…':'Clean Reference Audio';
  }
  if(originalBtn) originalBtn.disabled=jobRunning || cleaningReference || !cleaner.original_exists || !cleaner.using_cleaned;
  if(cleanedBtn) cleanedBtn.disabled=jobRunning || cleaningReference || !cleaner.cleaned_exists || cleaner.using_cleaned;
}

function loadReferencePlayers(cleaner={}, force=false){
  const original=$('originalReferenceAudio');
  const cleaned=$('cleanedReferenceAudio');
  if(original){
    if(cleaner.original_exists){
      if(force || !original.dataset.loaded){
        original.src='/audio/reference/original?t='+Date.now();
        original.dataset.loaded='1';
        original.load();
      }
    }else{
      original.removeAttribute('src');
      original.dataset.loaded='';
      original.load();
    }
  }
  if(cleaned){
    if(cleaner.cleaned_exists){
      if(force || !cleaned.dataset.loaded){
        cleaned.src='/audio/reference/cleaned?t='+Date.now();
        cleaned.dataset.loaded='1';
        cleaned.load();
      }
    }else{
      cleaned.removeAttribute('src');
      cleaned.dataset.loaded='';
      cleaned.load();
    }
  }
}

function setJob(job){
  jobRunning=!!(job&&job.running);
  $('jobPill').textContent=jobRunning?`${job.kind||'Job'} running`:(job?.status||'Idle');
  $('jobPill').style.color=jobRunning?'#62d6a4':'';
  updateGenerationButton();
}

function setReferenceState(status){
  referenceReady=!!status.reference_exists;
  const ref=status.reference||{};
  const cleaner=status.reference_cleaner||{};
  if(uploadingReference){
    $('referenceState').textContent='Uploading reference audio…';
  }else if(referenceReady){
    const kb=ref.size?` · ${(ref.size/1024).toFixed(1)} KB`:'';
    $('referenceState').textContent=`Active: ${ref.path||status.settings.REFERENCE_AUDIO}${kb}`;
  }else{
    $('referenceState').textContent=`Reference audio missing: ${ref.path||status.settings.REFERENCE_AUDIO||'input/reference.wav'}`;
  }

  const badge=$('referenceModeBadge');
  if(badge) badge.textContent=cleaner.using_cleaned?'Cleaned':'Original';
  const cleanStatus=$('cleanStatus');
  if(cleanStatus && !cleaningReference){
    if(cleaner.cleaned_exists) cleanStatus.textContent=cleaner.using_cleaned?'Cleaned reference is active.':'Cleaned reference is ready to use.';
    else if(cleaner.original_exists) cleanStatus.textContent='Ready to clean the uploaded reference.';
    else cleanStatus.textContent='Upload a reference to enable cleanup.';
  }
  updateCleanerButtons(cleaner);
  loadReferencePlayers(cleaner);
  updateGenerationButton();
}

async function refresh(){
  try{
    let s=await jfetch('/api/status');
    $('accepted').textContent=s.dataset.accepted.toLocaleString();
    $('rejected').textContent=s.dataset.rejected_ids.toLocaleString();
    $('corpusTotal').textContent=s.dataset.corpus_total.toLocaleString();
    $('percent').textContent=s.dataset.percent+'%';
    $('progressBar').style.width=s.dataset.percent+'%';
    $('modelBadge').textContent=s.settings.TTS_MODEL;
    $('supervisorStatus').textContent=s.supervisor.status||'not started';
    $('nextStart').textContent=s.supervisor.next_start??'—';
    setReferenceState(s);
    setJob(s.job);
    for(let k of ['TTS_MODEL','WHISPER_MODEL','MIN_ASR_SCORE','MAX_SILENCE_FRACTION','MAX_GENERATION_ATTEMPTS','GENERATION_CHUNK_SIZE']){
      if($(k)&&document.activeElement!==$(k))$(k).value=s.settings[k]??'';
    }
  }catch(e){console.error(e)}
}

async function startGeneration(){
  if(uploadingReference){alert('Please wait for the reference audio upload to finish.');return}
  if(!referenceReady){alert('Upload a reference voice before starting generation.');return}
  try{
    await jfetch('/api/jobs/generate',{method:'POST',body:JSON.stringify({})});
    startLogStream();
    refresh();
  }catch(e){alert(e.message);refresh()}
}

async function stopJob(){
  try{await jfetch('/api/jobs/stop',{method:'POST'});refresh()}catch(e){alert(e.message)}
}

async function startSimpleJob(kind){
  try{await jfetch(`/api/jobs/${kind}`,{method:'POST'});startLogStream();refresh()}catch(e){alert(e.message)}
}

function clearLogView(){$('log').textContent=''}

function startLogStream(){
  if(logSource)logSource.close();
  $('log').textContent='';
  logSource=new EventSource('/api/logs/stream');
  logSource.onmessage=e=>{
    let d=JSON.parse(e.data);
    if(!d.line)return;
    let p=$('log');
    p.textContent+=(p.textContent?'\n':'')+d.line;
    p.scrollTop=p.scrollHeight
  }
}

async function saveSettings(){
  let body={};
  for(let k of ['TTS_MODEL','WHISPER_MODEL','MIN_ASR_SCORE','MAX_SILENCE_FRACTION','MAX_GENERATION_ATTEMPTS','GENERATION_CHUNK_SIZE'])body[k]=$(k).value;
  try{await jfetch('/api/settings',{method:'POST',body:JSON.stringify(body)});await refresh();await loadRejects()}catch(e){alert(e.message)}
}

async function uploadReference(){
  const input=$('referenceFile');
  const button=$('uploadReferenceBtn');
  let f=input.files[0];
  if(!f){alert('Choose an audio file first.');return}
  if(jobRunning){alert('Stop the current generation job before replacing the reference voice.');return}

  let fd=new FormData();
  fd.append('file',f);
  uploadingReference=true;
  referenceReady=false;
  button.disabled=true;
  button.textContent='Uploading…';
  $('referenceState').textContent=`Uploading ${f.name}…`;
  updateGenerationButton();

  try{
    let r=await fetch('/api/reference',{method:'POST',body:fd});
    if(!r.ok)throw new Error(await r.text());
    let result=await r.json();

    let confirmed=false;
    for(let i=0;i<10;i++){
      let s=await jfetch('/api/status');
      if(s.reference_exists){
        confirmed=true;
        setReferenceState(s);
        break;
      }
      await new Promise(resolve=>setTimeout(resolve,250));
    }

    if(!confirmed)throw new Error(`Upload completed, but the server could not verify ${result.path}.`);
    input.value='';
    loadReferencePlayers((await jfetch('/api/status')).reference_cleaner||{},true);
  }catch(e){
    referenceReady=false;
    $('referenceState').textContent='Reference upload failed';
    alert(e.message);
  }finally{
    uploadingReference=false;
    button.disabled=false;
    button.textContent='Upload Reference Audio';
    await refresh();
    updateGenerationButton();
  }
}

async function cleanReference(){
  if(jobRunning){alert('Stop the current generation job before cleaning the reference voice.');return}
  const button=$('cleanReferenceBtn');
  cleaningReference=true;
  if(button)button.disabled=true;
  $('cleanStatus').textContent='Cleaning reference audio… the first run can take longer while DeepFilterNet loads.';
  updateGenerationButton();

  const body={
    strength:$('cleanStrength').value,
    normalize:$('cleanNormalize').checked,
    remove_rumble:$('cleanRumble').checked,
  };
  try{
    const result=await jfetch('/api/reference/clean',{method:'POST',body:JSON.stringify(body)});
    $('cleanStatus').textContent=`Cleaned successfully (${result.strength}). Cleaned reference is now active.`;
    const s=await jfetch('/api/status');
    setReferenceState(s);
    loadReferencePlayers(s.reference_cleaner||{},true);
  }catch(e){
    $('cleanStatus').textContent='Reference cleanup failed.';
    alert(e.message);
  }finally{
    cleaningReference=false;
    await refresh();
    updateGenerationButton();
  }
}

async function useReference(kind){
  if(jobRunning){alert('Stop the current generation job before changing the reference voice.');return}
  try{
    await jfetch(`/api/reference/use/${kind}`,{method:'POST'});
    await refresh();
  }catch(e){alert(e.message)}
}

async function loadRejects(){
  try{
    let d=await jfetch('/api/rejects');
    rejects=d.items;
    $('rejectCount').textContent=d.count;
    reviewPos=0;
    if(!rejects.length){
      $('reviewEmpty').textContent='No rejected samples waiting for review.';
      $('reviewEmpty').classList.remove('hidden');
      $('reviewCard').classList.add('hidden');
      return
    }
    $('reviewEmpty').classList.add('hidden');
    $('reviewCard').classList.remove('hidden');
    await loadReject()
  }catch(e){console.error(e)}
}

async function loadReject(){
  if(reviewPos>=rejects.length){await loadRejects();return}
  let item=rejects[reviewPos];
  currentDetail=await jfetch(`/api/rejects/${item.id}`);
  $('reviewIndex').textContent=`REJECT ${reviewPos+1} OF ${rejects.length} · ID ${item.id}`;
  $('reviewText').textContent=currentDetail.text;
  let sel=$('attemptSelect');
  sel.innerHTML='';
  currentDetail.attempts.forEach((a,i)=>{
    let o=document.createElement('option');
    o.value=i;
    o.textContent=`Attempt ${a.attempt} · ASR ${(a.asr_score??0).toFixed(1)}`;
    sel.appendChild(o)
  });
  renderAttempt()
}

function renderAttempt(){
  let a=currentDetail.attempts[+$('attemptSelect').value||0];
  $('reviewAudio').src=a.audio_url+'?t='+Date.now();
  $('heard').textContent=a.asr||'No transcription';
  $('asr').textContent=(a.asr_score??0).toFixed(1)+'%';
  $('silence').textContent=((a.silence_fraction??0)*100).toFixed(1)+'%';
  $('duration').textContent=(a.duration??0).toFixed(2)+'s'
}

async function acceptCurrent(){
  let a=currentDetail.attempts[+$('attemptSelect').value||0];
  if(!confirm(`Accept attempt ${a.attempt} for utterance ${currentDetail.id}?`))return;
  try{
    await jfetch(`/api/rejects/${currentDetail.id}/accept`,{method:'POST',body:JSON.stringify({attempt:a.attempt})});
    rejects.splice(reviewPos,1);
    if(reviewPos>=rejects.length)reviewPos=Math.max(0,rejects.length-1);
    await refresh();
    if(rejects.length)await loadReject();else await loadRejects()
  }catch(e){alert(e.message)}
}

async function rejectCurrent(){
  if(!confirm(`Permanently mark utterance ${currentDetail.id} as rejected?`))return;
  try{
    await jfetch(`/api/rejects/${currentDetail.id}/reject`,{method:'POST'});
    rejects.splice(reviewPos,1);
    if(reviewPos>=rejects.length)reviewPos=Math.max(0,rejects.length-1);
    if(rejects.length)await loadReject();else await loadRejects()
  }catch(e){alert(e.message)}
}

async function nextReject(){
  if(!rejects.length)return;
  reviewPos=(reviewPos+1)%rejects.length;
  await loadReject()
}

refresh();
loadRejects();
startLogStream();
setInterval(refresh,3000);
