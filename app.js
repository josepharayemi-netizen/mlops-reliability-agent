let documentData, report, aiEnabled=false, busy=false;
const $=id=>document.getElementById(id);
const labels={action_required:'Action required',investigate:'Investigate',insufficient_evidence:'Insufficient evidence',within_thresholds:'Within thresholds'};
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function percent(value){return value===null?'Unavailable':(value*100).toFixed(1)+'%';}
async function request(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const result=await r.json();if(!r.ok)throw Error(result.error||'Request failed');return result;}
function setBusy(value){busy=value;$('demo').disabled=value;$('upload').disabled=value;$('ask').disabled=value||!aiEnabled;}
async function load(data,label){if(busy)return;setBusy(true);try{const result=await request('/assess',data);report=result;documentData=data;$('workspace').hidden=false;$('status').textContent=label+' · '+report.model;$('answer').textContent='';render();}catch(e){$('status').textContent=e.message;}finally{setBusy(false);}}
function render(){
 const b=report.metrics.baseline,c=report.metrics.current;
 $('metrics').replaceChildren();
 for(const [label,value] of [['Overall status',labels[report.status]],['Current accuracy',percent(c.accuracy)],['Label coverage',percent(c.label_coverage)],['Investigation items',report.findings.length]]){const box=node('div',label,'metric');box.append(node('strong',String(value)));$('metrics').append(box);}
 $('comparison').replaceChildren();
 for(const [name,key,format] of [['Rows','rows',String],['Labelled rows','labelled_rows',String],['Label coverage','label_coverage',percent],['Accuracy','accuracy',percent],['Precision','precision',percent],['Recall','recall',percent],['F1 score','f1',percent],['P95 latency','p95_latency_ms',v=>v===null?'Unavailable':v.toFixed(1)+' ms'],['Inference error rate','error_rate',percent]]){const tr=node('tr');tr.append(node('th',name),node('td',format(b[key])),node('td',format(c[key])));$('comparison').append(tr);}
 $('drift').replaceChildren();
 for(const f of report.drift){const box=node('div',undefined,'driftrow');box.append(node('strong',f.feature),node('span',f.tv===null?'Unavailable':'TV '+f.tv.toFixed(3)));const bar=node('progress');bar.max=1;bar.value=f.tv??0;bar.setAttribute('aria-label',f.feature+' distribution change');box.append(bar,node('p','Missing: '+percent(f.baseline_missing_rate)+' → '+percent(f.current_missing_rate)+' · Observed samples: '+f.baseline_samples+' → '+f.current_samples,'muted'));$('drift').append(box);}
 $('findings').replaceChildren();const selected=report.findings.filter(f=>$('filter').value==='all'||f.severity===$('filter').value);
 if(!selected.length)$('findings').append(node('p','No investigation items in this view. Review coverage and configured thresholds before drawing conclusions.'));
 for(const f of selected){const box=node('article',undefined,'finding');box.append(node('span',f.severity.toUpperCase(),'tag '+f.severity),node('h3',f.title),node('p',f.evidence),node('p',f.recommendation),node('p','Check ID: '+f.id,'muted'));$('findings').append(box);}
 $('trace').textContent=JSON.stringify({checks:report.checks,trace:report.trace,evidence_sha256:report.evidence_sha256},null,2);$('limitations').textContent=report.limitations;
}
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=node('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('demo').onclick=async()=>{try{await load(await request('/sample'),'Synthetic degraded-model sample');}catch(e){$('status').textContent=e.message;}};
$('upload').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>5*1024*1024)throw Error('Maximum file size is 5 MB');await load(JSON.parse(await file.text()),'Imported: '+file.name);}catch(error){$('status').textContent=error.message;}};
$('filter').onchange=()=>report&&render();
$('download').onclick=()=>report&&download(JSON.stringify(report,null,2),'mlops-report.json','application/json');
$('markdown').onclick=()=>{if(!report)return;const lines=['# MLOps Reliability Assessment','','Model: '+report.model,'Status: '+report.status,'Generated: '+report.generated_at,'Evidence SHA-256: '+report.evidence_sha256,'',report.limitations,''];for(const f of report.findings)lines.push('## ['+f.severity+'] '+f.title,'Check: '+f.id,f.evidence,'',f.recommendation,'');download(lines.join('\n'),'mlops-report.md','text/markdown');};
$('askForm').onsubmit=async e=>{e.preventDefault();if(busy||!report||!aiEnabled)return;setBusy(true);$('answer').textContent='Reviewing monitoring evidence…';try{const result=await request('/ask',{document:documentData,question:$('question').value});$('answer').textContent=result.mode+'\n\n'+result.answer;}catch(error){$('answer').textContent=error.message;}finally{setBusy(false);}};
request('/health').then(h=>{aiEnabled=h.ai_enabled;$('aiState').textContent=aiEnabled?'Local AI advisor configured. Verify advice against check IDs.':'Optional local AI advisor is off. Set OLLAMA_MODEL and restart to enable it.';setBusy(busy);}).catch(e=>{$('aiState').textContent=e.message;});
