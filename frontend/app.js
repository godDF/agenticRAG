const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { jobTimer: null, uploadFilter: 'all', currentJob: null, embeddingPricing: null };

function headers(json = false) {
  const result = {};
  if (json) result['Content-Type'] = 'application/json';
  return result;
}
function toast(message, error = false) {
  const el = $('#toast'); el.textContent = message; el.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(el._timer); el._timer = setTimeout(() => el.className = 'toast', 3000);
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}
function safeUrl(value) {
  try { const url = new URL(value); return ['http:','https:'].includes(url.protocol) ? url.href : '#'; }
  catch (_) { return '#'; }
}
function formatMs(value) { return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value || 0} ms`; }
function kindForStage(stage) {
  if (stage === 'retrieval') return 'tool';
  if (['answer_generation','answer_revision'].includes(stage)) return 'llm';
  return 'agent';
}
const stageLabels = {
  query_planning:'查询规划 Agent', retrieval:'向量检索工具', evidence_grading:'证据评估 Agent',
  query_rewrite:'查询改写 Agent', answer_generation:'回答生成 LLM', answer_verification:'回答校验 Agent',
  answer_revision:'回答修正 LLM', answer_reverification:'回答复核 Agent'
};
const detailLabels = {
  round:'轮次', subquery_count:'子问题数', subqueries:'检索问题', query_count:'检索问题数', queries:'检索问题',
  category:'知识分类', selected_tool:'选择工具', retrieved_chunks:'召回片段数', top_score:'最高相似度',
  accepted_chunks:'接受片段数', accepted_titles:'接受证据', sufficient:'证据是否充分', missing_aspects:'缺失信息',
  original_queries:'原检索问题', rewritten_queries:'改写后问题', evidence_count:'证据数量', evidence_titles:'证据标题',
  verdict:'校验结论', revision_instruction:'修正要求', reason_code:'降级原因', status:'状态'
};
const categoryLabels = {student_ticket:'学生票规则',child_ticket:'儿童票规则',elderly_ticket:'老人票规则',flight_safety:'航班安全须知',highspeed_rail_safety:'高铁安全须知',attraction_notice:'景点注意事项'};
function readableValue(key, value) {
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (key === 'category') return categoryLabels[value] || value;
  if (Array.isArray(value)) return value.length ? value.join('；') : '无';
  if (value && typeof value === 'object') return JSON.stringify(value);
  return value === '' || value == null ? '无' : value;
}
function detailText(details) {
  if (!details) return '无额外参数';
  const hidden = new Set(['decision_summary','next_action']);
  const entries = Object.entries(details).filter(([key])=>!hidden.has(key));
  if (!entries.length) return '无额外参数';
  return entries.map(([key,value]) => `${detailLabels[key] || key}：${readableValue(key,value)}`).join(' · ');
}
function traceHtml(event, queryMode = false) {
  const kind = event.kind || kindForStage(event.stage);
  const name = event.label || stageLabels[event.stage] || event.stage || event.name;
  const status = event.status || 'completed';
  const latency = event.latency_ms == null ? '运行中' : formatMs(event.latency_ms);
  const error = event.error ? ` · ${event.error.type}: ${event.error.message}` : '';
  const statusLabel = {completed:'已完成',running:'执行中',failed:'失败',fallback:'降级执行'}[status] || status;
  const decision = event.details?.decision_summary ? `<p class="decision-line"><b>决策摘要</b>${escapeHtml(event.details.decision_summary)}</p>` : '';
  const next = event.details?.next_action ? `<p class="next-line"><b>下一步</b>${escapeHtml(event.details.next_action)}</p>` : '';
  return `<div class="trace-row" data-kind="${escapeHtml(kind)}" data-status="${escapeHtml(status)}">
    <div class="trace-icon ${escapeHtml(kind)}">${kind === 'agent' ? 'AG' : kind === 'llm' ? 'LLM' : kind === 'tool' ? 'TL' : 'SYS'}</div>
    <div class="trace-main"><strong>${escapeHtml(name)}</strong>${decision}${next}<p>${escapeHtml(detailText(event.details))}${escapeHtml(error)}</p></div>
    <div class="trace-meta"><span class="trace-state ${escapeHtml(status)}">${escapeHtml(statusLabel)}</span>${escapeHtml(latency)}</div>
  </div>`;
}
function setBar(id, percent) { $(id).style.width = `${Math.max(0, Math.min(100, percent))}%`; }

async function checkSystem() {
  try {
    const health = await fetch('/health').then(r => r.json());
    $('#serviceStatus').className = 'status-dot ok';
    $('#serviceText').textContent = `服务正常${health.ragaai_enabled ? ' · Catalyst 已启用' : ' · 本地 Trace'}${health.auth_required ? ' · Token 鉴权' : ' · 演示模式'}`;
    state.embeddingPricing = health.embedding_pricing || null;
    if (!state.currentJob && state.embeddingPricing?.configured) {
      $('#metricCost').textContent = state.embeddingPricing.free ? '¥0（官方免费）' : '等待计算';
      $('#metricCostHint').textContent = state.embeddingPricing.free
        ? `${health.embedding_model} · ${state.embeddingPricing.source}`
        : `${state.embeddingPricing.source} · 等待 Token`;
    }
  } catch (_) {
    $('#serviceStatus').className = 'status-dot bad'; $('#serviceText').textContent = '服务不可用';
  }
  try {
    const response = await fetch('/ready'); const data = await response.json();
    $('#qdrantText').textContent = response.ok ? 'Qdrant 已连接' : `Qdrant：${data.qdrant || '未就绪'}`;
  } catch (_) { $('#qdrantText').textContent = 'Qdrant 不可用'; }
}

function renderUploadJob(job) {
  state.currentJob = job;
  const status = $('#jobStatus'); status.textContent = job.status;
  status.className = `status-label ${job.status === 'queued' ? 'processing' : job.status}`;
  $('#stageLabel').textContent = job.current_stage;
  $('#processPercent').textContent = `${job.progress}%`; setBar('#processBar', job.progress);
  const m = job.metrics || {};
  $('#metricTokens').textContent = (m.token_total || 0).toLocaleString();
  $('#metricTokenHint').textContent = `Embedding ${m.embedding_tokens || 0} · LLM ${(m.llm_input_tokens || 0) + (m.llm_output_tokens || 0)}`;
  $('#metricLatency').textContent = formatMs(m.elapsed_ms || 0);
  const embeddingCost = Number(m.estimated_cost_cny || 0);
  const officiallyFree = m.cost_configured && embeddingCost === 0 && (m.pricing_free || state.embeddingPricing?.free);
  $('#metricCost').textContent = officiallyFree ? '¥0（官方免费）' : `¥${embeddingCost.toFixed(6)}`;
  $('#metricCostHint').textContent = officiallyFree
    ? `${m.pricing_source || state.embeddingPricing?.source || 'SiliconFlow 官方价格'} · 本次未调用 DeepSeek`
    : (m.cost_configured ? '按 BGE-M3 服务商单价估算' : '未配置 BGE-M3 单价');
  $('#metricCalls').textContent = `${m.agent_calls || 0} / ${m.error_count || 0}`;
  $('#metricCallHint').textContent = `Agent ${m.agent_calls || 0} · LLM ${m.llm_calls || 0} · Tool ${m.tool_calls || 0}`;
  let events = job.events || [];
  if (state.uploadFilter === 'error') events = events.filter(e => e.status === 'failed');
  else if (state.uploadFilter !== 'all') events = events.filter(e => e.kind === state.uploadFilter);
  $('#uploadTrace').innerHTML = events.length ? events.map(e => traceHtml(e)).join('') : '<div class="empty-state">当前筛选条件下没有记录</div>';
  if (job.status === 'completed') {
    const d = job.result;
    $('#jobSummary').className = 'job-result';
    $('#jobSummary').innerHTML = `<strong>索引构建完成</strong><br>文档：${escapeHtml(d.title)}<br>分片：${d.chunk_count} · 类别：${escapeHtml(d.category)}<br><span class="mono">${escapeHtml(d.document_id)}</span>`;
  } else if (job.status === 'failed') {
    $('#jobSummary').className = 'job-result job-error';
    $('#jobSummary').innerHTML = `<strong>处理失败</strong><br>${escapeHtml(job.error?.type)}：${escapeHtml(job.error?.message)}`;
  } else {
    $('#jobSummary').className = 'empty-state compact'; $('#jobSummary').textContent = job.current_stage;
  }
}

async function pollJob(jobId) {
  clearTimeout(state.jobTimer);
  try {
    const response = await fetch(`/api/v1/document-jobs/${jobId}`, { headers: headers() });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '任务查询失败');
    renderUploadJob(data);
    if (['completed','failed'].includes(data.status)) {
      $('#uploadButton').disabled = false;
      toast(data.status === 'completed' ? '知识库索引构建完成' : '知识库处理失败', data.status === 'failed');
      return;
    }
    state.jobTimer = setTimeout(() => pollJob(jobId), 550);
  } catch (error) { $('#uploadButton').disabled = false; toast(error.message, true); }
}

function submitUpload(event) {
  event.preventDefault();
  const file = $('#fileInput').files[0]; if (!file) return toast('请选择文件', true);
  const form = new FormData(event.target);
  $('#uploadButton').disabled = true; $('#jobStatus').textContent = 'uploading'; $('#jobStatus').className = 'status-label processing';
  $('#jobSummary').className = 'empty-state compact'; $('#jobSummary').textContent = '正在将文件传输到服务端…';
  setBar('#networkBar', 0); setBar('#processBar', 0); $('#networkPercent').textContent = '0%';
  const xhr = new XMLHttpRequest(); xhr.open('POST','/api/v1/document-jobs');
  xhr.upload.onprogress = e => { if (e.lengthComputable) { const p = Math.round(e.loaded/e.total*100); setBar('#networkBar',p); $('#networkPercent').textContent=`${p}%`; } };
  xhr.onerror = () => { $('#uploadButton').disabled=false; toast('文件上传网络错误',true); };
  xhr.onload = () => {
    let data={}; try { data=JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status < 200 || xhr.status >= 300) { $('#uploadButton').disabled=false; return toast(data.detail || `上传失败 HTTP ${xhr.status}`,true); }
    setBar('#networkBar',100); $('#networkPercent').textContent='100%'; pollJob(data.job_id);
  };
  xhr.send(form);
}

async function submitQuery(event) {
  event.preventDefault();
  const button=$('#queryButton'); button.disabled=true; $('#answerStatus').textContent='running'; $('#answerStatus').className='status-label processing';
  $('#answerContent').className='empty-state'; $('#answerContent').textContent='Agent 正在规划检索与校验证据…'; $('#queryTrace').innerHTML='<div class="empty-state">正在执行…</div>';
  const form=new FormData(event.target); const payload={session_id:`workbench-${Date.now()}`,query:form.get('query'),category:form.get('category')};
  try {
    const response=await fetch('/api/v1/query',{method:'POST',headers:headers(true),body:JSON.stringify(payload)}); const data=await response.json();
    if(!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    $('#answerStatus').textContent=data.found?'verified':'not found'; $('#answerStatus').className=`status-label ${data.found?'completed':'failed'}`;
    $('#answerContent').className='answer-body'; $('#answerContent').textContent=data.answer;
    $('#sourceList').innerHTML=(data.sources||[]).map((s,i)=>`<a class="source-chip" href="${escapeHtml(safeUrl(s.source_url))}" target="_blank" rel="noopener">[${i+1}] ${escapeHtml(s.title||s.source_name||'来源')}</a>`).join('');
    $('#queryTokens').textContent=`${data.meta.input_tokens} / ${data.meta.output_tokens}`; $('#queryTokenHint').textContent=`缓存命中 ${data.meta.cache_hit_input_tokens || 0} · 未命中 ${data.meta.cache_miss_input_tokens || 0}${data.meta.cache_usage_reported ? '' : '（保守估算）'}`; $('#queryRounds').textContent=data.meta.retrieval_rounds; $('#queryLatency').textContent=formatMs(data.meta.latency_ms);
    $('#queryCost').textContent=`¥${Number(data.meta.estimated_cost_cny||0).toFixed(6)}`; $('#queryCostHint').textContent=`${data.meta.pricing_model || 'deepseek-v4-flash'} · 官方人民币单价`;
    $('#traceId').textContent=`Trace ${data.trace_id}`; $('#queryTrace').innerHTML=(data.trace||[]).map(e=>traceHtml(e,true)).join('') || '<div class="empty-state">无 Trace</div>';
  } catch(error) {
    $('#answerStatus').textContent='error'; $('#answerStatus').className='status-label failed'; $('#answerContent').className='job-result job-error'; $('#answerContent').textContent=error.message; $('#queryTrace').innerHTML='<div class="empty-state">请求失败，详细错误已写入服务端日志。</div>'; toast(error.message,true);
  } finally { button.disabled=false; }
}

async function loadDocuments() {
  $('#documentList').innerHTML='<div class="panel empty-state">正在加载…</div>';
  try {
    const response=await fetch('/api/v1/documents',{headers:headers()}); const data=await response.json(); if(!response.ok) throw new Error(data.detail||'加载失败');
    $('#documentList').innerHTML=data.length?data.map(d=>`<article class="panel document-card"><span class="pill">${escapeHtml(d.category)}</span><h3>${escapeHtml(d.title)}</h3><p>${escapeHtml(d.source_name)} · ${escapeHtml(d.updated_at)}<br>${d.chunk_count} 个分片 · ${escapeHtml(d.status)}<br><span class="mono">${escapeHtml(d.document_id)}</span></p><div class="card-foot"><span class="muted mono">${escapeHtml(d.original_filename)}</span><button class="danger" data-delete="${escapeHtml(d.document_id)}">删除</button></div></article>`).join(''):'<div class="panel empty-state">还没有通过工作台上传的文档。</div>';
  } catch(error) { $('#documentList').innerHTML=`<div class="panel empty-state">${escapeHtml(error.message)}</div>`; }
}
async function deleteDocument(id) {
  if(!confirm('删除文档并同步删除 Qdrant 向量？')) return;
  const response=await fetch(`/api/v1/documents/${id}`,{method:'DELETE',headers:headers()}); const data=await response.json(); if(!response.ok) return toast(data.detail||'删除失败',true); toast('文档已删除'); loadDocuments();
}

$$('.nav-item').forEach(button=>button.onclick=()=>{ $$('.nav-item').forEach(b=>b.classList.remove('active')); $$('.tab-page').forEach(p=>p.classList.remove('active')); button.classList.add('active'); $(`#tab-${button.dataset.tab}`).classList.add('active'); if(button.dataset.tab==='documents') loadDocuments(); });
$('#fileInput').onchange=e=>{ const file=e.target.files[0]; $('#fileLabel').textContent=file?`${file.name} · ${(file.size/1024).toFixed(1)} KB`:'拖放文件，或点击选择'; $('#autoTitlePreview').textContent=file?file.name.replace(/\.[^.]+$/,'').replace(/[_-]+/g,' '):'选择文件后自动识别'; };
['dragenter','dragover'].forEach(name=>$('#dropZone').addEventListener(name,e=>{e.preventDefault();$('#dropZone').classList.add('dragging');}));
['dragleave','drop'].forEach(name=>$('#dropZone').addEventListener(name,e=>{$('#dropZone').classList.remove('dragging');}));
$('#uploadForm').addEventListener('submit',submitUpload); $('#queryForm').addEventListener('submit',submitQuery); $('#refreshDocuments').onclick=loadDocuments;
$('#documentList').addEventListener('click',e=>{ if(e.target.dataset.delete) deleteDocument(e.target.dataset.delete); });
$('#uploadFilters').addEventListener('click',e=>{ if(!e.target.dataset.filter)return; $$('#uploadFilters .filter').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');state.uploadFilter=e.target.dataset.filter;if(state.currentJob)renderUploadJob(state.currentJob); });
document.querySelector('[name="updated_at"]').value=new Date().toISOString().slice(0,10);
checkSystem(); setInterval(checkSystem,15000);
