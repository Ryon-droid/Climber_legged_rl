'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="climber-token"]').content;
const nf = new Intl.NumberFormat('zh-CN');
const phaseNames = {training:'正在训练', selection:'检查点评估中', level_baseline:'本档基线评估中',
  baseline_selection:'基线评估中', baseline_test:'基线验收中', final_independent_test:'独立验收中',
  completed:'本轮已结束', failed:'运行异常'};
const labels = {level:'本档推扰', accepted15:'10～15 N 抗推扰', standing:'无推力站立',
  nominal_getup:'正常趴姿起身', difficult_getup:'困难趴姿起身', push_0:'2～5 N 抗推扰',
  push_1:'5～10 N 抗推扰', push_2:'10～15 N 抗推扰', regression_standing:'历史偏航场景'};
let current = null, chosenRun = '', refreshing = false, terminalConnected = false;
let lastJob = null, jobsBusy = false, toastTimer = null, terminalBusy = false;
const history = [];
let historyIndex = 0;
const armNames = {standing_only:'仅站立回放', all_success:'站立＋抗扰恢复回放',
  weak_anchor:'弱回放约束', full_anchor:'强回放约束', no_rehearsal:'无动作回放',
  wheel_head:'仅轮子输出层', all_actor:'完整站立网络'};
const armColors = ['#72d9ac','#7daff5','#bd9aff','#ffab6e'];
let hiddenArms = new Set(), legendRun = '';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));

async function api(path, data) {
  const response = await fetch(path, {method:data === undefined ? 'GET':'POST',
    headers:{'X-Climber-Token':token, ...(data === undefined ? {}:{'Content-Type':'application/json'})},
    ...(data === undefined ? {}:{body:JSON.stringify(data)})});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `请求失败 ${response.status}`);
  return body;
}
function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => {$('toast').hidden = true;}, 6000);
}
function duration(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0));
  return [Math.floor(seconds/3600), Math.floor(seconds/60)%60, seconds%60].map(n => String(n).padStart(2,'0')).join(':');
}

async function refresh() {
  if (refreshing) return;
  refreshing = true; $('refresh').disabled = true;
  try {
    const data = await api('/api/status'+(chosenRun ? '?run='+encodeURIComponent(chosenRun):''));
    current = data; render(data);
  } catch (error) {
    if (current) render({...current, connected:false, error:error.message});
    $('connection').className = 'badge offline'; $('connection').innerHTML = '<i></i>控制台连接中断';
    $('error-banner').hidden = false; $('error-banner').textContent = error.message;
  } finally {refreshing = false; $('refresh').disabled = false;}
}

function render(data) {
  const connected = data.connected;
  $('connection').className = 'badge '+(connected ? '':'offline');
  $('connection').innerHTML = '<i></i>'+ (connected ? '训练设备已连接':'训练设备未连接');
  $('error-banner').hidden = connected;
  if (!connected) $('error-banner').textContent = 'SSH 连接暂不可用：'+(data.error || '')+'。以下保留最后一次读取的数据。';
  if (!data.state) return;
  const s = data.state, selected = data.selected_summary || {}, report = data.assessment;
  const active = connected && data.live;
  $('pulse').className = 'pulse'+(active ? ' live':'');
  $('phase').textContent = phaseNames[s.state] || s.state || '读取中';
  if (!data.live && !['completed','failed'].includes(s.state)) $('phase').textContent = '训练进程未运行';
  if (!connected) $('phase').textContent = '状态未知 · 显示上次数据';
  const runsKey = (data.runs || []).map(run => run.name).join('|');
  if ($('run-select').dataset.key !== runsKey) {
    $('run-select').innerHTML = (data.runs || []).map(run => `<option value="${esc(run.name)}">${esc(run.name)}</option>`).join('');
    $('run-select').dataset.key = runsKey;
  }
  $('run-select').value = data.run;
  const settled = ['completed','final_independent_test'].includes(s.state);
  const actualLevel = settled && data.levels?.length ? data.levels.at(-1).force_N : s.level_N;
  $('level').textContent = actualLevel ?? '—';
  const balanceFirst = ['balance_first18','sustained_balance18','rated_balance18'].includes(s.profile);
  $('target').textContent = balanceFirst ? `本轮 ${s.target_force_N} N · 先站稳，后收紧位置` : `本轮目标 ${s.target_force_N ?? 150} N · 长期目标 150 N`;
  const sampled = data.sampled_steps_lower_bound;
  $('steps').textContent = nf.format(sampled ?? s.steps ?? 0);
  $('step-note').textContent = data.protocol?.steps_per_arm ? '各组已写入日志与检查点的进度' : sampled != null ? '已完成回合的采样下限' : '已保存检查点的训练步数';
  const group = selected.exact18 || selected.level;
  $('success').textContent = group ? Math.round(group.success/group.count*100) : '—';
  $('success-note').textContent = group ? `${group.success}/${group.count} · ${selected.exact18 ? '固定 18 N':'本档随机推扰'}${balanceFirst ? ' · 当前课程标准':'选模集'}` : '尚未生成本档选模结果';
  $('elapsed').textContent = duration(data.elapsed_seconds);
  const start = new Date(s.started_unix*1000).toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit'});
  const estimate = balanceFirst ? '20～30（保守预留 45）' : ['recovery_replay18','wheel_recovery18'].includes(s.profile) ? '20～30（保守预留 47）' : s.profile === 'anchored30' ? '60～90' : s.profile === 'fine30' ? '45～70':'60～100';
  $('time-note').textContent = s.state === 'completed' ? `已结束 · ${start} 开始` : `${start} 开始 · 预计 ${estimate} 分钟`;
  if (!connected) $('time-note').textContent = '显示上次数据 · 恢复连接后重新估算';
  if ($('preview-force').dataset.run !== data.run) {
    $('preview-force').value = s.level_N || 15; $('preview-force').dataset.run = data.run;
  }
  const levels = data.protocol?.training_levels_N || (s.target_force_N === 18 ? [18,30,50,80,110,150] : ['fine30','anchored30'].includes(s.profile) ? [18,22,26,30]:[30,50,80,110,150]);
  $('curriculum').innerHTML = levels.map((level,index) => {
    const result = (data.levels || []).find(row => row.force_N === level);
    const cls = result ? (result.passed ? 'passed':'failed') : s.level_N === level ? (s.state === 'completed' ? (report?.target_passed ? 'passed':'failed') : 'current'):'';
    return `<div class="course ${cls}"><i>${result ? result.passed ? '✓':'×':String(index+1).padStart(2,'0')}</i><span>${level} N</span></div>`;
  }).join('');
  if (balanceFirst) {
    const names = {balance:'先站稳 · 忽略位移/朝向',return:'回位 · 15 cm / 15°',strict:'精确 · 5 cm / 5°'};
    $('curriculum').innerHTML = ['balance','return','strict'].map((stage,index) => {
      const currentStage = stage === s.balance_curriculum_stage;
      const cls = currentStage ? (s.state === 'completed' ? (report?.stage_passed ? 'passed':'failed') : 'current') : '';
      return `<div class="course ${cls}"><i>${String(index+1).padStart(2,'0')}</i><span>${names[stage]}</span></div>`;
    }).join('');
  }
  const used = s.max_steps ? Math.min(100, ((sampled ?? s.steps ?? 0)/s.max_steps)*100) : 0;
  $('budget-note').textContent = `训练步数预算 ${nf.format(s.max_steps || 0)} · 已使用 ${used.toFixed(1)}%`;
  renderProgress(data);
  renderLearning(data);
  renderEvaluation(data, report, selected);
  $('training-log').textContent = data.log || '尚无日志';
  $('updated').textContent = (connected ? '更新于 ':'上次更新 ')+new Date(data.fetched_at*1000).toLocaleTimeString('zh-CN');
  const hardware = data.hardware || {};
  $('vram-label').textContent = `${hardware.gpu_used_mb ?? '—'} / ${hardware.gpu_total_mb ?? '—'} MB`;
  $('vram-bar').value = hardware.gpu_total_mb ? 100*hardware.gpu_used_mb/hardware.gpu_total_mb : 0;
  $('ram-label').textContent = `${hardware.memory_used_gb ?? '—'} / ${hardware.memory_total_gb ?? '—'} GB`;
  $('ram-bar').value = hardware.memory_total_gb ? 100*hardware.memory_used_gb/hardware.memory_total_gb : 0;
  $('gpu-label').textContent = `GPU ${hardware.gpu_percent ?? '—'}%${connected ? '':' · 上次读取'}`;
  $('throughput').textContent = `采样速度 ${data.optimizer?.['time/fps'] ? nf.format(Number(data.optimizer['time/fps']))+' 步/秒':'—'}`;
  const pairs = [['设备',hardware.gpu || '2080 SUPER'],['显存',`${hardware.gpu_used_mb ?? '—'} / ${hardware.gpu_total_mb ?? '—'} MB`],
    ['GPU 使用率',`${hardware.gpu_percent ?? '—'}%`],['内存',`${hardware.memory_used_gb ?? '—'} / ${hardware.memory_total_gb ?? '—'} GB`],
    ['系统负载 / 1 min', hardware.load ?? '—'],['训练方式','CPU 并行 · PPO'],['连接',data.host_label || '—']];
  $('hardware').innerHTML = pairs.map(([key,value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('');
  $('checkpoint-path').textContent = '保留模型：'+(data.selected_checkpoint || '基线评估中')+(data.selected_sha256 ? '\nSHA256: '+data.selected_sha256:'');
  $('footer-host').textContent = data.host_label || 'Climber_legged_rl';
  renderLibrary(data.videos || []);
}

function branchList(data) {
  const configured = data.protocol?.arms;
  if (configured?.length) return configured;
  const found = [...new Set([...(data.training_series || []).map(r => r.arm), ...(data.curves || []).map(r => r.arm)].filter(Boolean))];
  return found.length ? found : ['training'];
}

function renderProgress(data) {
  const s = data.state, p = data.protocol || {}, branches = branchList(data);
  const total = s.max_steps || 1, done = Math.min(total,data.sampled_steps_lower_bound ?? s.steps ?? 0);
  const percentage = 100*done/total;
  $('progress-percent').textContent = percentage.toFixed(1)+'%';
  $('total-progress').value = percentage;
  $('env-count').textContent = `${p.num_envs ?? '—'} 个并行环境`;
  $('experiment-description').textContent = s.profile === 'rated_balance18' ? '站立腿电机 ±20 N·m／台 · 轮电机 ±2 N·m／台 · 4 个腿电机＋2 个轮电机共同学习' : s.profile === 'sustained_balance18' ? '加强持续站稳 · 新采集训练困难样本 · 4 个腿电机＋2 个轮电机共同学习' : s.profile === 'balance_first18' ? '先站稳再回位 · 4 个腿电机＋2 个轮电机共同学习' : s.profile === 'wheel_recovery18' ? '加强轮子恢复 · 对比可训练的参数范围' : s.profile === 'recovery_replay18' ? '降低更新幅度 · 对比成功动作回放范围' : `${s.target_force_N ?? '—'} N 抗推扰 · ${branches.length} 个训练分支`;
  const perArm = p.steps_per_arm || (s.max_steps || 0)/branches.length;
  $('arm-progress').innerHTML = branches.map((arm,i) => {
    const series = (data.training_series || []).find(r => r.arm === arm);
    const curves = (data.curves || []).filter(r => r.arm === arm);
    const localSteps = curves.length ? Math.max(...curves.map(r => r.steps-(p.steps_per_arm ? i*perArm : 0))) : 0;
    const steps = Math.min(perArm,Math.max(series?.latest?.steps || 0,localSteps));
    const active = data.connected && data.live && s.arm === arm;
    const status = steps >= perArm ? '已完成' : active ? (s.state === 'selection' ? '评估中':'训练中') : !data.connected && s.arm === arm ? '上次运行' : steps ? '已暂停':'等待';
    return `<div class="branch-progress"><div><span><i style="background:${armColors[i%4]}"></i>${esc(armNames[arm] || arm)}</span><small>${status} · ${nf.format(steps)} / ${nf.format(perArm)} 步</small></div><progress style="--arm-color:${armColors[i%4]}" max="${perArm || 1}" value="${steps}"></progress></div>`;
  }).join('');
  const elapsed = data.elapsed_seconds || 0;
  let remaining = '任务未运行';
  if (!data.connected) remaining = '连接恢复后重新估算';
  else if (s.state === 'completed') remaining = '训练与独立评估已结束';
  else if (s.state === 'failed') remaining = '运行异常，请查看日志';
  else if (data.live) {
    if (s.state === 'final_independent_test') remaining = '独立评估中，预计还需数分钟';
    else if (done > 0 && p.steps_per_arm) {
      const observed = elapsed/Math.max(done,1)*(total-done);
      const minutes = Math.ceil((observed+240)/60);
      remaining = `预计剩余约 ${minutes} 分钟（含评估）`;
    } else remaining = '初次评估中 · 总计预计 20–30 分钟';
  }
  $('remaining-time').textContent = remaining;
  $('progress-caption').textContent = `${nf.format(done)} / ${nf.format(total)} 步 · ${phaseNames[s.state] || s.state} · 评估耗时会影响预估`;
}

function metricFormat(value, metric) {
  if (metric === 'kl') return value.toFixed(5);
  return new Intl.NumberFormat('zh-CN',{maximumFractionDigits:metric === 'reward' || metric === 'value_loss' ? 1:2}).format(value);
}

function renderLearning(data) {
  const branches = branchList(data), p = data.protocol || {};
  if (legendRun !== data.run) {hiddenArms = new Set(); legendRun = data.run;}
  const legendKey = data.run+'|'+branches.join('|');
  if ($('arm-legend').dataset.key !== legendKey) {
    $('arm-legend').innerHTML = branches.map((arm,i) => `<button type="button" data-arm="${esc(arm)}" aria-pressed="true"><i style="background:${armColors[i%4]}"></i>${esc(armNames[arm] || arm)}</button>`).join('');
    $('arm-legend').dataset.key = legendKey;
    $('arm-legend').querySelectorAll('button').forEach(button => button.onclick = () => {
      const arm = button.dataset.arm;
      if (hiddenArms.has(arm)) hiddenArms.delete(arm); else hiddenArms.add(arm);
      button.setAttribute('aria-pressed',String(!hiddenArms.has(arm)));
      renderLearning(current);
    });
  }
  const available = Object.keys(data.parent_summary || {});
  if (available.length && !available.includes($('chart-group').value)) $('chart-group').value = 'level';
  const key = $('chart-group').value;
  const base = data.parent_summary?.[key];
  $('checkpoint-count').textContent = `${(data.curves || []).length} 次检查`;
  $('learning-note').textContent = `${data.connected ? '每 5 秒同步':'连接中断，保留上次数据'} · 横轴为各组独立采样步数 · 悬停查看数值，单击图例切换 · 训练奖励已缩放 0.01`;
  if (['balance_first18','sustained_balance18','rated_balance18'].includes(data.state.profile)) $('learning-note').textContent += ' · 成功率按当前课程标准，原严格标准另见下方对照';
  document.querySelectorAll('.metric-chart').forEach(element => {
    const metric = element.dataset.metric;
    const groups = branches.map((arm,index) => {
      let points;
      if (metric === 'selection') {
        points = (data.curves || []).filter(r => (r.arm || branches[r.attempt || 0]) === arm && r.summary?.[key]?.count)
          .map(r => ({x:r.steps-(p.steps_per_arm ? index*p.steps_per_arm:0),y:100*r.summary[key].success/r.summary[key].count,
            accepted:r.accepted,detail:`${r.summary[key].success}/${r.summary[key].count}${r.final_optimizer_state ? ' · 最终检查':''}`}));
      } else {
        points = (data.training_series || []).filter(r => r.arm === arm).flatMap(r => r.points)
          .filter(r => Number.isFinite(r[metric])).map(r => ({x:r.steps,y:r[metric]*(metric === 'success' ? 100:metric === 'episode_steps' ? .01:1)}));
      }
      return {arm,color:armColors[index%4],points:points.sort((a,b)=>a.x-b.x)};
    }).filter(g => !hiddenArms.has(g.arm));
    drawMetric(element, groups, metric, metric === 'selection' && base?.count ? 100*base.success/base.count:null);
  });
}

function drawMetric(element, groups, metric, baseline) {
  const points = groups.flatMap(g => g.points);
  if (!points.length && baseline === null) {
    element.innerHTML = `<div class="empty-chart">${groups.length ? '等待训练指标':'点击图例显示曲线'}<small>数据生成后自动显示</small></div>`;
    element.onpointermove = null; return;
  }
  const W = Math.max(320,element.clientWidth), H = 220, L = 58, R = 22, T = 18, B = 34;
  const maxX = Math.max(61440,...points.map(p=>p.x));
  let lo = Math.min(...points.map(p=>p.y),baseline ?? Infinity), hi = Math.max(...points.map(p=>p.y),baseline ?? -Infinity);
  if (['selection','success'].includes(metric)) {lo=0;hi=100;}
  else {const gap = Math.max((hi-lo)*.12,Math.abs(hi)*.02,.000001);lo-=gap;hi+=gap;if(metric !== 'reward')lo=Math.max(0,lo);}
  const x = v => L+v/maxX*(W-L-R), y = v => H-B-(v-lo)/(hi-lo || 1)*(H-T-B);
  let contents = '';
  for (let i=0;i<=4;i++) {
    const v = lo+(hi-lo)*i/4;
    contents += `<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#293748" stroke-dasharray="3 5"/><text x="${L-9}" y="${y(v)+3}" text-anchor="end" fill="#8195ae" font-size="10">${metricFormat(v,metric)}</text>`;
    const step = maxX*i/4;
    contents += `<text x="${x(step)}" y="${H-10}" text-anchor="middle" fill="#8195ae" font-size="10">${(step/1000).toFixed(0)}k</text>`;
  }
  if (baseline !== null) contents += `<line x1="${L}" y1="${y(baseline)}" x2="${W-R}" y2="${y(baseline)}" stroke="#a6b1c1" stroke-dasharray="5 5"/><text x="${W-R}" y="${Math.max(12,y(baseline)-6)}" text-anchor="end" fill="#a6b1c1" font-size="10">父模型 ${baseline.toFixed(1)}%</text>`;
  for (const group of groups) {
    if (!group.points.length) continue;
    contents += `<polyline points="${group.points.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}" fill="none" stroke="${group.color}" stroke-width="2" stroke-linejoin="round"/>`;
    const dots = metric === 'selection' ? group.points : [group.points.at(-1)];
    contents += dots.map(p=>`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3.5" fill="${group.color}" stroke="${p.accepted ? '#ffab6e':'#141c28'}" stroke-width="2"/>`).join('');
  }
  element.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(element.parentElement.querySelector('h2').textContent)}，横轴为各组训练步数">${contents}</svg><div class="chart-tooltip" hidden></div>`;
  const tip = element.querySelector('.chart-tooltip');
  element.onpointermove = event => {
    const rect = element.getBoundingClientRect(), px = event.clientX-rect.left;
    const target = (px-L)/(W-L-R)*maxX;
    const rows = [];
    for (const group of groups) {
      if (!group.points.length) continue;
      const nearest = group.points.reduce((a,b)=>Math.abs(a.x-target)<Math.abs(b.x-target)?a:b);
      rows.push(`${armNames[group.arm] || group.arm} · ${nf.format(nearest.x)} 步\n${metricFormat(nearest.y,metric)}${['selection','success'].includes(metric) ? '%':metric === 'episode_steps' ? ' 秒':''}${nearest.detail ? ' · '+nearest.detail:''}${nearest.accepted ? ' · 已保留':''}`);
    }
    if (!rows.length) {tip.hidden=true;return;}
    tip.textContent = rows.join('\n\n');tip.hidden=false;
    tip.style.left = Math.max(0,Math.min(px+10,W-tip.offsetWidth-8))+'px';
  };
  element.onpointerleave = () => {tip.hidden=true;};
}

function renderEvaluation(data, report, selected) {
  const balanceFirst = ['balance_first18','sustained_balance18','rated_balance18'].includes(data.state.profile);
  const final = Boolean(report?.selected_summary);
  const after = final ? report.selected_summary : selected;
  const before = final ? report.parent_summary : data.parent_summary;
  $('evaluation-kind').textContent = final ? '独立测试 · 同场景训练前后对比' : `${data.selected_level_N ?? data.state.level_N ?? '—'} N 档 · 选模场景`;
  $('eval-badge').textContent = final ? '独立验收':'选模集';
  const keys = Object.keys(after).sort((a,b) => {
    const order = ['level','exact18','exact30','exact150','accepted15','push_0','push_1','push_2','standing','nominal_getup','difficult_getup','regression_standing'];
    return order.indexOf(a)-order.indexOf(b);
  });
  $('eval-body').innerHTML = keys.length ? keys.map(key => {
    const a = before?.[key], b = after[key];
    const evaluatedLevel = final ? (report.target_force_N || 150) : (data.selected_level_N || data.state.level_N || 15);
    const name = key === 'level' ? `${Math.max(15,.7*evaluatedLevel)}～${evaluatedLevel} N 随机推扰` :
      key.startsWith('exact') ? `固定 ${key.slice(5)} N 推扰` : labels[key] || key;
    const cls = a && b.success < a.success ? 'bad' : b.success === b.count ? 'good':'neutral';
    return `<tr><td>${esc(name)}</td><td>${a ? `${a.success}/${a.count}`:'—'}</td><td class="${cls}">${b.success}/${b.count}</td></tr>`;
  }).join('') : '<tr><td colspan="3" class="table-empty">正在评估基线，结果稍后显示</td></tr>';
  if (balanceFirst) {
    $('evaluation-kind').textContent = final ? '独立测试 · 当前课程与原严格标准分别统计' : '选模场景 · 当前课程与原严格标准分别统计';
    const strictBefore = final ? report.parent_strict_summary : data.parent_strict_summary;
    const strictAfter = final ? report.selected_strict_summary : data.selected_strict_summary;
    for (const key of ['exact18','level','standing']) {
      const a = strictBefore?.[key], b = strictAfter?.[key];
      if (b) $('eval-body').insertAdjacentHTML('beforeend', `<tr><td>原严格标准 · ${key === 'exact18' ? '固定 18 N' : labels[key] || key}</td><td>${a ? `${a.success}/${a.count}` : '—'}</td><td>${b.success}/${b.count}</td></tr>`);
    }
  }
  if (final) {
    const target = report.target_force_N || 150;
    const passed = report.target_passed ?? report.target_150N_passed ?? report.gate_passed;
    $('evaluation-note').textContent = `本轮 ${target} N 目标${passed ? '通过':'未通过'}。${target < 150 ? '本轮结果不代表已经达到 150 N。':''}表内数据为独立测试，按完整回合统计成功数。`;
  } else {
    $('evaluation-note').textContent = '当前显示保留模型的选模成绩，不是刚完成更新的模型，也不是最终验收。独立测试将在本轮结束后进行。';
  }
  if (balanceFirst) $('evaluation-note').textContent = final ? `当前课程${report.stage_passed ? '通过，可评估下一阶段':'尚未通过'}；放宽课程的成绩不代表原严格 18 N 或 150 N 目标达标。` : '当前课程先学习推扰后站稳，忽略位移与朝向；仍要求低速、双轮支撑和车身竖直。原严格标准单独列出，未达标不阻止当前课程学习。';
}

function renderLibrary(videos) {
  $('video-count').textContent = videos.length;
  const signature = videos.map(v => v.path).join('|');
  if ($('video-list').dataset.signature === signature) return;
  $('video-list').dataset.signature = signature;
  $('video-list').innerHTML = '';
  if (!videos.length) {$('video-list').textContent = '暂无录像'; return;}
  for (const video of videos) {
    const button = document.createElement('button'); button.className = 'video-item';
    button.innerHTML = `${esc(video.name)}<span>${video.console_preview ? '控制台预览':'历史视频'}</span><small>${esc(video.path)} · ${video.size_mb} MB</small>`;
    button.onclick = async () => {
      try {const job = await api('/api/videos/fetch', {path:video.path}); lastJob = job.id; setJobBusy(true,'正在加载视频…'); await pollJobs();}
      catch (error) {toast(error.message);}
    };
    $('video-list').appendChild(button);
  }
}

function setJobBusy(busy, text='▷ 生成当前策略预览') {
  jobsBusy = busy; $('generate').disabled = busy; $('generate').textContent = text;
}
async function pollJobs() {
  try {
    const {jobs} = await api('/api/jobs');
    if (!lastJob && jobs.length) {
      const running = jobs.find(job => job.state === 'running');
      const latest = jobs.at(-1);
      lastJob = (running || latest).id;
    }
    const job = jobs.find(row => row.id === lastJob);
    if (!job) return;
    if (job.state === 'running') {setJobBusy(true, job.kind === 'preview' ? '正在录制并传回本机…':'正在加载视频…'); return;}
    if (job.state === 'completed' && $('video').dataset.job !== job.id) {
      $('video').src = job.media_url; $('video').dataset.job = job.id;
      $('video').hidden = false; $('video-placeholder').hidden = true;
      $('download-video').hidden = false; $('download-video').href = job.media_url;
      $('download-video').download = job.remote_path.split('/').at(-1);
      const m = job.metadata;
      $('video-meta').hidden = false;
      if (m?.checkpoint_sha256) {
        $('video-meta').textContent = `${m.run || '历史策略'} · ${m.force_N ?? '—'} N · seed ${m.seed}\n`+
          `${m.episode_complete ? (m.success ? '回合通过':'回合失败'):'预览片段'} · ${Number(m.seconds).toFixed(2)} 秒 · ${m.reason || '—'}\n`+
          `策略指纹 ${m.checkpoint_sha256.slice(0,16)}…`;
      } else {$('video-meta').textContent = '历史录像，不能作为当前模型成绩。\n'+job.remote_path;}
      $('video-note').textContent = '录像已传回本机，可播放、拖动进度或保存。单段演示不代表整体成功率。';
      toast('仿真视频已就绪'); refresh();
    }
    if (job.state === 'failed' && $('generate').dataset.failed !== job.id) {
      $('generate').dataset.failed = job.id; toast('视频任务失败：'+job.error);
      $('video-note').textContent = '视频任务失败：'+job.error;
    }
    setJobBusy(false);
  } catch (error) {if (jobsBusy) toast(error.message);}
}

async function connectTerminal() {
  try {await api('/api/terminal/connect', {}); terminalConnected = true; await pollTerminal();}
  catch (error) {toast(error.message);}
}
async function pollTerminal() {
  if (!terminalConnected || terminalBusy) return;
  terminalBusy = true;
  try {
    const data = await api('/api/terminal');
    const pre = $('terminal-output');
    const bottom = pre.scrollHeight-pre.scrollTop-pre.clientHeight < 40;
    if (pre.textContent !== data.text) {pre.textContent = data.text || '正在建立 SSH 会话…'; if (bottom) pre.scrollTop = pre.scrollHeight;}
    $('terminal-status').textContent = data.alive ? '已连接':'会话已结束';
    $('terminal-status').className = 'inline-status'+(data.alive ? ' connected':'');
    $('terminal-connect').textContent = data.alive ? '终端已连接':'重新连接';
    if (!data.alive) terminalConnected = false;
  } catch (error) {$('terminal-status').textContent = '连接中断';}
  finally {terminalBusy = false;}
}
async function sendCommand(command) {
  if (!command.trim()) return;
  try {
    await api('/api/terminal/input', {text:command+'\n'}); terminalConnected = true;
    history.push(command); historyIndex = history.length;
    $('command-input').value = ''; await pollTerminal();
    $('terminal-output').scrollTop = $('terminal-output').scrollHeight;
  } catch (error) {toast(error.message);}
}

$('run-select').onchange = () => {chosenRun = $('run-select').value; refresh();};
$('chart-group').onchange = () => {if (current) renderLearning(current);};
let resizeTimer;
const chartResize = new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {if (current?.state) renderLearning(current);},100);
});
document.querySelectorAll('.metric-chart').forEach(element => chartResize.observe(element));
$('refresh').onclick = refresh;
$('terminal-connect').onclick = connectTerminal;
$('command-form').onsubmit = event => {event.preventDefault(); sendCommand($('command-input').value);};
$('command-input').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey) {event.preventDefault(); sendCommand(event.currentTarget.value);}
  if (event.key === 'ArrowUp' && !event.currentTarget.value.includes('\n')) {
    event.preventDefault(); historyIndex = Math.max(0, historyIndex-1); event.currentTarget.value = history[historyIndex] || '';
  }
  if (event.key === 'ArrowDown' && !event.currentTarget.value.includes('\n')) {
    event.preventDefault(); historyIndex = Math.min(history.length, historyIndex+1); event.currentTarget.value = history[historyIndex] || '';
  }
  if (event.key === 'c' && event.ctrlKey && !event.currentTarget.value) {event.preventDefault(); $('interrupt').click();}
};
$('interrupt').onclick = async () => {try {if (!terminalConnected) return; await api('/api/terminal/input', {text:'\x03'}); await pollTerminal();} catch (error) {toast(error.message);}};
document.querySelectorAll('[data-command]').forEach(button => button.onclick = () => {
  if (!current?.run) return toast('请等待连接训练设备');
  const run = current.run;
  const commands = {status:`cat outputs/${run}/workflow.json`, log:`tail -n 12 outputs/${run}/console.log`,
    gpu:'nvidia-smi', files:'ls scripts/*play* scripts/*render* scripts/*preview* 2>/dev/null'};
  $('command-input').value = commands[button.dataset.command]; $('command-input').focus();
});
$('generate').onclick = async () => {
  if (!current?.connected || !current?.run) return toast('请先连接训练设备');
  try {
    const job = await api('/api/preview', {run:current.run, mode:$('preview-mode').value,
      force:Number($('preview-force').value), seed:Number($('preview-seed').value)});
    lastJob = job.id; setJobBusy(true,'正在录制并传回本机…');
    $('video-note').textContent = '正在用当前保留模型独立录制，通常需要 15～60 秒。训练继续运行。';
    await pollJobs();
  } catch (error) {toast(error.message);}
};
$('preview-mode').onchange = () => {if ($('preview-mode').value === 'folded') $('preview-force').value = 0;};
document.querySelectorAll('.nav-item').forEach(link => link.onclick = () => {
  document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item === link));
});
refresh(); pollJobs();
setInterval(refresh, 5000);
setInterval(pollTerminal, 800);
setInterval(pollJobs, 1500);
