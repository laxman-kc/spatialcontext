(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('study-data').textContent);
  const metrics = data.metrics;
  const events = data.history.events;
  const names = {original: 'Original', neutral: 'Neutral', competing: 'Competing', text_only: 'Text only'};
  const order = Object.keys(names);
  const colors = {base: '#275dce', tuned: '#17847c', regression: '#b95c12', ink: '#172b38', grid: '#dce3e6'};
  const descriptions = {
    original: 'The original prepared history. The question asks about an earlier clip.',
    neutral: 'One later clip is replaced by unrelated footage. Earlier target frames stay fixed.',
    competing: 'One later clip contains a relevant object category. Earlier target frames stay fixed.',
    text_only: 'Question, answer options and clip map only. No image tensors enter the model.'
  };
  const regression = metrics.paired_changes.neutral.regressed_ids[0];
  const state = {view: 'results', condition: 'competing', mode: 'accuracy', ci: true,
    question: regression || metrics.per_question[0].id, filter: 'all', step: events.length, metric: 'mean_loss'};
  const $ = id => document.getElementById(id);
  const percent = value => (value * 100).toFixed(1);
  const signed = value => (value > 0 ? '+' : '') + value.toFixed(2);
  const shortId = value => value.split('_').at(-1);
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const svgNS = 'http://www.w3.org/2000/svg';

  function node(tag, attributes = {}, text) {
    const element = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function add(parent, tag, attributes, text) {
    const element = node(tag, attributes, text);
    parent.append(element);
    return element;
  }
  function chart(container, height, description) {
    container.replaceChildren();
    const width = Math.max(280, container.clientWidth);
    const svg = node('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': description});
    add(svg, 'title', {}, description);
    container.append(svg);
    return {svg, width};
  }
  function whisker(svg, x1, x2, y, color) {
    add(svg, 'line', {x1, x2, y1: y, y2: y, stroke: color, 'stroke-width': 1.6, opacity: .63});
    [x1, x2].forEach(x => add(svg, 'line', {x1: x, x2: x, y1: y - 4, y2: y + 4, stroke: color, 'stroke-width': 1.6, opacity: .63}));
  }
  function renderComparison() {
    const isAccuracy = state.mode === 'accuracy';
    const description = order.map(condition => {
      const b = metrics.accuracies.base[condition], t = metrics.accuracies.tuned[condition];
      if (!isAccuracy) {
        const e = metrics.effects[condition + '_gain'];
        return `${names[condition]}: tuned minus base ${signed(e.estimate * 100)} percentage points; 95% paired interval ${signed(e.ci95[0] * 100)} to ${signed(e.ci95[1] * 100)} percentage points`;
      }
      return `${names[condition]}: base ${b.correct} of ${b.total}, ${percent(b.accuracy)} percent, 95% interval ${percent(b.ci95[0])} to ${percent(b.ci95[1])} percent; tuned ${t.correct} of ${t.total}, ${percent(t.accuracy)} percent, 95% interval ${percent(t.ci95[0])} to ${percent(t.ci95[1])} percent`;
    }).join('. ');
    const {svg, width} = chart($('comparison-chart'), 342, description);
    const left = width < 440 ? 77 : 91, right = 60, top = 24, bottom = 302;
    const span = width - left - right;
    const min = isAccuracy ? 0 : Math.floor(Math.min(0, ...order.map(c => metrics.effects[c + '_gain'].ci95[0] * 100)) / 5) * 5 - 5;
    const max = isAccuracy ? 100 : Math.max(5, Math.ceil(Math.max(...order.map(c => metrics.effects[c + '_gain'].ci95[1] * 100)) / 5) * 5 + 5);
    const x = value => left + (value - min) / (max - min) * span;
    const ticks = isAccuracy ? [0,25,50,75,100] : [min, -10, 0, max].filter((v, i, a) => a.indexOf(v) === i && v >= min && v <= max);
    const selectedIndex = order.indexOf(state.condition);
    add(svg, 'rect', {x: 0, y: top + selectedIndex * 66 + 2, width, height: 61, fill: '#eef5f4', rx: 3});
    ticks.forEach(value => {
      add(svg, 'line', {x1: x(value), x2: x(value), y1: top, y2: bottom, stroke: value === 0 && !isAccuracy ? '#8ca1ab' : colors.grid, 'stroke-dasharray': value === 0 && !isAccuracy ? '3 3' : ''});
      add(svg, 'text', {x: x(value), y: bottom + 18, 'text-anchor': 'middle'}, value > 0 && !isAccuracy ? '+' + value : value);
    });
    order.forEach((condition, index) => {
      const center = top + index * 66 + 32;
      const label = add(svg, 'text', {x: left - 12, y: center + 3, 'text-anchor': 'end'}, names[condition]);
      if (condition === state.condition) label.setAttribute('style', 'font-weight:700;fill:#172b38');
      if (condition === 'competing') add(svg, 'text', {x: left - 12, y: center + 17, 'text-anchor': 'end', style: 'font-size:8px'}, 'primary');
      if (isAccuracy) {
        ['base', 'tuned'].forEach((model, modelIndex) => {
          const value = metrics.accuracies[model][condition];
          const y = center + (modelIndex ? 8 : -8);
          if (state.ci) whisker(svg, x(value.ci95[0] * 100), x(value.ci95[1] * 100), y, colors[model]);
          const mark = model === 'base'
            ? add(svg, 'circle', {cx: x(value.accuracy * 100), cy: y, r: 4.7, fill: colors[model], stroke: 'white', 'stroke-width': 1.3})
            : add(svg, 'rect', {x: x(value.accuracy * 100) - 4.3, y: y - 4.3, width: 8.6, height: 8.6, fill: colors[model], stroke: 'white', 'stroke-width': 1.3});
          add(mark, 'title', {}, `${model}: ${value.correct}/${value.total}, ${percent(value.accuracy)}%; 95% interval ${percent(value.ci95[0])}–${percent(value.ci95[1])}%`);
          add(svg, 'text', {x: width - right + 11, y: y + 4, class: 'value', style: `fill:${colors[model]}`}, percent(value.accuracy) + '%');
        });
      } else {
        const effect = metrics.effects[condition + '_gain'];
        const value = effect.estimate * 100;
        const color = value < 0 ? colors.regression : colors.ink;
        if (state.ci) whisker(svg, x(effect.ci95[0] * 100), x(effect.ci95[1] * 100), center, color);
        add(svg, 'circle', {cx: x(value), cy: center, r: 5, fill: color, stroke: 'white', 'stroke-width': 1.3});
        add(svg, 'text', {x: width - right + 9, y: center + 4, class: 'value', style: `fill:${color}`}, signed(value));
      }
    });
    add(svg, 'text', {x: left + span / 2, y: 339, class: 'axis-label', 'text-anchor': 'middle'}, isAccuracy ? 'Test accuracy (%)' : 'Tuned − base accuracy (percentage points)');
    $('comparison-legend').innerHTML = isAccuracy
      ? '<span><i class="dot base"></i>Base model</span><span><i class="square tuned"></i>LoRA tuned</span>'
      : '<span>Paired change · identical questions and condition inputs</span>';
    $('comparison-note').textContent = isAccuracy
      ? 'Intervals: 95% accuracy intervals from 10,000 paired source/donor-cluster resamples. Conditional on 10 supplied groups; independence is unverified. These are not intervals for the paired gain.'
      : 'Intervals: paired source/donor-cluster bootstrap, 10,000 resamples. A [0, 0] interval reflects unchanged observed correctness; it does not establish population equivalence.';
  }
  function renderCondition() {
    $('conditions').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', button.dataset.condition === state.condition));
    $('condition-title').textContent = names[state.condition] + (state.condition === 'competing' ? ' · primary test' : '');
    $('condition-description').textContent = descriptions[state.condition];
    const hidden = state.condition === 'text_only' ? ' hidden-clip' : '';
    const last = state.condition === 'competing' ? 'competing' : state.condition === 'neutral' ? 'later neutral' : 'later';
    $('clip-schematic').innerHTML = `<div class="clip-box target${hidden}">Earlier target</div><div class="clip-box later${hidden}">Later clip</div><div class="clip-box ${last}${hidden}">${state.condition === 'competing' ? 'Competing' : state.condition === 'neutral' ? 'Unrelated' : 'Later clip'}</div>`;
    const b = metrics.accuracies.base[state.condition], t = metrics.accuracies.tuned[state.condition];
    const delta = t.correct - b.correct;
    $('selected-summary').innerHTML = `<strong>${b.correct}/${b.total} → ${t.correct}/${t.total}</strong><span class="${delta < 0 ? 'regression-text' : ''}">${delta === 0 ? 'No observed accuracy change.' : delta < 0 ? 'One fewer correct answer after tuning.' : 'More correct answers after tuning.'}</span>`;
  }
  function outcome(row) {
    const c = row.conditions[state.condition];
    return c.base_correct && c.tuned_correct ? 'correct' : !c.base_correct && !c.tuned_correct ? 'wrong' : c.base_correct ? 'regression' : 'improved';
  }
  const statusText = {correct: 'Both models correct', wrong: 'Both models wrong', regression: 'Regressed: correct → incorrect', improved: 'Corrected: incorrect → correct'};
  const symbols = {correct: '✓', wrong: '×', regression: '↘', improved: '↗'};
  function renderQuestions() {
    const rows = metrics.per_question.filter(row => state.filter === 'all' || (state.filter === 'wrong' ? outcome(row) === 'wrong' : ['regression','improved'].includes(outcome(row))));
    if (rows.length && !rows.some(row => row.id === state.question)) state.question = rows[0].id;
    $('question-count').textContent = `${rows.length} of ${metrics.per_question.length} · ${names[state.condition]}`;
    const grid = $('question-grid');
    grid.replaceChildren();
    rows.forEach(row => {
      const status = outcome(row);
      const button = document.createElement('button');
      button.className = 'q-tile ' + status;
      button.type = 'button';
      button.dataset.question = row.id;
      button.setAttribute('aria-pressed', row.id === state.question);
      button.setAttribute('aria-label', `Question ${shortId(row.id)}: ${statusText[status]}`);
      button.innerHTML = `<span class="mark" aria-hidden="true">${symbols[status]}</span><span>${escape(shortId(row.id))}</span>`;
      button.addEventListener('click', () => {
        state.question = row.id;
        grid.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.question === row.id));
        renderAnswer();
      });
      grid.append(button);
    });
    if (!rows.length) {
      const message = document.createElement('p');
      message.className = 'empty-state';
      message.textContent = `No questions match this filter under ${names[state.condition].toLowerCase()}.`;
      grid.append(message);
      $('answer-detail').innerHTML = '<p class="eyebrow">NO MATCHING QUESTIONS</p><h3>No outcome hidden.</h3><p>Choose another filter or condition to inspect saved answers.</p>';
    } else renderAnswer();
  }
  function renderAnswer() {
    const row = metrics.per_question.find(row => row.id === state.question);
    const c = row.conditions[state.condition], status = outcome(row);
    $('answer-detail').innerHTML = `<p class="eyebrow">${escape(names[state.condition])} / SAVED QUESTION</p><h3>#${escape(shortId(row.id))}</h3><p>Reviewed answer: <b>${escape(row.answer)}</b></p><dl><div><dt>Base model</dt><dd>${escape(c.base_prediction)} <span>${c.base_correct ? 'correct' : 'wrong'}</span></dd></div><div><dt>LoRA tuned</dt><dd>${escape(c.tuned_prediction)} <span>${c.tuned_correct ? 'correct' : 'wrong'}</span></dd></div></dl><p class="answer-status ${status === 'regression' ? 'regressed' : ''}">${statusText[status]}</p><p>Source/donor ${escape(row.cluster.replace('_', ' '))}.<br>Answer letters reference the original four options. Full question text and media are in the research handoff.</p>`;
  }
  function renderTable() {
    $('results-table').innerHTML = '<table><caption>Complete test comparison · 27 questions per condition · conditional paired cluster bootstrap intervals</caption><thead><tr><th scope="col">Condition</th><th scope="col">Base correct</th><th scope="col">Base 95% CI (%)</th><th scope="col">Tuned correct</th><th scope="col">Tuned 95% CI (%)</th><th scope="col">Change (pp)</th><th scope="col">Paired change 95% CI (pp)</th></tr></thead><tbody>' + order.map(condition => {
      const b = metrics.accuracies.base[condition], t = metrics.accuracies.tuned[condition];
      const e = metrics.effects[condition + '_gain'];
      return `<tr><th scope="row">${names[condition]}</th><td>${b.correct}/${b.total} (${percent(b.accuracy)}%)</td><td>${percent(b.ci95[0])}–${percent(b.ci95[1])}</td><td>${t.correct}/${t.total} (${percent(t.accuracy)}%)</td><td>${percent(t.ci95[0])}–${percent(t.ci95[1])}</td><td>${signed(100 * e.estimate)}</td><td>${signed(100 * e.ci95[0])} to ${signed(100 * e.ci95[1])}</td></tr>`;
    }).join('') + '</tbody></table>';
  }
  function selectCondition(condition) {
    state.condition = condition;
    renderCondition(); renderComparison(); renderQuestions();
  }
  function formatMetric(value, metric = state.metric) {
    return metric === 'lr_used' ? value === 0 ? '0' : value.toExponential(2) : value.toFixed(4);
  }
  const metricNames = {mean_loss: 'Accumulation-group mean loss', lr_used: 'Learning rate used', gradient_norm: 'Gradient norm before clipping'};
  function renderTraining() {
    if (state.view !== 'training') return;
    const event = events[state.step - 1];
    const {svg, width} = chart($('training-chart'), 332, `${metricNames[state.metric]} over all ${events.length} optimizer steps. Selected step ${state.step}: ${formatMetric(event[state.metric])}.`);
    const left = state.metric === 'lr_used' ? 62 : 48, right = 20, top = 20, bottom = 285;
    const maximum = Math.max(...events.map(e => e[state.metric])) * 1.12 || 1;
    const x = step => left + (step - 1) / (events.length - 1) * (width - left - right);
    const y = value => bottom - value / maximum * (bottom - top);
    for (let i = 0; i <= 4; i++) {
      const value = maximum * i / 4;
      add(svg, 'line', {x1: left, x2: width - right, y1: y(value), y2: y(value), stroke: colors.grid});
      add(svg, 'text', {x: left - 9, y: y(value) + 3, 'text-anchor': 'end'}, state.metric === 'lr_used' ? value === 0 ? '0' : value.toExponential(1) : value.toFixed(state.metric === 'mean_loss' ? 2 : 1));
    }
    [1, 7, 14, 21, 27].forEach(step => add(svg, 'text', {x: x(step), y: bottom + 18, 'text-anchor': 'middle'}, step));
    add(svg, 'path', {d: events.map((e, i) => `${i ? 'L' : 'M'}${x(e.step)},${y(e[state.metric])}`).join(' '), fill: 'none', stroke: colors.tuned, 'stroke-width': 1.8, 'stroke-linejoin': 'round'});
    events.forEach(e => add(svg, 'circle', {cx: x(e.step), cy: y(e[state.metric]), r: 2.5, fill: colors.tuned}));
    add(svg, 'line', {x1: x(event.step), x2: x(event.step), y1: top, y2: bottom, stroke: colors.ink, opacity: .45, 'stroke-dasharray': '3 3'});
    add(svg, 'circle', {cx: x(event.step), cy: y(event[state.metric]), r: 5.5, fill: colors.regression, stroke: 'white', 'stroke-width': 1.5});
    add(svg, 'text', {x: left, y: 11, class: 'axis-label'}, metricNames[state.metric]);
    add(svg, 'text', {x: left + (width - left - right) / 2, y: 329, class: 'axis-label', 'text-anchor': 'middle'}, 'Optimizer step');
    const hit = add(svg, 'rect', {x: left, y: top, width: width - left - right, height: bottom - top, fill: 'transparent', style: 'cursor:crosshair'});
    hit.addEventListener('click', e => {
      const local = (e.clientX - svg.getBoundingClientRect().left) * width / svg.getBoundingClientRect().width;
      state.step = Math.max(1, Math.min(events.length, Math.round(1 + (local - left) / (width - left - right) * (events.length - 1))));
      $('training-step').value = state.step;
      renderTraining();
    });
    $('step-output').textContent = `${state.step} / ${events.length}`;
    $('training-step').setAttribute('aria-valuetext', `Step ${state.step}, ${metricNames[state.metric]} ${formatMetric(event[state.metric])}`);
    $('step-title').textContent = 'Step ' + state.step;
    $('step-values').innerHTML = [
      ['Examples seen', `${event.examples_seen} / 108`], ['Batch mean loss', event.mean_loss.toFixed(5)],
      ['Learning rate used', formatMetric(event.lr_used, 'lr_used')], ['Gradient norm', event.gradient_norm.toFixed(4)],
      ['Elapsed loop time', event.elapsed_seconds.toFixed(2) + ' sec']
    ].map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join('');
  }
  function showView(view) {
    state.view = view;
    document.querySelectorAll('[data-view]').forEach(button => {
      const selected = button.dataset.view === view;
      button.setAttribute('aria-selected', selected); button.tabIndex = selected ? 0 : -1;
      $('view-' + button.dataset.view).hidden = !selected;
    });
    if (view === 'results') renderComparison();
    if (view === 'training') renderTraining();
  }
  order.forEach(condition => {
    const button = document.createElement('button');
    button.type = 'button'; button.dataset.condition = condition; button.textContent = names[condition];
    button.addEventListener('click', () => selectCondition(condition));
    $('conditions').append(button);
  });
  document.querySelectorAll('[data-view]').forEach((button, index, buttons) => {
    button.addEventListener('click', () => showView(button.dataset.view));
    button.addEventListener('keydown', e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? buttons.length - 1 : (index + (e.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next].focus(); showView(buttons[next].dataset.view);
    });
  });
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', b === button));
    renderComparison();
  }));
  $('show-ci').addEventListener('change', e => {state.ci = e.target.checked; renderComparison();});
  $('question-filter').addEventListener('change', e => {state.filter = e.target.value; renderQuestions();});
  $('show-regression').addEventListener('click', () => {
    state.question = regression; state.filter = 'changed'; $('question-filter').value = 'changed';
    selectCondition('neutral');
    $('question-grid').querySelector('button')?.focus({preventScroll: true});
  });
  $('training-metric').addEventListener('change', e => {state.metric = e.target.value; renderTraining();});
  $('training-step').addEventListener('input', e => {state.step = Number(e.target.value); renderTraining();});
  $('export-csv').addEventListener('click', () => {
    const rows = [['question_id','condition','reviewed_answer','base_prediction','tuned_prediction','base_correct','tuned_correct','source_donor_cluster']];
    metrics.per_question.forEach(q => {const c = q.conditions[state.condition]; rows.push([q.id,state.condition,q.answer,c.base_prediction,c.tuned_prediction,c.base_correct,c.tuned_correct,q.cluster]);});
    const csv = rows.map(row => row.map(value => '"' + String(value).replaceAll('"','""') + '"').join(',')).join('\r\n') + '\r\n';
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
    const a = document.createElement('a'); a.href = url; a.download = `earlier-clip-qa-${state.condition}.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $('protocol-short').textContent = data.summary.protocol_digest.slice(0, 16) + '…';
  const resize = new ResizeObserver(() => {if (state.view === 'results') renderComparison(); else if (state.view === 'training') renderTraining();});
  resize.observe($('comparison-chart')); resize.observe($('training-chart'));
  renderTable(); selectCondition(state.condition);
})();
