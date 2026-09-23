/* No report data is inserted as HTML. All names, formulas, and source text are
 * rendered with textContent; report expressions are never evaluated as code. */
(function () {
  'use strict';
  const M = window.DrperfModel;
  const host = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : null;
  const saved = host?.getState() || {};
  let loadedModel = null,
    rawCosts = saved.rawCosts || false;
  let model = null,
    selected = null,
    activeTab = saved.tab || 'interface',
    filter = '',
    result = null;
  let edits = [],
    assumptionIds = [],
    scenarioError = '',
    onlyAffected = true;
  let validationModel = null,
    validationName = '';
  let auditAlternatives = saved.auditAlternatives || false;
  let proposals = [],
    proposalDraft = null,
    proposalResult = null;
  const scenarioModel = () => M.withProposals(model, proposals);
  const root = document.getElementById('app');
  const send = (message) =>
    host
      ? host.postMessage(message)
      : window.dispatchEvent(new CustomEvent('drperfHostMessage', { detail: message }));
  const el = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const button = (text, action, className = '') => {
    const b = el('button', className, text);
    b.type = 'button';
    b.addEventListener('click', action);
    return b;
  };
  const append = (parent, ...children) => {
    children.filter(Boolean).forEach((child) => parent.append(child));
    return parent;
  };
  const fmt = M.format;
  const region = () => model?.regions.find((r) => r.id === selected) || model?.regions[0];
  const badge = (text, tone = '') => el('span', 'badge ' + tone, text);
  const stateLabel = (names, values) => names.map((name, i) => `${name}=${values[i]}`).join(', ');
  const persist = () =>
    host?.setState({
      modelId: model?.id,
      selected,
      tab: activeTab,
      edits,
      assumptionIds,
      proposals,
      rawCosts,
      auditAlternatives
    });
  function choose(id, tab = activeTab) {
    selected = id;
    activeTab = tab;
    persist();
    send({ type: 'select', region: id });
    render();
  }
  function heading(title, description) {
    return append(
      el('div', 'section-heading'),
      el('h2', '', title),
      description && el('p', 'muted', description)
    );
  }
  function stat(label, value) {
    return append(el('div', 'stat'), el('strong', '', value), el('span', 'muted', label));
  }
  function formulaCard(r) {
    const card = el('section', 'card formula-card');
    append(card, el('div', 'eyebrow', 'PERFORMANCE INTERFACE'), el('h1', '', r.name));
    const tags = el('div', 'tags');
    append(
      tags,
      badge('CPU instructions / call'),
      badge('Own work · nested regions excluded'),
      badge(rawCosts ? 'Marker API cost retained' : 'Marker estimate subtracted')
    );
    if (r.droppedCalls) tags.append(badge(`${r.droppedCalls} calls omitted from fitting`, 'warn'));
    append(card, tags);
    if (!r.regimes.length)
      append(
        card,
        el(
          'p',
          'empty',
          'There are not enough varied states to fit an interface yet. The measured calls are still available below.'
        )
      );
    for (const fit of r.regimes) {
      const block = el('div', 'regime');
      if (r.regimes.length > 1)
        block.append(
          el(
            'div',
            'muted range-label',
            fit.range.map(([lo, hi], i) => `${lo} ≤ ${r.states[i]} ≤ ${hi}`).join(' · ')
          )
        );
      block.append(el('div', 'formula', M.formula(r, fit)));
      const notes = el('div', 'tags');
      notes.append(
        badge(
          `${(100 * fit.unexplainedShare).toFixed(1)}% unexplained`,
          fit.unexplainedShare > 0.05 ? 'warn' : 'good'
        )
      );
      if (fit.dependent.length)
        notes.append(badge(`Tied PCVs: ${fit.dependent.join(', ')}`, 'warn'));
      append(block, notes);
      card.append(block);
    }
    for (const diagnostic of r.diagnostics || []) card.append(el('p', 'notice warn', diagnostic));
    if (
      !rawCosts &&
      (r.diagnostics || []).some((message) => /Marker-overhead|calibrated explained/.test(message))
    )
      card.append(
        button(
          'Inspect recorded counts before marker subtraction',
          () => {
            rawCosts = true;
            model = M.recordedCosts(loadedModel);
            render();
          },
          'secondary'
        )
      );
    const stats = el('div', 'stats');
    append(
      stats,
      stat(r.droppedCalls ? 'retained calls' : 'recorded calls', fmt(r.calls)),
      stat('observed states', fmt(r.points.length)),
      stat('declared PCVs', r.states.length)
    );
    card.append(stats);
    const sources = el('div', 'source-links');
    r.sources.forEach((source, i) =>
      sources.append(
        button(
          `${source.path}:${source.line} ↗`,
          () => send({ type: 'source', region: r.id, index: i }),
          'link-button'
        )
      )
    );
    if (!r.sources.length)
      sources.append(el('span', 'muted', 'No source mapping · export with --source-root'));
    if (r.sources.length > 1)
      sources.append(
        el(
          'span',
          'muted',
          'Multiple annotations share this region name; measurements are aggregated.'
        )
      );
    card.append(sources);
    return card;
  }
  function svg(tag, attributes = {}, text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function observations(r) {
    const card = append(
      el('section', 'card'),
      heading(
        'What the run observed',
        'Explained and unexplained work at the measured states. No claim between these points.'
      )
    );
    const points = r.regimes.length
      ? r.regimes.flatMap((fit) => fit.points)
      : r.points.map((p) => ({ ...p, explained: 0, unexplained: p.observed }));
    if (!points.length) {
      card.append(el('p', 'muted', 'No retained state points.'));
      return card;
    }
    const sampled =
      points.length <= 60
        ? points
        : points.filter((_, i) => i % Math.ceil(points.length / 60) === 0);
    const width = 760,
      height = 230,
      left = 62,
      right = 20,
      bottom = 40,
      top = 18;
    const max = Math.max(1, ...sampled.map((p) => Math.max(0, p.explained) + p.unexplained));
    const chart = svg('svg', {
      viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-label': 'Observed explained and unexplained instruction costs',
      class: 'observations-chart'
    });
    for (let i = 0; i <= 4; ++i) {
      const y = height - bottom - (i * (height - top - bottom)) / 4;
      chart.append(svg('line', { x1: left, y1: y, x2: width - right, y2: y, class: 'grid-line' }));
      chart.append(
        svg(
          'text',
          { x: left - 9, y: y + 4, 'text-anchor': 'end', class: 'chart-label' },
          compact((max * i) / 4)
        )
      );
    }
    const step = (width - left - right) / sampled.length,
      scale = (height - top - bottom) / max;
    sampled.forEach((point, i) => {
      const x = left + step * i + Math.min(4, step / 5),
        w = Math.max(1, step - Math.min(8, step / 3));
      const explained = Math.max(0, point.explained),
        y = height - bottom - explained * scale;
      const group = svg('g', { tabindex: 0 });
      group.append(
        svg(
          'title',
          {},
          `${stateLabel(r.states, point.state)}\nExplained: ${fmt(point.explained)}\nUnexplained: ${fmt(point.unexplained)}\nCalls: ${point.calls}`
        )
      );
      group.append(
        svg('rect', { x, y, width: w, height: explained * scale, class: 'bar-explained', rx: 2 })
      );
      group.append(
        svg('rect', {
          x,
          y: y - point.unexplained * scale,
          width: w,
          height: point.unexplained * scale,
          class: 'bar-unexplained',
          rx: 2
        })
      );
      if (sampled.length <= 12 || i % Math.ceil(sampled.length / 8) === 0)
        group.append(
          svg(
            'text',
            { x: x + w / 2, y: height - 17, 'text-anchor': 'middle', class: 'chart-label' },
            point.state.length === 1 ? String(point.state[0]) : String(i + 1)
          )
        );
      chart.append(group);
    });
    card.append(chart);
    append(
      card,
      append(
        el('div', 'legend'),
        el('span', 'legend-explained', 'Explained'),
        el('span', 'legend-unexplained', r.regimes.length ? 'Unexplained' : 'Not fitted'),
        el('span', 'muted', r.states.length === 1 ? r.states[0] : 'State combinations')
      )
    );
    if (points.length > sampled.length)
      card.append(
        el(
          'p',
          'muted small',
          `Showing ${sampled.length} of ${points.length} state points. The table retains all points.`
        )
      );
    const details = el('details');
    details.append(el('summary', '', `Inspect ${points.length} observed states`));
    const table = el('table');
    table.append(
      append(
        el('thead'),
        append(
          el('tr'),
          ...['PCVs', 'Calls', 'Explained', 'Unexplained'].map((h) => el('th', '', h))
        )
      )
    );
    const body = el('tbody'),
      controls = el('div', 'actions');
    let pageIndex = 0;
    const pageSize = 200;
    const drawPage = () => {
      body.replaceChildren();
      controls.replaceChildren();
      const start = pageIndex * pageSize;
      for (const p of points.slice(start, start + pageSize))
        body.append(
          append(
            el('tr'),
            el('td', 'mono', stateLabel(r.states, p.state)),
            el('td', 'number', fmt(p.calls)),
            el('td', 'number', fmt(p.explained)),
            el('td', 'number', fmt(p.unexplained))
          )
        );
      controls.append(
        el(
          'span',
          'small muted',
          `${start + 1}–${Math.min(start + pageSize, points.length)} of ${points.length} states`
        )
      );
      if (pageIndex > 0)
        controls.append(
          button(
            'Previous states',
            () => {
              --pageIndex;
              drawPage();
            },
            'quiet'
          )
        );
      if (start + pageSize < points.length)
        controls.append(
          button(
            'Next states',
            () => {
              ++pageIndex;
              drawPage();
            },
            'quiet'
          )
        );
    };
    details.addEventListener('toggle', () => {
      if (details.open && !body.children.length) drawPage();
    });
    append(table, body);
    append(details, controls, append(el('div', 'table-scroll'), table));
    card.append(details);
    return card;
  }
  function compact(value) {
    if (value >= 1e9) return fmt(value / 1e9) + 'G';
    if (value >= 1e6) return fmt(value / 1e6) + 'M';
    if (value >= 1e3) return fmt(value / 1e3) + 'k';
    return fmt(value);
  }
  function attribution(r) {
    const card = append(
      el('section', 'card'),
      heading(
        'Where the cost comes from',
        'Functions contributing to the interface and the unexplained part.'
      )
    );
    if (!r.regimes.length) {
      card.append(el('p', 'muted', 'Attribution will appear after enough states are measured.'));
      return card;
    }
    r.regimes.forEach((fit, i) => {
      if (r.regimes.length > 1) card.append(el('h3', '', `Regime ${i + 1}`));
      const groups = [
        ...r.states.map((state, j) => [state + ' coefficient', fit.attribution.coefficients[j]]),
        ['Constant', fit.attribution.constant],
        ['Unexplained', fit.attribution.unexplained]
      ];
      for (const [label, functions] of groups) {
        if (!functions.length) continue;
        const details = el('details');
        details.open = label === 'Unexplained' && fit.unexplainedShare > 0.05;
        details.append(el('summary', '', label));
        const list = el('ul', 'function-list');
        for (const fn of functions)
          list.append(
            append(
              el('li'),
              append(el('div'), el('code', '', fn.function), el('small', 'muted', fn.module)),
              el('span', 'number', fmt(fn.instructions))
            )
          );
        append(details, list);
        card.append(details);
      }
    });
    return card;
  }
  function interfaceView(r) {
    const main = el('div', 'interface-view');
    append(main, formulaCard(r), observations(r));
    if (Object.keys(r.nested || {}).length) {
      const nested = append(
        el('section', 'card'),
        heading(
          'Nested marked calls',
          'These measured calls are excluded from this region’s own instruction formula. Counts are means per parent call and include descendants at any depth; they are not a cost total.'
        )
      );
      const list = el('ul', 'function-list');
      for (const [name, count] of Object.entries(r.nested).sort((a, b) => b[1] - a[1])) {
        const target = model.regions.find((r) => r.id === name);
        list.append(
          append(
            el('li'),
            target
              ? button(target.name, () => choose(name), 'link-button')
              : el('span', 'mono', name),
            el('span', 'number', fmt(count) + ' calls')
          )
        );
      }
      nested.append(list);
      main.append(nested);
    }
    const related = model.relations.filter(
      (relation) =>
        relation.target.region === r.id || relation.terms.some((term) => term.region === r.id)
    );
    const relations = append(
      el('section', 'card'),
      heading('Connected state', `${related.length} observed relationships touch this region.`)
    );
    related.slice(0, 4).forEach((relation) =>
      relations.append(
        button(
          M.relationship(relation),
          () => {
            activeTab = 'relations';
            render();
          },
          'relation-preview'
        )
      )
    );
    relations.append(
      button(
        'Explore all relationships →',
        () => {
          activeTab = 'relations';
          render();
        },
        'link-button'
      )
    );
    const next = append(el('section', 'card'), heading('Next small experiment'));
    const list = el('ul', 'suggestions');
    M.suggestedExperiments(model, r).forEach((text) => list.append(el('li', '', text)));
    append(
      next,
      list,
      button(
        'Try a PCV scenario →',
        () => {
          activeTab = 'scenario';
          ensureEdit(r);
          render();
        },
        'secondary'
      )
    );
    append(main, append(el('div', 'two-columns'), relations, next), attribution(r));
    return main;
  }
  function connectionGraph() {
    const card = append(
      el('section', 'card'),
      heading(
        'Observed connections',
        'Arrows show equation dependencies in the recorded trace. They do not establish causality.'
      )
    );
    const all = M.graph(model);
    const connected = [
      ...new Set(all.edges.flatMap((e) => [JSON.parse(e.source)[0], JSON.parse(e.target)[0]]))
    ];
    const focus = connected.includes(selected) ? [selected] : [];
    for (let index = 0; index < focus.length && focus.length < 16; ++index) {
      const current = focus[index];
      for (const edge of all.edges) {
        const from = JSON.parse(edge.source)[0],
          to = JSON.parse(edge.target)[0];
        const other = from === current ? to : to === current ? from : null;
        if (other !== null && !focus.includes(other)) focus.push(other);
        if (focus.length >= 16) break;
      }
    }
    const names = [...focus, ...connected.filter((name) => !focus.includes(name))].slice(0, 16);
    if (!names.length) {
      card.append(
        el('p', 'muted', 'No cross-region relationship was found within the discovery budget.')
      );
      return card;
    }
    const edges = all.edges.filter(
      (e) => names.includes(JSON.parse(e.source)[0]) && names.includes(JSON.parse(e.target)[0])
    );
    const level = new Map(names.map((n) => [n, 0]));
    // Bound relaxation: cycles remain visible but never stall the layout.
    for (let i = 0; i < Math.min(names.length, 5); ++i)
      for (const edge of edges) {
        const from = JSON.parse(edge.source)[0],
          to = JSON.parse(edge.target)[0];
        if (from !== to) level.set(to, Math.min(4, Math.max(level.get(to), level.get(from) + 1)));
      }
    const columns = new Map();
    for (const name of names) {
      const column = level.get(name);
      if (!columns.has(column)) columns.set(column, []);
      columns.get(column).push(name);
    }
    const w = (Math.max(...columns.keys()) + 1) * 170 + 20,
      h = Math.max(160, ...[...columns.values()].map((c) => c.length * 84 + 40));
    const chart = svg('svg', {
      viewBox: `0 0 ${w} ${h}`,
      class: 'connection-graph',
      role: 'img',
      'aria-label': 'Relationships among measured regions'
    });
    const defs = svg('defs'),
      marker = svg('marker', {
        id: 'arrow',
        viewBox: '0 0 10 10',
        refX: 9,
        refY: 5,
        markerWidth: 5,
        markerHeight: 5,
        orient: 'auto-start-reverse'
      });
    marker.append(svg('path', { d: 'M 0 0 L 10 5 L 0 10 z', class: 'edge-arrow' }));
    defs.append(marker);
    chart.append(defs);
    const positions = new Map();
    for (const [column, members] of columns)
      members.forEach((name, row) =>
        positions.set(name, { x: column * 170 + 12, y: 25 + row * 84 })
      );
    const unique = new Set();
    for (const edge of edges) {
      const from = JSON.parse(edge.source)[0],
        to = JSON.parse(edge.target)[0],
        key = from + '\0' + to;
      if (unique.has(key)) continue;
      unique.add(key);
      const a = positions.get(from),
        b = positions.get(to);
      const d =
        from === to
          ? `M ${a.x + 20} ${a.y} C ${a.x + 10} ${a.y - 28}, ${a.x + 95} ${a.y - 28}, ${a.x + 90} ${a.y}`
          : `M ${a.x + 128} ${a.y + 23} C ${a.x + 155} ${a.y + 23}, ${b.x - 28} ${b.y + 23}, ${b.x} ${b.y + 23}`;
      const line = svg('path', {
        d,
        class: 'graph-edge' + (from === selected || to === selected ? ' highlighted' : ''),
        'marker-end': 'url(#arrow)'
      });
      line.append(svg('title', {}, `${from} → ${to} (${edge.kind})`));
      chart.append(line);
    }
    for (const [name, pos] of positions) {
      const node = svg('g', {
        class: 'graph-node' + (name === selected ? ' selected' : ''),
        tabindex: 0,
        role: 'button',
        'aria-label': 'Inspect ' + name
      });
      node.append(svg('rect', { x: pos.x, y: pos.y, width: 128, height: 46, rx: 8 }));
      node.append(
        svg(
          'text',
          { x: pos.x + 64, y: pos.y + 28, 'text-anchor': 'middle' },
          name.length > 16 ? name.slice(0, 15) + '…' : name
        )
      );
      node.addEventListener('click', () => choose(name));
      node.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') choose(name);
      });
      chart.append(node);
    }
    card.append(chart);
    if (connected.length > names.length)
      card.append(
        el(
          'p',
          'muted small',
          `Graph shows ${names.length} of ${connected.length} connected regions, prioritizing the selected region’s neighborhood. The list below includes every discovered relationship.`
        )
      );
    return card;
  }
  function relationshipsView() {
    const main = el('div');
    append(main, connectionGraph());
    const card = append(
      el('section', 'card'),
      heading(
        'All discovered relationships',
        `${model.relations.length} equalities held at their recorded target calls. Alternative equations can disagree after an intervention.`
      )
    );
    if (model.discovery?.truncated || model.discovery?.warnings?.length)
      card.append(
        el(
          'div',
          'notice warn',
          'Discovery was limited. ' + (model.discovery.warnings || []).join(' ')
        )
      );
    const search = el('input', 'search');
    search.type = 'search';
    search.placeholder = 'Filter by region, PCV, or history operation';
    search.setAttribute('aria-label', 'Filter relationships');
    const list = el('div', 'relationship-list');
    const draw = () => {
      list.replaceChildren();
      const matches = model.relations.filter((r) =>
        M.relationship(r).toLowerCase().includes(search.value.toLowerCase())
      );
      for (const relation of matches) {
        const row = el('article', 'relationship');
        append(
          row,
          el('code', 'equation', M.relationship(relation)),
          append(
            el('div', 'tags'),
            badge('observed · exact at sampled calls', 'observed'),
            el(
              'span',
              'muted small',
              `${relation.evidence?.calls ?? '?'} calls · ${relation.evidence?.groups ?? '?'} process/run groups`
            )
          )
        );
        const alternatives = relation.terms.reduce((n, t) => n + (t.aliases?.length || 0), 0);
        if (alternatives)
          row.append(
            el(
              'p',
              'muted small',
              `${alternatives} history features coincide with these terms in this trace. That equivalence is not guaranteed after a change.`
            )
          );
        append(
          row,
          button(
            'Inspect ' + relation.target.region,
            () => choose(relation.target.region, 'interface'),
            'link-button'
          )
        );
        list.append(row);
      }
      if (!matches.length) list.append(el('p', 'empty', 'No matching relationships.'));
    };
    search.addEventListener('input', draw);
    draw();
    append(card, search, list);
    main.append(card);
    return main;
  }
  function select(options, value, label, onChange) {
    const element = el('select');
    element.setAttribute('aria-label', label);
    options.forEach(([key, text]) => {
      const option = el('option', '', text);
      option.value = key;
      element.append(option);
    });
    element.value = value;
    element.addEventListener('change', () => onChange(element.value));
    return element;
  }
  function ensureEdit(r = region()) {
    if (!edits.length && r?.states.length)
      edits.push({ region: r.id, state: r.states[0], op: 'scale', value: 2 });
    if (!edits.length) {
      const first = model.regions.find((r) => r.states.length);
      if (first) edits.push({ region: first.id, state: first.states[0], op: 'scale', value: 2 });
    }
  }
  function updateScenario() {
    persist();
    try {
      result = M.replay(model, {
        edits: structuredClone(edits),
        relations: [...assumptionIds],
        proposals: structuredClone(proposals),
        costBasis: rawCosts ? 'recorded' : 'calibrated',
        auditAlternatives
      });
      scenarioError = '';
    } catch (error) {
      result = null;
      scenarioError = error.message;
    }
    const target = document.getElementById('scenario-result');
    if (target) target.replaceChildren(scenarioResults());
  }
  function scenarioResults() {
    const main = el('div');
    if (scenarioError) {
      main.append(el('div', 'notice warn', scenarioError));
      return main;
    }
    if (!result) return main;
    result.warnings.forEach((message) => main.append(el('p', 'scenario-note', message)));
    const card = append(
      el('section', 'card'),
      heading(
        'Changes by region',
        'Mean explained instructions per call. Rows are independent; they are not added into total cost or latency.'
      )
    );
    const label = el('label', 'check-label'),
      checkbox = el('input');
    checkbox.type = 'checkbox';
    checkbox.checked = onlyAffected;
    checkbox.addEventListener('change', () => {
      onlyAffected = checkbox.checked;
      updateScenario();
    });
    append(label, checkbox, el('span', '', 'Only regions with changed PCVs'));
    card.append(label);
    const table = el('table', 'scenario-table');
    table.append(
      append(
        el('thead'),
        append(
          el('tr'),
          ...['Region', 'Before', 'Scenario', 'Change', 'Support'].map((h) => el('th', '', h))
        )
      )
    );
    const body = el('tbody');
    const rows = result.regions
      .filter((row) => !onlyAffected || row.changedCalls)
      .sort((a, b) => Math.abs(b.delta || 0) - Math.abs(a.delta || 0));
    for (const row of rows) {
      const tr = el('tr');
      tr.dataset.region = row.region;
      const name = append(
        el('td'),
        button(row.region, () => choose(row.region, 'interface'), 'link-button'),
        el('small', 'muted', `${row.changedCalls}/${row.calls} calls have changed PCVs`)
      );
      const delta = row.delta === null ? 'Unknown' : (row.delta > 0 ? '+' : '') + fmt(row.delta);
      const support = el('td', 'support');
      row.support.forEach((value) =>
        support.append(badge(value, value === 'observed' ? 'observed' : 'warn'))
      );
      if (row.unknownCalls)
        support.append(badge(`${row.modelledCalls}/${row.calls} calls modelled`, 'warn'));
      if (row.unobservedBaselineCalls)
        support.append(badge(`${row.unobservedBaselineCalls} baseline states not in fit`, 'warn'));
      if (row.unexplainedCalls) support.append(badge('Unexplained cost not predicted', 'warn'));
      row.reasons.forEach((reason) => support.append(el('small', 'muted', reason)));
      append(
        tr,
        name,
        el('td', 'number', row.before === null ? '—' : fmt(row.before)),
        el('td', 'number', row.after === null ? '—' : fmt(row.after)),
        el('td', 'number ' + (row.delta > 0 ? 'increase' : row.delta < 0 ? 'decrease' : ''), delta),
        support
      );
      body.append(tr);
    }
    table.append(body);
    card.append(append(el('div', 'table-scroll'), table));
    if (!rows.length)
      card.append(
        el(
          'p',
          'empty',
          'No recorded PCVs changed. Choose an intervention or enable relationships as assumptions.'
        )
      );
    append(
      card,
      append(
        el('div', 'actions'),
        button(
          'Save scenario and per-region results',
          () => send({ type: 'exportScenario', scenario: result.scenario }),
          'secondary'
        ),
        button(
          'Compare against a new measured report',
          () => send({ type: 'loadValidationReport' }),
          'secondary'
        )
      )
    );
    main.append(card);
    if (result.alternativeChecks) {
      const disagreements = result.alternativeChecks.filter((row) => row.disagreements);
      const audit = append(
        el('section', 'card'),
        heading(
          'Which relationships need another experiment?',
          `${result.alternativeChecks.length} alternative equations checked in the selected scenario history. These are predictions, not measured counterexamples.`
        )
      );
      if (!disagreements.length)
        audit.append(
          el(
            'p',
            'muted',
            'No disagreements found for these interventions. Agreement does not establish causality.'
          )
        );
      for (const row of disagreements) {
        const detail = el('details', 'relationship');
        detail.open = disagreements.length <= 3;
        detail.append(
          el(
            'summary',
            '',
            `${row.target.region}.${row.target.state}: disagreement at ${row.disagreements}/${row.calls} calls`
          )
        );
        append(
          detail,
          el('code', 'equation', 'Selected: ' + row.selectedEquation),
          el('code', 'equation', 'Alternative: ' + row.candidateEquation)
        );
        for (const example of row.examples)
          detail.append(
            el(
              'p',
              'mono small',
              `call ${example.seq}: selected ${example.selectedValue}, alternative ${example.alternativeValue}`
            )
          );
        detail.append(
          el(
            'p',
            'small muted',
            'If this intervention can be realized in the program, a small new run can distinguish these equations.'
          )
        );
        audit.append(detail);
      }
      main.append(audit);
    }
    if (validationModel) main.append(validationView());
    const detail = el('details', 'card');
    detail.append(el('summary', '', 'Inspect propagated state changes'));
    for (const row of rows) {
      if (!row.changes.length) continue;
      detail.append(el('h3', '', row.region));
      for (const change of row.changes) {
        const call = el('details', 'change-explanation');
        call.append(
          el(
            'summary',
            '',
            `call ${change.seq}: ${JSON.stringify(change.before)} → ${JSON.stringify(change.after)}`
          )
        );
        for (const step of change.derivations || []) {
          if (step.equation) {
            call.append(el('code', 'equation', step.equation));
            for (const input of step.inputs)
              call.append(
                el(
                  'div',
                  'mono small',
                  `${input.kind}(${input.region}${input.state === null ? '' : '.' + input.state}) = ${input.value}`
                )
              );
            call.append(el('p', 'small', `Computed ${step.state} = ${step.result}`));
          } else
            call.append(
              el(
                'p',
                'small',
                `Direct ${step.intervention} intervention on ${step.state}: ${step.before} → ${step.result}`
              )
            );
        }
        detail.append(call);
      }
    }
    main.append(detail);
    return main;
  }
  function validationView() {
    const card = append(
      el('section', 'card'),
      heading(
        'Check against a new execution',
        validationName + ' · State propagation and cost prediction are checked separately.'
      )
    );
    let validation;
    try {
      validation = M.validateScenario(model, result.scenario, validationModel);
    } catch (error) {
      card.append(el('p', 'notice warn', error.message));
      return card;
    }
    if (!validation.structureMatches) {
      card.append(el('p', 'notice warn', validation.reason));
      return card;
    }
    card.append(
      el(
        'p',
        'small muted',
        'Calls, marker order, thread roles, and nesting match. Cost error uses only predictions with no unexplained blocks and compares against measured per-state means.'
      )
    );
    const table = el('table');
    table.append(
      append(
        el('thead'),
        append(
          el('tr'),
          ...[
            'Region',
            'State matches',
            'Cost checks',
            'Predicted',
            'Measured',
            'Absolute error'
          ].map((h) => el('th', '', h))
        )
      )
    );
    const body = el('tbody');
    for (const row of validation.regions) {
      const error =
        row.relativeAbsoluteError === null
          ? 'Not checked'
          : (100 * row.relativeAbsoluteError).toFixed(2) + '%';
      const tr = el('tr');
      tr.dataset.validationRegion = row.region;
      append(
        tr,
        append(
          el('td'),
          button(row.region, () => choose(row.region, 'interface'), 'link-button')
        ),
        el('td', 'number', `${row.stateMatches}/${row.calls}`),
        el('td', 'number', `${row.costChecks}/${row.calls}`),
        el('td', 'number', row.meanPredicted === null ? '—' : fmt(row.meanPredicted)),
        el('td', 'number', row.meanObserved === null ? '—' : fmt(row.meanObserved)),
        el('td', 'number ' + (row.relativeAbsoluteError > 0.05 ? 'increase' : ''), error)
      );
      body.append(tr);
      if (row.examples.length) {
        const examples = el('tr'),
          cell = el('td');
        cell.colSpan = 6;
        const detail = el('details');
        detail.append(el('summary', '', 'Inspect disagreements'));
        row.examples.forEach((example) =>
          detail.append(el('pre', 'state-change', JSON.stringify(example, null, 2)))
        );
        cell.append(detail);
        examples.append(cell);
        body.append(examples);
      }
    }
    table.append(body);
    card.append(append(el('div', 'table-scroll'), table));
    card.append(
      button(
        'Clear validation report',
        () => {
          validationModel = null;
          updateScenario();
        },
        'quiet'
      )
    );
    return card;
  }
  function proposalView() {
    const card = append(
      el('section', 'card'),
      heading(
        'Propose a state relationship',
        'State relationships may be nonlinear or conditional; each region’s cost interface stays affine in its declared PCVs. The expression is checked against every recorded target call.'
      )
    );
    const targets = model.regions.filter((r) => r.states.length);
    if (!targets.length) return card;
    if (!proposalDraft) {
      const r = region().states.length ? region() : targets[0];
      proposalDraft = { target: { region: r.id, state: r.states[0] }, expression: '' };
    }
    const row = el('div', 'actions');
    append(
      row,
      select(
        targets.map((r) => [r.id, r.name]),
        proposalDraft.target.region,
        'Proposed target region',
        (value) => {
          proposalDraft.target = {
            region: value,
            state: model.regions.find((r) => r.id === value).states[0]
          };
          proposalResult = null;
          render();
        }
      ),
      select(
        model.regions.find((r) => r.id === proposalDraft.target.region).states.map((s) => [s, s]),
        proposalDraft.target.state,
        'Proposed target PCV',
        (value) => {
          proposalDraft.target.state = value;
          proposalResult = null;
        }
      )
    );
    const input = el('input', 'search mono');
    input.setAttribute('aria-label', 'Proposed state expression');
    input.placeholder = 'last("dequeue", "items") ** 2';
    input.value = proposalDraft.expression;
    input.addEventListener('input', () => {
      proposalDraft.expression = input.value;
      proposalResult = null;
    });
    append(
      card,
      row,
      input,
      el(
        'p',
        'small muted',
        'History: last("region", "pcv"), cum(...), cumend(...), count("region"). Integer arithmetic: + − * // % **. Branches: condition ? then : else. Histories before the current call are used.'
      ),
      button(
        'Check proposed relationship',
        () => {
          try {
            proposalResult = M.proposeRelationship(
              model,
              proposalDraft.target,
              proposalDraft.expression
            );
          } catch (error) {
            proposalResult = { error: error.message };
          }
          render();
        },
        'secondary'
      )
    );
    if (proposalResult?.error) card.append(el('p', 'notice warn', proposalResult.error));
    else if (proposalResult) {
      const { relation, check } = proposalResult;
      card.append(
        el(
          'p',
          'notice ' + (check.holds ? '' : 'warn'),
          check.holds
            ? `Matches all ${check.calls} recorded target calls. This is evidence on this trace, not a proof for new inputs.`
            : `Failed at ${check.mismatches} of ${check.calls} recorded target calls.`
        )
      );
      for (const example of check.examples)
        card.append(el('pre', 'state-change', JSON.stringify(example, null, 2)));
      if (check.holds)
        card.append(
          button(
            'Use checked proposal as an assumption',
            () => {
              const available = scenarioModel().relations;
              assumptionIds = assumptionIds.filter(
                (id) =>
                  !available.some(
                    (r) =>
                      r.id === id &&
                      r.target.region === relation.target.region &&
                      r.target.state === relation.target.state
                  )
              );
              proposals = proposals.filter(
                (p) =>
                  M.id(p.target.region, p.target.state) !==
                  M.id(relation.target.region, relation.target.state)
              );
              proposals.push({ target: relation.target, expression: relation.expression });
              assumptionIds.push(relation.id);
              proposalResult = null;
              render();
            },
            'secondary'
          )
        );
    }
    if (proposals.length)
      card.append(
        button(
          'Remove proposed relationships',
          () => {
            const ids = new Set(
              scenarioModel()
                .relations.filter((r) => r.proposed)
                .map((r) => r.id)
            );
            assumptionIds = assumptionIds.filter((id) => !ids.has(id));
            proposals = [];
            proposalResult = null;
            render();
          },
          'quiet'
        )
      );
    return card;
  }
  function scenarioView() {
    if (model.validity?.errors?.length || model.validity?.traceErrors?.length)
      return el('p', 'empty', 'Correct the measurement errors before running scenarios.');
    ensureEdit();
    const main = el('div');
    const card = append(
      el('section', 'card'),
      heading(
        'What if a PCV changed?',
        'Change one or more declared states, then choose which observed relationships to use as assumptions.'
      )
    );
    const editing = el('div', 'interventions');
    edits.forEach((edit, index) => {
      const row = el('div', 'intervention');
      append(
        row,
        select(
          model.regions.filter((r) => r.states.length).map((r) => [r.id, r.name]),
          edit.region,
          'Region for intervention ' + (index + 1),
          (value) => {
            edit.region = value;
            edit.state = model.regions.find((r) => r.id === value).states[0];
            render();
          }
        ),
        select(
          model.regions.find((r) => r.id === edit.region).states.map((s) => [s, s]),
          edit.state,
          'PCV for intervention ' + (index + 1),
          (value) => {
            edit.state = value;
            updateScenario();
          }
        ),
        select(
          [
            ['scale', 'multiply by'],
            ['add', 'add'],
            ['set', 'set to']
          ],
          edit.op,
          'Operation ' + (index + 1),
          (value) => {
            edit.op = value;
            updateScenario();
          }
        )
      );
      const input = el('input', 'value-input');
      input.type = 'number';
      input.step = 'any';
      input.value = String(edit.value);
      input.setAttribute('aria-label', 'Intervention value ' + (index + 1));
      input.addEventListener('input', () => {
        edit.value = input.value === '' ? NaN : Number(input.value);
        updateScenario();
      });
      append(
        row,
        input,
        button(
          'Remove',
          () => {
            edits.splice(index, 1);
            render();
          },
          'quiet'
        )
      );
      editing.append(row);
    });
    append(
      card,
      editing,
      button(
        '+ Add another PCV',
        () => {
          const r = region().states.length ? region() : model.regions.find((r) => r.states.length);
          if (r) edits.push({ region: r.id, state: r.states[0], op: 'scale', value: 2 });
          render();
        },
        'secondary'
      )
    );
    main.append(card);
    main.append(proposalView());
    const availableRelations = scenarioModel().relations;
    const assumptions = append(
      el('section', 'card'),
      heading(
        'Relationship assumptions',
        `${assumptionIds.length} selected. Nothing propagates through an observed equation until you opt in.`
      )
    );
    const auditLabel = el('label', 'check-label'),
      audit = el('input');
    audit.type = 'checkbox';
    audit.checked = auditAlternatives;
    audit.addEventListener('change', () => {
      auditAlternatives = audit.checked;
      updateScenario();
    });
    append(auditLabel, audit, el('span', '', 'Compare alternative relationships'));
    assumptions.append(auditLabel);
    const actions = el('div', 'actions');
    append(
      actions,
      button(
        'Use first observed equation for each target',
        () => {
          const unique = new Map();
          for (const r of availableRelations) {
            const key = M.id(r.target.region, r.target.state);
            if (!unique.has(key)) unique.set(key, r.id);
          }
          assumptionIds = [...unique.values()];
          render();
        },
        'secondary'
      ),
      button(
        'Clear assumptions',
        () => {
          assumptionIds = [];
          render();
        },
        'quiet'
      )
    );
    assumptions.append(actions);
    const targets = new Map();
    for (const relation of availableRelations) {
      const key = M.id(relation.target.region, relation.target.state);
      if (!targets.has(key)) targets.set(key, []);
      targets.get(key).push(relation);
    }
    const details = el('details');
    details.open = targets.size <= 8;
    details.append(el('summary', '', `Choose equations for ${targets.size} target states`));
    for (const choices of targets.values()) {
      const target = choices[0].target;
      const row = el('label', 'assumption-row');
      append(
        row,
        el('span', 'mono', target.region + '.' + target.state),
        select(
          [['', 'No propagation'], ...choices.map((r) => [r.id, M.relationship(r)])],
          choices.find((r) => assumptionIds.includes(r.id))?.id || '',
          `Relationship for ${target.region}.${target.state}`,
          (value) => {
            const ids = new Set(choices.map((r) => r.id));
            assumptionIds = assumptionIds.filter((id) => !ids.has(id));
            if (value) assumptionIds.push(value);
            updateScenario();
          }
        )
      );
      details.append(row);
    }
    assumptions.append(details);
    main.append(assumptions);
    const results = el('div');
    results.id = 'scenario-result';
    main.append(results);
    // Update after insertion by render().
    return main;
  }
  function render() {
    root.replaceChildren();
    if (!model) {
      root.append(el('div', 'empty', 'Open a drperf report to explore its regions.'));
      return;
    }
    const top = el('header', 'topbar');
    append(
      top,
      append(
        el('div', 'brand'),
        el('span', 'brand-mark', 'd'),
        append(el('div'), el('strong', '', 'drperf'), el('span', 'muted', 'REGION EXPLORER'))
      ),
      append(
        el('div', 'top-meta'),
        el(
          'span',
          'muted',
          `${model.regions.length} regions · ${model.relations.length} relationships`
        ),
        button(
          rawCosts ? 'View calibrated formulas' : 'View recorded counts',
          () => {
            rawCosts = !rawCosts;
            model = rawCosts ? M.recordedCosts(loadedModel) : loadedModel;
            render();
          },
          'quiet'
        ),
        button('Reload', () => send({ type: 'reload' }), 'quiet')
      )
    );
    root.append(top);
    const errors = [...(model.validity?.errors || []), ...(model.validity?.traceErrors || [])];
    if (errors.length)
      root.append(
        el(
          'div',
          'notice error',
          'Measurement errors: ' + errors.join('; ') + '. Scenario predictions are disabled.'
        )
      );
    const shell = el('div', 'shell'),
      aside = el('aside', 'region-sidebar');
    aside.append(el('div', 'eyebrow', 'MEASURED REGIONS'));
    const search = el('input', 'search');
    search.type = 'search';
    search.placeholder = 'Find a region or PCV';
    search.value = filter;
    search.setAttribute('aria-label', 'Find a region or PCV');
    const list = el('nav', 'region-list');
    function fill() {
      list.replaceChildren();
      for (const r of model.regions.filter((r) =>
        (r.name + ' ' + r.states.join(' ')).toLowerCase().includes(filter.toLowerCase())
      )) {
        const b = button(
          '',
          () => choose(r.id),
          'region-button' + (r.id === selected ? ' active' : '')
        );
        b.setAttribute('aria-current', r.id === selected ? 'true' : 'false');
        const share = r.regimes.length
          ? Math.max(...r.regimes.map((f) => f.unexplainedShare))
          : null;
        append(
          b,
          el('span', 'region-name', r.name),
          append(
            el('span', 'region-meta'),
            el('span', 'muted', r.states.join(', ') || 'no PCVs'),
            share === null
              ? badge('needs states')
              : share > 0.05
                ? badge(`${Math.round(share * 100)}% unexplained`, 'warn')
                : el('span', 'good-dot', '●')
          )
        );
        list.append(b);
      }
    }
    search.addEventListener('input', () => {
      filter = search.value;
      fill();
    });
    fill();
    append(
      aside,
      search,
      list,
      el(
        'p',
        'sidebar-note',
        'Each interface describes one region. There is no total-cost or latency estimate.'
      )
    );
    const content = el('main', 'content');
    const tabs = el('nav', 'tabs');
    tabs.setAttribute('aria-label', 'Explorer views');
    for (const [key, title] of [
      ['interface', 'Interface'],
      ['relations', 'Relationships'],
      ['scenario', 'What-if']
    ]) {
      const b = button(
        title,
        () => {
          activeTab = key;
          persist();
          render();
        },
        'tab' + (activeTab === key ? ' active' : '')
      );
      b.setAttribute('aria-selected', String(activeTab === key));
      tabs.append(b);
    }
    content.append(tabs);
    const r = region();
    if (!r) {
      content.append(el('p', 'empty', 'No application regions in this report.'));
    } else {
      selected = r.id;
      content.append(
        activeTab === 'interface'
          ? interfaceView(r)
          : activeTab === 'relations'
            ? relationshipsView()
            : scenarioView()
      );
    }
    append(shell, aside, content);
    root.append(shell);
    if (activeTab === 'scenario') updateScenario();
    persist();
  }
  window.addEventListener('message', (event) => {
    const message = event.data;
    if (message?.type === 'model') {
      try {
        const previousId = model?.id;
        loadedModel = M.validate(message.model);
        model = rawCosts ? M.recordedCosts(loadedModel) : loadedModel;
        const current = host?.getState() || saved;
        const same = current.modelId === model.id;
        selected = message.selected || (same ? current.selected : null) || model.regions[0]?.id;
        if (previousId !== model.id) {
          validationModel = null;
          proposalResult = null;
          proposalDraft = null;
          proposals = same && Array.isArray(current.proposals) ? current.proposals : [];
          edits = same && Array.isArray(current.edits) ? current.edits : [];
          assumptionIds = same && Array.isArray(current.assumptionIds) ? current.assumptionIds : [];
        }
        render();
      } catch (error) {
        root.replaceChildren(el('div', 'notice error', 'Cannot display report: ' + error.message));
      }
    } else if (message?.type === 'validationModel') {
      try {
        validationModel = M.validate(message.model);
        validationName = message.name || 'Measured report';
        updateScenario();
      } catch (error) {
        scenarioError = error.message;
        updateScenario();
      }
    } else if (message?.type === 'select' && model?.regions.some((r) => r.id === message.region)) {
      selected = message.region;
      render();
    }
  });
  render();
  send({ type: 'ready' });
})();
