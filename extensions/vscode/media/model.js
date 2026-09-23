/* Shared by the VS Code webview, extension host, and Node tests.
 * Scenario execution is a conditional replay of marker order, never timing.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports)
    module.exports = factory(require('./expressions'));
  else root.DrperfModel = factory(root.DrperfExpressions);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (Expressions) {
  'use strict';
  const schema = 'drperf.explorer.v1';
  const compiledExpressions = new WeakMap();
  const id = (region, state) => JSON.stringify([region, state]);
  function keyFactory() {
    const regions = new Map();
    return (region, state) => {
      let states = regions.get(region);
      if (!states) {
        states = new Map();
        regions.set(region, states);
      }
      if (!states.has(state)) states.set(state, id(region, state));
      return states.get(state);
    };
  }
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const safe = (value) => Number.isSafeInteger(value);
  const sum = (values) => values.reduce((a, b) => a + b, 0);
  const coefficient = (value) => {
    const rounded = Math.round(value);
    if (rounded !== 0 && Math.abs(value - rounded) < 1e-9 * Math.max(1, Math.abs(value)))
      return String(rounded);
    return finite(value)
      ? new Intl.NumberFormat('en-US', { maximumSignificantDigits: 12, useGrouping: false }).format(
          value
        )
      : String(value);
  };
  const format = (value) =>
    finite(value)
      ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value)
      : String(value);

  function validate(model) {
    if (!model || model.schema !== schema)
      throw new Error(
        'Not a supported drperf explorer report. Export raw measurements with drperf-export.'
      );
    if (
      !Array.isArray(model.regions) ||
      !Array.isArray(model.relations) ||
      !Array.isArray(model.trace?.events)
    )
      throw new Error('Report is missing regions, relations, or trace events.');
    if (
      model.regions.length > 10000 ||
      model.relations.length > 100000 ||
      model.trace.events.length > 100000
    )
      throw new Error('Report exceeds viewer limits. Export fewer runs or lower the trace limit.');
    const integer = (x) => safe(x) || (typeof x === 'string' && /^[+-]?\d{1,128}$/.test(x));
    const names = new Map();
    const pointsValid = (points, n, fitted) =>
      Array.isArray(points) &&
      points.every(
        (p) =>
          p &&
          Array.isArray(p.state) &&
          p.state.length === n &&
          p.state.every(integer) &&
          safe(p.calls) &&
          p.calls >= 0 &&
          (fitted ? finite(p.explained) && finite(p.unexplained) : finite(p.observed))
      );
    for (const region of model.regions) {
      if (
        !region ||
        typeof region.id !== 'string' ||
        typeof region.name !== 'string' ||
        names.has(region.id) ||
        !Array.isArray(region.states) ||
        !region.states.every((s) => typeof s === 'string') ||
        new Set(region.states).size !== region.states.length ||
        !Array.isArray(region.regimes) ||
        !Array.isArray(region.sources)
      )
        throw new Error('Invalid or duplicate region schema.');
      names.set(region.id, region);
      if (
        !safe(region.calls) ||
        region.calls < 0 ||
        !pointsValid(region.points, region.states.length, false)
      )
        throw new Error(`Invalid observations for ${region.name}.`);
      for (const source of region.sources)
        if (
          !source ||
          typeof source.path !== 'string' ||
          !safe(source.line) ||
          source.line < 1 ||
          !safe(source.endLine) ||
          source.endLine < source.line ||
          typeof source.sha256 !== 'string'
        )
          throw new Error('Invalid source location.');
      for (const fit of region.regimes) {
        if (
          !fit ||
          !Array.isArray(fit.coefficients) ||
          fit.coefficients.length !== region.states.length ||
          !fit.coefficients.every(finite) ||
          !finite(fit.constant) ||
          !pointsValid(fit.points, region.states.length, true) ||
          !Array.isArray(fit.range) ||
          fit.range.length !== region.states.length ||
          !fit.range.every(
            (pair) =>
              Array.isArray(pair) &&
              pair.length === 2 &&
              pair.every(integer) &&
              asInteger(pair[0]) <= asInteger(pair[1])
          ) ||
          !Array.isArray(fit.dependent) ||
          !fit.dependent.every((s) => region.states.includes(s)) ||
          !finite(fit.unexplainedShare)
        )
          throw new Error(`Invalid formula for ${region.name}.`);
        const attribution = fit.attribution;
        if (
          !attribution ||
          !Array.isArray(attribution.coefficients) ||
          attribution.coefficients.length !== region.states.length ||
          ![...attribution.coefficients, attribution.constant, attribution.unexplained].every(
            (list) =>
              Array.isArray(list) &&
              list.every(
                (row) =>
                  row &&
                  typeof row.function === 'string' &&
                  typeof row.module === 'string' &&
                  finite(row.instructions)
              )
          )
        )
          throw new Error(`Invalid attribution for ${region.name}.`);
      }
    }
    const relationIds = new Set();
    for (const relation of model.relations) {
      if (
        !relation ||
        typeof relation.id !== 'string' ||
        relationIds.has(relation.id) ||
        !names.get(relation.target?.region)?.states.includes(relation.target?.state) ||
        !Array.isArray(relation.terms) ||
        relation.terms.length > 256 ||
        (!relation.terms.length && !relation.expression) ||
        !finite(relation.constant)
      )
        throw new Error('Invalid relationship.');
      relationIds.add(relation.id);
      for (const term of relation.terms) {
        if (
          !term ||
          !names.has(term.region) ||
          !['last', 'cum', 'cumend', 'count'].includes(term.kind) ||
          !finite(term.coefficient) ||
          (term.kind !== 'count' && !names.get(term.region).states.includes(term.state))
        )
          throw new Error('Invalid relationship term.');
        if (term.coefficientExact !== undefined) rational(term.coefficientExact);
      }
      if (relation.constantExact !== undefined) rational(relation.constantExact);
      if (relation.expression) {
        const compiled = Expressions.compile(relation.expression);
        const declared = new Set(
          relation.terms.map((t) => JSON.stringify([t.kind, t.region, t.state]))
        );
        if (
          compiled.features.length !== declared.size ||
          compiled.features.some((f) => !declared.has(JSON.stringify([f.kind, f.region, f.state])))
        )
          throw new Error('Expression dependencies do not match relationship terms.');
        compiledExpressions.set(relation, compiled);
      }
    }
    for (const event of model.trace.events)
      if (
        !event ||
        !names.has(event.region) ||
        typeof event.group !== 'string' ||
        typeof event.thread !== 'string' ||
        !safe(event.seq) ||
        !safe(event.end) ||
        event.end <= event.seq ||
        !event.values ||
        typeof event.values !== 'object' ||
        Array.isArray(event.values) ||
        !Object.values(event.values).every(integer)
      )
        throw new Error('Invalid trace event.');
    const markerIds = new Map(),
      stacks = new Map();
    for (const events of groupedEvents(model.trace.events))
      for (const event of events) {
        if (!markerIds.has(event.group)) markerIds.set(event.group, new Set());
        const used = markerIds.get(event.group);
        if (used.has(event.seq) || used.has(event.end))
          throw new Error('Duplicate marker sequence in trace.');
        used.add(event.seq);
        used.add(event.end);
        const thread = id(event.group, event.thread);
        if (!stacks.has(thread)) stacks.set(thread, []);
        const stack = stacks.get(thread);
        while (stack.length && stack.at(-1) < event.seq) stack.pop();
        if (stack.length && event.end > stack.at(-1))
          throw new Error('Crossing region boundaries on one thread.');
        stack.push(event.end);
      }
    return model;
  }

  function recordedCosts(model) {
    if (model.measurement?.markerAdjustment === 'none; marker API instructions retained')
      return model;
    const result = structuredClone(model);
    result.id = model.id + ':recorded';
    result.measurement = {
      ...model.measurement,
      markerAdjustment: 'none; marker API instructions retained'
    };
    for (const region of result.regions) {
      const calibration = region.markerCalibration ?? region.regimes[0]?.markerCalibration ?? 0;
      for (const point of region.points)
        point.observed = point.recorded ?? point.observed + calibration;
      for (const fit of region.regimes) {
        const correction = fit.markerCalibration ?? calibration;
        fit.constant += correction;
        for (const point of fit.points) point.explained += correction;
        const total = sum(fit.points.map((p) => p.explained + p.unexplained));
        fit.unexplainedShare = total ? sum(fit.points.map((p) => p.unexplained)) / total : 0;
        fit.markerCalibration = 0;
      }
      region.markerCalibration = 0;
      region.diagnostics = (region.diagnostics || []).filter(
        (message) => !/Marker-overhead|calibrated explained/.test(message)
      );
    }
    return result;
  }

  function formula(region, fit = region.regimes[0]) {
    if (!fit) return 'More varied states needed';
    const terms = fit.coefficients.flatMap((value, i) =>
      value !== 0 ? [`${coefficient(value)}*${region.states[i]}`] : []
    );
    terms.push(coefficient(fit.constant));
    return (
      terms.join(' + ').replace(/\+ -/g, '- ') + (fit.blocks?.unexplained ? ' + unexplained' : '')
    );
  }

  function relationship(relation) {
    if (relation.expression)
      return `${relation.target.region}.${relation.target.state} = ${relation.expression}`;
    const terms = relation.terms.map(
      (term) =>
        `${term.coefficientExact || String(term.coefficient)}*${term.kind}(${term.region}${term.state === null ? '' : '.' + term.state})`
    );
    if (relation.constant) terms.push(relation.constantExact || String(relation.constant));
    return `${relation.target.region}.${relation.target.state} = ${terms.join(' + ').replace(/\+ -/g, '- ')}`;
  }

  function rational(value) {
    const text = String(value);
    if (text.length > 128) throw new Error('Exact coefficient is too large.');
    const fraction = /^([+-]?\d+)\/([1-9]\d*)$/.exec(text);
    if (fraction) return [BigInt(fraction[1]), BigInt(fraction[2])];
    const decimal = /^([+-]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(text);
    if (!decimal) throw new Error('Invalid exact relationship coefficient.');
    const exponent = Number(decimal[4] || 0) - (decimal[3] || '').length;
    if (Math.abs(exponent) > 308) throw new Error('Coefficient exponent is too large.');
    let n = BigInt((decimal[1] || '') + decimal[2] + (decimal[3] || '')),
      d = 1n;
    if (exponent >= 0) n *= 10n ** BigInt(exponent);
    else d = 10n ** BigInt(-exponent);
    return [n, d];
  }

  function asInteger(value) {
    if (typeof value === 'bigint') return value;
    if (safe(value)) return BigInt(value);
    if (typeof value === 'string' && /^[+-]?\d{1,128}$/.test(value)) return BigInt(value);
    throw new Error('State is not an exact integer.');
  }

  function exactRelationshipInteger(relation, featureValue) {
    if (relation.expression) {
      if (!compiledExpressions.has(relation))
        compiledExpressions.set(relation, Expressions.compile(relation.expression));
      return Expressions.evaluate(compiledExpressions.get(relation), featureValue);
    }
    let [n, d] = rational(relation.constantExact ?? relation.constant);
    for (const term of relation.terms) {
      const [tn, td] = rational(term.coefficientExact ?? term.coefficient);
      const x = featureValue(term);
      n = n * td + tn * asInteger(x) * d;
      d *= td;
    }
    if (n % d !== 0n)
      throw new Error(
        `Relationship produces a non-integer state for ${relation.target.region}.${relation.target.state}: ${n}/${d}. Revise this assumption or intervention.`
      );
    return n / d;
  }

  function exactRelationshipValue(relation, featureValue) {
    const value = Number(exactRelationshipInteger(relation, featureValue));
    if (!safe(value)) throw new Error('Relationship exceeds exact integer range.');
    return value;
  }

  function verifyRelationships(model, selectedIds = null) {
    const stateId = keyFactory();
    const chosen = model.relations.filter((r) => !selectedIds || selectedIds.includes(r.id));
    if (!chosen.length) return [];
    const results = new Map(
      chosen.map((r) => [r.id, { id: r.id, calls: 0, mismatches: 0, examples: [] }])
    );
    const byRegion = new Map();
    for (const relation of chosen) {
      if (!byRegion.has(relation.target.region)) byRegion.set(relation.target.region, []);
      byRegion.get(relation.target.region).push(relation);
    }
    const groups = new Map();
    for (const event of model.trace.events) {
      if (!groups.has(event.group)) groups.set(event.group, []);
      groups.get(event.group).push(event);
    }
    for (const [group, events] of groups) {
      events.sort((a, b) => a.seq - b.seq);
      const ended = [...events].sort((a, b) => a.end - b.end);
      const cum = new Map(),
        cumend = new Map(),
        last = new Map(),
        count = new Map();
      let ei = 0;
      for (const event of events) {
        while (ei < ended.length && ended[ei].end < event.seq) {
          const previous = ended[ei++];
          for (const [state, value] of Object.entries(previous.values)) {
            const key = stateId(previous.region, state);
            cumend.set(key, (cumend.get(key) || 0n) + asInteger(value));
          }
        }
        for (const relation of byRegion.get(event.region) || []) {
          if (!Object.hasOwn(event.values, relation.target.state)) continue;
          const result = results.get(relation.id);
          ++result.calls;
          let predicted,
            expected = asInteger(event.values[relation.target.state]);
          try {
            predicted = exactRelationshipInteger(relation, (term) => {
              if (term.kind === 'count') return count.get(term.region) || 0n;
              return (
                (term.kind === 'last' ? last : term.kind === 'cum' ? cum : cumend).get(
                  stateId(term.region, term.state)
                ) || 0n
              );
            });
          } catch (error) {
            predicted = error.message;
          }
          if (predicted !== expected) {
            ++result.mismatches;
            if (result.examples.length < 3)
              result.examples.push({
                group,
                seq: event.seq,
                expected: String(expected),
                predicted: String(predicted)
              });
          }
        }
        count.set(event.region, (count.get(event.region) || 0n) + 1n);
        for (const [state, value] of Object.entries(event.values)) {
          const key = stateId(event.region, state),
            x = asInteger(value);
          last.set(key, x);
          cum.set(key, (cum.get(key) || 0n) + x);
        }
      }
    }
    return [...results.values()].map((r) => ({ ...r, holds: r.calls > 0 && r.mismatches === 0 }));
  }

  function proposeRelationship(model, target, expression) {
    if (
      !model.trace?.complete ||
      !model.trace.events.length ||
      model.validity?.errors?.length ||
      model.validity?.traceErrors?.length
    )
      throw new Error('Checking a proposal requires a complete, valid recorded trace.');
    if (!target || typeof target.region !== 'string' || typeof target.state !== 'string')
      throw new Error('Invalid proposal target.');
    const region = model.regions.find((r) => r.id === target.region);
    if (!region?.states.includes(target.state)) throw new Error('Unknown target PCV.');
    const compiled = Expressions.compile(expression);
    for (const feature of compiled.features) {
      const source = model.regions.find((r) => r.id === feature.region);
      if (!source || (feature.kind !== 'count' && !source.states.includes(feature.state)))
        throw new Error('Unknown history region or PCV in proposed relationship.');
    }
    const proposal = {
      id: 'proposed:' + JSON.stringify([target.region, target.state, expression]),
      target,
      expression,
      constant: 0,
      terms: compiled.features.map((f) => ({ ...f, coefficient: 1 })),
      proposed: true
    };
    compiledExpressions.set(proposal, compiled);
    const check = verifyRelationships({ ...model, relations: [proposal] })[0];
    proposal.evidence = {
      ...check,
      exact: check.holds,
      groups: new Set(model.trace.events.map((e) => e.group)).size
    };
    return { relation: proposal, check };
  }

  function withProposals(model, proposals) {
    if (!Array.isArray(proposals) || proposals.length > 100)
      throw new Error('At most 100 proposed relationships are supported.');
    if (!proposals.length) return model;
    const relations = [...model.relations],
      ids = new Set(relations.map((r) => r.id));
    for (const input of proposals) {
      const { relation, check } = proposeRelationship(model, input.target, input.expression);
      if (!check.holds)
        throw new Error(
          `Proposed relationship for ${input.target.region}.${input.target.state} fails at ${check.mismatches} recorded calls.`
        );
      if (!ids.has(relation.id)) {
        relations.push(relation);
        ids.add(relation.id);
      }
    }
    return { ...model, relations };
  }

  function evaluate(region, values) {
    const state = region.states.map((name) => values[name]);
    if (!state.every(safe))
      return { ok: false, reason: 'Missing state or integer outside JavaScript exact range' };
    if (!region.regimes.length) return { ok: false, reason: 'Insufficient observed states' };
    let fit, observed;
    for (const candidate of region.regimes) {
      const point = candidate.points.find(
        (p) => p.state.length === state.length && p.state.every((v, i) => v === state[i])
      );
      if (point) {
        fit = candidate;
        observed = point;
        break;
      }
    }
    const inRange = (candidate) =>
      candidate.range.every(
        ([lo, hi], i) => safe(lo) && safe(hi) && state[i] >= lo && state[i] <= hi
      );
    let support = observed ? 'observed' : 'unobserved';
    if (!fit) fit = region.regimes.find(inRange);
    if (!fit && region.regimes.length === 1) {
      fit = region.regimes[0];
      support = 'extrapolated';
    }
    if (!fit && state.length === 1) {
      const sorted = [...region.regimes].sort((a, b) => a.range[0][0] - b.range[0][0]);
      if (state[0] < sorted[0].range[0][0]) fit = sorted[0];
      else if (state[0] > sorted.at(-1).range[0][1]) fit = sorted.at(-1);
      if (fit) support = 'extrapolated';
    }
    if (!fit) return { ok: false, reason: 'Between fitted regimes; branch boundary is unknown' };
    if (!observed && fit.dependent.length)
      return { ok: false, reason: 'PCVs were correlated; this new state is not identifiable' };
    if (!observed && fit.range.some(([lo, hi], i) => lo === hi && state[i] !== lo))
      return { ok: false, reason: 'Changed a PCV that never varied in this regime' };
    const explained = sum(fit.coefficients.map((a, i) => a * state[i])) + fit.constant;
    if (!finite(explained) || explained < -1e-6)
      return { ok: false, reason: 'Formula predicts an invalid negative or non-finite cost' };
    return {
      ok: true,
      explained,
      support,
      fit,
      state,
      unexplained: observed ? observed.unexplained : null,
      hasUnexplained: !!fit.blocks?.unexplained
    };
  }

  function graph(model, relationIds = null) {
    const nodes = new Map(),
      edges = [];
    for (const region of model.regions)
      for (const state of region.states)
        nodes.set(id(region.id, state), { id: id(region.id, state), region: region.id, state });
    for (const relation of model.relations) {
      if (relationIds && !relationIds.includes(relation.id)) continue;
      const target = id(relation.target.region, relation.target.state);
      if (!nodes.has(target)) nodes.set(target, { id: target, ...relation.target });
      for (const term of relation.terms) {
        const source = id(term.region, term.state);
        if (!nodes.has(source))
          nodes.set(source, { id: source, region: term.region, state: term.state });
        edges.push({
          source,
          target,
          relation: relation.id,
          kind: term.kind,
          coefficient: term.coefficient
        });
      }
    }
    return { nodes: [...nodes.values()], edges };
  }

  function reachable(model, region, state, relationIds = null) {
    const { edges } = graph(model, relationIds),
      seen = new Set([id(region, state)]);
    let changed;
    do {
      changed = false;
      for (const edge of edges)
        if (seen.has(edge.source) && !seen.has(edge.target)) {
          seen.add(edge.target);
          changed = true;
        }
    } while (changed);
    return seen;
  }

  function replay(model, scenario, options = {}) {
    if (
      !scenario ||
      (scenario.edits !== undefined && !Array.isArray(scenario.edits)) ||
      (scenario.relations !== undefined && !Array.isArray(scenario.relations))
    )
      throw new Error('Invalid scenario.');
    if ((scenario.edits?.length || 0) > 1000 || (scenario.relations?.length || 0) > 10000)
      throw new Error('Scenario exceeds the supported size.');
    const stateId = keyFactory();
    if (scenario.costBasis === 'recorded') model = recordedCosts(model);
    model = withProposals(model, scenario.proposals || []);
    validate(model);
    if (model.validity?.errors?.length || model.validity?.traceErrors?.length)
      throw new Error('Measurement or trace validity errors disable scenario predictions.');
    if (!model.trace.complete || !model.trace.events.length)
      throw new Error('A complete recorded trace is required for scenario replay.');
    const regionMap = new Map(model.regions.map((r) => [r.id, r]));
    const selected = new Map();
    for (const selectedId of scenario.relations || []) {
      const relation = model.relations.find((r) => r.id === selectedId);
      if (!relation) throw new Error('Scenario refers to an unknown relationship.');
      const target = stateId(relation.target.region, relation.target.state);
      if (selected.has(target))
        throw new Error(
          `Choose one relationship for ${relation.target.region}.${relation.target.state}; alternatives are not interchangeable.`
        );
      selected.set(target, relation);
    }
    const selectedIds = new Set([...selected.values()].map((r) => r.id));
    const candidates = scenario.auditAlternatives
      ? model.relations.filter((r) => selected.has(stateId(r.target.region, r.target.state)))
      : [...selected.values()];
    const checks = verifyRelationships(
      model,
      candidates.map((r) => r.id)
    );
    const verified = new Set(checks.filter((c) => c.holds).map((c) => c.id));
    const invalidRelation = checks.find((check) => selectedIds.has(check.id) && !check.holds);
    if (invalidRelation)
      throw new Error(
        `Selected relationship ${invalidRelation.id} does not hold in the recorded trace. Inspect its counterexamples before using it.`
      );
    const alternatives = new Map(),
      ambiguities = new Map();
    if (scenario.auditAlternatives)
      for (const relation of candidates) {
        if (selectedIds.has(relation.id) || !verified.has(relation.id)) continue;
        const target = stateId(relation.target.region, relation.target.state);
        if (!alternatives.has(target)) alternatives.set(target, []);
        alternatives.get(target).push(relation);
        ambiguities.set(relation.id, {
          target: relation.target,
          selected: selected.get(target).id,
          candidate: relation.id,
          selectedEquation: relationship(selected.get(target)),
          candidateEquation: relationship(relation),
          calls: 0,
          disagreements: 0,
          examples: []
        });
      }
    const relationsByRegion = new Map();
    for (const [target, relation] of selected) {
      if (!relationsByRegion.has(relation.target.region))
        relationsByRegion.set(relation.target.region, []);
      relationsByRegion.get(relation.target.region).push([target, relation]);
    }
    const edits = new Map(),
      editsByRegion = new Map();
    for (const edit of scenario.edits || []) {
      if (
        !regionMap.get(edit.region)?.states.includes(edit.state) ||
        !['scale', 'add', 'set'].includes(edit.op) ||
        !finite(edit.value)
      )
        throw new Error('Invalid PCV intervention.');
      const key = stateId(edit.region, edit.state);
      if (edits.has(key)) throw new Error('Use one intervention per PCV.');
      edits.set(key, edit);
      if (!editsByRegion.has(edit.region)) editsByRegion.set(edit.region, []);
      editsByRegion.get(edit.region).push(edit);
    }
    if (!edits.size) throw new Error('Add at least one PCV intervention.');
    const summaries = new Map(
      model.regions.map((r) => [
        r.id,
        {
          region: r.id,
          calls: 0,
          changedCalls: 0,
          modelledCalls: 0,
          unknownCalls: 0,
          beforeSum: 0,
          afterSum: 0,
          changedStates: new Set(),
          reasons: new Set(),
          support: new Set(),
          unexplainedCalls: 0,
          unobservedBaselineCalls: 0,
          changes: []
        }
      ])
    );
    const warnings = new Set();
    const evaluations = new Map();
    const cachedEvaluation = (region, values) => {
      const key = JSON.stringify([region.id, region.states.map((state) => values[state])]);
      if (!evaluations.has(key)) evaluations.set(key, evaluate(region, values));
      return evaluations.get(key);
    };
    const predictedEvents = [];
    if (selected.size)
      warnings.add('Selected observed relationships are assumptions, not established causes.');
    warnings.add(
      'Call counts, nesting, and marker order are fixed. New branches/calls and elapsed time are not predicted.'
    );
    let group = null,
      cumulative,
      completed,
      latest,
      counts,
      pending;
    // End events are pre-sorted once per group; replayed values are filled at begin.
    const groups = new Map();
    for (const event of model.trace.events) {
      if (
        !regionMap.has(event.region) ||
        !safe(event.seq) ||
        !safe(event.end) ||
        event.end <= event.seq
      )
        throw new Error('Invalid trace event.');
      if (!groups.has(event.group)) groups.set(event.group, []);
      groups.get(event.group).push(event);
    }
    for (const [name, events] of groups) {
      group = name;
      cumulative = new Map();
      completed = new Map();
      latest = new Map();
      counts = new Map();
      pending = new Map();
      events.sort((a, b) => a.seq - b.seq);
      const ended = [...events].sort((a, b) => a.end - b.end);
      let endIndex = 0,
        lastSeq = -1;
      for (const event of events) {
        if (event.seq === lastSeq) throw new Error('Duplicate begin sequence in trace.');
        lastSeq = event.seq;
        while (endIndex < ended.length && ended[endIndex].end < event.seq) {
          const endedEvent = ended[endIndex++],
            values = pending.get(endedEvent.seq);
          if (!values) throw new Error('End event without a preceding begin event.');
          for (const [state, value] of Object.entries(values)) {
            const key = stateId(endedEvent.region, state);
            const next = (completed.get(key) || 0) + value;
            if (!safe(next))
              throw new Error('Scenario cumulative state exceeds exact integer range.');
            completed.set(key, next);
          }
        }
        const values = Object.assign(Object.create(null), event.values);
        for (const value of Object.values(values))
          if (!safe(value))
            throw new Error(
              'Trace contains state values outside JavaScript exact integer range. Values remain viewable, but replay is disabled.'
            );
        const derivations = [];
        const keepDerivations = summaries.get(event.region).changes.length < 8;
        for (const [target, relation] of relationsByRegion.get(event.region) || []) {
          const inputs = [];
          const feature = (term) =>
            term.kind === 'count'
              ? counts.get(term.region) || 0
              : (term.kind === 'last' ? latest : term.kind === 'cum' ? cumulative : completed).get(
                  stateId(term.region, term.state)
                ) || 0;
          const value = exactRelationshipValue(relation, (term) => {
            const history = feature(term);
            if (keepDerivations)
              inputs.push({
                kind: term.kind,
                region: term.region,
                state: term.state,
                value: history
              });
            return history;
          });
          for (const alternative of alternatives.get(target) || []) {
            const row = ambiguities.get(alternative.id);
            ++row.calls;
            let prediction;
            try {
              prediction = exactRelationshipValue(alternative, feature);
            } catch (error) {
              prediction = error.message;
            }
            if (prediction !== value) {
              ++row.disagreements;
              if (row.examples.length < 3)
                row.examples.push({
                  group,
                  seq: event.seq,
                  selectedValue: value,
                  alternativeValue: prediction
                });
            }
          }
          if (keepDerivations)
            derivations.push({
              state: relation.target.state,
              relationship: relation.id,
              equation: relationship(relation),
              inputs,
              result: value
            });
          values[relation.target.state] = value;
          if (edits.has(target))
            warnings.add(
              `Direct intervention overrides the relationship for ${relation.target.region}.${relation.target.state}.`
            );
        }
        for (const edit of editsByRegion.get(event.region) || []) {
          const before = values[edit.state];
          values[edit.state] =
            edit.op === 'scale'
              ? before * edit.value
              : edit.op === 'add'
                ? before + edit.value
                : edit.value;
          if (!safe(values[edit.state]))
            throw new Error('Intervention must produce exact integer PCV values.');
          if (keepDerivations)
            derivations.push({
              state: edit.state,
              intervention: edit.op,
              value: edit.value,
              before,
              result: values[edit.state]
            });
        }
        const region = regionMap.get(event.region),
          summary = summaries.get(event.region);
        ++summary.calls;
        const changed = region.states.filter((state) => values[state] !== event.values[state]);
        if (changed.length) {
          ++summary.changedCalls;
          changed.forEach((state) => summary.changedStates.add(state));
          if (summary.changes.length < 8)
            summary.changes.push({
              group,
              seq: event.seq,
              before: event.values,
              after: values,
              derivations: derivations.filter((d) => changed.includes(d.state))
            });
        }
        const before = cachedEvaluation(region, event.values),
          after = cachedEvaluation(region, values);
        if (before.ok && after.ok) {
          ++summary.modelledCalls;
          summary.beforeSum += before.explained;
          summary.afterSum += after.explained;
          summary.support.add(after.support);
          if (before.support !== 'observed') ++summary.unobservedBaselineCalls;
          if (before.hasUnexplained || after.hasUnexplained) ++summary.unexplainedCalls;
        } else {
          ++summary.unknownCalls;
          summary.reasons.add(!before.ok ? before.reason : after.reason);
        }
        pending.set(event.seq, values);
        if (options.includeTrace) predictedEvents.push({ ...event, values });
        counts.set(event.region, (counts.get(event.region) || 0) + 1);
        for (const [state, value] of Object.entries(values)) {
          const key = stateId(event.region, state),
            next = (cumulative.get(key) || 0) + value;
          if (!safe(next))
            throw new Error('Scenario cumulative state exceeds exact integer range.');
          cumulative.set(key, next);
          latest.set(key, value);
        }
      }
    }
    return {
      modelId: model.id,
      scenario,
      warnings: [...warnings],
      ...(scenario.auditAlternatives
        ? {
            alternativeChecks: [...ambiguities.values()],
            alternativeContract:
              'One-step comparisons in the selected scenario history, before direct target overrides; not measurements or full replays under each alternative.'
          }
        : {}),
      ...(options.includeTrace ? { predictedEvents } : {}),
      regions: [...summaries.values()].map((s) => ({
        region: s.region,
        calls: s.calls,
        changedCalls: s.changedCalls,
        modelledCalls: s.modelledCalls,
        unknownCalls: s.unknownCalls,
        changedStates: [...s.changedStates],
        support: [...s.support],
        reasons: [...s.reasons],
        unexplainedCalls: s.unexplainedCalls,
        unobservedBaselineCalls: s.unobservedBaselineCalls,
        changes: s.changes,
        before: s.modelledCalls ? s.beforeSum / s.modelledCalls : null,
        after: s.modelledCalls ? s.afterSum / s.modelledCalls : null,
        delta: s.modelledCalls ? (s.afterSum - s.beforeSum) / s.modelledCalls : null
      }))
    };
  }

  function groupedEvents(events) {
    const groups = new Map();
    for (const event of events) {
      if (!groups.has(event.group)) groups.set(event.group, []);
      groups.get(event.group).push(event);
    }
    return [...groups.values()].map((group) => [...group].sort((a, b) => a.seq - b.seq));
  }

  function traceStructure(events) {
    return groupedEvents(events).map((group) => {
      const threads = new Map();
      const ordered = [...group].sort((a, b) => a.seq - b.seq);
      const markers = [];
      ordered.forEach((event, i) => {
        if (!threads.has(event.thread)) threads.set(event.thread, threads.size);
        markers.push([event.seq, 'begin', event.region, i, threads.get(event.thread)]);
        markers.push([event.end, 'end', event.region, i, threads.get(event.thread)]);
      });
      return markers.sort((a, b) => a[0] - b[0]).map((m) => m.slice(1));
    });
  }

  function validateScenario(model, scenario, measured) {
    if (scenario.costBasis === 'recorded') {
      model = recordedCosts(model);
      measured = recordedCosts(measured);
    }
    validate(measured);
    if (
      measured.validity?.errors?.length ||
      measured.validity?.traceErrors?.length ||
      !measured.trace.complete
    )
      throw new Error('Validation report must have a complete, valid trace.');
    const signature = (report) => {
      const fields = [
        'version',
        'follow_unmarked_threads',
        'native_gx',
        'excluded_cuda_module',
        'rep_expand'
      ];
      const settings = (report.provenance?.runs || []).map((run) =>
        fields.map((key) => run.measurement?.[key] ?? null)
      );
      return JSON.stringify({
        unit: report.measurement?.unit,
        scope: report.measurement?.scope,
        markerAdjustment: report.measurement?.markerAdjustment,
        settings: [...new Set(settings.map((s) => JSON.stringify(s)))].sort()
      });
    };
    if (signature(model) !== signature(measured))
      throw new Error('Measurement scope or instrumentation settings differ between reports.');
    const prediction = replay(model, scenario, { includeTrace: true });
    if (
      JSON.stringify(traceStructure(model.trace.events)) !==
      JSON.stringify(traceStructure(measured.trace.events))
    ) {
      return {
        modelId: model.id,
        measuredModelId: measured.id,
        structureMatches: false,
        regions: [],
        reason:
          'Recorded call count, marker order, thread roles, or nesting changed. Fixed-trace scenario validation is not applicable.'
      };
    }
    const actualRegions = new Map(measured.regions.map((r) => [r.id, r]));
    const originalRegions = new Map(model.regions.map((r) => [r.id, r]));
    const summaries = new Map();
    const events = groupedEvents(measured.trace.events).flat();
    const predicted = groupedEvents(prediction.predictedEvents).flat();
    for (let i = 0; i < predicted.length; ++i) {
      const event = predicted[i],
        actual = events[i],
        original = originalRegions.get(event.region),
        measuredRegion = actualRegions.get(event.region);
      if (!summaries.has(event.region))
        summaries.set(event.region, {
          region: event.region,
          calls: 0,
          stateMatches: 0,
          costChecks: 0,
          unknownCosts: 0,
          absoluteError: 0,
          observedSum: 0,
          predictedSum: 0,
          examples: []
        });
      const summary = summaries.get(event.region);
      ++summary.calls;
      const matches =
        original.states.length === measuredRegion.states.length &&
        original.states.every(
          (state) =>
            measuredRegion.states.includes(state) &&
            String(event.values[state]) === String(actual.values[state])
        );
      if (matches) ++summary.stateMatches;
      else if (summary.examples.length < 3)
        summary.examples.push({ kind: 'state', predicted: event.values, measured: actual.values });
      const fitted = evaluate(original, event.values);
      const point = measuredRegion.points.find((point) =>
        point.state.every((v, j) => String(v) === String(actual.values[measuredRegion.states[j]]))
      );
      // A partial explained formula must not be compared to full observed cost.
      if (!matches || !fitted.ok || fitted.hasUnexplained || !point || !finite(point.observed)) {
        ++summary.unknownCosts;
        continue;
      }
      ++summary.costChecks;
      summary.absoluteError += Math.abs(fitted.explained - point.observed);
      summary.observedSum += point.observed;
      summary.predictedSum += fitted.explained;
      if (
        Math.abs(fitted.explained - point.observed) >
          Math.max(64, Math.abs(point.observed) * 0.05) &&
        summary.examples.length < 3
      )
        summary.examples.push({
          kind: 'cost',
          states: actual.values,
          predicted: fitted.explained,
          measured: point.observed,
          support: fitted.support
        });
    }
    return {
      modelId: model.id,
      measuredModelId: measured.id,
      structureMatches: true,
      comparison:
        'Changed-run observed per-state means versus baseline-formula predictions; excludes predictions with unexplained blocks.',
      regions: [...summaries.values()].map((row) => ({
        region: row.region,
        calls: row.calls,
        stateMatches: row.stateMatches,
        costChecks: row.costChecks,
        unknownCosts: row.unknownCosts,
        meanPredicted: row.costChecks ? row.predictedSum / row.costChecks : null,
        meanObserved: row.costChecks ? row.observedSum / row.costChecks : null,
        meanAbsoluteError: row.costChecks ? row.absoluteError / row.costChecks : null,
        relativeAbsoluteError:
          row.costChecks && row.observedSum > 0 ? row.absoluteError / row.observedSum : null,
        examples: row.examples
      }))
    };
  }

  function suggestedExperiments(model, region) {
    const suggestions = [];
    if (!region.regimes.length)
      suggestions.push('Exercise more distinct PCV states before fitting a formula.');
    for (const fit of region.regimes) {
      if (fit.dependent.length)
        suggestions.push(
          `Vary ${fit.dependent.join(', ')} independently to separate their coefficients.`
        );
      if (fit.unexplainedShare > 0.05)
        suggestions.push(
          'Inspect unexplained-function attribution; add a missing semantic PCV or split the region, then rerun small cases.'
        );
      for (let i = 0; i < fit.range.length; ++i)
        if (fit.range[i][0] === fit.range[i][1])
          suggestions.push(`Vary ${region.states[i]}: it was constant in this regime.`);
    }
    if (region.regimes.length > 1)
      suggestions.push('Probe the gap between fitted regimes to locate the branch transition.');
    const related = model.relations.filter((r) => r.target.region === region.id);
    if (related.length)
      suggestions.push(
        'Perturb a source PCV in a small execution and check whether the observed relationship survives.'
      );
    if (!suggestions.length)
      suggestions.push(
        'Hold other PCVs fixed and test a new small state to challenge the formula outside its observed points.'
      );
    return [...new Set(suggestions)];
  }

  return {
    schema,
    id,
    validate,
    format,
    formula,
    relationship,
    evaluate,
    graph,
    reachable,
    replay,
    suggestedExperiments,
    exactRelationshipValue,
    verifyRelationships,
    validateScenario,
    proposeRelationship,
    withProposals,
    recordedCosts
  };
});
