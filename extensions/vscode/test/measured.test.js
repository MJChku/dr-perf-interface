'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../media/model');
const baseline = require('../demo/pipeline.drperf.json');
const changed = require('../demo/changed.drperf.json');
const refined = require('../demo/refined.drperf.json');
const refinedChanged = require('../demo/refined-changed.drperf.json');
function scenario(model) {
  const targets = new Map();
  for (const r of model.relations) {
    const k = M.id(r.target.region, r.target.state);
    if (!targets.has(k)) targets.set(k, r.id);
  }
  return {
    edits: [{ region: 'enqueue', state: 'items', op: 'scale', value: 2 }],
    relations: [...targets.values()]
  };
}
test('measured changed program confirms propagation and exposes narrow quadratic-fit extrapolation', () => {
  const comparison = M.validateScenario(baseline, scenario(baseline), changed);
  assert.equal(comparison.structureMatches, true);
  for (const name of ['enqueue', 'dequeue', 'decode', 'copy']) {
    const row = comparison.regions.find((r) => r.region === name);
    assert.equal(row.stateMatches, row.calls);
    assert.equal(row.costChecks, row.calls);
    assert.ok(row.relativeAbsoluteError < (name === 'enqueue' ? 0.01 : 1e-12), name);
  }
  const lookup = comparison.regions.find((r) => r.region === 'lookup');
  assert.equal(lookup.stateMatches, 24);
  assert.ok(lookup.relativeAbsoluteError > 0.2 && lookup.relativeAbsoluteError < 0.25);
  assert.ok(lookup.unknownCosts > 0);
});
test('semantic squared PCV and checked nonlinear relationship predict all measured lookup costs', () => {
  const input = scenario(refined);
  const proposal = {
    target: { region: 'lookup', state: 'pairs' },
    expression: 'last("dequeue", "items") ** 2'
  };
  const { relation, check } = M.proposeRelationship(refined, proposal.target, proposal.expression);
  assert.equal(check.holds, true);
  input.proposals = [proposal];
  input.relations.push(relation.id);
  const comparison = M.validateScenario(refined, input, refinedChanged);
  assert.equal(comparison.structureMatches, true);
  const lookup = comparison.regions.find((r) => r.region === 'lookup');
  assert.equal(lookup.stateMatches, 24);
  assert.equal(lookup.costChecks, 24);
  assert.equal(lookup.unknownCosts, 0);
  assert.ok(lookup.relativeAbsoluteError < 1e-12);
  const region = refined.regions.find((r) => r.id === 'lookup');
  assert.equal(region.regimes.length, 1);
  assert.equal(region.regimes[0].blocks.unexplained, 0);
  assert.match(M.formula(region), /^128\*pairs \+ 22$/);
});

test('a realized expansion-factor intervention distinguishes competing state relationships', () => {
  const measured = require('../demo/expansion-changed.drperf.json');
  const input = scenario(refined);
  input.edits = [{ region: 'decode', state: 'tokens', op: 'scale', value: 2 }];
  input.auditAlternatives = true;
  const lookup = {
    target: { region: 'lookup', state: 'pairs' },
    expression: 'last("dequeue", "items") ** 2'
  };
  const dispatch = {
    target: { region: 'dispatch', state: 'items' },
    expression: 'last("dequeue", "items")'
  };
  input.proposals = [lookup];
  input.relations.push(
    M.proposeRelationship(refined, lookup.target, lookup.expression).relation.id
  );
  const initial = M.validateScenario(refined, input, measured);
  assert.equal(initial.regions.find((r) => r.region === 'dispatch').stateMatches, 0);
  assert.ok(
    M.replay(refined, input).alternativeChecks.some(
      (r) => r.target.region === 'dispatch' && r.disagreements > 0
    )
  );
  input.proposals.push(dispatch);
  input.relations = input.relations.filter(
    (id) => !refined.relations.some((r) => r.id === id && r.target.region === 'dispatch')
  );
  input.relations.push(
    M.proposeRelationship(refined, dispatch.target, dispatch.expression).relation.id
  );
  const revised = M.validateScenario(refined, input, measured);
  assert.ok(revised.regions.every((r) => r.stateMatches === r.calls));
  assert.ok(
    revised.regions
      .filter((r) => r.region !== 'startup')
      .every(
        (r) =>
          r.costChecks === r.calls &&
          (r.region === 'enqueue' ? r.relativeAbsoluteError < 0.01 : r.meanAbsoluteError < 1e-5)
      )
  );
});
