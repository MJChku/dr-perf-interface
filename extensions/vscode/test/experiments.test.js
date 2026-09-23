'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../media/model');
const model = require('../demo/refined.drperf.json');
const measured = require('../demo/expansion-changed.drperf.json');
function assumptions() {
  const targets = new Map();
  for (const r of model.relations) {
    const key = M.id(r.target.region, r.target.state);
    if (!targets.has(key)) targets.set(key, r.id);
  }
  return { relations: [...targets.values()] };
}
test('bounded search finds the independently measured expansion experiment', () => {
  const result = M.findDistinguishingExperiments(model, assumptions());
  const probe = result.suggestions.find((s) => s.edit.region === 'decode' && s.edit.op === 'scale');
  assert.ok(probe);
  assert.deepEqual(probe.edit, { region: 'decode', state: 'tokens', op: 'scale', value: 2 });
  assert.ok(probe.targets.some((t) => t.region === 'copy' && t.state === 'bytes'));
  const validation = M.validateScenario(model, probe.scenario, measured);
  const copy = validation.regions.find((r) => r.region === 'copy');
  assert.equal(copy.stateMatches, 24);
  assert.equal(copy.costChecks, 24);
  assert.ok(copy.relativeAbsoluteError < 1e-12);
  // This is a useful experiment precisely because a baseline-valid dispatch
  // explanation fails in the independent execution.
  assert.equal(validation.regions.find((r) => r.region === 'dispatch').stateMatches, 0);
  assert.equal(Object.hasOwn(result, 'totalCost'), false);
});
test('search reports its bound and non-replayable probes instead of silently rounding', () => {
  const limited = M.findDistinguishingExperiments(model, assumptions(), { maxProbes: 1 });
  assert.equal(limited.search.tried, 1);
  assert.equal(limited.search.truncated, true);
  const full = M.findDistinguishingExperiments(model, assumptions());
  assert.ok(full.rejected.some((r) => /non-integer/.test(r.reason)));
  assert.throws(
    () => M.findDistinguishingExperiments(model, assumptions(), { maxProbes: 1000 }),
    /64 probes/
  );
});
test('search needs explicit, valid assumptions and starts from baseline', () => {
  assert.throws(
    () => M.findDistinguishingExperiments(model, { relations: [] }),
    /Choose relationship/
  );
  assert.throws(
    () => M.findDistinguishingExperiments(model, { ...assumptions(), edits: [{}] }),
    /baseline/
  );
  assert.throws(
    () => M.findDistinguishingExperiments(model, { ...assumptions(), costBasis: 'seconds' }),
    /cost basis/
  );
  const invalid = structuredClone(model);
  invalid.trace.complete = false;
  assert.throws(
    () => M.findDistinguishingExperiments(invalid, assumptions()),
    /valid recorded trace/
  );
  const wrong = structuredClone(model);
  const selected = wrong.relations.find((r) => r.id === assumptions().relations[0]);
  selected.constant += 1;
  selected.constantExact = String(selected.constant);
  assert.throws(() => M.findDistinguishingExperiments(wrong, assumptions()), /does not hold/);
});
test('proposals survive in suggested scenarios and each suggestion can be replayed independently', () => {
  const input = assumptions();
  const proposal = {
    target: { region: 'lookup', state: 'pairs' },
    expression: 'last("dequeue", "items") ** 2'
  };
  input.proposals = [proposal];
  input.relations.push(
    M.proposeRelationship(model, proposal.target, proposal.expression).relation.id
  );
  input.costBasis = 'recorded';
  const result = M.findDistinguishingExperiments(model, input);
  assert.ok(result.suggestions.length);
  for (const suggestion of result.suggestions) {
    assert.deepEqual(suggestion.scenario.proposals, [proposal]);
    const replay = M.replay(model, suggestion.scenario);
    assert.match(replay.modelId, /:recorded$/);
    assert.deepEqual(
      replay.regions.filter((r) => r.changedCalls).map((r) => r.region),
      suggestion.changedRegions.map((r) => r.region)
    );
  }
});
