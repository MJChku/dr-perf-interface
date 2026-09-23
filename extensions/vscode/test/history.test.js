'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../media/model');

function overlapping() {
  // Producer p starts first and finishes last. Producer q finishes while p is
  // still active. Observer calls occur on a third thread.
  const template = [
    { region: 'source', thread: 'p', seq: 1, end: 10, values: { n: 3 } },
    { region: 'observer', thread: 'o', seq: 2, end: 3, values: { completed: 0 } },
    { region: 'source', thread: 'q', seq: 4, end: 5, values: { n: 5 } },
    { region: 'observer', thread: 'o', seq: 6, end: 7, values: { completed: 5 } },
    { region: 'observer', thread: 'o', seq: 11, end: 12, values: { completed: 8 } }
  ];
  const events = ['first', 'second'].flatMap((group) =>
    template.map((event) => ({ ...structuredClone(event), group }))
  );
  const regions = [
    ['source', 'n', 4],
    ['observer', 'completed', 6]
  ].map(([id, state, calls]) => ({
    id,
    name: id,
    states: [state],
    calls,
    sources: [],
    regimes: [],
    points: []
  }));
  return {
    schema: M.schema,
    id: 'overlapping-test',
    measurement: { unit: 'CPU instructions per call', scope: 'exclusive region work' },
    validity: { errors: [], traceErrors: [] },
    trace: { complete: true, recordCount: events.length, events },
    regions,
    relations: [
      {
        id: 'completed-work',
        target: { region: 'observer', state: 'completed' },
        constant: 0,
        terms: [{ kind: 'cumend', region: 'source', state: 'n', coefficient: 1 }]
      }
    ]
  };
}

test('completion histories respect cross-thread overlap and reset between runs', () => {
  const model = overlapping();
  assert.doesNotThrow(() => M.validate(model));
  assert.equal(M.verifyRelationships(model)[0].holds, true);
  const scenario = {
    edits: [{ region: 'source', state: 'n', op: 'scale', value: 2 }],
    relations: ['completed-work']
  };
  const result = M.replay(model, scenario, { includeTrace: true });
  assert.deepEqual(
    result.predictedEvents.filter((e) => e.region === 'observer').map((e) => e.values.completed),
    [0, 10, 16, 0, 10, 16]
  );
  assert.equal(result.regions.find((r) => r.region === 'observer').changedCalls, 4);
  // A state prediction is still useful when no cost interface is available.
  assert.equal(result.regions.find((r) => r.region === 'observer').modelledCalls, 0);
});

test('identity replay preserves states, while substituting begin history fails', () => {
  const model = overlapping();
  const identity = M.replay(
    model,
    {
      edits: [{ region: 'source', state: 'n', op: 'scale', value: 1 }],
      relations: ['completed-work']
    },
    { includeTrace: true }
  );
  assert.deepEqual(
    identity.predictedEvents.map((e) => ({ ...e.values })),
    model.trace.events.map((e) => e.values)
  );
  for (const kind of ['last', 'cum']) {
    const other = structuredClone(model);
    other.relations[0].terms[0].kind = kind;
    assert.equal(M.verifyRelationships(other)[0].holds, false);
    assert.throws(
      () =>
        M.replay(other, {
          edits: [{ region: 'source', state: 'n', op: 'scale', value: 2 }],
          relations: ['completed-work']
        }),
      /does not hold/
    );
  }
});
