'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../media/model');
const measured = require('../demo/refined.drperf.json');
const changed = require('../demo/expansion-changed.drperf.json');
function small(
  states = ['n', 'm'],
  points = [
    [1, 10],
    [2, 20],
    [3, 30]
  ]
) {
  return {
    schema: M.schema,
    id: 'test',
    measurement: { unit: 'CPU instructions per call', scope: 'exclusive region work' },
    validity: { errors: [], traceErrors: [] },
    trace: { complete: false, events: [] },
    relations: [],
    regions: [
      {
        id: 'r',
        name: 'r',
        originalName: 'r',
        states,
        calls: 3,
        sources: [],
        regimes: [],
        points: points.map((state, i) => ({ state, calls: 1, observed: 10 * (i + 1) }))
      }
    ]
  };
}
test('interface comparison pairs PCVs by name and uses equal state weights', () => {
  const before = small(),
    after = small(
      ['m', 'n'],
      [
        [10, 1],
        [20, 2],
        [30, 3]
      ]
    );
  after.regions[0].points[2].calls = 1000;
  const comparison = M.compareInterfaces(before, after);
  assert.equal(comparison.regions[0].pairedStates, 3);
  assert.equal(comparison.regions[0].beforeMean, 20);
  assert.equal(comparison.regions[0].afterMean, 20);
  assert.equal(comparison.regions[0].deltaMean, 0);
});
test('schema changes and disjoint coverage never get invented paired costs', () => {
  const before = small(),
    after = small(['n', 'other']);
  const unmatched = M.compareInterfaces(before, after).regions;
  assert.deepEqual(
    unmatched.map((r) => r.status),
    ['only-before', 'only-after']
  );
  const shifted = small(
    ['n', 'm'],
    [
      [11, 10],
      [12, 20],
      [13, 30]
    ]
  );
  const row = M.compareInterfaces(before, shifted).regions[0];
  assert.equal(row.status, 'no-common-states');
  assert.equal(row.deltaMean, null);
});
test('interface identity survives a later report introducing schema variants', () => {
  const before = small(),
    after = small();
  after.regions[0].id = 'r@pcv-variant';
  const row = M.compareInterfaces(before, after).regions[0];
  assert.equal(row.pairedStates, 3);
  assert.equal(row.afterRegion, 'r@pcv-variant');
});
test('a measured workload change leaves shared-state region costs unchanged', () => {
  const comparison = M.compareInterfaces(measured, changed);
  for (const name of ['dequeue', 'lookup', 'dispatch']) {
    const row = comparison.regions.find((r) => r.region === name);
    assert.ok(row.pairedStates > 0);
    assert.equal(row.deltaMean, 0);
  }
  const decode = comparison.regions.find((r) => r.region === 'decode');
  assert.equal(decode.pairedStates, 0);
  assert.ok(decode.coefficientChanges.every((c) => c.delta === 0));
});
test('comparison flags changed source expressions and rejects invalid counters', () => {
  const before = small(),
    after = small();
  const source = {
    path: 'a.py',
    line: 1,
    endLine: 1,
    sha256: 'a'.repeat(64),
    expressions: { n: 'size' }
  };
  before.regions[0].sources = [source];
  after.regions[0].sources = [{ ...source, expressions: { n: 'size * size' } }];
  assert.deepEqual(M.compareInterfaces(before, after).regions[0].changedExpressions, ['n']);
  after.validity.errors = ['counter overflow'];
  assert.throws(() => M.compareInterfaces(before, after), /Invalid instruction/);
});

test('PCV names that coincide with object properties remain ordinary data', () => {
  const before = small(['constructor', '__proto__']),
    after = small(['constructor', '__proto__']);
  const expressions = JSON.parse('{"constructor":"size", "__proto__":"bytes"}');
  before.regions[0].sources = [
    { path: 'a.py', line: 1, endLine: 1, sha256: 'a'.repeat(64), expressions }
  ];
  after.regions[0].sources = [
    { ...before.regions[0].sources[0], expressions: { ...expressions, constructor: 'size * size' } }
  ];
  assert.deepEqual(M.compareInterfaces(before, after).regions[0].changedExpressions, [
    'constructor'
  ]);
});
