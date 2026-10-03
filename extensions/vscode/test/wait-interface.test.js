'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../media/model');

test('null refinement retains its indicator and reason without a producer link', () => {
  const m = {eventModel:{interfaceChecks:{claims:[{region:'B', producer:null, event:null,
    indicator:'need == 1', prefix:'I[need == 1] * ', reference:'waited(null)',
    refinement:'null', reason:'Counter bookkeeping.', reasonReview:'manual', status:'checked'}],
    unexplained:[{region:'A', producer:null, refinement:'null', kind:'undeclared-null'}]}}};
  const term = M.waitTerms(m,'B')[0];
  assert.equal(term.prefix+term.reference+term.suffix, 'I[need == 1] * waited(null)');
  assert.equal(term.producer, null);
  assert.equal(term.reasonReview, 'manual');
  assert.equal(term.reason, 'Counter bookkeeping.');
  assert.equal(M.waitTerms(m,'A')[0].reference, 'waited(null)');
});

test('explicit indicator and unresolved waits are both interface terms', () => {
  const m = {eventModel:{interfaceChecks:{claims:[{region:'A', producer:'B', indicator:'need == 1',
    prefix:'I[need == 1] * ', reference:'Wait[B]', term:'I[need == 1] * Wait[B]', status:'checked'}],
    unexplained:[{region:'A', producer:'?', count:2, term:'unexplained(Wait[?])'}]}}};
  const terms = M.waitTerms(m, 'A');
  assert.deepEqual(terms.map(t => t.prefix+t.reference+t.suffix), ['I[need == 1] * Wait[B]', 'unexplained(Wait[?])']);
  assert.equal(terms[0].producer, 'B');
  assert.deepEqual(M.waitTerms(m,'B'), []);
});

test('failed indicator stays failed instead of becoming an accepted fit', () => {
  const m = {eventModel:{interfaceChecks:{claims:[{region:'A', producer:'B', indicator:'False',
    prefix:'I[False] * ', reference:'Wait[B]', status:'invalid'}], unexplained:[]}}};
  assert.equal(M.waitTerms(m,'A')[0].status, 'invalid');
  assert.equal(M.waitTerms(m,'A')[0].prefix, 'I[False] * ');
});

test('legacy occurrence fits cannot silently supply a declared indicator', () => {
  const m = {eventModel:{interfaces:[{region:'A',countFormula:{constant:'1'}}],
    edges:[{consumer:'A',producer:'B'}],unexplained:[{region:'A'}]}};
  assert.deepEqual(M.waitTerms(m,'A').map(t=>t.term), ['unexplained(Wait[B])','unexplained(Wait[?])']);
});
