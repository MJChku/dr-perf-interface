'use strict';
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const base = path.resolve(__dirname, '..');
const output = path.resolve(__dirname, '../../../out/explorer-ui');
const model = JSON.parse(
  require('node:fs').readFileSync(path.join(base, 'demo/pipeline.drperf.json'), 'utf8')
);
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-test'; worker-src blob:; connect-src 'self'; style-src 'self'; img-src 'self' data:"><link rel="stylesheet" href="/media/explorer.css"></head><body><div id="app"></div><script nonce="test" src="/media/expressions.js"></script><script nonce="test" src="/media/model.js"></script><script nonce="test" src="/media/analysis-client.js" data-expressions="/media/expressions.js" data-model="/media/model.js" data-worker="/media/analysis-worker.js"></script><script nonce="test" src="/media/explorer.js"></script></body></html>`;
(async () => {
  await fs.mkdir(output, { recursive: true });
  const server = http.createServer(async (req, res) => {
    if (req.url === '/') {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
      return;
    }
    const target = path.resolve(base, '.' + req.url);
    if (!target.startsWith(base + path.sep)) {
      res.writeHead(403);
      res.end();
      return;
    }
    try {
      const data = await fs.readFile(target);
      res.writeHead(200, {
        'Content-Type': target.endsWith('.js') ? 'text/javascript' : 'text/css'
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    const page = await browser.newPage({
      viewport: { width: 1280, height: 900 },
      deviceScaleFactor: 1
    });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.evaluate((m) => {
      window.hostMessages = [];
      window.addEventListener('drperfHostMessage', (e) => window.hostMessages.push(e.detail));
      window.postMessage({ type: 'model', model: m, selected: 'enqueue' }, '*');
    }, model);
    await page.getByRole('heading', { name: 'enqueue', exact: true }).waitFor();
    assert.match(await page.locator('.formula').first().textContent(), /64\*items/);
    await page.evaluate(
      (id) =>
        window.postMessage(
          {
            type: 'sourceStatus',
            modelId: id,
            region: 'enqueue',
            sources: [{ path: 'pipeline.c', state: 'changed' }]
          },
          '*'
        ),
      model.id
    );
    await page.getByText(/Source differs from the exported snapshot/).waitFor();
    await page.evaluate(
      (id) =>
        window.postMessage(
          {
            type: 'sourceStatus',
            modelId: id,
            region: 'enqueue',
            sources: [{ path: 'pipeline.c', state: 'matches' }]
          },
          '*'
        ),
      model.id
    );
    await page.getByText(/Source differs from the exported snapshot/).waitFor({ state: 'hidden' });
    await page.screenshot({ path: path.join(output, 'interface.png'), fullPage: true });
    await page.getByRole('button', { name: 'Relationships', exact: true }).click();
    assert.equal(await page.locator('.relationship').count(), model.relations.length);
    await page.getByLabel('Filter relationships').fill('enqueue');
    assert.ok((await page.locator('.relationship').count()) > 0);
    await page.getByLabel('Filter relationships').fill('');
    await page.screenshot({ path: path.join(output, 'relationships.png'), fullPage: true });
    await page.getByRole('button', { name: 'What-if', exact: true }).click();
    await page.locator('.scenario-table tbody tr').first().waitFor();
    assert.equal(await page.locator('.scenario-table tbody tr').count(), 1);
    assert.equal(await page.evaluate(() => window.DrperfAnalysis.backend), 'worker');
    await page.getByRole('button', { name: 'Use first observed equation for each target' }).click();
    await page.locator('.scenario-table tr[data-region="copy"]').waitFor();
    assert.ok((await page.locator('.scenario-table tbody tr').count()) >= 5);
    await page
      .getByRole('button', { name: 'Find distinguishing experiments', exact: true })
      .click();
    await page
      .locator(
        '.experiment-suggestion[data-experiment-region="decode"][data-experiment-op="scale"]'
      )
      .waitFor();
    assert.match(
      await page
        .locator(
          '.experiment-suggestion[data-experiment-region="decode"][data-experiment-op="scale"]'
        )
        .textContent(),
      /copy.bytes/
    );
    const copyEquation = await page.getByLabel('Relationship for copy.bytes').inputValue();
    await page.getByLabel('Relationship for copy.bytes').selectOption('');
    assert.equal(
      await page.locator('.experiment-suggestion').count(),
      0,
      'changed assumptions invalidate displayed experiment suggestions'
    );
    await page.getByLabel('Relationship for copy.bytes').selectOption(copyEquation);
    await page.locator('.scenario-table tr[data-region="copy"]').waitFor();
    const copy = page.locator('.scenario-table tr[data-region="copy"]');
    assert.ok((await copy.textContent()).includes('extrapolated'));
    await page.screenshot({ path: path.join(output, 'scenario.png'), fullPage: true });
    const changed = JSON.parse(
      await fs.readFile(path.join(base, 'demo/changed.drperf.json'), 'utf8')
    );
    await page.evaluate(
      (m) =>
        window.postMessage(
          { type: 'validationModel', model: m, name: 'Changed measured program' },
          '*'
        ),
      changed
    );
    await page.locator('tr[data-validation-region="lookup"]').waitFor();
    assert.match(
      await page.locator('tr[data-validation-region="lookup"]').textContent(),
      /22\.26%/
    );
    await page.screenshot({ path: path.join(output, 'validation.png'), fullPage: true });
    await page.getByRole('button', { name: 'Compare runs', exact: true }).click();
    await page.locator('.comparison-table').waitFor();
    assert.match(
      await page.locator('tr[data-comparison-region="enqueue"]').textContent(),
      /paired/
    );
    assert.match(
      await page.locator('tr[data-comparison-region="copy"]').textContent(),
      /No paired states/
    );
    await page.screenshot({ path: path.join(output, 'comparison.png'), fullPage: true });
    await page.getByRole('button', { name: 'Relationships', exact: true }).click();
    await page.locator('.relationship-check-summary').waitFor();
    assert.equal(
      await page.locator('.relationship-check[data-relationship-status="fails"]').count(),
      0
    );
    await page.getByRole('button', { name: 'What-if', exact: true }).click();
    await page.getByRole('button', { name: 'Save scenario and per-region results' }).click();
    assert.ok(
      await page.evaluate(() =>
        window.hostMessages.some(
          (m) => m.type === 'exportScenario' && m.scenario.edits[0].value === 2
        )
      )
    );
    await page.getByRole('button', { name: '+ Add another PCV' }).click();
    await page.getByLabel('Region for intervention 2', { exact: true }).selectOption('decode');
    await page.getByLabel('Intervention value 2', { exact: true }).fill('1.5');
    assert.equal(await page.locator('.notice.error').count(), 0);
    await page.getByLabel('Compare alternative relationships', { exact: true }).check();
    await page.getByText('copy.bytes: disagreement at 24/24 calls').waitFor();
    await page.screenshot({ path: path.join(output, 'alternatives.png'), fullPage: true });
    await page.setViewportSize({ width: 520, height: 850 });
    await page.screenshot({ path: path.join(output, 'narrow.png'), fullPage: true });
    assert.ok(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)
    );
    await page.setViewportSize({ width: 1280, height: 900 });
    // Proposals are independently checked, retain counterexamples, and require opt-in.
    await page.getByLabel('Proposed target region').selectOption('decode');
    await page.getByLabel('Proposed state expression').fill('3 * last("dequeue", "items")');
    await page.getByRole('button', { name: 'Check proposed relationship', exact: true }).click();
    await page.getByText('Failed at 24 of 24 recorded target calls.').waitFor();
    await page.getByLabel('Proposed state expression').fill('2 * last("dequeue", "items")');
    await page.getByRole('button', { name: 'Check proposed relationship', exact: true }).click();
    await page
      .getByRole('button', { name: 'Use checked proposal as an assumption', exact: true })
      .click();
    assert.match(
      await page.getByLabel('Relationship for decode.tokens').inputValue(),
      /^proposed:/
    );
    await page.getByRole('button', { name: 'Save scenario and per-region results' }).click();
    assert.ok(
      await page.evaluate(() =>
        window.hostMessages.some(
          (m) => m.type === 'exportScenario' && m.scenario.proposals?.length === 1
        )
      )
    );
    // The latest edit wins even when worker work is already queued.
    await page.evaluate(() => {
      const input = document.querySelector('[aria-label="Intervention value 1"]');
      for (const value of ['3', '4', '2']) {
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    await page.locator('#scenario-result .analysis-pending').waitFor({ state: 'hidden' });
    await page.getByRole('button', { name: 'Save scenario and per-region results' }).click();
    const savedScenario = await page.evaluate(
      () => window.hostMessages.filter((m) => m.type === 'exportScenario').at(-1).scenario
    );
    assert.equal(savedScenario.edits[0].value, 2);
    // A saved scenario is checked again, and importing it cannot overwrite a
    // subsequent edit made while that check is running.
    await page.evaluate(
      ({ id, scenario }) => {
        window.dispatchEvent(
          new MessageEvent('message', { data: { type: 'scenario', modelId: id, scenario } })
        );
        const input = document.querySelector('[aria-label="Intervention value 1"]');
        input.value = '3';
        input.dispatchEvent(new Event('input', { bubbles: true }));
      },
      { id: model.id, scenario: savedScenario }
    );
    await page.locator('#scenario-result .analysis-pending').waitFor({ state: 'hidden' });
    assert.equal(await page.getByLabel('Intervention value 1', { exact: true }).inputValue(), '3');
    await page.evaluate(
      ({ id, scenario }) => window.postMessage({ type: 'scenario', modelId: id, scenario }, '*'),
      { id: model.id, scenario: savedScenario }
    );
    await page.waitForFunction(
      () =>
        document.querySelector('[aria-label="Intervention value 1"]')?.value === '2' &&
        !document.querySelector('#scenario-result .analysis-pending')
    );
    assert.equal(
      await page.getByLabel('Intervention value 2', { exact: true }).inputValue(),
      '1.5'
    );
    assert.match(
      await page.getByLabel('Relationship for decode.tokens').inputValue(),
      /^proposed:/
    );
    const bad = structuredClone(model);
    bad.validity.errors = ['slot overflow'];
    await page.evaluate((m) => window.postMessage({ type: 'model', model: m }, '*'), bad);
    await page
      .getByText('Measurement errors: slot overflow. Scenario predictions are disabled.')
      .waitFor();
    assert.equal(await page.locator('.scenario-table').count(), 0);
    const hostile = structuredClone(model);
    hostile.regions[0].name = '<img src=x onerror="window.exploited=1">';
    await page.evaluate(
      (m) => window.postMessage({ type: 'model', model: m, selected: m.regions[0].id }, '*'),
      hostile
    );
    await page.getByRole('button', { name: 'Interface', exact: true }).click();
    assert.equal(await page.locator('img').count(), 0);
    assert.equal(await page.evaluate(() => window.exploited), undefined);
    const refined = JSON.parse(
      await fs.readFile(path.join(base, 'demo/refined.drperf.json'), 'utf8')
    );
    const joint = JSON.parse(
      await fs.readFile(path.join(base, 'demo/joint-changed.drperf.json'), 'utf8')
    );
    const jointScenario = JSON.parse(
      await fs.readFile(path.join(base, 'demo/joint.scenario.json'), 'utf8')
    );
    await page.evaluate(
      ({ model, measured, saved }) => {
        window.postMessage({ type: 'model', model }, '*');
        window.postMessage(
          {
            type: 'scenario',
            modelId: model.id,
            scenario: saved.scenario,
            measured,
            measuredName: 'Measured joint producer/expansion change'
          },
          '*'
        );
      },
      { model: refined, measured: joint, saved: jointScenario }
    );
    await page.locator('tr[data-validation-region="lookup"]').waitFor();
    assert.match(await page.locator('tr[data-validation-region="lookup"]').textContent(), /24\/24/);
    assert.match(
      await page.locator('tr[data-validation-region="dispatch"]').textContent(),
      /17\/24/
    );
    assert.equal(
      await page.getByLabel('Intervention value 2', { exact: true }).inputValue(),
      '1.5'
    );
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, 'joint.png'), fullPage: false });
    assert.deepEqual(errors, []);
    console.log(
      'UI PASS: formulas, all relations, opt-in propagation, multiple PCVs, measured-run validation, checked proposals, narrow layout, invalid-data guard, safe rendering.'
    );
  } finally {
    await browser?.close();
    server.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
