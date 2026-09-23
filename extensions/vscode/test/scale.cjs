/* Optional browser smoke test against a user's exported large report.
 * node test/scale.cjs /path/report.drperf.json [region]
 */
'use strict';
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const M = require('../media/model');
(async () => {
  if (!process.argv[2]) throw new Error('Supply a portable drperf report path.');
  const model = M.validate(JSON.parse(await fs.readFile(process.argv[2], 'utf8')));
  const selected =
    process.argv[3] ||
    model.regions.find((r) => r.regimes.length && r.states.length)?.id ||
    model.regions[0].id;
  const root = path.resolve(__dirname, '..');
  const output = path.resolve(__dirname, '../../../out/explorer-ui');
  await fs.mkdir(output, { recursive: true });
  const html =
    '<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="/media/explorer.css"></head><body><div id="app"></div><script src="/media/expressions.js"></script><script src="/media/model.js"></script><script src="/media/explorer.js"></script></body></html>';
  const server = http.createServer(async (req, res) => {
    if (req.url === '/') {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
      return;
    }
    const filename = path.resolve(root, '.' + req.url);
    if (!filename.startsWith(root + path.sep)) {
      res.writeHead(403);
      res.end();
      return;
    }
    try {
      res.writeHead(200, {
        'Content-Type': filename.endsWith('.js') ? 'text/javascript' : 'text/css'
      });
      res.end(await fs.readFile(filename));
    } catch {
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }),
      errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    let start = performance.now();
    await page.evaluate(
      ({ model, selected }) => window.postMessage({ type: 'model', model, selected }, '*'),
      { model, selected }
    );
    await page
      .getByRole('heading', {
        name: model.regions.find((r) => r.id === selected).name,
        exact: true
      })
      .waitFor();
    const interfaceMs = performance.now() - start;
    await page.screenshot({ path: path.join(output, 'large-interface.png'), fullPage: true });
    await page.getByRole('button', { name: 'Relationships', exact: true }).click();
    assert.equal(await page.locator('.relationship').count(), model.relations.length);
    await page.getByRole('button', { name: 'View recorded counts', exact: true }).click();
    await page.getByRole('button', { name: 'What-if', exact: true }).click();
    await page
      .getByRole('button', { name: 'Use first observed equation for each target', exact: true })
      .click();
    start = performance.now();
    await page.getByLabel('Intervention value 1', { exact: true }).fill('1');
    await page.waitForFunction(() => !document.querySelector('.scenario-table tbody tr'));
    const identityMs = performance.now() - start;
    await page.getByLabel('Intervention value 1', { exact: true }).fill('2');
    await page.locator('.scenario-table tbody tr').first().waitFor();
    await page.screenshot({ path: path.join(output, 'large-scenario.png'), fullPage: true });
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify(
        {
          regions: model.regions.length,
          relations: model.relations.length,
          calls: model.trace.events.length,
          interfaceMs,
          identityMs,
          renderedScenarioRows: await page.locator('.scenario-table tbody tr').count(),
          errors
        },
        null,
        2
      )
    );
  } finally {
    await browser?.close();
    server.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
