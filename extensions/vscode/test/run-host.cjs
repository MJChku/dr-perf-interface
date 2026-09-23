'use strict';
const path = require('node:path');
const os = require('node:os');
const fs = require('node:fs');
const { runTests } = require('@vscode/test-electron');
(async () => {
  // A fresh editor profile and copied source avoid changing the developer's
  // files or inheriting dirty editor buffers from an earlier failed test.
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'drperf-vscode-test-'));
  const workspace = path.join(temporary, 'workspace');
  fs.mkdirSync(path.join(workspace, 'examples/explorer'), { recursive: true });
  fs.copyFileSync(
    path.resolve(__dirname, '../../../examples/explorer/pipeline.c'),
    path.join(workspace, 'examples/explorer/pipeline.c')
  );
  try {
    await runTests({
      extensionDevelopmentPath: path.resolve(__dirname, '..'),
      extensionTestsPath: path.join(__dirname, 'extension-host.cjs'),
      cachePath: path.resolve(__dirname, '../.vscode-test'),
      launchArgs: [
        workspace,
        '--user-data-dir',
        path.join(temporary, 'profile'),
        '--extensions-dir',
        path.join(temporary, 'extensions'),
        '--no-sandbox',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-workspace-trust',
        '--skip-welcome',
        '--skip-release-notes'
      ]
    });
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
