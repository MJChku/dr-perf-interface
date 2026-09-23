/* Preserve monorepo-relative README links in a portable VSIX. */
'use strict';
const { execFileSync, spawnSync } = require('node:child_process');
const commit = execFileSync('git', ['rev-parse', 'HEAD'], {
  cwd: __dirname,
  encoding: 'utf8'
}).trim();
const repository = 'MJChku/dr-perf-interface';
const result = spawnSync(
  process.execPath,
  [
    require.resolve('@vscode/vsce/vsce'),
    'package',
    '--skip-license',
    '--baseContentUrl',
    `https://github.com/${repository}/blob/${commit}/extensions/vscode`,
    '--baseImagesUrl',
    `https://raw.githubusercontent.com/${repository}/${commit}/extensions/vscode`,
    ...process.argv.slice(2)
  ],
  { cwd: __dirname, stdio: 'inherit' }
);
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
