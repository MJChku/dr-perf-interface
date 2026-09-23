'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const vscode = require('vscode');

async function run() {
  const extension = vscode.extensions.getExtension('MJChku.drperf-explorer');
  assert.ok(extension, 'extension is installed in the development host');
  const api = await extension.activate();
  const root = vscode.workspace.workspaceFolders[0].uri.fsPath;
  await api.load(vscode.Uri.file(path.join(__dirname, '../demo/pipeline.drperf.json')));
  const model = api.getReport();
  assert.equal(model.regions.length, 7);
  const document = await vscode.workspace.openTextDocument(
    path.join(root, 'examples/explorer/pipeline.c')
  );
  const editor = await vscode.window.showTextDocument(document);
  const lenses = await vscode.commands.executeCommand(
    'vscode.executeCodeLensProvider',
    document.uri
  );
  assert.ok(lenses.length >= 7, `expected source-linked regions, got ${lenses.length}`);
  assert.ok(lenses.some((l) => l.command?.title.includes('64*items')));
  const enqueue = model.regions.find((r) => r.id === 'enqueue');
  const line = enqueue.sources[0].line - 1;
  editor.selection = new vscode.Selection(line, 0, line, 0);
  assert.equal(api.currentRegion(editor), 'enqueue');
  const hovers = await vscode.commands.executeCommand(
    'vscode.executeHoverProvider',
    document.uri,
    new vscode.Position(line, 8)
  );
  assert.ok(hovers.length);
  await vscode.commands.executeCommand('drperf.inspectRegion', 'enqueue');
  await editor.edit((edit) =>
    edit.insert(new vscode.Position(0, 0), '/* changed after measurement */\n')
  );
  const stale = await vscode.commands.executeCommand(
    'vscode.executeCodeLensProvider',
    document.uri
  );
  assert.ok(stale.some((l) => l.command?.title.includes('source changed')));
  await vscode.window.showTextDocument(document);
  await vscode.commands.executeCommand('undo');
  for (let i = 0; i < 30 && document.isDirty; ++i)
    await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(document.isDirty, false);
  const invalidUri = vscode.Uri.file(path.join(root, 'invalid.drperf.json'));
  await vscode.workspace.fs.writeFile(
    invalidUri,
    Buffer.from(JSON.stringify({ schema: 'wrong', regions: 'not an array' }))
  );
  await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(invalidUri));
  let diagnostics = [];
  for (let i = 0; i < 100; ++i) {
    diagnostics = vscode.languages.getDiagnostics(invalidUri);
    if (diagnostics.length) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  console.log(
    'SCHEMA DIAGNOSTICS:',
    JSON.stringify(diagnostics.map((d) => ({ severity: d.severity, message: d.message })))
  );
  assert.ok(
    diagnostics.some((d) => /required|array|drperf.explorer.v1/.test(d.message)),
    'bundled JSON schema validates exported-report documents'
  );
  await vscode.commands.executeCommand('workbench.action.closeAllEditors');
  console.log(
    'EXTENSION HOST PASS: activation, report loading, source CodeLens, hover, cursor mapping, webview command, stale-source detection, JSON schema diagnostics.'
  );
}
module.exports = { run };
