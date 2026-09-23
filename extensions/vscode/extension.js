'use strict';
const vscode = require('vscode');
const path = require('node:path');
const fs = require('node:fs/promises');
const crypto = require('node:crypto');
const { execFile } = require('node:child_process');
const Model = require('./media/model');

function activate(context) {
  let report = null,
    reportUri = null,
    panel = null,
    selected = null,
    demo = false,
    watcher = null;
  const changed = new vscode.EventEmitter();
  const lensesChanged = new vscode.EventEmitter();
  const documentHashes = new WeakMap();
  const output = vscode.window.createOutputChannel('drperf');
  context.subscriptions.push(changed, lensesChanged, output, {
    dispose: () => {
      watcher?.dispose();
      panel?.dispose();
    }
  });
  const configuration = () => vscode.workspace.getConfiguration('drperf');
  const sourceRoot = () => {
    const configured = configuration().get('sourceRoot', '');
    const folder =
      (reportUri && vscode.workspace.getWorkspaceFolder(reportUri)) ||
      vscode.workspace.workspaceFolders?.[0];
    return configured ? path.resolve(folder?.uri.fsPath || '', configured) : folder?.uri.fsPath;
  };
  function sourceUri(location) {
    if (demo) return vscode.Uri.joinPath(context.extensionUri, 'demo', 'pipeline.c');
    const root = sourceRoot();
    if (!root || typeof location.path !== 'string' || path.isAbsolute(location.path)) return null;
    const candidate = path.resolve(root, location.path),
      relative = path.relative(root, candidate);
    if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative))
      return null;
    return vscode.Uri.file(candidate);
  }
  function locations(document) {
    if (!report || document.uri.scheme !== 'file') return [];
    const result = [];
    for (const region of report.regions)
      for (const location of region.sources) {
        if (sourceUri(location)?.toString() === document.uri.toString())
          result.push({ region, location });
      }
    return result;
  }
  function stale(document, location) {
    let cached = documentHashes.get(document);
    if (!cached || cached.version !== document.version) {
      cached = {
        version: document.version,
        hash: crypto.createHash('sha256').update(document.getText()).digest('hex')
      };
      documentHashes.set(document, cached);
    }
    return location.sha256 !== cached.hash;
  }
  function currentRegion(editor = vscode.window.activeTextEditor) {
    if (!editor) return null;
    const line = editor.selection.active.line + 1;
    const candidates = locations(editor.document).filter(
      ({ location }) => location.line <= line && line <= location.endLine
    );
    candidates.sort(
      (a, b) => a.location.endLine - a.location.line - (b.location.endLine - b.location.line)
    );
    return candidates[0]?.region.id || null;
  }
  function sendModel() {
    panel?.webview.postMessage({
      type: 'model',
      model: report,
      selected,
      demo,
      sourceAvailable: !!sourceRoot() || demo,
      reportPath: reportUri?.fsPath || 'Bundled example'
    });
  }
  function selectRegion(regionId, reveal = true) {
    if (!report?.regions.some((r) => r.id === regionId)) return;
    selected = regionId;
    if (reveal) showExplorer();
    panel?.webview.postMessage({ type: 'select', region: regionId });
  }
  async function openSource(regionId, index = 0) {
    const region = report?.regions.find((r) => r.id === regionId);
    const location = region?.sources[index];
    const uri = location && sourceUri(location);
    if (!uri) {
      vscode.window.showInformationMessage(
        'No source location in this workspace. Set drperf.sourceRoot or export with --source-root.'
      );
      return;
    }
    try {
      const document = await vscode.workspace.openTextDocument(uri);
      const line = Math.min(Math.max(0, location.line - 1), document.lineCount - 1);
      await vscode.window.showTextDocument(document, {
        viewColumn: vscode.ViewColumn.One,
        selection: new vscode.Range(line, 0, line, 0)
      });
      if (stale(document, location))
        vscode.window.showWarningMessage(
          'Source has changed since this report was exported. Recorded formulas may be stale.'
        );
    } catch (error) {
      vscode.window.showErrorMessage('Cannot open region source: ' + error.message);
    }
  }
  function showExplorer() {
    if (!report) {
      openReport();
      return;
    }
    if (panel) {
      panel.reveal(vscode.ViewColumn.Beside, true);
      return;
    }
    panel = vscode.window.createWebviewPanel(
      'drperf.explorer',
      'drperf · Region Explorer',
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')]
      }
    );
    const webview = panel.webview;
    const nonce = crypto.randomBytes(24).toString('base64');
    const media = (file) =>
      webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'media', file));
    webview.html = `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
      <link rel="stylesheet" href="${media('explorer.css')}"><title>drperf Region Explorer</title></head>
      <body><div id="app" aria-live="polite"></div><script nonce="${nonce}" src="${media('expressions.js')}"></script><script nonce="${nonce}" src="${media('model.js')}"></script><script nonce="${nonce}" src="${media('explorer.js')}"></script></body></html>`;
    panel.onDidDispose(
      () => {
        panel = null;
      },
      null,
      context.subscriptions
    );
    webview.onDidReceiveMessage(
      async (message) => {
        if (!message || typeof message.type !== 'string') return;
        if (message.type === 'ready') sendModel();
        else if (message.type === 'source' && typeof message.region === 'string')
          await openSource(message.region, Number.isInteger(message.index) ? message.index : 0);
        else if (message.type === 'select' && typeof message.region === 'string')
          selectRegion(message.region, false);
        else if (message.type === 'reload') await reload();
        else if (message.type === 'loadValidationReport') {
          const uri = (
            await vscode.window.showOpenDialog({
              canSelectMany: false,
              openLabel: 'Compare against measured run',
              filters: { 'drperf report': ['drperf.json', 'json'] }
            })
          )?.[0];
          if (!uri) return;
          try {
            const measured = await readModel(uri);
            panel?.webview.postMessage({
              type: 'validationModel',
              model: measured,
              name: path.basename(uri.fsPath)
            });
          } catch (error) {
            vscode.window.showErrorMessage('Cannot load validation report: ' + error.message);
          }
        } else if (message.type === 'exportScenario' && message.scenario && report) {
          try {
            const result = Model.replay(report, message.scenario);
            const uri = await vscode.window.showSaveDialog({
              filters: { 'drperf scenario': ['json'] },
              saveLabel: 'Save scenario'
            });
            if (uri)
              await vscode.workspace.fs.writeFile(
                uri,
                Buffer.from(
                  JSON.stringify({ schema: 'drperf.scenario.v1', ...result }, null, 2) + '\n'
                )
              );
          } catch (error) {
            vscode.window.showErrorMessage(error.message);
          }
        }
      },
      null,
      context.subscriptions
    );
  }
  async function readModel(uri) {
    const info = await vscode.workspace.fs.stat(uri);
    if (info.size > 50 * 1024 * 1024)
      throw new Error('Report is larger than 50 MiB. Export fewer runs or lower --max-trace.');
    const bytes = await vscode.workspace.fs.readFile(uri);
    return Model.validate(JSON.parse(Buffer.from(bytes).toString('utf8')));
  }
  async function load(uri, isDemo = false) {
    const parsed = await readModel(uri);
    report = parsed;
    reportUri = uri;
    demo = isDemo;
    if (!report.regions.some((r) => r.id === selected)) selected = report.regions[0]?.id;
    context.workspaceState.update('drperf.lastReport', isDemo ? null : uri.toString());
    vscode.commands.executeCommand('setContext', 'drperf.reportLoaded', true);
    changed.fire();
    lensesChanged.fire();
    sendModel();
    watcher?.dispose();
    if (!isDemo) {
      watcher = vscode.workspace.createFileSystemWatcher(
        new vscode.RelativePattern(path.dirname(uri.fsPath), path.basename(uri.fsPath))
      );
      watcher.onDidChange(() => reload());
    }
  }
  async function reload() {
    if (!reportUri) return;
    try {
      await load(reportUri, demo);
    } catch (error) {
      vscode.window.showErrorMessage('Could not reload drperf report: ' + error.message);
    }
  }
  async function openReport(uri) {
    try {
      if (!(uri instanceof vscode.Uri)) {
        const chosen = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: { 'drperf report': ['drperf.json', 'json'] }
        });
        uri = chosen?.[0];
      }
      if (!uri) return;
      await load(uri);
      showExplorer();
    } catch (error) {
      vscode.window.showErrorMessage('Cannot open drperf report: ' + error.message);
    }
  }
  async function exportReport() {
    if (!vscode.workspace.isTrusted) {
      vscode.window.showWarningMessage(
        'Trust this workspace before running the drperf exporter. Existing reports can still be viewed.'
      );
      return;
    }
    const raw = (
      await vscode.window.showOpenDialog({
        canSelectFiles: false,
        canSelectFolders: true,
        canSelectMany: false,
        openLabel: 'Select raw drperf measurements'
      })
    )?.[0];
    if (!raw) return;
    const destination = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.joinPath(raw, '..', 'regions.drperf.json'),
      filters: { 'drperf report': ['drperf.json'] }
    });
    if (!destination) return;
    const root =
      sourceRoot() ||
      vscode.workspace.getWorkspaceFolder(raw)?.uri.fsPath ||
      path.dirname(raw.fsPath);
    const script =
      configuration().get('exporterPath', '') || path.join(root, 'bin', 'drperf-export');
    const python = configuration().get('pythonPath', 'python3');
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'Exporting drperf region interfaces',
        cancellable: true
      },
      async (_, cancellation) => {
        try {
          await new Promise((resolve, reject) => {
            const child = execFile(
              python,
              [script, raw.fsPath, '--source-root', root, '-o', destination.fsPath],
              { cwd: root, maxBuffer: 8 * 1024 * 1024, timeout: 300000 },
              (error, stdout, stderr) => {
                output.append(stdout);
                output.append(stderr);
                if (error) reject(error);
                else resolve();
              }
            );
            const cancel = cancellation.onCancellationRequested(() => child.kill());
            child.once('exit', () => cancel.dispose());
          });
          await load(destination);
          showExplorer();
        } catch (error) {
          output.show();
          vscode.window.showErrorMessage('drperf export failed: ' + error.message);
        }
      }
    );
  }
  const provider = {
    onDidChangeTreeData: changed.event,
    getChildren(element) {
      if (!report || element?.leaf) return [];
      if (!element) return report.regions.map((region) => ({ region }));
      if (element.region)
        return element.region.states.map((state) => ({
          region: element.region,
          state,
          leaf: true
        }));
      return [];
    },
    getTreeItem(element) {
      const { region, state } = element;
      const item = new vscode.TreeItem(
        state || region.name,
        element.leaf || !region.states.length
          ? vscode.TreeItemCollapsibleState.None
          : vscode.TreeItemCollapsibleState.Collapsed
      );
      item.description = state ? 'PCV' : `${region.calls} calls`;
      item.tooltip =
        region.regimes.map((fit) => Model.formula(region, fit)).join('\n') ||
        'Insufficient varied states';
      item.command = {
        command: 'drperf.inspectRegion',
        title: 'Inspect performance interface',
        arguments: [region.id]
      };
      return item;
    }
  };
  context.subscriptions.push(vscode.window.registerTreeDataProvider('drperf.regions', provider));
  const selectors = ['python', 'c', 'cpp', 'rust'].map((language) => ({
    language,
    scheme: 'file'
  }));
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(selectors, {
      onDidChangeCodeLenses: lensesChanged.event,
      provideCodeLenses(document) {
        if (!configuration().get('codeLens', true)) return [];
        return locations(document)
          .filter(({ location }) => location.line <= document.lineCount)
          .map(({ region, location }) => {
            const title = stale(document, location)
              ? 'drperf: source changed · inspect recorded interface'
              : `drperf: ${Model.formula(region)} · ${region.calls} calls`;
            return new vscode.CodeLens(
              new vscode.Range(location.line - 1, 0, location.line - 1, 0),
              { command: 'drperf.inspectRegion', title, arguments: [region.id] }
            );
          });
      }
    })
  );
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(selectors, {
      provideHover(document, position) {
        const match = locations(document).find(
          ({ location }) => location.line === position.line + 1
        );
        if (!match) return;
        const markdown = new vscode.MarkdownString();
        markdown.appendText(match.region.name + ' — CPU instructions per call\n\n');
        match.region.regimes.forEach((fit) =>
          markdown.appendCodeblock(Model.formula(match.region, fit), 'text')
        );
        markdown.appendText(
          stale(document, match.location)
            ? 'Source changed since export.'
            : 'Recorded interface; valid only at observed states within checker tolerance.'
        );
        return new vscode.Hover(markdown);
      }
    })
  );
  context.subscriptions.push(
    vscode.window.onDidChangeTextEditorSelection((event) => {
      if (panel && configuration().get('followCursor', true)) {
        const region = currentRegion(event.textEditor);
        if (region) selectRegion(region, false);
      }
    })
  );
  context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(() => lensesChanged.fire()));
  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(() => lensesChanged.fire()));
  for (const [command, callback] of Object.entries({
    'drperf.openReport': openReport,
    'drperf.showExplorer': showExplorer,
    'drperf.exportReport': exportReport,
    'drperf.refresh': reload,
    'drperf.openRefinedDemo': async () => {
      await load(vscode.Uri.joinPath(context.extensionUri, 'demo', 'refined.drperf.json'), true);
      selected = 'lookup';
      showExplorer();
      sendModel();
    },
    'drperf.openDemo': async () => {
      await load(vscode.Uri.joinPath(context.extensionUri, 'demo', 'pipeline.drperf.json'), true);
      showExplorer();
    },
    'drperf.inspectRegion': async (region) => {
      if (!report) {
        await openReport();
        return;
      }
      const candidate = typeof region === 'string' ? region : currentRegion();
      if (candidate) selectRegion(candidate);
      else showExplorer();
    }
  }))
    context.subscriptions.push(vscode.commands.registerCommand(command, callback));
  const saved = context.workspaceState.get('drperf.lastReport');
  if (saved)
    load(vscode.Uri.parse(saved)).catch((error) =>
      output.appendLine('Previous report unavailable: ' + error.message)
    );
  return { load, getReport: () => report, currentRegion };
}

function deactivate() {}
module.exports = { activate, deactivate };
