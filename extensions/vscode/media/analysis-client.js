/* Local background analysis with one running request and one queued request
 * per operation. Repeated edits replace queued work instead of building a
 * backlog. Only this extension's fixed source assets are fetched. */
(function () {
  'use strict';
  const assets = document.currentScript.dataset;
  const queue = new Map();
  let worker = null,
    starting = null,
    active = null,
    model = null,
    nextId = 0,
    failure = null;
  const aborted = () =>
    Object.assign(new Error('Superseded analysis request.'), { name: 'AbortError' });
  async function start() {
    if (worker) return worker;
    if (failure) throw failure;
    if (!starting)
      starting = (async () => {
        const sources = await Promise.all(
          [assets.expressions, assets.model, assets.worker].map(async (url) => {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Cannot load the local analysis worker.');
            return response.text();
          })
        );
        const url = URL.createObjectURL(new Blob(sources, { type: 'text/javascript' }));
        worker = new Worker(url);
        URL.revokeObjectURL(url);
        worker.onmessage = (event) => {
          if (!active || event.data.id !== active.id) return;
          const done = active;
          active = null;
          if (event.data.error) done.reject(new Error(event.data.error));
          else done.resolve(event.data.value);
          void drain();
        };
        worker.onerror = (event) => {
          failure = new Error(
            event.message || 'Background analysis failed. Close and reopen the explorer.'
          );
          active?.reject(failure);
          active = null;
          for (const request of queue.values()) request.reject(failure);
          queue.clear();
          worker.terminate();
          worker = null;
        };
        return worker;
      })().catch((error) => {
        failure = error;
        throw error;
      });
    return starting;
  }
  async function drain() {
    if (active || !queue.size) return;
    const request = queue.values().next().value;
    queue.delete(request.kind);
    active = request;
    try {
      const target = await start();
      if (model !== request.model) {
        target.postMessage({ type: 'model', model: request.model });
        model = request.model;
      }
      target.postMessage({ id: request.id, kind: request.kind, payload: request.payload });
    } catch (error) {
      if (active === request) active = null;
      request.reject(error);
      void drain();
    }
  }
  window.DrperfAnalysis = {
    request(kind, currentModel, payload) {
      return new Promise((resolve, reject) => {
        queue.get(kind)?.reject(aborted());
        queue.set(kind, { id: ++nextId, kind, model: currentModel, payload, resolve, reject });
        void drain();
      });
    },
    get backend() {
      return worker ? 'worker' : failure ? 'failed' : 'starting';
    }
  };
  window.addEventListener('pagehide', () => worker?.terminate(), { once: true });
})();
