/* Appended to expressions.js and model.js in one local blob. No dynamic imports. */
'use strict';
let currentModel;
self.onmessage = function (event) {
  const message = event.data;
  if (message.type === 'model') {
    currentModel = message.model;
    return;
  }
  const { id, kind, payload } = message;
  try {
    let value;
    if (kind === 'scenario') {
      value = { result: DrperfModel.replay(currentModel, payload.scenario) };
      if (payload.measured) {
        try {
          value.validation = DrperfModel.validateScenario(
            currentModel,
            payload.scenario,
            payload.measured
          );
        } catch (error) {
          value.validationError = error.message;
        }
      }
    } else if (kind === 'proposal')
      value = DrperfModel.proposeRelationship(currentModel, payload.target, payload.expression);
    else if (kind === 'comparison')
      value = DrperfModel.compareInterfaces(currentModel, payload.measured, payload.options);
    else if (kind === 'experiments')
      value = DrperfModel.findDistinguishingExperiments(
        currentModel,
        payload.assumptions,
        payload.options
      );
    else if (kind === 'relationship-check')
      value = DrperfModel.checkRelationships(currentModel, payload.measured, payload.options);
    else throw new Error('Unknown analysis operation.');
    self.postMessage({ id, value });
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
