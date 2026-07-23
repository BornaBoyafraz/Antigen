"use strict";

(function exposeDecisionLogic(root) {
  function normalizeUnit(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.min(Math.max(number, 0), 1);
  }

  function decisionForScore(score, threshold) {
    return normalizeUnit(score) >= normalizeUnit(threshold) ? "injection" : "benign";
  }

  function operatingPointForThreshold(threshold) {
    const normalizedThreshold = normalizeUnit(threshold);
    if (normalizedThreshold < 0.35) {
      return {
        name: "Recall-first",
        guidance: "Flags more borderline text, usually catching more attacks while increasing false positives."
      };
    }
    if (normalizedThreshold > 0.65) {
      return {
        name: "Selective",
        guidance: "Flags fewer borderline cases, usually reducing false positives while missing more attacks."
      };
    }
    return {
      name: "Balanced",
      guidance: "Keeps the decision line near the evaluated default operating range."
    };
  }

  const decisionLogic = Object.freeze({
    decisionForScore,
    normalizeUnit,
    operatingPointForThreshold
  });

  if (typeof module === "object" && module.exports) {
    module.exports = decisionLogic;
  } else {
    root.AntigenDecision = decisionLogic;
  }
})(globalThis);
