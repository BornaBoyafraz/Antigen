"use strict";

(function exposeConversationLogic(root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./decision.js"));
  } else {
    root.AntigenConversation = factory(root.AntigenDecision);
  }
})(globalThis, function buildConversationLogic(decisionLogic) {
  const { normalizeUnit } = decisionLogic;

  function normalizeConversationPayload(payload) {
    const rawTurns = Array.isArray(payload && payload.turns) ? payload.turns : [];
    return {
      label: payload && payload.label === "injection" ? "injection" : "benign",
      score: normalizeUnit(payload && payload.score),
      escalationDetected: Boolean(payload && payload.escalation_detected),
      turns: rawTurns.map((turn) => ({
        text: String(turn && turn.text || ""),
        label: turn && turn.label === "injection" ? "injection" : "benign",
        score: normalizeUnit(turn && turn.score),
        discussionOverrideApplied: Boolean(turn && turn.discussion_override_applied),
        codewordSetups: Array.isArray(turn && turn.codeword_setups)
          ? turn.codeword_setups.map(String)
          : [],
        codewordInvoked: turn && turn.codeword_invoked
          ? String(turn.codeword_invoked)
          : null
      }))
    };
  }

  function escalationSummary(result) {
    if (!result.escalationDetected) {
      return {
        title: "No codeword-smuggling escalation detected",
        detail: "No later short turn invoked a trigger token defined earlier in this conversation."
      };
    }

    const invokedCodewords = result.turns
      .map((turn) => turn.codewordInvoked)
      .filter(Boolean);
    const uniqueCodewords = [...new Set(invokedCodewords)];
    const tokenText = uniqueCodewords.length
      ? ` Triggered codeword: ${uniqueCodewords.map((word) => `“${word}”`).join(", ")}.`
      : "";
    return {
      title: "Codeword-smuggling escalation detected",
      detail: `A later short turn invoked a trigger token established earlier.${tokenText}`
    };
  }

  return Object.freeze({
    escalationSummary,
    normalizeConversationPayload
  });
});
