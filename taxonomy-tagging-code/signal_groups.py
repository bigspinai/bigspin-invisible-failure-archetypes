"""
Signal groupings for two-pass tagging architecture.

Data-driven assignment tuned on test_1k, validated on score_10k (held-out).
Default is PASS1 (naive); signals are promoted to PASS2 (calibrated) only when:
  1. Calibration yields positive Δκ on the held-out score_10k validation set
  2. The signal has a calibration block in calibration.json (operational
     definitions, decision steps, worked examples)

Exception: ai_references_user_words has no calibration block but validated
well (+0.098 Δκ on 10k). silent_assumption validated slightly negative (-0.013)
but the calibration meaningfully tightens the definition (prevalence 25% → 6.5%).

  - PASS1_SIGNALS (42): sourced from naive pass
  - PASS2_SIGNALS (8): sourced from calibrated pass

Removed from PASS2 after score_10k validation:
  - ai_summarizes: Δκ = -0.214 on 10k (calibration hurts badly), no cal block
  - over_delivered: Δκ = -0.068 on 10k, no cal block
  - ai_flags_complexity: Δκ = +0.012 on 10k (negligible), no cal block
  - ai_normalizes_difficulty: Δκ = +0.045 on 10k (sub-threshold), no cal block
"""

# ── PASS1: sourced from naive pass ──────────────────────────────────────
#
# Δκ values from test_1k tuning set (2,129 turns, Fleiss' κ, sonnet + gpt5).
# 10k Δκ noted where signal was demoted from PASS2.

PASS1_SIGNALS = {
    # ── Strong naive advantage (Δκ < -0.10) ──────────────────────────
    "intent_addressed",               # Δκ = -0.428
    "scope_matched",                  # Δκ = -0.422
    "ai_provides_caveats",            # Δκ = -0.362
    "ai_offered_options",             # Δκ = -0.292
    "generate_without_clarifying",    # Δκ = -0.286
    "adaptation",                     # Δκ = -0.237
    "ai_acknowledges_correction",     # Δκ = -0.202
    "problem_ignored",                # Δκ = -0.190
    "ai_refuses_or_declines",         # Δκ = -0.162
    "ai_self_contradiction",          # Δκ = -0.143
    "ai_provides_alternatives",       # Δκ = -0.139
    "error_commitment",               # Δκ = -0.135
    "off_topic_drift",                # Δκ = -0.135
    "ai_offers_to_elaborate",         # Δκ = -0.133
    "ai_warns_user",                  # Δκ = -0.130
    "appropriate_confidence",         # Δκ = -0.118
    "ethical_tension",                # Δκ = -0.112
    "ai_provides_example",            # Δκ = -0.102
    "false_confidence",               # Δκ = -0.100
    "ai_asked_probing_question",      # Δκ = -0.100
    "ai_hedges_uncertainty",          # Δκ = -0.095
    "ai_validates_user",              # Δκ = -0.087
    # ── Moderate naive advantage ─────────────────────────────────────
    "plow_through",                   # Δκ = -0.052
    "under_delivered",                # Δκ = -0.046
    "ai_asserts_knowledge_limit",     # Δκ = -0.046
    "intent_missed",                  # Δκ = -0.043
    "conversation_advanced",          # Δκ = -0.041
    "conversation_stalled",           # Δκ = -0.039
    "ai_structured_response",         # Δκ = -0.014
    # ── Effectively tied / sub-threshold — default to naive ──────────
    "ai_implicit_refusal",            # Δκ = -0.003
    "ai_empathy_expressed",           # Δκ = -0.001
    "ai_asks_for_feedback",           # Δκ = -0.001
    "problem_surfaced",               # Δκ = +0.004
    "ai_asked_clarifying_question",   # Δκ = +0.005
    "ai_provides_step_by_step",       # Δκ = +0.008
    "ai_stated_interpretation",       # Δκ = +0.021
    "ai_cites_source",                # Δκ = +0.024
    "appropriate_hedge",              # Δκ = +0.048
    # ── Demoted from PASS2 after score_10k validation ────────────────
    "ai_summarizes",                  # 10k Δκ = -0.214  (no cal block)
    "over_delivered",                 # 10k Δκ = -0.068  (no cal block)
    "ai_flags_complexity",            # 10k Δκ = +0.012  (no cal block, negligible)
    "ai_normalizes_difficulty",       # 10k Δκ = +0.045  (no cal block, sub-threshold)
}

# ── PASS2: sourced from calibrated pass ─────────────────────────────────
#
# All have calibration blocks except ai_references_user_words.
# All validated with positive Δκ on held-out score_10k.

PASS2_SIGNALS = {
    "performative_hedge",             # 1k Δκ=+0.656  10k Δκ=+0.452  (cal block)
    "ai_references_prior_turn",       # 1k Δκ=+0.197  10k Δκ=+0.135  (cal block)
    "ai_malfunction",                 # 1k Δκ=+0.277  10k Δκ=+0.132  (cal block)
    "error_recovery",                 # 1k Δκ=+0.126  10k Δκ=+0.114  (cal block)
    "ai_references_user_words",       # 1k Δκ=+0.178  10k Δκ=+0.098  (no cal block)
    "repetition",                     # 1k Δκ=+0.186  10k Δκ=+0.091  (cal block)
    "factual_error",                  # 1k Δκ=+0.102  10k Δκ=+0.047  (cal block)
    "silent_assumption",              # 1k Δκ=+0.080  10k Δκ=-0.013  (cal block, tightens defn)
}


# ── Legacy groupings (for reference) ────────────────────────────────────

_LEGACY_SELF_EVIDENT = {
    "ai_asks_for_feedback", "ai_empathy_expressed", "ai_refuses_or_declines",
    "ai_provides_caveats", "ai_structured_response", "ai_warns_user",
    "ai_provides_step_by_step", "ai_provides_example", "ai_hedges_uncertainty",
    "ai_offered_options", "problem_ignored", "under_delivered", "adaptation",
    "scope_matched", "intent_missed", "conversation_advanced",
    "generate_without_clarifying", "ai_implicit_refusal", "ethical_tension",
}

_LEGACY_CONSTRUCTED = {
    "performative_hedge", "error_commitment", "error_recovery",
    "appropriate_hedge", "plow_through", "repetition", "off_topic_drift",
    "conversation_stalled", "silent_assumption", "ai_cites_source",
    "ai_references_prior_turn", "ai_asked_clarifying_question",
    "ai_validates_user", "ai_flags_complexity", "ai_acknowledges_correction",
    "ai_asserts_knowledge_limit",
}

_LEGACY_NEUTRAL = {
    "ai_malfunction", "ai_offers_to_elaborate", "ai_summarizes",
    "factual_error", "false_confidence",
}

_LEGACY_TIER2_NAIVE = {
    "ai_asked_probing_question", "ai_normalizes_difficulty",
    "ai_provides_alternatives", "ai_references_user_words",
    "ai_self_contradiction", "ai_stated_interpretation",
    "appropriate_confidence", "intent_addressed", "over_delivered",
    "problem_surfaced",
}

# Validation
assert len(PASS1_SIGNALS & PASS2_SIGNALS) == 0, "Overlap between pass 1 and pass 2"
assert len(PASS1_SIGNALS | PASS2_SIGNALS) == 50, f"Expected 50 signals, got {len(PASS1_SIGNALS | PASS2_SIGNALS)}"
