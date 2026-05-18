"""
Prompt builder for taxonomy tagging.

Single source of truth for turning a taxonomy JSON + calibration JSON into
the final prompts sent to tagger models. No text parsing, no regex, no
implicit file loading — everything is explicit and validated.

Architecture:
    taxonomy.json   ─┐
                     ├─→  PromptBuilder  ─→  prompt string
    calibration.json ─┘        ↑
                          turn_pair dict

Usage:
    builder = PromptBuilder.from_files("taxonomy.json", "calibration.json")
    prompt  = builder.build_ai_prompt(turn_pair)
    prompt  = builder.build_user_prompt(turn_pair)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── Signal routing ──────────────────────────────────────────────────────

AI_L3_SIGNALS = frozenset({
    "ai_self_contradiction",
    "ai_implicit_refusal",
    "ai_malfunction",
    "ethical_tension",
})

USER_L3_SIGNALS = frozenset({
    "user_asks_clarification",
    "user_corrects_ai",
    "user_implicit_correction",
    "user_expresses_frustration",
    "user_expresses_dissatisfaction",
    "user_scope_change",
    "user_validation_seeking",
    "user_passive_acceptance",
    "user_ambiguous_request",
    "user_provides_invalid_input",
    "user_positive_feedback",
    "user_multi_request",
    "user_abandons_thread",
    "user_repeats_request",
})


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class SignalDef:
    """A single signal definition from the taxonomy."""
    name: str
    layer: str              # "L1", "L2", "L3"
    description: str
    category: str = ""
    valence: str = ""       # "positive", "negative", "neutral", or ""
    structural_gate: str = ""
    boundary_notes: str = ""
    counts: list[str] = field(default_factory=list)
    does_not_count: list[str] = field(default_factory=list)
    tier: int = 1           # 1 = reliable, 2 = not yet reliable


@dataclass
class CalibrationBlock:
    """Structured calibration data for one signal."""
    signal: str
    definition: str                          # one-line definition recap
    decision_steps: list[str]                # ordered decision tree steps
    examples: list[dict]                     # {"turn_id": str, "label": int, "category": str, "rationale": str}
    source_round: str = ""                   # which round produced this calibration
    profile_type: str = ""                   # "threshold_dominant", "definition_dominant", "mixed"


@dataclass
class Taxonomy:
    """Parsed taxonomy with signal definitions."""
    name: str
    version: str
    signals: dict[str, SignalDef]            # signal_name → SignalDef

    @property
    def l1_signals(self) -> dict[str, SignalDef]:
        return {k: v for k, v in self.signals.items() if v.layer == "L1"}

    @property
    def l2_signals(self) -> dict[str, SignalDef]:
        return {k: v for k, v in self.signals.items() if v.layer == "L2"}

    @property
    def ai_l3_signals(self) -> dict[str, SignalDef]:
        return {k: v for k, v in self.signals.items()
                if v.layer == "L3" and k in AI_L3_SIGNALS}

    @property
    def user_l3_signals(self) -> dict[str, SignalDef]:
        return {k: v for k, v in self.signals.items()
                if v.layer == "L3" and k in USER_L3_SIGNALS}

    @property
    def ai_signals(self) -> dict[str, SignalDef]:
        """All signals tagged in the AI pass (L1 + L2 + AI-L3)."""
        return {**self.l1_signals, **self.l2_signals, **self.ai_l3_signals}

    @property
    def user_signals(self) -> dict[str, SignalDef]:
        """All signals tagged in the user pass (user-L3)."""
        return self.user_l3_signals


@dataclass
class Calibration:
    """All calibration blocks, keyed by signal name."""
    blocks: dict[str, CalibrationBlock]
    version: str = ""

    def get(self, signal_name: str) -> CalibrationBlock | None:
        return self.blocks.get(signal_name)


# ── Loading ─────────────────────────────────────────────────────────────

def load_taxonomy(path: str | Path) -> Taxonomy:
    """Load a taxonomy JSON file and return a structured Taxonomy object."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    signals = {}
    for layer_key, layer_label in [
        ("layer1_signals", "L1"),
        ("layer2_signals", "L2"),
        ("layer3_signals", "L3"),
    ]:
        for sig_name, sig_def in raw.get(layer_key, {}).items():
            signals[sig_name] = SignalDef(
                name=sig_name,
                layer=layer_label,
                description=sig_def.get("description", ""),
                category=sig_def.get("category", ""),
                valence=sig_def.get("valence", ""),
                structural_gate=sig_def.get("structural_gate", ""),
                boundary_notes=sig_def.get("boundary_notes", ""),
                counts=sig_def.get("counts", []),
                does_not_count=sig_def.get("does_not_count", []),
                tier=sig_def.get("tier", 1),
            )

    return Taxonomy(
        name=raw.get("name", ""),
        version=raw.get("version", ""),
        signals=signals,
    )


def load_calibration(path: str | Path) -> Calibration:
    """Load a calibration JSON file and return a structured Calibration object."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    blocks = {}
    for sig_name, block_data in raw.get("signals", {}).items():
        blocks[sig_name] = CalibrationBlock(
            signal=sig_name,
            definition=block_data.get("definition", ""),
            decision_steps=block_data.get("decision_steps", []),
            examples=block_data.get("examples", []),
            source_round=block_data.get("source_round", ""),
            profile_type=block_data.get("profile_type", ""),
        )

    return Calibration(
        blocks=blocks,
        version=raw.get("version", ""),
    )


# ── Validation ──────────────────────────────────────────────────────────

class ValidationError:
    def __init__(self, level: str, message: str):
        self.level = level  # "error" or "warning"
        self.message = message

    def __repr__(self):
        return f"[{self.level.upper()}] {self.message}"


def validate(taxonomy: Taxonomy, calibration: Calibration) -> list[ValidationError]:
    """
    Validate that taxonomy and calibration are consistent.
    Returns a list of issues. Empty list = all good.
    """
    issues = []

    # 1. Every calibration block must reference a signal that exists in the taxonomy
    for sig_name in calibration.blocks:
        if sig_name not in taxonomy.signals:
            issues.append(ValidationError(
                "error",
                f"Calibration block for '{sig_name}' has no matching signal in taxonomy"
            ))

    # 2. Every calibration block must have at least one decision step
    for sig_name, block in calibration.blocks.items():
        if not block.decision_steps:
            issues.append(ValidationError(
                "error",
                f"Calibration block for '{sig_name}' has zero decision steps"
            ))

    # 3. Warn about calibrated signals that are Tier 2
    for sig_name, block in calibration.blocks.items():
        sig = taxonomy.signals.get(sig_name)
        if sig and sig.tier == 2:
            issues.append(ValidationError(
                "warning",
                f"'{sig_name}' is Tier 2 but has calibration — intentional?"
            ))

    # 4. Warn about dead calibration (signal not in any pass)
    for sig_name in calibration.blocks:
        sig = taxonomy.signals.get(sig_name)
        if sig:
            is_ai = sig_name in taxonomy.ai_signals
            is_user = sig_name in taxonomy.user_signals
            if not is_ai and not is_user:
                issues.append(ValidationError(
                    "error",
                    f"'{sig_name}' has calibration but isn't routed to AI or user pass"
                ))

    # 5. Check for L3 signals not in either routing set
    for sig_name, sig in taxonomy.signals.items():
        if sig.layer == "L3" and sig_name not in AI_L3_SIGNALS and sig_name not in USER_L3_SIGNALS:
            issues.append(ValidationError(
                "warning",
                f"L3 signal '{sig_name}' not in AI_L3_SIGNALS or USER_L3_SIGNALS — won't be tagged"
            ))

    return issues


# ── Prompt building ─────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds tagger prompts from taxonomy + calibration.

    Immutable after construction. All prompt text is generated
    deterministically from the inputs.
    """

    def __init__(self, taxonomy: Taxonomy, calibration: Calibration,
                 meta_calibration_text: str = None, naive: bool = False):
        self.taxonomy = taxonomy
        self.calibration = calibration
        self.meta_calibration_text = meta_calibration_text
        self.naive = naive

        # Validate at construction time
        issues = validate(taxonomy, calibration)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]

        if warnings:
            for w in warnings:
                print(f"  PromptBuilder WARNING: {w.message}")
        if errors:
            error_msgs = "\n".join(f"  - {e.message}" for e in errors)
            raise ValueError(
                f"Cannot build prompts — {len(errors)} validation error(s):\n{error_msgs}"
            )

        # Pre-build the taxonomy text sections (they don't change per turn)
        self._ai_taxonomy_text = self._build_ai_taxonomy_text()
        self._user_taxonomy_text = self._build_user_taxonomy_text()

        if meta_calibration_text:
            print(f"  PromptBuilder: meta-calibration guidance loaded ({len(meta_calibration_text)} chars)")

    @classmethod
    def from_files(cls, taxonomy_path: str | Path, calibration_path: str | Path,
                   meta_calibration_path: str | Path = None,
                   naive: bool = False) -> "PromptBuilder":
        """Load from files and construct."""
        taxonomy = load_taxonomy(taxonomy_path)
        calibration = load_calibration(calibration_path)
        meta_text = None
        if meta_calibration_path:
            import json as _json
            with open(meta_calibration_path) as f:
                mc_data = _json.load(f)
            meta_text = mc_data.get("guidance_text")
            if meta_text:
                print(f"  Loaded meta-calibration from {Path(meta_calibration_path).name}")
        return cls(taxonomy, calibration, meta_text, naive=naive)

    # ── Taxonomy text building ──────────────────────────────────────────

    def _format_signal_base(self, sig: SignalDef, max_examples: int = 3) -> list[str]:
        """Format a signal's base definition (no calibration)."""
        lines = []
        header = f"### {sig.name}"
        if sig.valence:
            header += f" [{sig.valence}]"
        lines.append(header)
        lines.append(f"  {sig.description}")
        if sig.structural_gate:
            lines.append(f"  GATE: {sig.structural_gate}")
        if sig.boundary_notes:
            lines.append(f"  BOUNDARY: {sig.boundary_notes}")
        if sig.counts:
            lines.append("  Counts:")
            for ex in sig.counts[:max_examples]:
                lines.append(f"    - {ex}")
        if sig.does_not_count:
            lines.append("  Does NOT count:")
            for ex in sig.does_not_count[:max_examples]:
                lines.append(f"    - {ex}")
        lines.append("")
        return lines

    def _format_calibration(self, block: CalibrationBlock) -> list[str]:
        """Format a calibration block as indented text."""
        lines = []
        if block.definition:
            lines.append(f"  Definition: {block.definition}")
            lines.append("")
        lines.append("  decision path:")
        for i, step in enumerate(block.decision_steps, 1):
            lines.append(f"    {i}. {step}")
        if block.examples:
            lines.append("")
            lines.append("  Examples:")
            for ex in block.examples:
                turn_id = ex.get("turn_id", "?")
                label = ex.get("label", "?")
                label_str = "1 (present)" if label == 1 else "0 (absent)"
                category = ex.get("category", "")
                cat_str = f" [{category}]" if category else ""
                rationale = ex.get("rationale", "")
                lines.append(f"    {turn_id}: {label_str}{cat_str} — {rationale}")
        return lines

    def _format_signal_with_calibration(self, sig: SignalDef, block: CalibrationBlock) -> list[str]:
        """Format a signal with its calibration inlined."""
        lines = []
        header = f"### {sig.name}"
        if sig.valence:
            header += f" [{sig.valence}]"
        header += " [CALIBRATED]"
        lines.append(header)
        lines.append(f"  {sig.description}")
        if sig.structural_gate:
            lines.append(f"  GATE: {sig.structural_gate}")
        if sig.boundary_notes:
            lines.append(f"  BOUNDARY: {sig.boundary_notes}")
        lines.append("")
        lines.append(f"  CALIBRATION for {sig.name}:")
        lines.extend(self._format_calibration(block))
        lines.append("")
        return lines

    def _format_signal(self, sig: SignalDef, max_examples: int = 3) -> list[str]:
        """Format a signal, with calibration if available."""
        block = self.calibration.get(sig.name)
        if block:
            return self._format_signal_with_calibration(sig, block)
        return self._format_signal_base(sig, max_examples)

    def _build_ai_taxonomy_text(self) -> str:
        """Build the two-phase AI taxonomy text."""
        lines = []

        # Phase 1: L1 structural
        lines.append("## PHASE 1 — STRUCTURAL SIGNALS (tag these first)")
        lines.append("Observable AI response behaviors. No inference required — point at specific tokens.\n")
        for sig_name in sorted(self.taxonomy.l1_signals):
            sig = self.taxonomy.signals[sig_name]
            lines.extend(self._format_signal(sig, max_examples=3))

        # Phase 2: L2 evaluative
        lines.append("\n## PHASE 2 — INTERACTION OUTCOMES & FAILURE MODES (tag using Phase 1 as context)")
        lines.append("Evaluative judgments. Use your Phase 1 tags to inform these decisions.")
        lines.append("Signals marked [CALIBRATED] have decision rules and examples — follow them strictly.\n")
        for sig_name in sorted(self.taxonomy.l2_signals):
            sig = self.taxonomy.signals[sig_name]
            lines.extend(self._format_signal(sig, max_examples=2))

        # AI-focused L3
        lines.append("\n## FAILURE MODES (AI-specific)")
        for sig_name in sorted(self.taxonomy.ai_l3_signals):
            sig = self.taxonomy.signals[sig_name]
            lines.extend(self._format_signal(sig, max_examples=3))

        return "\n".join(lines)

    def _build_user_taxonomy_text(self) -> str:
        """Build the user-turn taxonomy text."""
        lines = []
        lines.append("## USER BEHAVIOR SIGNALS")
        lines.append("Observable user behaviors and patterns in the user message.\n")
        for sig_name in sorted(self.taxonomy.user_l3_signals):
            sig = self.taxonomy.signals[sig_name]
            lines.extend(self._format_signal(sig, max_examples=3))
        return "\n".join(lines)

    # ── Turn formatting ─────────────────────────────────────────────────

    @staticmethod
    def _build_turn_section(turn: dict, tag_target: str = "ai") -> str:
        """
        Build the turn content section.

        tag_target: "ai" or "user" — controls which part is labeled TAG THIS.
        """
        prior = turn.get("prior_context", "")
        prior_section = ""
        if prior:
            prior_section = (
                f"PRIOR CONVERSATION CONTEXT (for reference — do NOT tag these turns):\n"
                f"{prior}\n\n---\n"
            )

        tn = turn["turn_number"]

        if tag_target == "ai":
            user_label = "USER MESSAGE (context for the AI response)"
            ai_label = "AI RESPONSE (TAG THIS)"
        else:
            user_label = "USER MESSAGE (TAG THIS)"
            ai_label = "AI RESPONSE (context for the user message)"

        return (
            f"{prior_section}"
            f"CURRENT TURN (Turn {tn}):\n\n"
            f"{user_label}:\n{turn['user_content']}\n\n"
            f"{ai_label}:\n{turn['ai_content']}"
        )

    # ── Full prompt building ────────────────────────────────────────────

    def build_ai_prompt(self, turn: dict) -> str:
        """Build the complete AI-turn tagging prompt (phased output).
        If self.naive is True, delegates to build_naive_ai_prompt instead."""
        if self.naive:
            return self.build_naive_ai_prompt(turn)
        turn_section = self._build_turn_section(turn, tag_target="ai")
        tn = turn["turn_number"]

        # Build optional meta-calibration section
        meta_section = ""
        if self.meta_calibration_text:
            meta_section = (
                f"ANNOTATION DENSITY GUIDANCE:\n"
                f"{self.meta_calibration_text}\n\n"
            )

        return (
            f"You are an expert evaluator of AI responses in human-AI conversations. "
            f"Tag the AI RESPONSE below for behavioral signals using a two-phase process.\n\n"
            f"{meta_section}"
            f"PROCESS:\n"
            f"1. FIRST, tag all Phase 1 (Structural) signals by examining the text directly.\n"
            f"2. THEN, tag all Phase 2 (Evaluative) signals, using your Phase 1 tags to inform your judgment.\n"
            f"   Example: if you tagged ai_provides_step_by_step=present in Phase 1, factor that into your\n"
            f"   evaluation of under_delivered in Phase 2.\n\n"
            f"RULES:\n"
            f"1. Tag ONLY the AI response — not the user message or prior turns.\n"
            f"2. Check GATE conditions FIRST. If the gate fails, the signal CANNOT fire.\n"
            f"3. Apply BOUNDARY rules strictly. For [CALIBRATED] signals, follow the decision tree step by step.\n"
            f"4. Only tag signals you are confident are present. When in doubt, do NOT tag.\n"
            f"5. Base judgment on observable text features, not subjective impressions.\n\n"
            f"SIGNALS:\n{self._ai_taxonomy_text}\n\n"
            f"{turn_section}\n\n"
            f'Return a JSON object with your tags organized by phase:\n'
            f'{{\n'
            f'  "turn_number": {tn},\n'
            f'  "phase1_signals": [\n'
            f'    {{"signal": "signal_name", "evidence": "brief quote or description"}}\n'
            f'  ],\n'
            f'  "phase2_signals": [\n'
            f'    {{"signal": "signal_name", "evidence": "brief quote or description"}}\n'
            f'  ],\n'
            f'  "signal_count": <total integer across both phases>\n'
            f'}}\n\n'
            f"If no signals are present in a phase, use an empty list.\n"
            f"Respond with ONLY the JSON object."
        )

    def build_user_prompt(self, turn: dict) -> str:
        """Build the complete user-turn tagging prompt."""
        turn_section = self._build_turn_section(turn, tag_target="user")
        tn = turn["turn_number"]

        return (
            f"You are an expert evaluator of human-AI conversations. "
            f"Tag the USER MESSAGE below for behavioral signals.\n\n"
            f"RULES:\n"
            f"1. Tag ONLY the user message — not the AI response or prior turns.\n"
            f"2. Apply BOUNDARY rules strictly. For [CALIBRATED] signals, follow the decision rules step by step.\n"
            f"3. Only tag signals you are confident are present. When in doubt, do NOT tag.\n"
            f"4. Consider the conversation context (prior turns and AI response) when interpreting user behavior.\n\n"
            f"SIGNALS:\n{self._user_taxonomy_text}\n\n"
            f"{turn_section}\n\n"
            f'Return a JSON object:\n'
            f'{{\n'
            f'  "turn_number": {tn},\n'
            f'  "signals": [\n'
            f'    {{"signal": "signal_name", "evidence": "brief quote or description"}}\n'
            f'  ],\n'
            f'  "signal_count": <integer>\n'
            f'}}\n\n'
            f"If no signals are present, return an empty list with count 0.\n"
            f"Respond with ONLY the JSON object."
        )

    # ── Naive baseline prompt ────────────────────────────────────────────

    def build_naive_ai_prompt(self, turn: dict) -> str:
        """Build a naive/minimal AI-turn tagging prompt.

        Strips out all prompt engineering: no phased approach, no gates,
        no boundaries, no calibration blocks, no counts/does_not_count.
        Just signal names + descriptions in a flat list, with a
        straightforward instruction.
        """
        turn_section = self._build_turn_section(turn, tag_target="ai")
        tn = turn["turn_number"]

        # Build flat signal list: just name + description
        signal_lines = []
        for sig_name in sorted(self.taxonomy.ai_signals):
            sig = self.taxonomy.signals[sig_name]
            valence_str = f" [{sig.valence}]" if sig.valence else ""
            signal_lines.append(f"- {sig.name}{valence_str}: {sig.description}")

        signal_text = "\n".join(signal_lines)

        return (
            f"You are tagging AI responses in human-AI conversations for behavioral signals.\n\n"
            f"Below is a taxonomy of signals. For each signal, decide whether it is present "
            f"in the AI response. Only tag signals that are clearly present.\n\n"
            f"SIGNALS:\n{signal_text}\n\n"
            f"{turn_section}\n\n"
            f'Return a JSON object with your tags:\n'
            f'{{\n'
            f'  "turn_number": {tn},\n'
            f'  "signals": [\n'
            f'    {{"signal": "signal_name", "evidence": "brief quote or description"}}\n'
            f'  ],\n'
            f'  "signal_count": <integer>\n'
            f'}}\n\n'
            f"If no signals are present, return an empty list with count 0.\n"
            f"Respond with ONLY the JSON object."
        )

    # ── Two-pass prompt building ─────────────────────────────────────────

    def build_naive_subset_prompt(self, turn: dict, signal_subset: set[str]) -> str:
        """Build a naive prompt for a SUBSET of signals (Pass 1 of two-pass).

        Like build_naive_ai_prompt but only includes signals in signal_subset.
        """
        turn_section = self._build_turn_section(turn, tag_target="ai")
        tn = turn["turn_number"]

        signal_lines = []
        for sig_name in sorted(signal_subset & set(self.taxonomy.ai_signals)):
            sig = self.taxonomy.signals[sig_name]
            valence_str = f" [{sig.valence}]" if sig.valence else ""
            signal_lines.append(f"- {sig.name}{valence_str}: {sig.description}")

        signal_text = "\n".join(signal_lines)

        return (
            f"You are tagging AI responses in human-AI conversations for behavioral signals.\n\n"
            f"Below is a taxonomy of signals. For each signal, decide whether it is present "
            f"in the AI response. Only tag signals that are clearly present.\n\n"
            f"SIGNALS:\n{signal_text}\n\n"
            f"{turn_section}\n\n"
            f'Return a JSON object with your tags:\n'
            f'{{\n'
            f'  "turn_number": {tn},\n'
            f'  "signals": [\n'
            f'    {{"signal": "signal_name", "evidence": "brief quote or description"}}\n'
            f'  ],\n'
            f'  "signal_count": <integer>\n'
            f'}}\n\n'
            f"If no signals are present, return an empty list with count 0.\n"
            f"Respond with ONLY the JSON object."
        )

    def build_calibrated_subset_prompt(self, turn: dict, signal_subset: set[str],
                                        pass1_signals: list[dict] = None) -> str:
        """Build a calibrated prompt for a SUBSET of signals (Pass 2 of two-pass).

        Uses full structured prompt with phases/gates/calibration, but only for
        signals in signal_subset. Optionally includes Pass 1 results as context.
        """
        turn_section = self._build_turn_section(turn, tag_target="ai")
        tn = turn["turn_number"]

        # Build taxonomy text for just the subset signals
        lines = []
        for sig_name in sorted(signal_subset & set(self.taxonomy.ai_signals)):
            sig = self.taxonomy.signals.get(sig_name)
            if not sig:
                continue
            lines.extend(self._format_signal(sig, max_examples=3))

        taxonomy_text = "\n".join(lines)

        # Build optional meta-calibration section
        meta_section = ""
        if self.meta_calibration_text:
            meta_section = (
                f"ANNOTATION DENSITY GUIDANCE:\n"
                f"{self.meta_calibration_text}\n\n"
            )

        # Build optional Pass 1 context section
        pass1_section = ""
        if pass1_signals is not None:
            if pass1_signals:
                sig_list = ", ".join(s["signal"] for s in pass1_signals if s.get("signal"))
                pass1_section = (
                    f"ALREADY-TAGGED SIGNALS (from an initial structural pass):\n"
                    f"  {sig_list}\n"
                    f"Use these as context for your evaluative judgments, but do not re-tag them.\n\n"
                )
            else:
                pass1_section = (
                    f"ALREADY-TAGGED SIGNALS (from an initial structural pass):\n"
                    f"  (none)\n\n"
                )

        return (
            f"You are an expert evaluator of AI responses in human-AI conversations. "
            f"Tag the AI RESPONSE below for behavioral signals.\n\n"
            f"{meta_section}"
            f"{pass1_section}"
            f"RULES:\n"
            f"1. Tag ONLY the AI response — not the user message or prior turns.\n"
            f"2. Check GATE conditions FIRST. If the gate fails, the signal CANNOT fire.\n"
            f"3. Apply BOUNDARY rules strictly. For [CALIBRATED] signals, follow the decision tree step by step.\n"
            f"4. Only tag signals you are confident are present. When in doubt, do NOT tag.\n"
            f"5. Base judgment on observable text features, not subjective impressions.\n\n"
            f"SIGNALS:\n{taxonomy_text}\n\n"
            f"{turn_section}\n\n"
            f'Return a JSON object with your tags:\n'
            f'{{\n'
            f'  "turn_number": {tn},\n'
            f'  "signals": [\n'
            f'    {{"signal": "signal_name", "evidence": "brief quote or description"}}\n'
            f'  ],\n'
            f'  "signal_count": <integer>\n'
            f'}}\n\n'
            f"If no signals are present, return an empty list with count 0.\n"
            f"Respond with ONLY the JSON object."
        )

    # ── Diagnostics ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of what this builder will produce."""
        tax = self.taxonomy
        cal = self.calibration
        n_ai = len(tax.ai_signals)
        n_user = len(tax.user_signals)
        n_cal = len(cal.blocks)
        cal_ai = sum(1 for s in cal.blocks if s in tax.ai_signals)
        cal_user = sum(1 for s in cal.blocks if s in tax.user_signals)

        return (
            f"PromptBuilder v2\n"
            f"  Taxonomy: {tax.name} v{tax.version} ({len(tax.signals)} signals)\n"
            f"  AI pass:   {n_ai} signals ({cal_ai} calibrated)\n"
            f"  User pass: {n_user} signals ({cal_user} calibrated)\n"
            f"  AI prompt:   {len(self._ai_taxonomy_text):,} chars\n"
            f"  User prompt: {len(self._user_taxonomy_text):,} chars\n"
            f"  Calibration: v{cal.version} ({n_cal} blocks)"
        )
