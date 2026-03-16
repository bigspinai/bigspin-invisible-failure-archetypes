#!/usr/bin/env python3
"""
Reproducible statistics for the capability/interaction validation experiment.

Computes every number drawn from capability_interaction_results.json that
appears anywhere in the paper (intro, Section 5 methods/findings/archetypes).
Run with:

    python paper_stats_capability_interaction.py

Requires:
    - capability_interaction_results.json  (Opus evaluation output)
    - wildchat_big_ux_tagged.csv           (main tagged dataset, for archetypes)
    - techreport_utils.py                  (signal mapping + archetype logic)

Output format:
    Each stat is annotated with [SECTION → "quoted text"] so you can trace
    every number back to the exact sentence in the paper.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import techreport_utils as tu


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

ARCHETYPE_ORDER = [
    "The confidence trap",
    "The silent mismatch",
    "The drift",
    "The death spiral",
    "The contradiction unravel",
    "The walkaway",
    "The partial recovery",
    "The mystery failure",
    "Visible failure",
]


def parse_signals(sj):
    if pd.isna(sj):
        return []
    d = json.loads(sj) if isinstance(sj, str) else sj
    sigs = sorted({re.sub(r"_\d$", "", s) for s in d.keys()})
    return sorted([tu.signal_mapping.get(s, s) for s in sigs])


def load_data():
    with open(SCRIPT_DIR / "capability_interaction_results.json") as f:
        all_results = json.load(f)
    valid = [r for r in all_results if "overall_classification" in r]
    errors = [r for r in all_results if "overall_classification" not in r]
    df = pd.read_csv(SCRIPT_DIR / "wildchat_big_ux_tagged.csv")
    df["signals"] = df["signals_json"].apply(parse_signals)
    df["archetypes"] = df["signals"].apply(tu.assign_archetypes)
    lookup = {r["conversation_id"]: r for r in valid}
    return valid, all_results, errors, df, lookup


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def primary_layer(r):
    return r["overall_classification"]["primary_layer"]

def has_cap_gap(r):
    return r["capability_layer"]["has_capability_gap"] is True

def persist_level(r):
    return r["interaction_layer"]["would_persist_with_better_model"]

def persist_yes(r):
    return persist_level(r) in ("definitely_yes", "probably_yes")

def validation_status(r):
    return r["validation"]["is_actual_failure"]

def dynamics(r):
    return r["interaction_layer"]["interaction_dynamics_present"]


# ---------------------------------------------------------------------------
# Per-archetype helper
# ---------------------------------------------------------------------------

def get_archetype_records(arch, main_df, lookup):
    cids = set()
    for _, row in main_df.iterrows():
        if arch in row["archetypes"] and row["conversation_id"] in lookup:
            cids.add(row["conversation_id"])
    return [lookup[c] for c in cids]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}\n")


def stat(label, value, where=""):
    where_str = f"  [{where}]" if where else ""
    print(f"  {label:<55s} {value}{where_str}")


def check(label, computed, cited, where):
    """Print stat and flag mismatches.

    Handles rounding: if computed is e.g. '57.6%' and cited is '57%',
    that's a match (rounds correctly). Also handles '<1%'.
    """
    def _rounds_to(computed_str, cited_str):
        # Extract numeric value from computed
        try:
            val = float(computed_str.replace("%", "").replace(",", ""))
        except ValueError:
            return computed_str == cited_str
        # Handle "<1%"
        if cited_str == "<1%":
            return val < 1.0
        # Extract cited integer
        try:
            cited_val = int(cited_str.replace("%", "").replace(",", ""))
        except ValueError:
            return computed_str == cited_str
        return round(val) == cited_val

    match = "✓" if _rounds_to(computed, cited) else f"✗ MISMATCH (paper says {cited})"
    print(f"  {label:<55s} {computed:>6s}  {match}  [{where}]")


# ---------------------------------------------------------------------------
# INTRO — numbers from the validation that appear in the introduction
# ---------------------------------------------------------------------------

def stats_intro(valid):
    section("INTRODUCTION")

    n = len(valid)

    # "91% of failures involve interaction-layer dynamics"
    # Intro says 58% + 33% = 91%, consistent with standard rounding.
    inter = sum(1 for r in valid if primary_layer(r) == "interaction")
    both = sum(1 for r in valid if primary_layer(r) == "both")
    pct = (inter + both) / n * 100
    check("interaction-layer dynamics (int+both)",
          f"{pct:.1f}%", "91%",
          'Intro → "91% of failures involve interaction-layer dynamics"')

    # "94% of these dynamics would persist"
    py = sum(1 for r in valid if persist_yes(r))
    pct = py / n * 100
    check("persist (def_yes+prob_yes)",
          f"{pct:.1f}%", "94%",
          'Intro → "94% of these dynamics would persist"')

    # "79% ... generate rather than clarify"
    grc = sum(1 for r in valid if "generate_rather_than_clarify" in dynamics(r))
    pct = grc / n * 100
    check("generate rather than clarify",
          f"{pct:.1f}%", "79%",
          'Intro → "appearing in 79% of failures"')

    # "20M" derived claim
    stat("20M claim", f"91% of 26M = {0.91 * 26:.1f}M",
         'Intro → "over 20M of those to involve persistent behavioral dynamics"')


# ---------------------------------------------------------------------------
# SECTION 5 METHODS — sample design numbers
# ---------------------------------------------------------------------------

def stats_methods(all_results, errors):
    section("SECTION 5: METHODS")

    check("Sample size", f"{len(all_results):,}", "1,044",
          'Methods → "1,044 transcripts"')
    check("Min per archetype", "100", "100",
          'Methods → "minimum of 100 transcripts per archetype"')
    check("Parse errors", str(len(errors)), "6",
          'Footnote → "Six transcripts produced malformed evaluator outputs"')
    check("Valid evaluations", f"{len(all_results) - len(errors):,}", "1,038",
          'Footnote → "yielding 1,038 valid evaluations"')


# ---------------------------------------------------------------------------
# SECTION 5: OVERALL FINDINGS
# ---------------------------------------------------------------------------

def stats_overall(valid):
    section("SECTION 5: OVERALL FINDINGS")

    n = len(valid)

    # Upstream validation
    confirmed = sum(1 for r in valid if validation_status(r) == "confirmed")
    plausible = sum(1 for r in valid if validation_status(r) == "plausible")
    validated = confirmed + plausible

    check("upstream validated (confirmed+plausible)",
          f"{validated/n*100:.1f}%", "95%",
          'Overall → "95% of cases"')
    check("  confirmed",
          f"{confirmed/n*100:.1f}%", "93%",
          'Overall → "93% confirmed"')
    check("  plausible",
          f"{plausible/n*100:.1f}%", "2%",
          'Overall → "2% plausible"')

    # Four-way classification
    layers = Counter(primary_layer(r) for r in valid)
    inter = layers.get("interaction", 0)
    both = layers.get("both", 0)
    cap = layers.get("capability", 0)

    check("primarily interaction",
          f"{inter/n*100:.1f}%", "58%",
          'Overall → "58% of failures are classified as primarily interaction"')
    check("substantially both",
          f"{both/n*100:.1f}%", "33%",
          'Overall → "33% as substantially both"')
    check("interaction component (int+both)",
          f"{(inter+both)/n*100:.1f}%", "91%",
          'Overall → "meaning 91% have an interaction component"')
    check("primarily capability",
          f"{cap/n*100:.1f}%", "7%",
          'Overall → "Only 7% are classified as primarily capability"')

    # Persistence
    p = Counter(persist_level(r) for r in valid)
    py = sum(1 for r in valid if persist_yes(r))
    check("persist (def_yes+prob_yes)",
          f"{py/n*100:.1f}%", "94%",
          'Overall → "94% of failures are judged as likely to persist"')

    # Footnote distribution
    def_yes = p.get("definitely_yes", 0)
    prob_yes = p.get("probably_yes", 0)
    uncertain = p.get("uncertain", 0)
    prob_no = p.get("probably_no", 0)
    def_no = p.get("definitely_no", 0)

    check("  definitely yes",
          f"{def_yes/n*100:.1f}%", "4%",
          'Overall footnote → "4% definitely"')
    check("  probably yes",
          f"{prob_yes/n*100:.1f}%", "90%",
          'Overall footnote → "90% probably"')
    check("  uncertain",
          f"{uncertain/n*100:.1f}%", "1%",
          'Overall footnote → "1% uncertain"')
    check("  probably no",
          f"{prob_no/n*100:.1f}%", "5%",
          'Overall footnote → "5% probably not"')
    check("  definitely no",
          f"{def_no/n*100:.1f}%", "<1%",
          'Overall footnote → "<1% definitely not"')

    # Interaction dynamics
    dyn = Counter()
    for r in valid:
        for d in dynamics(r):
            dyn[d] += 1

    check("generate_rather_than_clarify",
          f"{dyn['generate_rather_than_clarify']/n*100:.1f}%", "79%",
          'Overall → "79% of all failure transcripts"')
    check("false_confidence_masking",
          f"{dyn['false_confidence_masking']/n*100:.1f}%", "46%",
          'Overall → "false confidence masking (46%)"')
    check("ambiguity_underspecification",
          f"{dyn['ambiguity_underspecification']/n*100:.1f}%", "45%",
          'Overall → "ambiguity/underspecification (45%)"')


# ---------------------------------------------------------------------------
# SECTION 5: TABLE 1
# ---------------------------------------------------------------------------

def stats_table(main_df, lookup):
    section("SECTION 5: TABLE 1 — FOUR-WAY CLASSIFICATION BY ARCHETYPE")

    # Paper's table values (Cap%, Int%, Both%, Unclear%, Persist%)
    # for verification
    TABLE_CITED = {
        "The confidence trap":       (140,  5, 19, 76, 0, 97),
        "The silent mismatch":       (124,  4, 69, 27, 0, 97),
        "The drift":                 (229,  4, 67, 28, 1, 97),
        "The death spiral":          (144,  3, 63, 33, 1, 96),
        "The contradiction unravel": (141, 15, 45, 40, 1, 90),
        "The walkaway":              (169,  1, 79, 17, 3, 97),
        "The partial recovery":      (160,  5, 54, 39, 2, 96),
        "The mystery failure":       (117,  5, 74, 15, 6, 90),
        "Visible failure":           (132, 20, 27, 53, 0, 89),
    }

    header = (f"  {'Archetype':<27s} {'n':>4s}  "
              f"{'Cap':>5s} {'Int':>5s} {'Both':>5s} {'Unc':>5s}  "
              f"{'Sum':>5s}  {'Per':>5s}")
    print(header)
    print(f"  {'-' * 72}")

    all_ok = True
    for arch in ARCHETYPE_ORDER:
        matched = get_archetype_records(arch, main_df, lookup)
        n = len(matched)
        if n == 0:
            continue

        layers = Counter(primary_layer(r) for r in matched)
        cap = layers.get("capability", 0)
        inter = layers.get("interaction", 0)
        both = layers.get("both", 0)
        unc = layers.get("unclear", 0)
        persist = sum(1 for r in matched if persist_yes(r))
        row_sum = cap + inter + both + unc

        cap_p = round(cap / n * 100)
        int_p = round(inter / n * 100)
        both_p = round(both / n * 100)
        unc_p = round(unc / n * 100)
        per_p = round(persist / n * 100)

        cited = TABLE_CITED[arch]
        mismatches = []
        if n != cited[0]:
            mismatches.append(f"n={cited[0]}")
        if cap_p != cited[1]:
            mismatches.append(f"Cap={cited[1]}")
        if int_p != cited[2]:
            mismatches.append(f"Int={cited[2]}")
        if both_p != cited[3]:
            mismatches.append(f"Both={cited[3]}")
        if unc_p != cited[4]:
            mismatches.append(f"Unc={cited[4]}")
        if per_p != cited[5]:
            mismatches.append(f"Per={cited[5]}")

        flag = " ✓" if not mismatches else f" ✗ paper says {', '.join(mismatches)}"
        if mismatches:
            all_ok = False

        print(f"  {arch:<27s} {n:4d}  "
              f"{cap_p:4d}% {int_p:4d}% {both_p:4d}% {unc_p:4d}%  "
              f"{round(row_sum/n*100):4d}%  {per_p:4d}%{flag}")

    if all_ok:
        print("\n  All table values match. ✓")
    else:
        print("\n  ✗ TABLE MISMATCHES FOUND — see above.")


# ---------------------------------------------------------------------------
# SECTION 5: PER-ARCHETYPE PROSE CLAIMS
# ---------------------------------------------------------------------------

def stats_archetype_prose(main_df, lookup):
    section("SECTION 5: PER-ARCHETYPE PROSE CLAIMS")

    # --- Confidence trap ---
    ct = get_archetype_records("The confidence trap", main_df, lookup)
    n = len(ct)

    check("CT: 'both' classification",
          f"{sum(1 for r in ct if primary_layer(r) == 'both')/n*100:.0f}%", "76%",
          'CT → "dominated by the both classification (76%)"')
    check("CT: has_capability_gap",
          f"{sum(1 for r in ct if has_cap_gap(r))/n*100:.0f}%", "81%",
          'CT → "81% of confidence trap cases involve a genuine capability gap"')
    check("CT: primarily capability",
          f"{sum(1 for r in ct if primary_layer(r) == 'capability')/n*100:.0f}%", "5%",
          'CT → "only 5% are primarily capability"')
    check("CT: false_confidence_masking",
          f"{sum(1 for r in ct if 'false_confidence_masking' in dynamics(r))/n*100:.0f}%", "97%",
          'CT → "false confidence masking (97%)"')

    # --- Drift ---
    dr = get_archetype_records("The drift", main_df, lookup)
    n = len(dr)

    check("Drift: interaction-primary",
          f"{sum(1 for r in dr if primary_layer(r) == 'interaction')/n*100:.0f}%", "67%",
          'Drift → "67% interaction-primary"')
    check("Drift: persist",
          f"{sum(1 for r in dr if persist_yes(r))/n*100:.0f}%", "97%",
          'Drift → "97% persistence"')

    # --- Silent mismatch ---
    sm = get_archetype_records("The silent mismatch", main_df, lookup)
    n = len(sm)

    check("SM: interaction-primary",
          f"{sum(1 for r in sm if primary_layer(r) == 'interaction')/n*100:.0f}%", "69%",
          'SM → "69% interaction-primary"')
    check("SM: persist",
          f"{sum(1 for r in sm if persist_yes(r))/n*100:.0f}%", "97%",
          'SM → "97% persistence"')

    # --- Walkaway ---
    wa = get_archetype_records("The walkaway", main_df, lookup)
    n = len(wa)

    check("WA: primarily interaction",
          f"{sum(1 for r in wa if primary_layer(r) == 'interaction')/n*100:.0f}%", "79%",
          'WA → "79% primarily interaction"')
    check("WA: primarily capability",
          f"{sum(1 for r in wa if primary_layer(r) == 'capability')/n*100:.0f}%", "1%",
          'WA → "only 1% primarily capability"')
    check("WA: persist",
          f"{sum(1 for r in wa if persist_yes(r))/n*100:.0f}%", "97%",
          'WA → "97% persistent"')

    # --- Contradiction unravel ---
    cu = get_archetype_records("The contradiction unravel", main_df, lookup)
    n = len(cu)

    check("CU: primarily capability",
          f"{sum(1 for r in cu if primary_layer(r) == 'capability')/n*100:.0f}%", "15%",
          'CU → "15% primarily capability"')
    check("CU: persist",
          f"{sum(1 for r in cu if persist_yes(r))/n*100:.0f}%", "90%",
          'CU → "lowest persistence rate (90%)"')

    # --- Visible failure ---
    vf = get_archetype_records("Visible failure", main_df, lookup)
    n = len(vf)

    check("VF: primarily capability",
          f"{sum(1 for r in vf if primary_layer(r) == 'capability')/n*100:.0f}%", "20%",
          'VF → "20%"')

    # --- "1-5% for most invisible archetypes" range ---
    print("\n  --- Primarily-capability range across invisible archetypes ---")
    invisible = [a for a in ARCHETYPE_ORDER if a != "Visible failure"]
    for arch in invisible:
        recs = get_archetype_records(arch, main_df, lookup)
        n = len(recs)
        cap = sum(1 for r in recs if primary_layer(r) == "capability")
        print(f"    {arch:<27s} {cap}/{n} = {cap/n*100:.1f}%")
    print('  [Paper cites "1-5% for most invisible archetypes other than '
          'contradiction unravel"]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    valid, all_results, errors, main_df, lookup = load_data()
    print(f"  Total results: {len(all_results)}")
    print(f"  Parse errors:  {len(errors)}")
    print(f"  Valid:         {len(valid)}")
    print(f"  Main dataset:  {len(main_df):,} rows")

    stats_intro(valid)
    stats_methods(all_results, errors)
    stats_overall(valid)
    stats_table(main_df, lookup)
    stats_archetype_prose(main_df, lookup)

    section("DONE")


if __name__ == "__main__":
    main()
