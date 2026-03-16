"""
Copyright 2026 Bigspin AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
from matplotlib import pyplot as plt
import numpy as np
import re


signal_mapping = {
    # --- AI ERRORS ---
    'ai_calculation_error':             'ai_error_factual',
    'ai_conceptual_error':              'ai_error_factual',
    'ai_error':                         'ai_error_factual',
    'ai_error_windows_filenames':       'ai_error_factual',
    'ai_hallucination':                 'ai_error_factual',
    'ai_incorrect_diagnosis':           'ai_error_factual',
    'ai_misdiagnosis':                  'ai_error_factual',
    'ai_provides_incorrect_solution':   'ai_error_factual',
    'potential_error':                  'ai_error_factual',

    'ai_false_confidence':              'ai_false_confidence',
    'ai_hallucination_as_false_confidence': 'ai_false_confidence',
    'false_confidence':                 'ai_false_confidence',
    'false_confidence_colonial':        'ai_false_confidence',

    'ai_fundamental_misunderstanding':  'ai_misunderstanding',
    'ai_misinterpretation':             'ai_misunderstanding',
    'ai_misses_existing_solution':      'ai_misunderstanding',
    'ai_misunderstanding':              'ai_misunderstanding',
    'ai_misunderstood_question':        'ai_misunderstanding',
    'ai_misuse_of_term':                'ai_misunderstanding',
    'ai_response_not_targeted':         'ai_misunderstanding',
    'initial_misunderstanding':         'ai_misunderstanding',
    'misunderstanding':                 'ai_misunderstanding',
    'scope_misunderstanding':           'ai_misunderstanding',
    'confusion':                        'ai_misunderstanding',

    'ai_malfunction':                   'ai_malfunction',

    # --- AI RESPONSE QUALITY ---
    'ai_incomplete_excel_answer':       'ai_incomplete_response',
    'incomplete_guidance':              'ai_incomplete_response',
    'incomplete_output':                'ai_incomplete_response',
    'incomplete_response':              'ai_incomplete_response',

    'excessive_brevity':                'ai_poor_response_quality',
    'excessive_verbosity':              'ai_poor_response_quality',
    'excessive_hedging':                'ai_poor_response_quality',
    'invalid_formatting':               'ai_poor_response_quality',
    'minimal_improvement':              'ai_poor_response_quality',

    'ai_off_topic_drift':               'ai_off_topic_drift',
    'off_topic_drift':                  'ai_off_topic_drift',
    'off_topic_drift_second':           'ai_off_topic_drift',

    'ai_self_contradiction':            'ai_self_contradiction',
    'self_contradiction':               'ai_self_contradiction',

    'ai_self_correction':               'ai_self_correction',
    'ai_self_correction_second':        'ai_self_correction',
    'self_correction':                  'ai_self_correction',

    # --- AI REFUSALS ---
    'ai_false_rejection':               'ai_explicit_refusal',
    'ai_ignores_instruction':           'ai_implicit_refusal',
    'ai_implicit_refusal':              'ai_implicit_refusal',
    'ai_refusal':                       'ai_explicit_refusal',

    # --- USER REQUESTS & CLARIFICATION ---
    'ambiguity_in_request':             'ambiguous_request',
    'ambiguous_intent':                 'ambiguous_request',
    'ambiguous_request':                'ambiguous_request',

    'clarification_request':            'user_clarification',
    'user_clarification':               'user_clarification',
    'user_clarification_after_initial_refusal': 'user_clarification',
    'user_clarification_request':       'user_clarification',

    'user_correction':                  'explicit_user_correction',
    'user_correction_second':           'explicit_user_correction',
    'implicit_user_correction':         'implicit_user_correction',

    'escalation_in_asks':               'user_scope_change',
    'escalation_request':               'user_scope_change',
    'scope_expansion':                  'ai_misunderstanding',
    'scope_narrowing':                  'ai_misunderstanding',

    # --- USER SENTIMENT ---
    'explicit_frustration':             'user_frustration',
    'frustration':                      'user_frustration',
    'user_frustration':                 'user_frustration',

    'implicit_dissatisfaction':         'implicit_user_dissatisfaction',
    'negative_sentiment_shift':         'explicit_user_dissatisfaction',

    'user_passive_acceptance':          'user_passive_acceptance',
    'validation_seeking':               'user_validation_seeking',

    # --- CONVERSATION DYNAMICS ---
    'circular_conversation':            'repetition',
    'repetition':                       'repetition',
    'repetition_ai':                    'repetition',
    'user_repetition':                  'repetition',    

    'ethical_tension':                  'ethical_tension',

    # --- OUTCOMES ---
    'abandonment':                      'user_abandonment',
    'goal_failure':                     'goal_failure',
    'goal_failure_colonial':            'goal_failure',

    'implicit_task_mismatch':           'goal_failure',

    'partial_success':                  'partial_success',
    'recovery':                         'recovery',

    'implicit_success':                 'task_success',
    'implicit_task_completion':         'task_success',

    # --- INVALID ---
    'invalid_input':                    'invalid_input'
}


def get_signals(d):
    sigs = list(d.keys())
    sigs = sorted({re.sub(r"_\d$", "", s) for s in sigs})
    sigs = sorted([signal_mapping.get(s, s) for s in sigs])
    return sigs


visible_signals = {
    'ai_explicit_refusal',
    'ai_malfunction',
    'explicit_user_correction',
    'explicit_user_dissatisfaction',
    'user_scope_change',
    'user_clarification',
    'user_frustration',
    'user_validation_seeking'
}


archetype_rules = {
    'The confidence trap': lambda sigs: 'ai_false_confidence' in sigs,
    'The silent mismatch': lambda sigs: {'ai_misunderstanding', 'ai_implicit_refusal', 'ai_error_factual'} & sigs,
    'The drift': lambda sigs: 'ai_off_topic_drift' in sigs,
    'The death spiral': lambda sigs: 'repetition' in sigs,
    'The contradiction unravel': lambda sigs: 'ai_self_contradiction' in sigs,
    'The walkaway': lambda sigs: 'user_abandonment' in sigs,
    'The mystery failure': lambda sigs: sigs == {'goal_failure'},
    'The partial recovery': lambda sigs: {'recovery', 'ai_self_correction', 'task_success', 'partial_success'} & sigs,
}

def assign_archetypes(sigs):
    if 'goal_failure' not in sigs:
        return ["No failure detected"]
    if  set(sigs) & visible_signals:
        return ['Visible failure']
    a = []
    for n, f in archetype_rules.items():
        if f(set(sigs)):
            a.append(n)    
    return sorted(set(a))


def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = df / expected
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df