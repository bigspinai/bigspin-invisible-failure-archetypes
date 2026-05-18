import gzip
import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DATA_DIRNAME = "data"
FIGURE_OUTPUT_DIRNAME = "fig"

def data_loader(filename, filename_column='filename'):
    with gzip.open(filename, mode="rt", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    df = pd.DataFrame(data)
    df[filename_column] = os.path.basename(filename)
    return df


def get_archetype_tags(vals):
    """Extract archetype names and format each one nicely."""
    if pd.isnull(vals) is True:
        return []    
    names = [d["archetype"] for d in vals]
    names = [n.replace("_", " ").capitalize() for n in names]
    names = ["The mystery failure" if n == "None" else n for n in names]
    return names

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


def prepare_agreement_series(df, colname, name):   
    df = df.copy()
    df.set_index("conversation_id", inplace=True, drop=True)
    if colname == "archetype_tags":
        df["archetype_tags"] = df.archetypes.apply(get_archetype_tags)    
    df = df[df[colname].isnull() == False]
    series = df[colname]        
    series.name = name
    return series


def align_agreement_series(series1, series2):
    merged = pd.merge(
        series1.to_frame(), 
        series2.to_frame(), 
        how="inner", 
        left_index=True, 
        right_index=True)
    return merged[series1.name], merged[series2.name]


def failure_agreement_report(condition="C", colname="failure_visibility", to_latex=True):    
    dirname = os.path.join(DATA_DIRNAME, "interannotator")

    opus = data_loader(f"{dirname}/score_10k_quality_{condition}_opus.jsonl.gz")    
    opus = prepare_agreement_series(opus, colname, "Opus 4.6")

    gpt = data_loader(f"{dirname}/score_10k_quality_{condition}_gpt5.jsonl.gz")
    gpt = prepare_agreement_series(gpt, colname, "GPT-5.2")
        
    labels = set(opus.unique()) | set(gpt.unique())

    opus, gpt = align_agreement_series(opus, gpt)
    
    assert opus.index.identical(gpt.index)

    all_opus = []
    all_gpt = []
    report = []
    for label in sorted(labels):
        o = label == opus
        all_opus.append(o)
        g = label == gpt  
        all_gpt.append(g)
        k = cohen_kappa_score(o, g)
        agr = (o == g).value_counts()
        a = agr[True] / agr.sum()
        d = {"label": label, "kappa": k, "Agreement": a}
        report.append(d)

    macro_k = np.mean([d['kappa'] for d in report])
    macro_a = np.mean([d['Agreement'] for d in report])
    
    report.append({"label": "Macro", "kappa": macro_k, "Agreement": macro_a})

    all_opus = pd.concat(all_opus)    
    all_gpt = pd.concat(all_gpt)
    kappa = cohen_kappa_score(all_opus, all_gpt)
    agr = (all_opus == all_gpt).value_counts()
    agreement = agr[True] / agr.sum()    

    report.append({"label": "Overall", "kappa": kappa, "Agreement": agreement})
    
    report = pd.DataFrame(report)
    if to_latex:
        print(report.to_latex(index=False, float_format="%.2f"))
    else:
        return report

        
def archetype_agreement_report(opus, gpt, colname="archetype_tags", to_latex=True):

    opus = prepare_agreement_series(opus, colname, "Opus 4.6")
    gpt = prepare_agreement_series(gpt, colname, "GPT-5.2")    
    opus, gpt = align_agreement_series(opus, gpt)
    
    assert opus.index.identical(gpt.index)

    labels = {x for vals in opus for x in vals} | {x for vals in gpt for x in vals}
        
    all_opus = []
    all_gpt = []
    report = []
    for label in sorted(labels):
        o = opus.apply(lambda x: label in x)
        all_opus.append(o)
        g = gpt.apply(lambda x: label in x)
        all_gpt.append(g)
        k = cohen_kappa_score(o, g)
        agr = (o == g).value_counts()
        a = agr[True] / agr.sum()
        d = {"label": label, "kappa": k, "Agreement": a}
        report.append(d)

    macro_k = np.mean([d['kappa'] for d in report])
    macro_a = np.mean([d['Agreement'] for d in report])
    
    report.append({"label": "Macro", "kappa": macro_k, "Agreement": macro_a})

    all_opus = pd.concat(all_opus)    
    all_gpt = pd.concat(all_gpt)
    kappa = cohen_kappa_score(all_opus, all_gpt)
    agr = (all_opus == all_gpt).value_counts()
    agreement = agr[True] / agr.sum()    

    report.append({"label": "Micro", "kappa": kappa, "Agreement": agreement})
    
    report = pd.DataFrame(report)
    if to_latex:
        print(report.to_latex(index=False, float_format="%.2f"))
    else:
        return report    