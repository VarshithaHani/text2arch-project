"""Re-score saved model outputs with different post-processing (no Ollama calls).
Usage:  python rescore.py --run results\\run_*.json
"""
import argparse, csv, glob, json, re

from evaluate import evaluate_prediction, parse_dot
from metrics import macro_average
from run_evaluation import extract_dot, postprocess, clean_description, find_dataset

csv.field_size_limit(10 ** 8)
MODES = ("exact", "normalized", "fuzzy")
SUFFIX = re.compile(r"\s+(module|block)$", re.IGNORECASE)


def _q(x):
    return '"' + x.replace('"', '\\"') + '"'


def strip_suffix(dot):
    g = parse_dot(dot)
    edges, seen = [], set()
    for s, t in sorted(g["edges"]):
        s2, t2 = SUFFIX.sub("", s), SUFFIX.sub("", t)
        if s2 != t2 and (s2, t2) not in seen:
            seen.add((s2, t2))
            edges.append((s2, t2))
    return "\n".join(["digraph G {"] + [f"  {_q(s)} -> {_q(t)}" for s, t in edges] + ["}"])


VARIANTS = {
    "raw model output": lambda d, desc: d,
    "safe postprocess (acronyms + markdown only)": lambda d, desc: postprocess(d, desc),
    "safe + drop-ungrounded": lambda d, desc: postprocess(d, desc, True),
    "safe + fuzzy snap 0.9": lambda d, desc: postprocess(d, desc, False, 0.9, True),
    "safe + strip Module/Block": lambda d, desc: postprocess(strip_suffix(d), desc),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    dataset, tried = find_dataset(args.dataset)
    if not dataset:
        print("manual.tsv not found; pass --dataset path")
        return 1
    with open(dataset, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    records = {}
    for pattern in args.run:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for rec in json.load(f)["samples"]:
                    records[rec["sample"]] = rec      # later runs override earlier ones
    print(f"Re-scoring {len(records)} saved samples\n")

    for name, fn in VARIANTS.items():
        scores = {m: {"node": [], "edge": []} for m in MODES}
        zero_edges = 0
        for num, rec in sorted(records.items()):
            row = rows[num - 1]
            desc = clean_description(row["Cleaned Description"])
            try:
                dot, _ = extract_dot(rec["raw_output"])
                dot = fn(dot, desc)
            except Exception:
                continue
            for m in MODES:
                r = evaluate_prediction(row["Dot code"], dot, m)
                if not r["valid"]:
                    continue
                scores[m]["node"].append(r["node_metrics"])
                scores[m]["edge"].append(r["edge_metrics"])
                if m == "normalized" and r["edge_metrics"]["tp"] == 0:
                    zero_edges += 1
        print(f"=== {name} ===")
        for m in MODES:
            n, e = macro_average(scores[m]["node"]), macro_average(scores[m]["edge"])
            print(f"  {m:<11} node F1={n['f1']*100:5.1f}  edge F1={e['f1']*100:5.1f}  "
                  f"(edge P={e['precision']*100:5.1f} R={e['recall']*100:5.1f})")
        print(f"  samples with 0 correct edges (normalized): {zero_edges}/{len(scores['normalized']['edge'])}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())