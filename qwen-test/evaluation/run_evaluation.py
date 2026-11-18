"""
Text2Arch evaluation with a local Ollama model (default: qwen3:8b).

Works with any Ollama chat model; prompt style and thinking switches are chosen
automatically (qwen3:8b -> "flow" prompt + thinking off, qwen2.5:1.5b -> "small" prompt).

Examples
    python run_evaluation.py                      # first 5 samples, qwen3:8b
    python run_evaluation.py --model qwen2.5:1.5b-instruct
    python run_evaluation.py --samples 20
    python run_evaluation.py --all                # all rows of manual.tsv
    python run_evaluation.py --dataset D:\\path\\manual.tsv --prompt strict

Output: console report + results/run_<time>.json + results/run_<time>.csv
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

from difflib import SequenceMatcher
from evaluate import evaluate_prediction, extract_ground_truth, parse_dot, repair_dot, normalize_name
from metrics import macro_average, micro_average

# Windows consoles choke on characters like 'ΔH' / 'ŷ' -> never crash on print
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

csv.field_size_limit(10 ** 8)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
MODES = ("exact", "normalized", "fuzzy")

# ------------------------------------------------------------------
# prompts
# ------------------------------------------------------------------
PROMPTS = {
    # Closest to your original prompt (only explicitly stated edges)
    "strict": """You are a Graphviz DOT code generator for software architecture diagrams.
Convert the architecture description into a directed Graphviz DOT graph.

Rules:
1. Output ONLY valid Graphviz DOT code, starting with `digraph {` and ending with `}`.
2. No explanations and no Markdown code fences.
3. Create nodes only for components explicitly mentioned.
4. Create an edge ONLY when that relationship is explicitly stated. Never invent relationships.
5. Preserve the direction of every relationship. No duplicate edges.
6. Put every node name in double quotes. One statement per line. No attributes.""",

    # Better match for manual.tsv: descriptions narrate a *flow*, ground truth contains every flow edge
    "flow": """You convert a textual description of an architecture diagram into Graphviz DOT.

Rules:
1. Output ONLY DOT code, starting with `digraph {` and ending with `}`. No explanations, no Markdown fences.
2. Make one node per component / module / input / output / data item named in the description.
   Use short names exactly as written in the description (e.g. "Flow Estimator", "I_t").
   Do not add words such as "Module" or "Block" and do not invent components.
3. Make a directed edge A -> B for every data flow, input/output or dependency that the description
   states, including the narrated flow of information ("X is fed into Y", "Y produces Z",
   "A and B are combined by C" gives A -> C and B -> C). Direction = direction of data flow.
4. Do not connect components merely because they are in the same group. No duplicate edges.
5. Put every node name in double quotes. One statement per line. No attributes, edge labels or clusters.

Example output:
digraph {
  "Input Image" -> "Encoder"
  "Encoder" -> "Decoder"
  "Decoder" -> "Output Image"
}""",
    # flow + exact-wording rules + fan-in/fan-out example
    
    "flow2": """You convert a textual description of an architecture diagram into Graphviz DOT.

Rules:
1. Output ONLY valid DOT code, starting with digraph { and ending with }.
2. Represent every distinct component, module, layer, input, output, operation and data item mentioned in the description as a node.
3. Preserve technical component details from the description. For example, if Feature Extraction uses Improved Local Binary Patterns (ILBP), name the node "Feature Extraction (ILBP)". If a Classifier uses an Evaluated Kullback-Leibler method, name it "Classifier (Evaluated Kullback-Leibler)". Do not discard technical methods, algorithms or distinguishing details when naming nodes.
4. Include all inputs and outputs as separate nodes.
5. Create directed edges according to the described data flow. If multiple inputs enter one component, connect every input to that component.
6. Track every intermediate data item between components. If extracted features are stored in a database and later passed to a classifier, include the database as a node and connect it to the classifier.
7. Preserve all branches, parallel paths, sequential connections, skip connections, feedback connections and merging operations explicitly mentioned.
8. If a component produces multiple outputs, create an edge to each output.
9. If multiple components contribute to one operation, connect each contributing component to that operation.
10. For repeated or numbered components, create separate nodes when the description distinguishes them.
11. Do not omit components simply because they appear inside a repeated block or group.
12. Do not invent components or connections that are not supported by the description.
13. Do not connect components merely because they are mentioned together.
14. Do not create nodes for section headings or decorative group names.
15. Keep useful details in parentheses and preserve dimensions and identifiers when they distinguish components.
16. Put every node name in double quotes. Use one edge per line. Do not use attributes, edge labels, subgraphs or clusters.
17. Do not create isolated nodes unless they are explicitly described as independent inputs, outputs or components.

Example description:
Input Image is passed to Encoder. Encoder produces Features. Features are passed to Decoder, which generates Output Image.

Example output:
digraph {
  "Input Image" -> "Encoder"
  "Encoder" -> "Features"
  "Features" -> "Decoder"
  "Decoder" -> "Output Image"
}""",
"qwen4b": """You are an architecture diagram generator.
Convert the given description into a directed Graphviz DOT graph.

OUTPUT:
- Return only valid DOT code.
- Start with digraph G {
- End with }
- Use double quotes around every node name.
- Write one directed edge per line.
- Do not include explanations, Markdown, labels, attributes, or clusters.

NODE RULES:
1. Include every explicitly mentioned component, module, layer,
   input, output, operation, and intermediate data item.
2. Preserve technical names, abbreviations, algorithms,
   dimensions, and identifiers.
3. Use the component names from the description.
4. Do not invent components.
5. Do not create nodes for headings or decorative groups.

EDGE RULES:
1. Follow the direction of data flow.
2. Connect components only when a relationship is described.
3. Connect every input to the operation that uses it.
4. Connect every output produced by a component.
5. Preserve sequential, parallel, branching, merging,
   feedback, and skip connections.
6. Include intermediate data items when they are explicitly
   described as being passed between components.
7. Do not create duplicate edges.
8. Do not connect components just because they are mentioned
   together.

EXAMPLE:

Description:
The Input Image is passed to the Encoder.
The Encoder extracts Features.
The Features are processed by the Decoder.
The Decoder generates the Output Image.

Output:
digraph G {
    "Input Image" -> "Encoder"
    "Encoder" -> "Features"
    "Features" -> "Decoder"
    "Decoder" -> "Output Image"
}

Now convert the architecture description into DOT.
""",
    # Short prompt + one worked example: what 0.5B-3B models follow reliably
    "small": """Convert the description into Graphviz DOT.

Output format (nothing else - no explanations, no code fences):
digraph G {
  "Component A" -> "Component B"
}

Rules:
- Every node name is in double quotes and copied from the description.
- One edge per line: "Source" -> "Target". The arrow follows the data flow.
- Use only components named in the description. No attributes, no labels, no clusters.
- If A and B both go into C, write "A" -> "C" and "B" -> "C".

Example
Description: The Input Image is processed by the Encoder. The Encoder output is passed to the Decoder, which produces the Output Image.
Output:
digraph G {
  "Input Image" -> "Encoder"
  "Encoder" -> "Decoder"
  "Decoder" -> "Output Image"
}""",
}

SMALL_MODEL = re.compile(r"[:\\\-/ ](0\.5|1\.5|1|2|3)b(?![a-z0-9.])", re.I)


def resolve_prompt(args):
    if args.prompt != "auto":
        return args.prompt
    return "small" if SMALL_MODEL.search(args.model) else "flow"


def thinking_switch(args):
    """True / False = send think flag, None = model has no thinking mode (send nothing)."""
    if args.think == "on":
        return True
    if args.think == "off":
        return False
    name = args.model.lower()
    return False if ("qwen3" in name or "deepseek-r1" in name) else None


# ------------------------------------------------------------------
# Ollama helpers
# ------------------------------------------------------------------
def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def installed_models():
    tags = http_json(f"{OLLAMA_HOST}/api/tags", timeout=10)
    return [m.get("name", "") for m in tags.get("models", [])]


def model_available(wanted, installed):
    names = set(installed)
    if wanted in names:
        return True
    return ":" not in wanted and f"{wanted}:latest" in names


def clean_description(text):
    # manual.tsv stores line breaks as '|'
    return text.replace("||", "\n\n").replace("|", "\n").strip()


def extract_dot(raw):
    """Turn raw model text into clean DOT. Returns (dot, notes)."""
    notes = []
    text = raw or ""

    # Qwen3 reasoning block
    if "<think>" in text and "</think>" not in text:
        raise ValueError("Output is only an unfinished <think> block (raise --max-tokens or disable thinking).")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    fence = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        text = text.replace("```dot", "").replace("```graphviz", "").replace("```", "")

    pos = text.lower().find("digraph")
    if pos == -1:
        raise ValueError("No 'digraph' found in model output.")
    text = text[pos:]

    last = text.rfind("}")
    if last != -1:
        text = text[: last + 1]
    if text.count("{") > text.count("}"):
        text += "\n}" * (text.count("{") - text.count("}"))
        notes.append("added missing closing brace (output was probably cut off)")

    # quote names with spaces, fix arrows, one statement per line
    fixed = repair_dot(text.strip())
    if fixed.split() != text.split():
        notes.append("normalised DOT (quoted names / arrows / layout)")
    return fixed, notes


def chat(payload, args):
    """One Ollama /api/chat call. Drops the 'think' flag if the model rejects it."""
    for _ in range(2):
        try:
            return http_json(f"{OLLAMA_HOST}/api/chat", payload, timeout=args.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 400 and "think" in body.lower() and "think" in payload:
                payload.pop("think")          # e.g. qwen2.5 does not know the flag
                continue
            raise RuntimeError(f"Ollama HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {OLLAMA_HOST}: {e}")
    raise RuntimeError("Ollama request failed")


def generate_dot(description, args):
    system_prompt = PROMPTS[resolve_prompt(args)]
    think = thinking_switch(args)
    base_msg = (
        "Architecture description:\n\n"
        f"{clean_description(description)}\n\n"
        "Output only the DOT code."
    )
    if think is False and "qwen3" in args.model.lower():
        base_msg += " /no_think"      # Qwen3 soft switch; never sent to other models

    last_error = None
    for attempt in range(args.retries + 1):
        user_msg = base_msg
        if attempt:                   # retry: slightly warmer + explicit reminder
            user_msg += ('\n\nYour previous answer was not valid. Reply with ONLY a DOT graph that '
                         'starts with "digraph G {" and has one "A" -> "B" edge per line.')
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.2 if attempt else 0, "seed": 0,
                        "num_predict": args.max_tokens, "num_ctx": args.num_ctx},
        }
        if think is not None:
            payload["think"] = think

        result = chat(payload, args)
        message = result.get("message", {})
        content = (message.get("content") or "").strip()
        meta = {
            "done_reason": result.get("done_reason"),
            "eval_count": result.get("eval_count"),
            "had_thinking_field": bool(message.get("thinking")),
            "attempts": attempt + 1,
        }
        try:
            if not content:
                why = ("the model spent all tokens on 'thinking' (use --think off or raise --max-tokens)"
                       if meta["had_thinking_field"] else "empty response")
                raise ValueError(f"Ollama returned empty content: {why}.")
            dot, notes = extract_dot(content)
            edges = parse_dot(dot)["edges"]
            if not edges:
                raise ValueError("generated DOT contains no edges.")
        except ValueError as e:
            last_error = e
            continue
        if attempt:
            notes.append(f"needed {attempt + 1} attempts")
        if meta["done_reason"] == "length":
            notes.append("hit the token limit - raise --max-tokens")
        return content, dot, notes, meta
    raise ValueError(f"{last_error} (after {args.retries + 1} attempts)")


def dot_renders(dot):
    """True/False if Graphviz `dot` is installed, else None."""
    exe = shutil.which("dot")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "-Tcanon"], input=dot.encode("utf-8"),
                              capture_output=True, timeout=20)
        return proc.returncode == 0
    except Exception:
        return False
def _flat(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", normalize_name(s))).strip()


def acronym_map(description):
    """'Sparse Point Prediction Network (SPPN)' -> {'sppn': 'Sparse Point Prediction Network'}"""
    acro = {}
    for m in re.finditer(r"\(([A-Za-z]{2,10})\)", description):
        abbr = m.group(1)
        before = re.findall(r"[\w\-]+", description[:m.start()])
        k = len(abbr)
        if len(before) >= k:
            cand = before[-k:]
            if "".join(w[0] for w in cand).lower() == abbr.lower():
                acro[abbr.lower()] = " ".join(cand)
    return acro

def postprocess(dot, description, drop_ungrounded=False, threshold=0.9, fuzzy=False):
    graph = parse_dot(dot)
    acro = acronym_map(description)
    text = description.replace("*", " ")
    desc_flat = " " + _flat(text) + " "

    grams = {}
    if fuzzy:
        words = [w.strip(".,;:-") for w in re.findall(r"[\w.\-]+", text)]
        words = [w for w in words if w]
        for n in range(1, 6):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i:i + n])
                grams.setdefault(_flat(g), g)

    mapping = {}
    for node in graph["nodes"]:
        name = node.replace("*", "").strip().strip('"')
        flat = _flat(name)
        if not flat:
            mapping[node] = None
        elif name.lower() in acro:                    # SPPN -> Sparse Point Prediction Network
            mapping[node] = acro[name.lower()]
        elif f" {flat} " in desc_flat:                # phrase is in the text: keep the model's name
            mapping[node] = name
        elif fuzzy:
            best, best_score = None, 0.0
            for gflat, g in grams.items():
                if abs(len(gflat.split()) - len(flat.split())) > 1:
                    continue
                s = SequenceMatcher(None, flat, gflat).ratio()
                if s > best_score:
                    best, best_score = g, s
            mapping[node] = best if best_score >= threshold else (None if drop_ungrounded else name)
        else:
            mapping[node] = None if drop_ungrounded else name

    edges, seen = [], set()
    for s, t in sorted(graph["edges"]):
        s2, t2 = mapping.get(s), mapping.get(t)
        if not s2 or not t2 or s2 == t2 or (s2, t2) in seen:
            continue
        seen.add((s2, t2))
        edges.append((s2, t2))

    q = lambda x: '"' + x.replace('"', '\\"') + '"'
    return "\n".join(["digraph G {"] + [f"  {q(s)} -> {q(t)}" for s, t in edges] + ["}"]) 
# ------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------
def find_dataset(cli_path):
    candidates = [cli_path, os.environ.get("TEXT2ARCH_DATASET")]
    for base in (SCRIPT_DIR, os.path.dirname(SCRIPT_DIR), os.getcwd()):
        candidates += [os.path.join(base, "manual.tsv"), os.path.join(base, "data", "manual.tsv")]
    tried = []
    for c in candidates:
        if c:
            tried.append(c)
            if os.path.isfile(c):
                return c, tried
    return None, tried


# ------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------
def fmt_row(label, m):
    return (f"  {label:<14} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  Jaccard={m['jaccard']:.3f}")


def print_metrics(title, m):
    print(f"\n--- {title} ---")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1 Score  : {m['f1']:.4f}")
    print(f"Jaccard   : {m['jaccard']:.4f}")
    print(f"(TP={m['tp']}  FP={m['fp']}  FN={m['fn']})")


def summarise(records):
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for level in ("node", "edge"):
            ms = [r["metrics"][mode][level] for r in records]
            summary[mode][level] = {"macro": macro_average(ms), "micro": micro_average(ms)}
    return summary


def print_summary(records, summary, attempted, skipped):
    line = "=" * 60
    print(f"\n{line}\nFINAL RESULTS\n{line}")
    print(f"Evaluated {len(records)} / {attempted} samples ({len(skipped)} failed or skipped)")
    for s in skipped:
        print(f"   sample {s['sample']}: {s['reason']}")

    if not records:
        print("\nNo samples were successfully evaluated - see the reasons above.")
        return

    for mode in MODES:
        tag = {"exact": "EXACT names", "normalized": "NORMALIZED names (case/_/- ignored)  <- main score",
               "fuzzy": "FUZZY names (similarity >= 0.75)"}[mode]
        print(f"\n[{tag}]")
        for level in ("node", "edge"):
            print(f" {level.upper()}S")
            print(fmt_row("macro average", summary[mode][level]["macro"]))
            print(fmt_row("micro average", summary[mode][level]["micro"]))

    by_diff = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty") or "?", []).append(r)
    if len(by_diff) > 1:
        print("\nEdge F1 by difficulty (normalized, macro):")
        for diff, rs in sorted(by_diff.items()):
            f1 = macro_average([r["metrics"]["normalized"]["edge"] for r in rs])["f1"]
            print(f"  {diff:<8} n={len(rs):<3} F1={f1:.3f}")

    ren = [r["renders"] for r in records if r["renders"] is not None]
    if ren:
        print(f"\nPredicted DOT that Graphviz can render: {sum(ren)}/{len(ren)}")


def jsonable(x):
    if isinstance(x, set):
        return sorted([list(i) if isinstance(i, tuple) else i for i in x])
    raise TypeError


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate an Ollama model on Text2Arch manual.tsv")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--dataset", default=None, help="path to manual.tsv")
    ap.add_argument("--samples", type=int, default=99, help="how many rows to evaluate")
    ap.add_argument("--start", type=int, default=1, help="first row (1-based)")
    ap.add_argument("--all", action="store_true", help="evaluate every row")
    ap.add_argument("--prompt", choices=["auto"] + sorted(PROMPTS), default="auto",
                    help="auto = 'small' for <=3B models, 'flow' otherwise")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--think", nargs="?", const="on", choices=["auto", "on", "off"], default="auto",
                    help="auto = off for qwen3/deepseek-r1, not sent to other models")
    ap.add_argument("--retries", type=int, default=0, help="extra attempts if the output is unusable")
    ap.add_argument("--postprocess", action="store_true", help="snap predicted names to phrases in the description")
    ap.add_argument("--drop-ungrounded", action="store_true", help="drop nodes that match nothing in the description")
    args = ap.parse_args()

    print("=" * 60)
    print("TEXT2ARCH EVALUATION")
    print("=" * 60)

    dataset, tried = find_dataset(args.dataset)
    if not dataset:
        print("ERROR: manual.tsv not found. Looked in:")
        for t in tried:
            print("   ", t)
        print("Pass it explicitly:  python run_evaluation.py --dataset path\\to\\manual.tsv")
        return 1

    try:
        models = installed_models()
    except Exception as e:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_HOST}  ({e})")
        print("Start it (open the Ollama app, or run `ollama serve`) and try again.")
        return 1
    if not model_available(args.model, models):
        print(f"ERROR: model '{args.model}' is not installed. Installed models:")
        for m in models:
            print("   ", m)
        print(f"Install with:  ollama pull {args.model}")
        return 1

    sw = thinking_switch(args)
    print(f"Model   : {args.model}   (think={'n/a' if sw is None else 'on' if sw else 'off'}, prompt={resolve_prompt(args)})")
    print(f"Dataset : {dataset}")

    with open(dataset, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"Cleaned Description", "Dot code"}
    if not rows or not required <= set(rows[0].keys()):
        print(f"ERROR: expected columns {sorted(required)}, found {list(rows[0].keys()) if rows else 'none'}")
        return 1

    first = max(args.start, 1) - 1
    last = len(rows) if args.all else min(len(rows), first + args.samples)
    selected = list(enumerate(rows[first:last], start=first + 1))
    print(f"Samples : {len(selected)} (rows {first + 1}..{last} of {len(rows)})")
    print("=" * 60)

    records, skipped = [], []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    try:
        for done, (num, row) in enumerate(selected, start=1):
            print(f"\n{'=' * 60}\nSAMPLE {num}  ({done}/{len(selected)})  "
                  f"difficulty={row.get('Difficulty', '?')}  file={row.get('File Name', '')}\n{'=' * 60}")
            try:
                description = (row["Cleaned Description"] or "").strip()
                gt_dot = row["Dot code"] or ""
                if not description:
                    raise ValueError("empty description")
                ground_truth = extract_ground_truth(gt_dot)
                exp_nodes = ground_truth["nodes"]
                exp_edges = ground_truth["edges"]
                if not exp_edges:
                    raise ValueError("ground truth has no parsable edges (bad row in dataset)")

                t1 = time.time()
                print("Generating DOT ...", flush=True)
                raw, pred_dot, notes, meta = generate_dot(description, args)
                if args.postprocess:pred_dot = postprocess(pred_dot, clean_description(description), args.drop_ungrounded)
                print(f"({time.time() - t1:.0f}s, {meta['eval_count']} tokens, stop={meta['done_reason']})")
                for n in notes:
                    print("  note:", n)

                print("\n===== GENERATED DOT =====")
                print(pred_dot)

                results = {m: evaluate_prediction(gt_dot, pred_dot, m) for m in MODES}
                if not results["normalized"]["valid"]:
                    raise ValueError(results["normalized"]["error"])

                norm = results["normalized"]
                print("\n===== GROUND TRUTH =====")
                print("Nodes:", sorted(norm["expected_nodes"]))
                print("Edges:", sorted(norm["expected_edges"]))
                print("\n===== PREDICTED =====")
                print("Nodes:", sorted(norm["predicted_nodes"]))
                print("Edges:", sorted(norm["predicted_edges"]))
                print("\nMissing edges  :", sorted(norm["expected_edges"] - norm["predicted_edges"]))
                print("Extra edges    :", sorted(norm["predicted_edges"] - norm["expected_edges"]))

                print_metrics("NODE METRICS (normalized)", norm["node_metrics"])
                print_metrics("EDGE METRICS (normalized)", norm["edge_metrics"])

                records.append({
                    "sample": num,
                    "file": row.get("File Name", ""),
                    "difficulty": row.get("Difficulty", ""),
                    "predicted_dot": pred_dot,
                    "raw_output": raw,
                    "renders": dot_renders(pred_dot),
                    "expected_nodes": norm["expected_nodes"],
                    "expected_edges": norm["expected_edges"],
                    "predicted_nodes": norm["predicted_nodes"],
                    "predicted_edges": norm["predicted_edges"],
                    "metrics": {m: {"node": results[m]["node_metrics"],
                                    "edge": results[m]["edge_metrics"]} for m in MODES},
                })
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\nFAILED SAMPLE {num}: {e}")
                skipped.append({"sample": num, "reason": str(e)})
    except KeyboardInterrupt:
        print("\n\nInterrupted - reporting the samples finished so far.")

    summary = summarise(records) if records else {}
    print_summary(records, summary, len(selected), skipped)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")

    # ---- save ----
    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"run_{run_id}.json")
    csv_path = os.path.join(out_dir, f"run_{run_id}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": summary, "skipped": skipped, "samples": records},
                  f, indent=2, ensure_ascii=False, default=jsonable)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        cols = ["sample", "file", "difficulty"] + [
            f"{lvl}_{k}" for lvl in ("node", "edge") for k in ("precision", "recall", "f1", "jaccard")]
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            nm, em = r["metrics"]["normalized"]["node"], r["metrics"]["normalized"]["edge"]
            w.writerow([r["sample"], r["file"], r["difficulty"]] +
                       [round(m[k], 4) for m in (nm, em) for k in ("precision", "recall", "f1", "jaccard")])
    print(f"\nSaved: {json_path}\n       {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())