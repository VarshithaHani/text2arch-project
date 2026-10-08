"""
DOT parsing + comparison of a predicted graph against a ground-truth graph.

Fixes compared with the previous regex version:
  * edge attributes  (A -> B [label="calls"])  no longer corrupt node labels
  * chained edges    (A -> B -> C)             give (A,B) and (B,C)
  * bare node lines  (Customer;)               are recognised as nodes
  * graph-level lines (rankdir=LR;)            are ignored
  * ground-truth cells in manual.tsv use '|' instead of newlines -> handled
  * names are compared in 'exact', 'normalized' or 'fuzzy' mode
"""
import re
from difflib import SequenceMatcher

from metrics import calculate_metrics

HEADER_WORDS = {"digraph", "graph", "strict", "subgraph"}
DEFAULT_WORDS = {"node", "edge"}

_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"')


# ------------------------------------------------------------------
# low-level scanning helpers (all quote-aware)
# ------------------------------------------------------------------
def _split_statements(text, pipe_is_newline=False):
    stmts, buf = [], []
    i, n = 0, len(text)
    in_q, depth = False, 0

    def flush():
        s = "".join(buf).strip()
        if s:
            stmts.append(s)
        buf.clear()

    while i < n:
        c = text[i]
        if in_q:
            buf.append(c)
            if c == "\\" and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_q = False
            i += 1
            continue
        if c == '"':
            in_q = True
            buf.append(c)
            i += 1
            continue
        if pipe_is_newline and text.startswith("\\n", i):
            # a few ground-truth cells use a literal backslash-n as line break
            flush()
            i += 2
            continue
        if text.startswith("//", i):
            # a comment ends at a real newline (or at '|' in manual.tsv cells)
            while i < n and text[i] != "\n" and not (pipe_is_newline and (text[i] == "|" or text.startswith("\\n", i))):
                i += 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == "#" and not "".join(buf).strip():
            while i < n and text[i] != "\n" and not (pipe_is_newline and (text[i] == "|" or text.startswith("\\n", i))):
                i += 1
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth = max(0, depth - 1)
        if depth == 0 and (c in ";\n{}" or (pipe_is_newline and c == "|")):
            flush()
            i += 1
            continue
        buf.append(c)
        i += 1
    flush()
    return stmts


def _split_attrs(stmt):
    """'A [label="x"]' -> ('A', 'label="x"')"""
    out, attrs = [], []
    depth, in_q, i = 0, False, 0
    while i < len(stmt):
        c = stmt[i]
        tgt = attrs if depth > 0 else out
        if in_q:
            tgt.append(c)
            if c == "\\" and i + 1 < len(stmt):
                tgt.append(stmt[i + 1])
                i += 2
                continue
            if c == '"':
                in_q = False
        elif c == '"':
            in_q = True
            tgt.append(c)
        elif c == "[":
            depth += 1
            if depth > 1:
                attrs.append(c)
        elif c == "]":
            if depth > 0:
                depth -= 1
                attrs.append(c if depth > 0 else " ")
        else:
            tgt.append(c)
        i += 1
    return "".join(out).strip(), "".join(attrs)


def _split_arrow(body):
    parts, buf, in_q, i = [], [], False, 0
    while i < len(body):
        c = body[i]
        if in_q:
            buf.append(c)
            if c == "\\" and i + 1 < len(body):
                buf.append(body[i + 1])
                i += 2
                continue
            if c == '"':
                in_q = False
            i += 1
            continue
        if c == '"':
            in_q = True
        if not in_q and body.startswith("->", i):
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def _parse_id(tok):
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1].replace('\\"', '"').strip()
    return tok.split(":")[0].strip()  # drop ports like  node:port


_LABEL = re.compile(r'\blabel\s*=\s*("(?:\\.|[^"\\])*"|[^\s,;\]]+)')


def _label_from_attrs(attrs):
    m = _LABEL.search(attrs)
    if not m:
        return None
    label = m.group(1)
    if label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    label = label.replace('\\"', '"').replace("\\n", " ").strip()
    return label or None


# ------------------------------------------------------------------
# public parsing API
# ------------------------------------------------------------------
def parse_dot(dot, pipe_is_newline=False):
    """Return (nodes, edges): set of display names, set of (src, dst)."""
    if not isinstance(dot, str) or not dot.strip():
        raise ValueError("DOT input is empty.")
    if "digraph" not in dot.lower():
        raise ValueError("Invalid DOT: 'digraph' not found.")

    labels, order, raw_edges = {}, [], []

    def see(node_id):
        if node_id and node_id not in order:
            order.append(node_id)

    for stmt in _split_statements(dot, pipe_is_newline):
        words = stmt.split()
        if not words:
            continue
        first = words[0].lower()
        if first in HEADER_WORDS:
            continue
        body, attrs = _split_attrs(stmt)
        if not body:
            continue
        if body.lower() in DEFAULT_WORDS:          # node [shape=box]
            continue
        plain = _QUOTED.sub("", body)
        if "->" in plain:
            ids = [_parse_id(p) for p in _split_arrow(body) if p.strip()]
            for a, b in zip(ids, ids[1:]):
                if a and b:
                    raw_edges.append((a, b))
            for node_id in ids:
                see(node_id)
        elif "=" in plain:                          # rankdir=LR
            continue
        else:
            node_id = _parse_id(body)
            if not node_id or node_id.lower() in DEFAULT_WORDS | HEADER_WORDS:
                continue
            see(node_id)
            label = _label_from_attrs(attrs)
            if label and pipe_is_newline:
                label = " ".join(label.replace("|", " ").split())
            if label:
                labels[node_id] = label

    def name(node_id):
        return labels.get(node_id, node_id)

    nodes = {name(n) for n in order}
    edges = {(name(a), name(b)) for a, b in raw_edges}
    if not nodes and not edges:
        raise ValueError("No nodes or edges could be extracted from DOT.")
    return nodes, edges


_ARROW_VARIANTS = re.compile(r"\s*(?:-{1,2}>|=>|\u2192|\u27f6|\u2794|\u279c)\s*")
_KEYWORDS = DEFAULT_WORDS | HEADER_WORDS


def _quote(tok):
    tok = tok.strip().rstrip(";,").strip()
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok
    return '"' + tok.replace('"', '\\"') + '"'


def repair_dot(dot):
    """
    Rebuild model output as clean, renderable DOT.
      * 'MGGN -> Flow Estimator;'   -> '"MGGN" -> "Flow Estimator"'
      * '-->' / '=>' / unicode arrows -> '->'
      * several statements on one line, 'digraph G { A -> B; B -> C }' -> one per line
      * subgraph/cluster wrappers are flattened (they never matter for scoring)
    Returns the repaired DOT string (raises ValueError if there is no 'digraph').
    """
    if "digraph" not in (dot or "").lower():
        raise ValueError("Invalid DOT: 'digraph' not found.")
    lines = ["digraph G {"]
    for stmt in _split_statements(dot):
        first = stmt.split()[0].lower()
        if first in HEADER_WORDS:
            continue
        head, attrs = _split_attrs(stmt)
        if not head:
            continue
        suffix = f" [{attrs.strip()}]" if attrs.strip() else ""
        plain = _QUOTED.sub("", head)
        if not _QUOTED.search(head):
            head = _ARROW_VARIANTS.sub(" -> ", head)
            plain = head
        if "->" in plain:
            toks = [t for t in _split_arrow(head) if t.strip().rstrip(";,").strip()]
            if len(toks) >= 2:
                lines.append("  " + " -> ".join(_quote(t) for t in toks) + suffix)
            elif len(toks) == 1:
                lines.append("  " + _quote(toks[0]))
        elif "=" in plain or head.lower() in DEFAULT_WORDS:
            lines.append("  " + head + suffix)
        else:
            lines.append("  " + _quote(head) + suffix)
    lines.append("}")
    return "\n".join(lines)


def extract_ground_truth(ground_truth_dot):
    # manual.tsv stores line breaks as '|'
    return parse_dot(ground_truth_dot, pipe_is_newline=True)


# ------------------------------------------------------------------
# name matching modes
# ------------------------------------------------------------------
def normalize_name(s):
    """'Flow_Estimator' / 'flow estimator' / 'Flow-Estimator' -> 'flow estimator'"""
    return re.sub(r"[\W_]+", " ", s.lower()).strip() or s.lower()


def _best_match(name, candidates, threshold):
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, name, c).ratio()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None


def _apply_mode(exp_nodes, exp_edges, pred_nodes, pred_edges, mode):
    if mode == "exact":
        return exp_nodes, exp_edges, pred_nodes, pred_edges

    en = {normalize_name(n) for n in exp_nodes}
    ee = {(normalize_name(a), normalize_name(b)) for a, b in exp_edges}
    pn = {normalize_name(n) for n in pred_nodes}
    pe = {(normalize_name(a), normalize_name(b)) for a, b in pred_edges}

    if mode == "fuzzy":
        mapping = {}
        for p in pn:
            if p in en:
                mapping[p] = p
            else:
                mapping[p] = _best_match(p, en, 0.75) or p
        pn = {mapping[p] for p in pn}
        pe = {(mapping.get(a, a), mapping.get(b, b)) for a, b in pe}
    return en, ee, pn, pe


def evaluate_prediction(ground_truth_dot, predicted_dot, mode="normalized"):
    """mode: 'exact' | 'normalized' | 'fuzzy'"""
    try:
        exp_nodes, exp_edges = extract_ground_truth(ground_truth_dot)
    except Exception as e:
        return {"valid": False, "error": f"Ground-truth parsing failed: {e}"}
    try:
        pred_nodes, pred_edges = parse_dot(predicted_dot)
    except Exception as e:
        return {"valid": False, "error": f"Prediction parsing failed: {e}"}

    en, ee, pn, pe = _apply_mode(exp_nodes, exp_edges, pred_nodes, pred_edges, mode)
    return {
        "valid": True,
        "mode": mode,
        "expected_nodes": en,
        "expected_edges": ee,
        "predicted_nodes": pn,
        "predicted_edges": pe,
        "node_metrics": calculate_metrics(en, pn),
        "edge_metrics": calculate_metrics(ee, pe),
    }