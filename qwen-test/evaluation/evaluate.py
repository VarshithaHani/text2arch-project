
import re
from difflib import SequenceMatcher


_QUOTED = re.compile(r'"((?:\\.|[^"\\])*)"')
_ARROW_VARIANTS = re.compile(r"\s*(?:-->|-{1,2}>|=>|→|⟶|➔|➜)\s*")



def normalize_name(name):
    """Normalize labels for case and separator differences."""
    name = str(name).strip().strip('"').strip("'")
    name = name.replace("\\n", " ").replace("\n", " ")
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.lower().strip()



def _split_statements(dot):
    """Split DOT into statements without splitting inside quotes."""
    statements = []
    current = []
    quoted = False
    escaped = False

    for char in dot:
        if escaped:
            current.append(char)
            escaped = False
            continue

        if char == "\\" and quoted:
            current.append(char)
            escaped = True
            continue

        if char == '"':
            quoted = not quoted

        if char in ";\n" and not quoted:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)

    return statements


def repair_dot(dot):
    """Repair common arrow variants and remove Markdown fences."""
    if not dot:
        return ""

    dot = re.sub(r"```(?:dot|graphviz)?", "", dot, flags=re.IGNORECASE)
    dot = dot.replace("```", "")
    dot = dot.replace("\r", "\n")
    dot = dot.replace("\\n", "\n")
    dot = dot.replace('R"t-tree', 'R-tree')

    repaired = []
    for line in dot.splitlines():
        line = line.strip()
        if not line:
            continue

        # Normalize arrows outside quoted labels only.
        parts = []
        quoted = False
        escaped = False
        buffer = []

        for char in line:
            if escaped:
                buffer.append(char)
                escaped = False
                continue

            if char == "\\" and quoted:
                buffer.append(char)
                escaped = True
                continue

            if char == '"':
                quoted = not quoted
                buffer.append(char)
                continue

            if not quoted:
                buffer.append(char)
            else:
                buffer.append(char)

        line = "".join(buffer)
        repaired.append(line)

    return "\n".join(repaired)


def _unquote(value):
    value = value.strip()
    match = _QUOTED.fullmatch(value)
    if match:
        value = match.group(1)
    return value.replace('\\"', '"').strip()


def parse_dot(dot):
    """Return node labels and directed edges from Graphviz DOT."""
    dot = repair_dot(dot)

    nodes = set()
    edges = set()
    id_to_label = {}

    for statement in _split_statements(dot):
        statement = statement.strip().rstrip(";").strip()

        if not statement:
            continue

        statement = statement.strip("{} \t\r\n")

        if not statement:
            continue

        if statement.startswith("//") or statement.startswith("#"):
            continue

        if re.match(
            r"^(strict\s+)?(di)?graph\b|^(node|edge|graph)\s*\[",
            statement,
            re.IGNORECASE,
        ):
            continue

        # Find directed arrows outside quoted strings.
        arrow_positions = []
        quoted = False
        escaped = False
        i = 0

        while i < len(statement):
            char = statement[i]

            if escaped:
                escaped = False
                i += 1
                continue

            if char == "\\" and quoted:
                escaped = True
                i += 1
                continue

            if char == '"':
                quoted = not quoted
                i += 1
                continue

            if not quoted:
                match = re.match(r"\s*(?:->|→|⟶|➔|➜|=>)\s*", statement[i:])
                if match:
                    arrow_positions.append((i, i + match.end()))
                    i += match.end()
                    continue

            i += 1

        if arrow_positions:
            pieces = []
            start = 0

            for left, right in arrow_positions:
                pieces.append(statement[start:left].strip())
                start = right

            pieces.append(statement[start:].strip())

            ids = []

            for piece in pieces:
                match = re.match(
                    r'^\s*("(?:\\.|[^"\\])*"|[A-Za-z_][\w.]*|-?\d+)',
                    piece,
                )

                if match:
                    ids.append(_unquote(match.group(1)))

            for node_id in ids:
                id_to_label.setdefault(node_id, node_id)

            for source, target in zip(ids, ids[1:]):
                edges.add((source, target))

            continue

        # Parse ordinary node declarations.
        match = re.match(
            r'^\s*("(?:\\.|[^"\\])*"|[A-Za-z_][\w.]*|-?\d+)\s*(?:\[(.*)\])?\s*$',
            statement,
        )

        if not match:
            continue

        node_id = _unquote(match.group(1))
        attrs = match.group(2) or ""

        label_match = re.search(
            r'\blabel\s*=\s*("(?:\\.|[^"\\])*"|[^,\]]+)',
            attrs,
            re.IGNORECASE,
        )

        label = (
            _unquote(label_match.group(1).strip())
            if label_match
            else node_id
        )

        id_to_label[node_id] = label
        nodes.add(label)

    # Convert edge IDs into display labels.
    display_edges = set()

    for source, target in edges:
        display_edges.add(
            (
                id_to_label.get(source, source),
                id_to_label.get(target, target),
            )
        )

    # Include edge endpoints in the node set.
    for source, target in display_edges:
        nodes.add(source)
        nodes.add(target)

    return {"nodes": nodes, "edges": display_edges}

def extract_ground_truth(ground_truth_dot):
    return parse_dot(ground_truth_dot.replace("|", "\n"))

def _best_match(name, candidates):
    if not candidates:
        return None, 0.0

    normalized = normalize_name(name)
    best_candidate = None
    best_score = 0.0

    for candidate in candidates:
        score = SequenceMatcher(
            None, normalized, normalize_name(candidate)
        ).ratio()

        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate, best_score


def _score_sets(expected, predicted):
    expected = set(expected)
    predicted = set(predicted)

    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    union = expected | predicted
    jaccard = tp / len(union) if union else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "matched_count": tp,
    }


def _normalized_nodes(nodes):
    return {normalize_name(node) for node in nodes}


def _normalized_edges(edges):
    return {
        (normalize_name(source), normalize_name(target))
        for source, target in edges
    }



def evaluate_prediction(expected, predicted, mode="normalized"):
    """
    Compare expected and predicted DOT strings.
    Fuzzy matching uses one-to-one node assignments.
    """
    try:
        from scipy.optimize import linear_sum_assignment

        expected_data = extract_ground_truth(expected)
        predicted_data = parse_dot(predicted)

        expected_nodes = set(expected_data["nodes"])
        expected_edges = set(expected_data["edges"])
        predicted_nodes = set(predicted_data["nodes"])
        predicted_edges = set(predicted_data["edges"])

        if mode == "exact":
            en, ee = expected_nodes, expected_edges
            pn, pe = predicted_nodes, predicted_edges

        elif mode == "normalized":
            en = _normalized_nodes(expected_nodes)
            ee = _normalized_edges(expected_edges)
            pn = _normalized_nodes(predicted_nodes)
            pe = _normalized_edges(predicted_edges)

        elif mode == "fuzzy":
            expected_list = sorted(expected_nodes)
            predicted_list = sorted(predicted_nodes)

            node_mapping = {}

            if expected_list and predicted_list:
                scores = [
                    [
                        SequenceMatcher(
                            None,
                            normalize_name(pred),
                            normalize_name(exp)
                        ).ratio()
                        for exp in expected_list
                    ]
                    for pred in predicted_list
                ]

                rows, cols = linear_sum_assignment(
                    [[-score for score in row] for row in scores]
                )

                for row, col in zip(rows, cols):
                    if scores[row][col] >= 0.75:
                        node_mapping[predicted_list[row]] = expected_list[col]

            en = expected_nodes
            pn = {
                node_mapping.get(node, f"__unmatched__:{normalize_name(node)}")
                for node in predicted_nodes
            }

            ee = expected_edges
            pe = {
                (
                    node_mapping.get(source, f"__unmatched__:{normalize_name(source)}"),
                    node_mapping.get(target, f"__unmatched__:{normalize_name(target)}")
                )
                for source, target in predicted_edges
            }

        else:
            raise ValueError("Invalid mode")

        return {
            "valid": True,
            "mode": mode,
            "expected_nodes": en,
            "expected_edges": ee,
            "predicted_nodes": pn,
            "predicted_edges": pe,
            "node_metrics": _score_sets(en, pn),
            "edge_metrics": _score_sets(ee, pe),
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}
