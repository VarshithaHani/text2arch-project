
import re
from difflib import SequenceMatcher


_QUOTED = re.compile(r'"((?:\\.|[^"\\])*)"')
_ARROW_VARIANTS = re.compile(r"\s*(?:-->|-{1,2}>|=>|→|⟶|➔|➜)\s*")


def normalize_name(name):
    """Normalize labels for fair comparison."""
    name = str(name).strip().strip('"').strip("'")
    name = name.replace("\\n", " ").replace("\n", " ")
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
        statement = statement.strip().rstrip(";")
        if not statement:
            continue

        if statement.startswith("//") or statement.startswith("#"):
            continue

        # Ignore graph declarations and attributes.
        if re.match(
            r"^(strict\s+)?(di)?graph\b|^(node|edge|graph)\s*\[",
            statement,
            re.IGNORECASE,
        ):
            continue

        # Split a statement into directed edges, preserving quoted text.
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
                match = re.match(r'^\s*("[^"]+"|[^\s\[]+)', piece)
                if match:
                    ids.append(_unquote(match.group(1)))

            for node_id in ids:
                nodes.add(node_id)
                id_to_label.setdefault(node_id, node_id)

            for source, target in zip(ids, ids[1:]):
                edges.add((source, target))
            continue

        # Parse ordinary node declarations, including label attributes.
        match = re.match(
            r'^\s*("[^"]+"|[^\s\[]+)\s*(?:\[(.*)\])?\s*$',
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
        label = _unquote(label_match.group(1)) if label_match else node_id

        nodes.add(label)
        id_to_label[node_id] = label

    # Convert edge identifiers to display labels where possible.
    display_edges = set()
    for source, target in edges:
        display_edges.add(
            (
                id_to_label.get(source, source),
                id_to_label.get(target, target),
            )
        )

    return {"nodes": nodes, "edges": display_edges}


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

    true_positive = len(expected & predicted)
    precision = (
        true_positive / len(predicted) if predicted else 0.0
    )
    recall = (
        true_positive / len(expected) if expected else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    union = expected | predicted
    jaccard = len(expected & predicted) / len(union) if union else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "matched_count": true_positive,
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
    Compare parsed DOT structures.

    expected and predicted should each contain:
        {"nodes": set(...), "edges": set((source, target), ...)}
    """
    expected_nodes = set(expected.get("nodes", set()))
    predicted_nodes = set(predicted.get("nodes", set()))
    expected_edges = set(expected.get("edges", set()))
    predicted_edges = set(predicted.get("edges", set()))

    if mode == "exact":
        nodes = _score_sets(expected_nodes, predicted_nodes)
        edges = _score_sets(expected_edges, predicted_edges)

    elif mode == "normalized":
        nodes = _score_sets(
            _normalized_nodes(expected_nodes),
            _normalized_nodes(predicted_nodes),
        )
        edges = _score_sets(
            _normalized_edges(expected_edges),
            _normalized_edges(predicted_edges),
        )

    elif mode == "fuzzy":
        node_mapping = {}
        for node in predicted_nodes:
            match, score = _best_match(node, expected_nodes)
            if match is not None and score >= 0.75:
                node_mapping[node] = match

        mapped_nodes = {
            node_mapping.get(node, f"__unmatched__:{normalize_name(node)}")
            for node in predicted_nodes
        }

        mapped_edges = set()
        for source, target in predicted_edges:
            mapped_source = node_mapping.get(
                source, f"__unmatched__:{normalize_name(source)}"
            )
            mapped_target = node_mapping.get(
                target, f"__unmatched__:{normalize_name(target)}"
            )
            mapped_edges.add((mapped_source, mapped_target))

        nodes = _score_sets(expected_nodes, mapped_nodes)
        edges = _score_sets(expected_edges, mapped_edges)

    else:
        raise ValueError(
            "mode must be 'exact', 'normalized', or 'fuzzy'"
        )

    return {"nodes": nodes, "edges": edges}
