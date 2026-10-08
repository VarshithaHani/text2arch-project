import re

from metrics import calculate_metrics


# ============================================================
# Graphviz keywords that must NOT become nodes
# ============================================================

GRAPHVIZ_KEYWORDS = {
    "digraph",
    "graph",
    "strict",
    "node",
    "edge",
    "subgraph"
}


# ============================================================
# Cleaning helpers
# ============================================================

def strip_wrappers(dot):
    """
    Remove Markdown fences and comments from DOT text.
    """

    dot = dot.strip()

    # ```dot / ``` fence lines
    dot = re.sub(
        r"^```[A-Za-z]*\s*$",
        "",
        dot,
        flags=re.MULTILINE
    )

    # /* block comments */
    dot = re.sub(
        r"/\*.*?\*/",
        "",
        dot,
        flags=re.DOTALL
    )

    # // and # comment lines
    dot = re.sub(
        r"^\s*(//|#).*$",
        "",
        dot,
        flags=re.MULTILINE
    )

    return dot.strip()


def split_top_level(text, delimiters):
    """
    Split text on any delimiter, but ONLY when the delimiter
    is outside "quotes" and outside [ attribute lists ].
    """

    parts = []
    current = []
    in_quote = False
    depth = 0
    i = 0

    while i < len(text):

        ch = text[i]

        if in_quote:

            current.append(ch)

            if ch == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue

            if ch == '"':
                in_quote = False

            i += 1
            continue

        if ch == '"':
            in_quote = True
            current.append(ch)
            i += 1
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)

        if depth == 0:

            matched = None

            for delimiter in delimiters:
                if text.startswith(delimiter, i):
                    matched = delimiter
                    break

            if matched is not None:
                parts.append("".join(current))
                current = []
                i += len(matched)
                continue

        current.append(ch)
        i += 1

    parts.append("".join(current))

    return parts


def find_attribute_start(text):
    """
    Index of the first '[' that is outside quotes, or -1.
    """

    in_quote = False
    i = 0

    while i < len(text):

        ch = text[i]

        if in_quote:

            if ch == "\\":
                i += 2
                continue

            if ch == '"':
                in_quote = False

        elif ch == '"':
            in_quote = True

        elif ch == "[":
            return i

        i += 1

    return -1


def clean_id(raw):
    """
    Strip whitespace and surrounding quotes from a DOT ID.
    Works for:  WebServer   "Web Server"   Flow Estimator
    """

    raw = raw.strip()

    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        raw = raw[1:-1].replace('\\"', '"')

    return raw.strip()


def get_label(attributes):
    """
    Read label=... from an attribute string, or return None.
    """

    quoted = re.search(
        r'(?<![A-Za-z_])label\s*=\s*"((?:\\.|[^"])*)"',
        attributes,
        re.DOTALL
    )

    if quoted:
        return quoted.group(1).replace('\\"', '"').strip()

    unquoted = re.search(
        r'(?<![A-Za-z_])label\s*=\s*([^\s,\]]+)',
        attributes
    )

    if unquoted:
        return unquoted.group(1).strip()

    return None


def normalize(name):
    """
    Make names comparable between ground truth and prediction:
    lowercase, underscores/newlines -> spaces, collapse whitespace.
    """

    name = name.replace("\\n", " ").replace("\\l", " ")
    name = name.replace("\\r", " ")
    name = name.replace("_", " ")
    name = name.lower()

    return " ".join(name.split())


# ============================================================
# Parse a complete DOT graph
# ============================================================

def parse_dot(dot):
    """
    Parse a DOT graph and return:

        nodes  (set of normalized labels)
        edges  (set of (source_label, target_label))

    The SAME function is used for ground truth and prediction.
    """

    if not isinstance(dot, str):
        raise ValueError("DOT output is not a string.")

    dot = strip_wrappers(dot)

    if not dot:
        raise ValueError("DOT output is empty.")

    if not re.search(r"\bdigraph\b", dot, re.IGNORECASE):
        raise ValueError(
            "Output is not a valid directed Graphviz DOT graph."
        )

    start = dot.find("{")

    if start == -1:
        raise ValueError("DOT graph has no opening '{'.")

    end = dot.rfind("}")

    if end > start:
        body = dot[start + 1:end]
    else:
        body = dot[start + 1:]

    statements = split_top_level(body, [";", "\n", "{", "}"])

    node_map = {}       # id -> label
    raw_edges = []      # (source_id, target_id)

    for statement in statements:

        statement = statement.strip()

        if not statement:
            continue

        bracket = find_attribute_start(statement)

        if bracket == -1:
            head = statement
            attributes = ""
        else:
            head = statement[:bracket].strip()
            attributes = statement[bracket:]

        if not head:
            continue

        # Skip keyword statements: node [...], edge [...], subgraph x
        if head.lower() in GRAPHVIZ_KEYWORDS:
            continue

        if head.split()[0].lower() == "subgraph":
            continue

        # Skip graph attributes such as rankdir=LR
        if bracket == -1 and re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*\s*=", head
        ):
            continue

        # ------------------------------------------------
        # Edge statement (supports chains: a -> b -> c)
        # ------------------------------------------------

        parts = split_top_level(head, ["->"])

        if len(parts) > 1:

            ids = [clean_id(part) for part in parts]

            for source, target in zip(ids, ids[1:]):
                if source and target:
                    raw_edges.append((source, target))

            continue

        # ------------------------------------------------
        # Node statement
        # ------------------------------------------------

        node_id = clean_id(head)

        if not node_id or node_id.lower() in GRAPHVIZ_KEYWORDS:
            continue

        label = get_label(attributes)

        if label:
            node_map[node_id] = label
        elif node_id not in node_map:
            node_map[node_id] = node_id

    # ----------------------------------------------------
    # Convert IDs to labels
    # ----------------------------------------------------

    edges = set()

    for source, target in raw_edges:

        source_label = normalize(node_map.get(source, source))
        target_label = normalize(node_map.get(target, target))

        if source_label and target_label:
            edges.add((source_label, target_label))

    nodes = {
        normalize(label) for label in node_map.values()
    }

    for source, target in edges:
        nodes.add(source)
        nodes.add(target)

    nodes.discard("")

    if not nodes and not edges:
        raise ValueError("DOT graph contains no nodes or edges.")

    return nodes, edges


# ============================================================
# Ground-truth parser
# ============================================================

def extract_ground_truth(ground_truth_dot):
    """
    Expected nodes and edges from the dataset's Dot code,
    using human-readable labels instead of numeric IDs.
    """

    return parse_dot(ground_truth_dot)


# ============================================================
# Evaluate one prediction
# ============================================================

def evaluate_prediction(
    expected_nodes,
    expected_edges,
    predicted_dot
):

    predicted_nodes, predicted_edges = parse_dot(predicted_dot)

    node_metrics = calculate_metrics(
        expected_nodes,
        predicted_nodes
    )

    edge_metrics = calculate_metrics(
        expected_edges,
        predicted_edges
    )

    return {
        "valid": True,
        "predicted_nodes": predicted_nodes,
        "predicted_edges": predicted_edges,
        "node_metrics": node_metrics,
        "edge_metrics": edge_metrics
    }