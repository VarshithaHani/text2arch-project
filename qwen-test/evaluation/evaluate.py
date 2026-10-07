import re

from metrics import calculate_metrics


# ============================================================
# DOT ID
# Supports:
#   WebServer
#   web_server
#   0
#   "Web Server"
# ============================================================

DOT_ID = (
    r'(?:"((?:\\.|[^"])*)"|'
    r'([A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?))'
)


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
# Convert regex match into node ID
# ============================================================

def get_dot_id(match):
    """
    Return the actual node ID from a regex match.
    """

    quoted_id = match.group(1)
    unquoted_id = match.group(2)

    if quoted_id is not None:
        return quoted_id.replace('\\"', '"')

    return unquoted_id


# ============================================================
# Extract node declarations and labels
# ============================================================

def extract_node_map(dot):
    """
    Extract node IDs and their labels.

    Example:

        0 [label="Frontend"];
        1 [label="Backend"];

    becomes:

        {
            "0": "Frontend",
            "1": "Backend"
        }

    If a node has no label, its ID is used.
    """

    node_map = {}

    # --------------------------------------------------------
    # Node declaration pattern
    #
    # We require the node declaration to start after:
    #   beginning of file
    #   {
    #   ;
    #   newline
    #
    # This prevents attributes such as:
    #   label="Frontend"
    #
    # from being interpreted as nodes.
    # --------------------------------------------------------

    node_pattern = re.compile(
        rf'(?:^|[{{;\n])\s*'
        rf'(?P<node>{DOT_ID})'
        rf'\s*'
        rf'(?:\[(?P<attributes>[^\]]*)\])?'
        rf'\s*;',
        re.MULTILINE | re.DOTALL
    )

    for match in node_pattern.finditer(dot):

        node_id_match = match.group("node")

        node_id = get_dot_id(node_id_match)

        if node_id is None:
            continue

        if node_id in GRAPHVIZ_KEYWORDS:
            continue

        attributes = match.group("attributes") or ""

        # ----------------------------------------------------
        # Look for label="..."
        # ----------------------------------------------------

        label_match = re.search(
            r'label\s*=\s*"((?:\\.|[^"])*)"',
            attributes,
            re.DOTALL
        )

        if label_match:

            label = label_match.group(1)

            label = label.replace(
                '\\"',
                '"'
            )

            node_map[node_id] = label.strip()

        else:

            # No label → use node ID
            node_map[node_id] = node_id.strip()

    return node_map


# ============================================================
# Extract directed edges
# ============================================================

def extract_edges(dot, node_map=None):
    """
    Extract directed edges from DOT.

    Example:

        frontend -> backend;

    becomes:

        ("frontend", "backend")

    If node_map is supplied, numeric/technical IDs are
    converted to their human-readable labels.
    """

    edges = set()

    edge_pattern = re.compile(
        rf'{DOT_ID}'
        rf'\s*->\s*'
        rf'{DOT_ID}',
        re.DOTALL
    )

    for match in edge_pattern.finditer(dot):

        source = get_dot_id(
            match
        )

        target = get_dot_id(
            match
        )

        if source is None or target is None:
            continue

        if node_map is not None:

            source = node_map.get(
                source,
                source
            )

            target = node_map.get(
                target,
                target
            )

        source = source.strip()
        target = target.strip()

        if source and target:

            edges.add(
                (
                    source,
                    target
                )
            )

    return edges


# ============================================================
# Extract nodes
# ============================================================

def extract_nodes(
    dot,
    edges=None,
    node_map=None
):
    """
    Extract nodes from a DOT graph.

    Nodes are obtained from:
    1. Explicit node declarations
    2. Nodes appearing in edges
    """

    nodes = set()

    if node_map is None:
        node_map = extract_node_map(dot)

    # --------------------------------------------------------
    # Explicitly declared nodes
    # --------------------------------------------------------

    for node in node_map.values():

        node = node.strip()

        if node and node not in GRAPHVIZ_KEYWORDS:

            nodes.add(node)

    # --------------------------------------------------------
    # Nodes appearing in edges
    # --------------------------------------------------------

    if edges:

        for source, target in edges:

            source = source.strip()
            target = target.strip()

            if source:
                nodes.add(source)

            if target:
                nodes.add(target)

    return nodes


# ============================================================
# Parse a complete DOT graph
# ============================================================

def parse_dot(dot):
    """
    Parse a DOT graph and return:

        nodes
        edges

    The SAME function is used for:
        - ground truth DOT
        - predicted DOT
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(dot, str):

        raise ValueError(
            "DOT output is not a string."
        )

    dot = dot.strip()

    if not dot:

        raise ValueError(
            "DOT output is empty."
        )

    # --------------------------------------------------------
    # Validate directed Graphviz graph
    # --------------------------------------------------------

    if not re.search(
        r'\bdigraph\b',
        dot,
        re.IGNORECASE
    ):

        raise ValueError(
            "Output is not a valid directed Graphviz DOT graph."
        )

    # --------------------------------------------------------
    # Extract node declarations
    # --------------------------------------------------------

    node_map = extract_node_map(
        dot
    )

    # --------------------------------------------------------
    # Extract edges
    # --------------------------------------------------------

    edges = extract_edges(
        dot,
        node_map
    )

    # --------------------------------------------------------
    # Extract nodes
    # --------------------------------------------------------

    nodes = extract_nodes(
        dot,
        edges,
        node_map
    )

    # --------------------------------------------------------
    # Make sure the graph actually contains something
    # --------------------------------------------------------

    if not nodes and not edges:

        raise ValueError(
            "DOT graph contains no nodes or edges."
        )

    return nodes, edges


# ============================================================
# Ground-truth parser
# ============================================================

def extract_ground_truth(ground_truth_dot):
    """
    Extract expected nodes and expected edges from the
    dataset's Dot code.

    Example dataset DOT:

        digraph {
            0 [label="Frontend"];
            1 [label="Backend"];
            2 [label="Database"];

            0 -> 1;
            1 -> 2;
        }

    Returns:

        nodes = {
            "Frontend",
            "Backend",
            "Database"
        }

        edges = {
            ("Frontend", "Backend"),
            ("Backend", "Database")
        }
    """

    return parse_dot(
        ground_truth_dot
    )


# ============================================================
# Evaluate one prediction
# ============================================================

def evaluate_prediction(
    expected_nodes,
    expected_edges,
    predicted_dot
):
    """
    Compare one predicted DOT graph against the
    ground-truth nodes and edges.
    """

    # --------------------------------------------------------
    # Parse prediction using the SAME parser
    # --------------------------------------------------------

    predicted_nodes, predicted_edges = parse_dot(
        predicted_dot
    )

    # --------------------------------------------------------
    # Node metrics
    # --------------------------------------------------------

    node_metrics = calculate_metrics(
        expected_nodes,
        predicted_nodes
    )

    # --------------------------------------------------------
    # Edge metrics
    # --------------------------------------------------------

    edge_metrics = calculate_metrics(
        expected_edges,
        predicted_edges
    )

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {
        "valid": True,

        "predicted_nodes":
            predicted_nodes,

        "predicted_edges":
            predicted_edges,

        "node_metrics":
            node_metrics,

        "edge_metrics":
            edge_metrics
    }