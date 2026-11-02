import json
import os
import re
import urllib.request
import urllib.error

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

SYSTEM_PROMPT = """Extract software architecture from natural language.

Return only valid JSON in exactly this format:
{
  "nodes": [
    {"id": "frontend", "label": "Frontend", "shape": "box"},
    {"id": "backend", "label": "Backend API", "shape": "box"},
    {"id": "database", "label": "Database", "shape": "cylinder"}
  ],
  "edges": [
    {"source": "frontend", "target": "backend"},
    {"source": "backend", "target": "database"}
  ]
}

Allowed shapes: ellipse, box, diamond, cylinder, parallelogram, hexagon, doublecircle.

Rules:
1. Include all explicitly mentioned software components as nodes.
2. Add an edge for every explicitly described connection, dependency,
   API call, communication, data flow, read/write, or storage relationship.
3. Examples: A sends data to B, A calls B, A uses B, A connects to B.
4. Preserve the direction of every relationship.
5. Every node must have id, label, and shape.
6. Every edge must have source and target.
7. Edge endpoints must match node IDs exactly.
8. Node IDs must contain lowercase letters, numbers, and underscores only.
9. Do not invent unsupported components or relationships.
10. Do not omit edges when the description clearly states a connection.
11. Use ellipse if a shape is not specified.
12. Return JSON only, without markdown or explanations.
"""


# ---------------------------------------------------------
# JSON EXTRACTION
# ---------------------------------------------------------

def extract_json(text: str):
    """Extract a JSON object from the model response."""

    if not isinstance(text, str):
        return None

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    ).strip()

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None

    except json.JSONDecodeError:
        start = text.find("{")

        if start < 0:
            return None

        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            return value if isinstance(value, dict) else None

        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------
# NODE ID NORMALIZATION
# ---------------------------------------------------------

def _normalise_id(value):
    """Convert a node identifier into a safe normalized ID."""

    value = str(value or "").strip().lower()

    value = re.sub(
        r"[^a-z0-9_]+",
        "_",
        value
    )

    return re.sub(
        r"_+",
        "_",
        value
    ).strip("_")


def _endpoint_key(value):
    """Normalize IDs and labels for tolerant endpoint matching."""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value or "").strip().lower()
    )


# ---------------------------------------------------------
# CLEAN AND VALIDATE ARCHITECTURE
# ---------------------------------------------------------

def clean_architecture(data):
    """
    Validate nodes and edges.

    Edge endpoints may use node IDs or labels.
    Endpoint matching ignores capitalization, spaces, and punctuation.
    Invalid endpoints and duplicate directed edges are discarded.
    """

    if not isinstance(data, dict):
        return {"nodes": [], "edges": []}

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    if not isinstance(raw_nodes, list):
        raw_nodes = []

    if not isinstance(raw_edges, list):
        raw_edges = []

    allowed_shapes = {
        "ellipse",
        "oval",
        "box",
        "rectangle",
        "diamond",
        "cylinder",
        "parallelogram",
        "hexagon",
        "doublecircle"
    }

    nodes = []
    used_ids = set()
    aliases = {}

    def add_alias(value, final_id):
        value = str(value or "").strip()

        if not value:
            return

        aliases[value.casefold()] = final_id
        aliases[_normalise_id(value)] = final_id
        aliases[_endpoint_key(value)] = final_id

    # -----------------------------------------------------
    # CLEAN NODES
    # -----------------------------------------------------

    for item in raw_nodes:

        if not isinstance(item, dict):
            continue

        old_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or "").strip()

        if not label or label.casefold() == "null":
            continue

        base = _normalise_id(old_id or label)

        if not base or base == "null":
            base = _normalise_id(label)

        if not base:
            continue

        # Ensure unique node IDs.
        final_id = base
        number = 2

        while final_id in used_ids:
            final_id = f"{base}_{number}"
            number += 1

        shape = str(
            item.get("shape") or "ellipse"
        ).strip().lower()

        if shape not in allowed_shapes:
            shape = "ellipse"

        if shape == "oval":
            shape = "ellipse"

        elif shape == "rectangle":
            shape = "box"

        nodes.append({
            "id": final_id,
            "label": label,
            "shape": shape
        })

        used_ids.add(final_id)

        # Save original and normalized names for edge resolution.
        add_alias(old_id, final_id)
        add_alias(base, final_id)
        add_alias(final_id, final_id)
        add_alias(label, final_id)

    # -----------------------------------------------------
    # RESOLVE EDGE ENDPOINTS
    # -----------------------------------------------------

    def resolve_endpoint(value):
        value = str(value or "").strip()

        if not value:
            return None

        # Exact case-insensitive match.
        found = aliases.get(value.casefold())

        if found:
            return found

        # Match normalized node IDs.
        found = aliases.get(_normalise_id(value))

        if found:
            return found

        # Match labels without spaces or punctuation.
        key = _endpoint_key(value)

        found = aliases.get(key)

        if found:
            return found

        # Last attempt: compare against existing node IDs and labels.
        for node in nodes:

            node_id_key = _endpoint_key(node["id"])
            label_key = _endpoint_key(node["label"])

            if key and (
                key == node_id_key
                or key == label_key
            ):
                return node["id"]

        return None

    # -----------------------------------------------------
    # CLEAN EDGES
    # -----------------------------------------------------

    edges = []
    seen_edges = set()

    for item in raw_edges:

        if isinstance(item, str):

            match = re.match(
                r"^\s*(.+?)\s*(?:->|-->| to )\s*(.+?)\s*$",
                item,
                re.IGNORECASE
            )

            if not match:
                continue

            source_value, target_value = match.groups()

        elif isinstance(item, dict):

            source_value = (
                item.get("source")
                or item.get("from")
                or item.get("source_id")
                or item.get("from_node")
                or ""
            )

            target_value = (
                item.get("target")
                or item.get("to")
                or item.get("target_id")
                or item.get("to_node")
                or ""
            )

        else:
            continue

        source_value = str(source_value or "").strip()
        target_value = str(target_value or "").strip()

        if not source_value or not target_value:
            continue

        if source_value.casefold() == "null":
            continue

        if target_value.casefold() == "null":
            continue

        source = resolve_endpoint(source_value)
        target = resolve_endpoint(target_value)

        # Both endpoints must refer to existing nodes.
        if source not in used_ids or target not in used_ids:
            continue

        edge_key = (source, target)

        # Avoid duplicate directed edges.
        if edge_key in seen_edges:
            continue

        edges.append({
            "source": source,
            "target": target
        })

        seen_edges.add(edge_key)

    return {
        "nodes": nodes,
        "edges": edges
    }


# ---------------------------------------------------------
# JSON TO GRAPHVIZ DOT
# ---------------------------------------------------------

def architecture_to_dot(data):
    """Convert architecture JSON into Graphviz DOT."""

    lines = [
        "digraph Architecture {",
        "    rankdir=LR;",
        ""
    ]

    allowed_shapes = {
        "ellipse",
        "box",
        "diamond",
        "cylinder",
        "parallelogram",
        "hexagon",
        "doublecircle"
    }

    node_ids = set()

    # Generate nodes.
    for node in data.get("nodes", []):

        node_id = _normalise_id(node.get("id"))

        if not node_id:
            continue

        node_ids.add(node.get("id"))

        label = str(
            node.get("label", node_id)
        )

        # Escape characters that could break DOT syntax.
        label = label.replace("\\", "\\\\")
        label = label.replace('"', '\\"')

        shape = str(
            node.get("shape", "ellipse")
        ).lower()

        if shape not in allowed_shapes:
            shape = "ellipse"

        lines.append(
            f'    {node_id} [label="{label}", shape="{shape}"];'
        )

    lines.append("")

    # Generate edges only when both nodes exist.
    for edge in data.get("edges", []):

        source = edge.get("source")
        target = edge.get("target")

        if source not in node_ids:
            continue

        if target not in node_ids:
            continue

        source_id = _normalise_id(source)
        target_id = _normalise_id(target)

        lines.append(
            f"    {source_id} -> {target_id};"
        )

    lines.append("}")

    return "\n".join(lines)


# ---------------------------------------------------------
# RELATIONSHIP RECOVERY
# ---------------------------------------------------------
# Qwen can return nodes correctly but omit edges.
# This fallback recovers relationships explicitly described
# in the original text between recognized components.
# It does not connect every pair of nodes automatically.

_RELATIONSHIP_CUES = re.compile(
    r"\b(?:"
    r"connect(?:s|ed|ing)?\s+(?:to|with)?|"
    r"communicat(?:es|ed|ing)?\s+(?:with|to)?|"
    r"call(?:s|ed|ing)?|"
    r"invoke(?:s|d)?|"
    r"send(?:s|ing)?\s+(?:data\s+)?to|"
    r"forward(?:s|ed|ing)?\s+(?:data\s+)?to|"
    r"pass(?:es|ed|ing)?\s+(?:data\s+)?to|"
    r"fetch(?:es|ed|ing)?\s+(?:data\s+)?from|"
    r"read(?:s|ing)?\s+(?:data\s+)?from|"
    r"write(?:s|ing)?\s+(?:data\s+)?to|"
    r"store(?:s|d|ing)?\s+(?:data\s+)?in|"
    r"save(?:s|d|ing)?\s+(?:data\s+)?to|"
    r"quer(?:y|ies|ied|ying)|"
    r"depend(?:s|ed|ing)?\s+on|"
    r"use(?:s|d|ing)?|"
    r"interact(?:s|ed|ing)?\s+with|"
    r"send(?:s|ing)?\s+requests?\s+to|"
    r"receive(?:s|d|ing)?\s+(?:data\s+)?from"
    r")\b",
    re.IGNORECASE
)


def _find_node_mentions(text: str, nodes: list):
    """
    Find component mentions in their original text order.

    Longer labels are preferred over shorter overlapping labels.
    All non-overlapping occurrences are considered.
    """

    mentions = []

    candidates = sorted(
        nodes,
        key=lambda node: max(
            len(str(node.get("label", ""))),
            len(str(node.get("id", "")))
        ),
        reverse=True
    )

    occupied = []

    for node in candidates:

        node_id = node.get("id", "")

        values = {
            str(node.get("label", "")).strip(),
            str(node_id).strip()
        }

        for value in sorted(
            (v for v in values if v),
            key=len,
            reverse=True
        ):

            pattern = re.compile(
                r"(?<![A-Za-z0-9_])"
                + re.escape(value)
                + r"(?![A-Za-z0-9_])",
                re.IGNORECASE
            )

            for match in pattern.finditer(text):

                start = match.start()
                end = match.end()

                # Do not overlap an already recognized mention.
                if any(
                    start < old_end and end > old_start
                    for old_start, old_end in occupied
                ):
                    continue

                mentions.append((
                    start,
                    end,
                    node_id
                ))

                occupied.append((start, end))

    mentions.sort(key=lambda item: item[0])

    return mentions


def recover_explicit_edges(
    description: str,
    nodes: list,
    existing_edges: list = None
):
    """
    Recover edges only when the description contains a relationship cue
    between two recognized components in the same sentence.
    """

    edges = list(existing_edges or [])

    seen = {
        (
            str(edge.get("source", "")),
            str(edge.get("target", ""))
        )
        for edge in edges
        if isinstance(edge, dict)
    }

    if not description or len(nodes) < 2:
        return edges

    # Keep relationships local to each sentence.
    sentences = re.split(
        r"(?<=[.!?;])\s+|\n+",
        description
    )

    for sentence in sentences:

        mentions = _find_node_mentions(
            sentence,
            nodes
        )

        if len(mentions) < 2:
            continue

        # Check neighboring component mentions.
        for left, right in zip(
            mentions,
            mentions[1:]
        ):

            left_id = left[2]
            right_id = right[2]

            if left_id == right_id:
                continue

            between = sentence[
                left[1]:right[0]
            ]

            if _RELATIONSHIP_CUES.search(between):

                key = (left_id, right_id)

                if key not in seen:

                    edges.append({
                        "source": left_id,
                        "target": right_id
                    })

                    seen.add(key)

    return edges


def _add_missing_relationship_edges(
    description: str,
    architecture: dict
):
    """Apply text-based relationship recovery."""

    nodes = architecture.get("nodes", [])
    edges = architecture.get("edges", [])

    recovered = recover_explicit_edges(
        description,
        nodes,
        edges
    )

    return {
        "nodes": nodes,
        "edges": recovered
    }


# ---------------------------------------------------------
# GENERATE ARCHITECTURE USING OLLAMA
# ---------------------------------------------------------

def generate_architecture(description: str):
    """
    Call Ollama, parse the model response, clean nodes and edges,
    and recover explicitly described missing relationships.
    """

    description = (description or "").strip()

    if not description:
        return {
            "nodes": [],
            "edges": []
        }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Extract all software components and relationships "
                    "from the description below. Include every explicitly "
                    "stated connection as a directed edge. Ensure that "
                    "each edge endpoint matches a node ID. "
                    "Return valid JSON only.\n\n"
                    + description
                )
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 768
        }
    }

    def call_ollama(body):

        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=600
        ) as response:

            return json.loads(
                response.read().decode("utf-8")
            )

    try:

        try:
            result = call_ollama(payload)

        except urllib.error.HTTPError as exc:

            error_text = exc.read().decode(
                "utf-8",
                "replace"
            )

            # Retry without the "think" field if unsupported.
            if (
                exc.code != 400
                or "think" not in error_text.lower()
            ):
                raise

            payload.pop("think", None)

            result = call_ollama(payload)

        answer = (
            result.get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise ValueError(
                "Ollama returned an empty response."
            )

        architecture = extract_json(answer)

        if architecture is None:
            raise ValueError(
                "Qwen did not return valid JSON."
            )

        cleaned = clean_architecture(architecture)

        if not cleaned["nodes"]:
            raise ValueError(
                "No valid architecture nodes were generated."
            )

        # Recover explicitly described connections if Qwen omitted them.
        cleaned = _add_missing_relationship_edges(
            description,
            cleaned
        )

        return cleaned

    except Exception as exc:

        print(
            f"Architecture generation failed: {exc}"
        )

        raise RuntimeError(
            f"Could not generate architecture: {exc}"
        ) from exc


# ---------------------------------------------------------
# COMPATIBILITY FUNCTION
# ---------------------------------------------------------

def generate_dot(description: str) -> str:
    """Generate architecture DOT directly from a description."""

    architecture = generate_architecture(
        description
    )

    dot = architecture_to_dot(
        architecture
    )

    print("\nGenerated DOT:")
    print(dot)

    return dot