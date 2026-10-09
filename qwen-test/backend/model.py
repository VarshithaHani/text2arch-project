
import json
import re
import os
import urllib.request
import urllib.error

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = "http://localhost:11434/api/chat"

print(f"Using Ollama model: {MODEL_NAME}")



# ---------------------------------------------------------
# SHORT, DIRECT PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """You extract software architecture from text.

Return ONLY valid JSON in exactly this format:

{
  "nodes": [
    {"id": "unique_id", "label": "Component Name"}
  ],
  "edges": [
    {"source": "node_id", "target": "node_id"}
  ]
}

Rules:
- Create nodes only for components explicitly mentioned.
- Components include frontend, backend, API, database, cache, queue,
  server, user, external service, worker, gateway, etc.
- Do not invent components.
- Do not create nodes for actions, protocols, or data unless explicitly
  described as components.
- Create an edge only when the text explicitly says two components
  communicate, connect, call, access, use, send to, store in, retrieve
  from, or forward to each other.
- Preserve relationship direction.
- Do not infer indirect relationships.
- Do not create duplicate nodes or edges.
- Node IDs must use lowercase letters, numbers, and underscores only.
- Every edge endpoint must exactly match a node ID.
- Never use null.
- Return JSON only. No markdown, explanation, or comments.
"""


# ---------------------------------------------------------
# JSON EXTRACTION
# ---------------------------------------------------------

def extract_json(text: str):
    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    ).strip()

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object inside extra text
    start = text.find("{")

    if start == -1:
        return None

    # Try closing braces from the end
    for end in range(len(text) - 1, start, -1):

        if text[end] != "}":
            continue

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------
# CLEAN / VALIDATE ARCHITECTURE
# ---------------------------------------------------------

def clean_architecture(data):

    if not isinstance(data, dict):
        return {
            "nodes": [],
            "edges": []
        }

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    if not isinstance(raw_nodes, list):
        raw_nodes = []

    if not isinstance(raw_edges, list):
        raw_edges = []

    nodes = []
    node_ids = set()

    # -------------------------
    # Nodes
    # -------------------------

    for node in raw_nodes:

        if not isinstance(node, dict):
            continue

        node_id = str(
            node.get("id", "")
        ).strip()

        label = str(
            node.get("label", "")
        ).strip()

        if not node_id or not label:
            continue

        if node_id.lower() == "null":
            continue

        node_id = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            node_id
        )

        node_id = re.sub(
            r"_+",
            "_",
            node_id
        ).strip("_").lower()

        if not node_id:
            continue

        original_id = node_id
        counter = 2

        while node_id in node_ids:
            node_id = f"{original_id}_{counter}"
            counter += 1

        nodes.append({
            "id": node_id,
            "label": label
        })

        node_ids.add(node_id)

    # -------------------------
    # Edges
    # -------------------------

    edges = []
    edge_set = set()

    for edge in raw_edges:

        if not isinstance(edge, dict):
            continue

        source = str(
            edge.get("source", "")
        ).strip()

        target = str(
            edge.get("target", "")
        ).strip()

        if not source or not target:
            continue

        if source.lower() == "null":
            continue

        if target.lower() == "null":
            continue

        if source not in node_ids:
            continue

        if target not in node_ids:
            continue

        edge_key = (source, target)

        if edge_key in edge_set:
            continue

        edges.append({
            "source": source,
            "target": target
        })

        edge_set.add(edge_key)

    return {
        "nodes": nodes,
        "edges": edges
    }


# ---------------------------------------------------------
# JSON -> DOT
# ---------------------------------------------------------

def architecture_to_dot(data):

    lines = [
        "digraph Architecture {",
        "    rankdir=LR;",
        ""
    ]

    for node in data["nodes"]:

        node_id = node["id"]
        label = node["label"].replace('"', '\\"')

        lines.append(
            f'    {node_id} [label="{label}"];'
        )

    lines.append("")

    for edge in data["edges"]:

        source = edge["source"]
        target = edge["target"]

        lines.append(
            f"    {source} -> {target};"
        )

    lines.append("}")

    return "\n".join(lines)


# ---------------------------------------------------------
# GENERATE ARCHITECTURE
# ---------------------------------------------------------

def generate_architecture(description: str):
    description = description.strip()

    if not description:
        return {"nodes": [], "edges": []}

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract the architecture from this description. "
                    "Return only JSON.\n\n" + description
                )
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 512
        }
    }

    def call_ollama(data):
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        try:
            result = call_ollama(payload)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise

            error_text = exc.read().decode("utf-8", "replace")
            if "think" not in error_text.lower():
                raise

            payload.pop("think", None)
            result = call_ollama(payload)

        answer = result.get("message", {}).get("content", "").strip()

        if not answer:
            raise ValueError("Ollama returned an empty response.")

        architecture = extract_json(answer)

        if architecture is None:
            raise ValueError("Qwen did not return valid JSON.")

        cleaned = clean_architecture(architecture)

        if not cleaned["nodes"]:
            raise ValueError("No valid architecture nodes were generated.")

        return cleaned

    except Exception as exc:
        print(f"Architecture generation failed: {exc}")
        raise RuntimeError(
            f"Could not generate architecture: {exc}"
        ) from exc



# ---------------------------------------------------------
# COMPATIBILITY FUNCTION
# ---------------------------------------------------------

def generate_dot(description: str) -> str:

    architecture = generate_architecture(
        description
    )

    dot = architecture_to_dot(
        architecture
    )

    print("\nGenerated DOT:")
    print(dot)

    return dot