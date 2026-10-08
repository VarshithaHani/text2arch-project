from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto"
)

print("Qwen model loaded successfully!")


SYSTEM_PROMPT = """
You are a software architecture extraction model.

Your task is to convert a natural language software architecture
description into a structured architecture graph.

Return ONLY valid JSON.

The JSON must have exactly this structure:

{
  "nodes": [
    {
      "id": "unique_id",
      "label": "Human Readable Component Name"
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id"
    }
  ]
}

NODE EXTRACTION RULES:

1. Create a node for every distinct software or system component
   explicitly mentioned in the description.

2. Components may include:
   - users
   - customers
   - browsers
   - frontend applications
   - mobile applications
   - backend services
   - APIs
   - servers
   - databases
   - caches
   - message queues
   - authentication services
   - payment services
   - external services
   - workers
   - load balancers
   - API gateways

3. Do NOT create components that are not explicitly mentioned.

4. Do NOT create a separate node for an action, request, operation,
   protocol, or data item unless it is explicitly described as a
   component.

5. If the same component is mentioned multiple times, create only
   one node for it.

EDGE EXTRACTION RULES:

6. Create an edge only when the description explicitly states that
   one component communicates with, sends requests to, calls,
   connects to, accesses, uses, stores data in, retrieves data from,
   forwards requests to, or otherwise directly interacts with another
   component.

7. Preserve the direction of the relationship.

8. Examples:

   "Frontend communicates with Backend"
   means:
   Frontend -> Backend

   "Backend connects to MySQL"
   means:
   Backend -> MySQL

   "Backend stores data in Database"
   means:
   Backend -> Database

   "API retrieves information from Database"
   means:
   API -> Database

   "Load balancer forwards requests to Server"
   means:
   Load_Balancer -> Server

9. NEVER infer a relationship merely because two components appear
   in the same description.

10. NEVER create indirect relationships.
    If A connects to B and B connects to C, do NOT automatically
    create A -> C.

11. Do NOT create duplicate edges.

12. Do NOT reverse the direction of an explicitly stated relationship.

IDENTIFIER RULES:

13. Every node must have a unique ID.

14. IDs must contain only lowercase letters, numbers, and underscores.

15. Convert component names into simple IDs.

   Examples:

   "React Frontend" -> "react_frontend"
   "Spring Boot Backend" -> "spring_boot_backend"
   "MySQL Database" -> "mysql_database"
   "API Gateway" -> "api_gateway"

16. Every edge source and target must exactly match an existing
    node ID.

17. NEVER use "null" as a node ID or edge endpoint.

OUTPUT RULES:

18. Return ONLY JSON.

19. Do NOT return Markdown code fences.

20. Do NOT return explanations.

21. Do NOT return comments.

22. The final output must be valid JSON that can be parsed directly
    by a Python JSON parser.
"""

def extract_json(text: str):
    """
    Extract and parse the JSON object from the model response.
    Handles Markdown fences and extra text around the JSON.
    """

    text = text.strip()

    # Remove Markdown code fences
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
    )

    text = text.strip()

    # First attempt: parse the entire response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: find the JSON object inside extra text
    start = text.find("{")

    if start == -1:
        return None

    # Try possible closing braces from the end
    for end in range(len(text) - 1, start, -1):

        if text[end] != "}":
            continue

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def clean_architecture(data):
    """
    Validate and clean the model-generated architecture.
    """

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
    # Clean nodes
    # -------------------------

    for node in raw_nodes:

        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id", "")).strip()
        label = str(node.get("label", "")).strip()

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
        ).strip("_")

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
    # Clean edges
    # -------------------------

    edges = []
    edge_set = set()

    for edge in raw_edges:

        if not isinstance(edge, dict):
            continue

        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()

        if not source or not target:
            continue

        if source.lower() == "null" or target.lower() == "null":
            continue

        if source not in node_ids or target not in node_ids:
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


def architecture_to_dot(data):
    """
    Convert the cleaned architecture JSON into Graphviz DOT.
    """

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


def generate_architecture(description: str):
    """
    Generate and return the cleaned structured architecture.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
Extract the software architecture from this description.

Architecture description:

{description}

Return only the required JSON.
"""
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1
    )

    generated = outputs[0][inputs.input_ids.shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    print("\nQwen raw output:")
    print(answer)

    architecture = extract_json(answer)

    if architecture is None:
        print("Could not parse Qwen output as JSON.")

        return {
            "nodes": [],
            "edges": []
        }

    cleaned_architecture = clean_architecture(
        architecture
    )

    print("\nCleaned architecture:")
    print(
        json.dumps(
            cleaned_architecture,
            indent=2
        )
    )

    return cleaned_architecture


def generate_dot(description: str) -> str:
    """
    Generate DOT from the structured architecture.

    This function is kept so the existing FastAPI
    endpoint continues to work.
    """

    architecture = generate_architecture(
        description
    )

    dot = architecture_to_dot(
        architecture
    )

    print("\nGenerated DOT:")
    print(dot)

    return dot