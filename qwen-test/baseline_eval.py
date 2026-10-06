from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# ============================================================
# 1. LOAD QWEN
# ============================================================

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto"
)

print("Model loaded successfully!\n")


# ============================================================
# 2. EVALUATION DATASET
# ============================================================

examples = [

    {
        "description": """
A customer uses a web frontend.
The web frontend communicates with a backend server.
The backend stores data in a MySQL database.
""",
        "nodes": {
            "Customer",
            "Web_Frontend",
            "Backend_Server",
            "MySQL_Database"
        },
        "edges": {
            ("Customer", "Web_Frontend"),
            ("Web_Frontend", "Backend_Server"),
            ("Backend_Server", "MySQL_Database")
        }
    },

    {
        "description": """
A mobile application communicates with a REST API.
The REST API retrieves information from a PostgreSQL database.
""",
        "nodes": {
            "Mobile_Application",
            "REST_API",
            "PostgreSQL_Database"
        },
        "edges": {
            ("Mobile_Application", "REST_API"),
            ("REST_API", "PostgreSQL_Database")
        }
    },

    {
        "description": """
A client sends requests to a load balancer.
The load balancer forwards requests to an application server.
The application server reads data from a database.
""",
        "nodes": {
            "Client",
            "Load_Balancer",
            "Application_Server",
            "Database"
        },
        "edges": {
            ("Client", "Load_Balancer"),
            ("Load_Balancer", "Application_Server"),
            ("Application_Server", "Database")
        }
    },

    {
        "description": """
A user accesses a React frontend.
The React frontend communicates with a Node.js backend.
The Node.js backend stores information in MongoDB.
""",
        "nodes": {
            "User",
            "React_Frontend",
            "Node.js_Backend",
            "MongoDB"
        },
        "edges": {
            ("User", "React_Frontend"),
            ("React_Frontend", "Node.js_Backend"),
            ("Node.js_Backend", "MongoDB")
        }
    },

    {
        "description": """
A customer uses an e-commerce website.
The website communicates with an application server.
The application server connects to a payment service.
The application server stores orders in a database.
""",
        "nodes": {
            "Customer",
            "Ecommerce_Website",
            "Application_Server",
            "Payment_Service",
            "Database"
        },
        "edges": {
            ("Customer", "Ecommerce_Website"),
            ("Ecommerce_Website", "Application_Server"),
            ("Application_Server", "Payment_Service"),
            ("Application_Server", "Database")
        }
    },

    {
        "description": """
A user sends a request to an API gateway.
The API gateway forwards the request to an authentication service.
The authentication service accesses a user database.
""",
        "nodes": {
            "User",
            "API_Gateway",
            "Authentication_Service",
            "User_Database"
        },
        "edges": {
            ("User", "API_Gateway"),
            ("API_Gateway", "Authentication_Service"),
            ("Authentication_Service", "User_Database")
        }
    },

    {
        "description": """
A browser communicates with a web server.
The web server uses Redis for caching.
The web server retrieves persistent data from PostgreSQL.
""",
        "nodes": {
            "Browser",
            "Web_Server",
            "Redis",
            "PostgreSQL"
        },
        "edges": {
            ("Browser", "Web_Server"),
            ("Web_Server", "Redis"),
            ("Web_Server", "PostgreSQL")
        }
    },

    {
        "description": """
A customer uses a mobile application.
The mobile application communicates with a backend API.
The backend API connects to a payment gateway.
The backend API stores transaction data in a database.
""",
        "nodes": {
            "Customer",
            "Mobile_Application",
            "Backend_API",
            "Payment_Gateway",
            "Database"
        },
        "edges": {
            ("Customer", "Mobile_Application"),
            ("Mobile_Application", "Backend_API"),
            ("Backend_API", "Payment_Gateway"),
            ("Backend_API", "Database")
        }
    },

    {
        "description": """
A user interacts with a frontend.
The frontend sends requests to a backend.
The backend communicates with a message queue.
The message queue sends messages to a worker service.
""",
        "nodes": {
            "User",
            "Frontend",
            "Backend",
            "Message_Queue",
            "Worker_Service"
        },
        "edges": {
            ("User", "Frontend"),
            ("Frontend", "Backend"),
            ("Backend", "Message_Queue"),
            ("Message_Queue", "Worker_Service")
        }
    },

    {
        "description": """
A customer accesses a web application.
The web application communicates with a backend service.
The backend service uses Redis for caching.
The backend service stores permanent data in MySQL.
The backend service connects to an external notification API.
""",
        "nodes": {
            "Customer",
            "Web_Application",
            "Backend_Service",
            "Redis",
            "MySQL",
            "Notification_API"
        },
        "edges": {
            ("Customer", "Web_Application"),
            ("Web_Application", "Backend_Service"),
            ("Backend_Service", "Redis"),
            ("Backend_Service", "MySQL"),
            ("Backend_Service", "Notification_API")
        }
    }
]


# ============================================================
# 3. PROMPT
# ============================================================

system_prompt = """
You are a Graphviz DOT code generator for software architecture diagrams.

Convert the architecture description into a directed Graphviz DOT graph.

Rules:

1. Output ONLY valid Graphviz DOT code.
2. Do not output explanations.
3. Do not output Markdown code fences.
4. Create nodes only for components explicitly mentioned.
5. Create an edge ONLY when that relationship is explicitly stated.
6. NEVER infer or invent relationships.
7. Do not create duplicate edges.
8. Preserve the direction of every relationship.
9. Every component used in an edge must be represented as a node.
10. Use a directed graph with digraph.
"""


# ============================================================
# 4. GENERATE DOT
# ============================================================

def generate_dot(description):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
Convert this software architecture description into Graphviz DOT code.

Architecture description:

{description}

Output only DOT code.
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

    return answer


# ============================================================
# 5. EXTRACT EDGES FROM DOT
# ============================================================

def extract_edges(dot):

    edges = set()

    # Handles:
    # A -> B
    # "A" -> "B"
    # A -> "B"
    # "A" -> B

    pattern = r'"([^"]+)"\s*->\s*"([^"]+)"|([A-Za-z0-9_.-]+)\s*->\s*([A-Za-z0-9_.-]+)'

    matches = re.findall(pattern, dot)

    for match in matches:

        if match[0] and match[1]:
            source = match[0]
            target = match[1]
        else:
            source = match[2]
            target = match[3]

        edges.add(
            (
                source.strip(),
                target.strip()
            )
        )

    return edges


# ============================================================
# 6. EXTRACT NODES
# ============================================================

def extract_nodes(dot, edges):

    nodes = set()

    # Every endpoint of an edge is necessarily a node
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)

    # Also detect explicitly declared nodes
    lines = dot.splitlines()

    for line in lines:

        line = line.strip()

        if "->" in line:
            continue

        if line.startswith("digraph"):
            continue

        if line.startswith("{") or line.startswith("}"):
            continue

        # Quoted node declaration
        match = re.match(r'"([^"]+)"\s*\[', line)

        if match:
            nodes.add(match.group(1))

        # Unquoted node declaration
        match = re.match(r'([A-Za-z0-9_.-]+)\s*\[', line)

        if match:
            nodes.add(match.group(1))

    return nodes


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

def calculate_metrics(expected, predicted):

    true_positive = len(expected & predicted)

    false_positive = len(predicted - expected)

    false_negative = len(expected - predicted)

    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / (
            true_positive + false_positive
        )

    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / (
            true_positive + false_negative
        )

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (
            precision + recall
        )

    union = expected | predicted

    if len(union) == 0:
        jaccard = 1
    else:
        jaccard = len(expected & predicted) / len(union)

    return precision, recall, f1, jaccard


# ============================================================
# 8. RUN EVALUATION
# ============================================================

node_results = []
edge_results = []

print("\n")
print("=" * 70)
print("        QWEN TEXT2ARCH BASELINE EVALUATION")
print("=" * 70)


for i, example in enumerate(examples):

    print(f"\n\nTEST {i + 1}/10")
    print("-" * 70)

    print("Generating DOT...")

    dot = generate_dot(example["description"])

    print("\nGenerated DOT:")
    print(dot)

    predicted_edges = extract_edges(dot)

    predicted_nodes = extract_nodes(
        dot,
        predicted_edges
    )

    expected_nodes = example["nodes"]
    expected_edges = example["edges"]

    # Node metrics
    np_, nr_, nf1, nj = calculate_metrics(
        expected_nodes,
        predicted_nodes
    )

    # Edge metrics
    ep, er, ef1, ej = calculate_metrics(
        expected_edges,
        predicted_edges
    )

    node_results.append(
        (np_, nr_, nf1, nj)
    )

    edge_results.append(
        (ep, er, ef1, ej)
    )

    print("\nExpected Nodes:")
    print(expected_nodes)

    print("\nPredicted Nodes:")
    print(predicted_nodes)

    print("\nExpected Edges:")
    print(expected_edges)

    print("\nPredicted Edges:")
    print(predicted_edges)

    print("\nNode Metrics:")
    print(f"Precision : {np_:.4f}")
    print(f"Recall    : {nr_:.4f}")
    print(f"F1        : {nf1:.4f}")
    print(f"Jaccard   : {nj:.4f}")

    print("\nEdge Metrics:")
    print(f"Precision : {ep:.4f}")
    print(f"Recall    : {er:.4f}")
    print(f"F1        : {ef1:.4f}")
    print(f"Jaccard   : {ej:.4f}")


# ============================================================
# 9. FINAL AVERAGES
# ============================================================

def average(results):

    return [
        sum(x[i] for x in results) / len(results)
        for i in range(4)
    ]


avg_nodes = average(node_results)
avg_edges = average(edge_results)


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n\n")
print("=" * 70)
print("                 FINAL BASELINE RESULTS")
print("=" * 70)

print("\nNODE-LEVEL RESULTS")
print("-" * 70)

print(f"Average Node Precision : {avg_nodes[0] * 100:.2f}%")
print(f"Average Node Recall    : {avg_nodes[1] * 100:.2f}%")
print(f"Average Node F1        : {avg_nodes[2] * 100:.2f}%")
print(f"Average Node Jaccard   : {avg_nodes[3] * 100:.2f}%")

print("\nEDGE-LEVEL RESULTS")
print("-" * 70)

print(f"Average Edge Precision : {avg_edges[0] * 100:.2f}%")
print(f"Average Edge Recall    : {avg_edges[1] * 100:.2f}%")
print(f"Average Edge F1        : {avg_edges[2] * 100:.2f}%")
print(f"Average Edge Jaccard   : {avg_edges[3] * 100:.2f}%")

print("\n")
print("=" * 70)
print("Evaluation completed.")
print("=" * 70)