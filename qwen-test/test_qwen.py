from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# 1. Load Qwen model
# --------------------------------------------------

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto"
)

print("Model loaded successfully!")


# --------------------------------------------------
# 2. Architecture description
# --------------------------------------------------

prompt = """Convert the following software architecture description into valid Graphviz DOT code.

IMPORTANT:
Only create components and relationships explicitly stated in the description.
Do not infer, assume, or invent any additional relationships.
Preserve the direction of every relationship exactly as described.
Do not create duplicate edges.
Output only the DOT code.
Do not output explanations, comments, or Markdown code fences.

Architecture description:

A customer uses a React frontend.
The React frontend communicates with a Spring Boot backend.
The backend stores data in a MySQL database.
The backend uses Redis for caching.
The backend connects to an external payment API.
"""


# --------------------------------------------------
# 3. Instructions given to Qwen
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": """You are a Graphviz DOT code generator for software architecture diagrams.

Your job is to convert an architecture description into a directed Graphviz DOT graph.

Follow these rules strictly:

1. Output ONLY valid Graphviz DOT code.
2. Do not output explanations.
3. Do not output Markdown code fences.
4. Create nodes only for components explicitly mentioned in the description.
5. Create an edge ONLY when that relationship is explicitly stated.
6. NEVER infer or invent relationships.
7. Do not connect two components merely because they exist in the same architecture.
8. Preserve the direction of each relationship exactly as described.
9. Do not create duplicate edges.
10. Every component used in an edge must be represented as a node.
11. Use a directed graph with digraph.
12. The final response must be directly usable as a .dot file.

Example:

Description:
A frontend communicates with a backend.
The backend stores data in a database.

Correct relationships:
Frontend -> Backend
Backend -> Database

Do NOT create:
Frontend -> Database

because that relationship was not explicitly stated."""
    },
    {
        "role": "user",
        "content": prompt
    }
]


# --------------------------------------------------
# 4. Convert messages to Qwen's input format
# --------------------------------------------------

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    [text],
    return_tensors="pt"
)


# --------------------------------------------------
# 5. Generate DOT code
# --------------------------------------------------

print("Generating...")

outputs = model.generate(
    **inputs,
    max_new_tokens=400,
    do_sample=False,
    num_beams=1
)


# --------------------------------------------------
# 6. Extract only Qwen's generated response
# --------------------------------------------------

generated = outputs[0][inputs.input_ids.shape[1]:]

answer = tokenizer.decode(
    generated,
    skip_special_tokens=True
).strip()


# --------------------------------------------------
# 7. Display result
# --------------------------------------------------

print("\n================ QWEN OUTPUT ================\n")
print(answer)
print("\n==============================================")