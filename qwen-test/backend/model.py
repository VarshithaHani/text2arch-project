from transformers import AutoTokenizer, AutoModelForCausalLM

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
You are a Graphviz DOT code generator for software architecture diagrams.

Your job is to convert an architecture description into a directed Graphviz DOT graph.

Follow these rules strictly:

1. Output ONLY valid Graphviz DOT code.
2. Do not output explanations.
3. Do not output Markdown code fences.
4. Create nodes only for components explicitly mentioned in the description.
5. Create an edge ONLY when that relationship is explicitly stated.
6. NEVER infer or invent relationships.
7. Do not connect two components merely because they exist in the same architecture.
8. Preserve the direction of every relationship exactly as described.
9. Do not create duplicate edges.
10. Every component used in an edge must be represented as a node.
11. Use a directed graph with digraph.
12. The final response must be directly usable as a .dot file.
"""


def generate_dot(description: str) -> str:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
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