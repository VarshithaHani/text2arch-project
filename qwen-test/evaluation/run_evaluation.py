import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate_prediction, extract_ground_truth


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = input(
    "Enter Hugging Face model name "
    "(example: Qwen/Qwen3-8B): "
).strip()


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = "data/manual.tsv"

# Keep testing limited to 5 samples for now
MAX_SAMPLES = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a Graphviz DOT code generator for software architecture diagrams.

Your job is to convert an architecture description into a directed Graphviz DOT graph.

Rules:

1. Output ONLY valid Graphviz DOT code.
2. Do not output explanations.
3. Do not output Markdown code fences.
4. Create nodes only for components explicitly mentioned.
5. Create an edge ONLY when that relationship is explicitly stated.
6. NEVER infer or invent relationships.
7. Preserve the direction of every relationship exactly as described.
8. Do not create duplicate edges.
9. Every component used in an edge must be represented as a node.
10. Use a directed graph with digraph.
"""


# ============================================================
# GENERATE DOT
# ============================================================

def generate_dot(description):

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

    # Move inputs to the same device as the model
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # Remove Markdown fences if the model produces them
    if answer.startswith("```"):

        lines = answer.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        answer = "\n".join(lines).strip()

    return answer


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(title, metrics):

    print(f"\n--- {title} ---")

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['f1']:.4f}"
    )

    print(
        f"Jaccard   : {metrics['jaccard']:.4f}"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n======================================")
    print("TEXT2ARCH EVALUATION")
    print("======================================")

    print("Model   :", MODEL_NAME)
    print("Dataset :", DATASET_PATH)
    print("Samples :", MAX_SAMPLES)

    print("======================================\n")

    results = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for sample_number, row in enumerate(
            reader,
            start=1
        ):

            # Keep only first 5 samples
            if sample_number > MAX_SAMPLES:
                break

            description = row[
                "Cleaned Description"
            ]

            ground_truth_dot = row[
                "Dot code"
            ]

            print("\n======================================")
            print(
                f"SAMPLE {sample_number}"
            )
            print("======================================")

            try:

                # --------------------------------
                # 1. Generate prediction
                # --------------------------------

                print("\nGenerating DOT...")

                predicted_dot = generate_dot(
                    description
                )

                print("\n===== GENERATED DOT =====")
                print(predicted_dot)

                # --------------------------------
                # 2. Extract ground truth
                # --------------------------------

                expected_nodes, expected_edges = (
                    extract_ground_truth(
                        ground_truth_dot
                    )
                )

                # --------------------------------
                # 3. ACTUALLY EVALUATE
                # --------------------------------

                evaluation_result = (
                    evaluate_prediction(
                        expected_nodes,
                        expected_edges,
                        predicted_dot
                    )
                )

                # --------------------------------
                # 4. Get metrics
                # --------------------------------

                node_metrics = (
                    evaluation_result[
                        "node_metrics"
                    ]
                )

                edge_metrics = (
                    evaluation_result[
                        "edge_metrics"
                    ]
                )

                # --------------------------------
                # 5. Display prediction
                # --------------------------------

                print(
                    "\n===== PREDICTED NODES ====="
                )

                print(
                    evaluation_result[
                        "predicted_nodes"
                    ]
                )

                print(
                    "\n===== PREDICTED EDGES ====="
                )

                print(
                    evaluation_result[
                        "predicted_edges"
                    ]
                )

                # --------------------------------
                # 6. Display NODE metrics
                # --------------------------------

                print_metrics(
                    "NODE METRICS",
                    node_metrics
                )

                # --------------------------------
                # 7. Display EDGE metrics
                # --------------------------------

                print_metrics(
                    "EDGE METRICS",
                    edge_metrics
                )

                # --------------------------------
                # 8. Save result
                # --------------------------------

                results.append({
                    "sample": sample_number,
                    "node_precision":
                        node_metrics["precision"],
                    "node_recall":
                        node_metrics["recall"],
                    "node_f1":
                        node_metrics["f1"],
                    "node_jaccard":
                        node_metrics["jaccard"],
                    "edge_precision":
                        edge_metrics["precision"],
                    "edge_recall":
                        edge_metrics["recall"],
                    "edge_f1":
                        edge_metrics["f1"],
                    "edge_jaccard":
                        edge_metrics["jaccard"]
                })

            except Exception as error:

                print(
                    f"\nERROR in sample "
                    f"{sample_number}:"
                )

                print(error)

    # ========================================================
    # AVERAGES
    # ========================================================

    if not results:

        print(
            "\nNo samples were successfully evaluated."
        )

        return

    average_node_precision = sum(
        r["node_precision"]
        for r in results
    ) / len(results)

    average_node_recall = sum(
        r["node_recall"]
        for r in results
    ) / len(results)

    average_node_f1 = sum(
        r["node_f1"]
        for r in results
    ) / len(results)

    average_node_jaccard = sum(
        r["node_jaccard"]
        for r in results
    ) / len(results)

    average_edge_precision = sum(
        r["edge_precision"]
        for r in results
    ) / len(results)

    average_edge_recall = sum(
        r["edge_recall"]
        for r in results
    ) / len(results)

    average_edge_f1 = sum(
        r["edge_f1"]
        for r in results
    ) / len(results)

    average_edge_jaccard = sum(
        r["edge_jaccard"]
        for r in results
    ) / len(results)

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("======================================")
    print("FINAL RESULTS")
    print("======================================")

    print(
        f"\nSuccessful samples: "
        f"{len(results)}"
    )

    print("\nNODE RESULTS")

    print(
        f"Precision : "
        f"{average_node_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{average_node_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{average_node_f1:.4f}"
    )

    print(
        f"Jaccard   : "
        f"{average_node_jaccard:.4f}"
    )

    print("\nEDGE RESULTS")

    print(
        f"Precision : "
        f"{average_edge_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{average_edge_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{average_edge_f1:.4f}"
    )

    print(
        f"Jaccard   : "
        f"{average_edge_jaccard:.4f}"
    )


if __name__ == "__main__":
    main()