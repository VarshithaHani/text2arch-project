import csv
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from evaluate import (
    evaluate_prediction,
    extract_ground_truth
)


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

# Works from any launch directory.
# Assumes this file is in qwen-test/evaluation/
# and data is in qwen-test/data/
DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "manual.tsv"
)

# Keep testing limited to 5 samples
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
# SYSTEM PROMPT
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

    # Move inputs to the model device
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            num_beams=1
        )

    generated_tokens = (
        outputs[0][
            inputs["input_ids"].shape[1]:
        ]
    )

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------------
    # Remove Markdown code fences if generated
    # --------------------------------------------------------

    if answer.startswith("```"):

        lines = answer.splitlines()

        if (
            lines
            and lines[0].strip().startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        answer = "\n".join(
            lines
        ).strip()

    return answer


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics
):

    print(
        f"\n--- {title} ---"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Jaccard   : "
        f"{metrics['jaccard']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("TEXT2ARCH EVALUATION")
    print("======================================")

    print(
        f"Model   : {MODEL_NAME}"
    )

    print(
        f"Dataset : {DATASET_PATH}"
    )

    print(
        f"Samples : {MAX_SAMPLES}"
    )

    print("======================================")

    results = []

    # --------------------------------------------------------
    # Read dataset
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Evaluate first 5 samples
        # ----------------------------------------------------

        for sample_number, row in enumerate(
            reader,
            start=1
        ):

            if sample_number > MAX_SAMPLES:
                break

            print("\n======================================")
            print(
                f"SAMPLE {sample_number}"
            )
            print("======================================")

            try:

                description = row[
                    "Cleaned Description"
                ]

                ground_truth_dot = row[
                    "Dot code"
                ]

                # ------------------------------------------------
                # Temporary debugging:
                # show the actual ground-truth DOT
                # ------------------------------------------------

                print(
                    "\n===== RAW GROUND-TRUTH DOT "
                    "(first 400 chars) ====="
                )

                print(
                    ground_truth_dot[:400]
                )

                # ------------------------------------------------
                # 1. Generate prediction
                # ------------------------------------------------

                print(
                    "\nGenerating DOT..."
                )

                predicted_dot = generate_dot(
                    description
                )

                # ------------------------------------------------
                # 2. Validate generated output
                # ------------------------------------------------

                if not predicted_dot.strip():

                    raise ValueError(
                        "Model returned empty output."
                    )

                print(
                    "\n===== GENERATED DOT ====="
                )

                print(
                    predicted_dot
                )

                # ------------------------------------------------
                # 3. Extract ground truth
                # ------------------------------------------------

                expected_nodes, expected_edges = (
                    extract_ground_truth(
                        ground_truth_dot
                    )
                )

                print(
                    "\n===== GROUND TRUTH ====="
                )

                print(
                    "Expected nodes:",
                    expected_nodes
                )

                print(
                    "Expected edges:",
                    expected_edges
                )

                # ------------------------------------------------
                # 4. ACTUALLY EVALUATE PREDICTION
                # ------------------------------------------------

                evaluation_result = (
                    evaluate_prediction(
                        expected_nodes,
                        expected_edges,
                        predicted_dot
                    )
                )

                # ------------------------------------------------
                # 5. Predicted graph
                # ------------------------------------------------

                print(
                    "\n===== PREDICTED GRAPH ====="
                )

                print(
                    "Predicted nodes:",
                    evaluation_result[
                        "predicted_nodes"
                    ]
                )

                print(
                    "Predicted edges:",
                    evaluation_result[
                        "predicted_edges"
                    ]
                )

                # ------------------------------------------------
                # 6. Node metrics
                # ------------------------------------------------

                node_metrics = (
                    evaluation_result[
                        "node_metrics"
                    ]
                )

                print_metrics(
                    "NODE METRICS",
                    node_metrics
                )

                # ------------------------------------------------
                # 7. Edge metrics
                # ------------------------------------------------

                edge_metrics = (
                    evaluation_result[
                        "edge_metrics"
                    ]
                )

                print_metrics(
                    "EDGE METRICS",
                    edge_metrics
                )

                # ------------------------------------------------
                # 8. Store results
                # ------------------------------------------------

                results.append({

                    "sample":
                        sample_number,

                    "node_precision":
                        node_metrics[
                            "precision"
                        ],

                    "node_recall":
                        node_metrics[
                            "recall"
                        ],

                    "node_f1":
                        node_metrics[
                            "f1"
                        ],

                    "node_jaccard":
                        node_metrics[
                            "jaccard"
                        ],

                    "edge_precision":
                        edge_metrics[
                            "precision"
                        ],

                    "edge_recall":
                        edge_metrics[
                            "recall"
                        ],

                    "edge_f1":
                        edge_metrics[
                            "f1"
                        ],

                    "edge_jaccard":
                        edge_metrics[
                            "jaccard"
                        ]
                })

            # ----------------------------------------------------
            # Failed/invalid prediction
            # ----------------------------------------------------

            except Exception as error:

                print(
                    f"\nFAILED SAMPLE "
                    f"{sample_number}"
                )

                print(
                    f"Reason: {error}"
                )

                print(
                    "Skipping this sample "
                    "and continuing."
                )

                continue

    # ========================================================
    # FINAL AVERAGES
    # ========================================================

    print("\n======================================")
    print("FINAL RESULTS")
    print("======================================")

    if not results:

        print(
            "\nNo samples were successfully evaluated."
        )

        return

    # --------------------------------------------------------
    # Node averages
    # --------------------------------------------------------

    avg_node_precision = sum(
        result["node_precision"]
        for result in results
    ) / len(results)

    avg_node_recall = sum(
        result["node_recall"]
        for result in results
    ) / len(results)

    avg_node_f1 = sum(
        result["node_f1"]
        for result in results
    ) / len(results)

    avg_node_jaccard = sum(
        result["node_jaccard"]
        for result in results
    ) / len(results)

    # --------------------------------------------------------
    # Edge averages
    # --------------------------------------------------------

    avg_edge_precision = sum(
        result["edge_precision"]
        for result in results
    ) / len(results)

    avg_edge_recall = sum(
        result["edge_recall"]
        for result in results
    ) / len(results)

    avg_edge_f1 = sum(
        result["edge_f1"]
        for result in results
    ) / len(results)

    avg_edge_jaccard = sum(
        result["edge_jaccard"]
        for result in results
    ) / len(results)

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print(
        f"\nSuccessfully evaluated: "
        f"{len(results)} / {MAX_SAMPLES}"
    )

    print("\nNODE RESULTS")

    print(
        f"Precision : "
        f"{avg_node_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{avg_node_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{avg_node_f1:.4f}"
    )

    print(
        f"Jaccard   : "
        f"{avg_node_jaccard:.4f}"
    )

    print("\nEDGE RESULTS")

    print(
        f"Precision : "
        f"{avg_edge_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{avg_edge_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{avg_edge_f1:.4f}"
    )

    print(
        f"Jaccard   : "
        f"{avg_edge_jaccard:.4f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()