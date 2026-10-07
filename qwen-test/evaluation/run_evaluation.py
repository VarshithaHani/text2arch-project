import csv
import json
import urllib.request

from evaluate import evaluate_prediction


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = input(
    "Enter Ollama model name (example: qwen3:8b): "
).strip()


def generate_with_ollama(description):

    prompt = f"""
You are a Graphviz DOT code generator for software architecture diagrams.

Convert the following architecture description into a directed Graphviz DOT graph.

Rules:
1. Output only valid DOT code.
2. Do not use Markdown code fences.
3. Do not provide explanations.
4. Create nodes only for components explicitly mentioned.
5. Create an edge only when the relationship is explicitly stated.
6. Do not invent relationships.
7. Preserve relationship direction.
8. Do not create duplicate edges.

Architecture description:

{description}
"""

    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


def main():

    dataset_path = "data/manual.tsv"

    with open(
        dataset_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for i, row in enumerate(reader):

            description = row["Cleaned Description"]
            ground_truth_dot = row["Dot code"]

            print(f"\n===== SAMPLE {i + 1} =====")

            print("\nDescription:")
            print(description[:300])

            print("\nGenerating DOT...")

            predicted_dot = generate_with_ollama(
                description
            )

            print("\n===== GENERATED DOT =====")
            print(predicted_dot)

            print("\n===== GROUND TRUTH DOT =====")
            print(ground_truth_dot[:500])

            print("\nEvaluation completed.")

            # First test only 5 samples
            if i == 4:
                break


if __name__ == "__main__":
    main()