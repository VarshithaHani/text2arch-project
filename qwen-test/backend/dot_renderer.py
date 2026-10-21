
from pathlib import Path
from graphviz import Source


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_diagram(
    dot_code: str,
    output_name: str = "architecture",
    output_format: str = "svg",
) -> str:
    """Render DOT code locally and return the generated file path."""

    if not dot_code.strip():
        raise ValueError("DOT code cannot be empty.")

    if output_format not in {"svg", "png"}:
        raise ValueError("Output format must be svg or png.")

    output_path = OUTPUT_DIR / output_name

    graph = Source(dot_code)
    rendered_path = graph.render(
        filename=str(output_path),
        format=output_format,
        cleanup=True,
    )

    return rendered_path
