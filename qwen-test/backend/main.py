
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model import generate_architecture, architecture_to_dot
from dot_renderer import render_diagram


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Text2Arch",
    description="Natural Language to Software Architecture Diagram",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


class ArchitectureRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=5000)


@app.get("/")
def root():
    return {"message": "Text2Arch backend is running"}


@app.post("/generate")
def generate_architecture_endpoint(request: ArchitectureRequest):
    description = request.description.strip()

    if not description:
        raise HTTPException(
            status_code=422,
            detail="Description cannot be empty.",
        )

    try:
        architecture = generate_architecture(description)
        dot = architecture_to_dot(architecture)

        image_path = render_diagram(
            dot,
            output_name="architecture",
            output_format="svg",
        )

        return {
            "architecture": architecture,
            "dot": dot,
            "image_url": f"/outputs/{Path(image_path).name}",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Architecture generation failed: {exc}",
        ) from exc
