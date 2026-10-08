from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import generate_architecture, architecture_to_dot


app = FastAPI(
    title="Text2Arch",
    description="Natural Language to Software Architecture Diagram",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ArchitectureRequest(BaseModel):
    description: str


@app.get("/")
def root():
    return {
        "message": "Text2Arch backend is running"
    }


@app.post("/generate")
def generate_architecture_endpoint(
    request: ArchitectureRequest
):
    architecture = generate_architecture(
        request.description
    )

    dot = architecture_to_dot(
        architecture
    )

    return {
        "architecture": architecture,
        "dot": dot
    }