async function generateArchitecture() {
const description = document.getElementById("description").value.trim();
const button = document.getElementById("generateBtn");
const status = document.getElementById("status");
const diagramContainer = document.getElementById("diagramContainer");

if (!description) {
    status.textContent = "Please enter an architecture description.";
    return;
}

button.disabled = true;
button.textContent = "Generating...";
status.textContent = "Generating architecture...";
diagramContainer.innerHTML = "";

try {
    // Send description to FastAPI backend
    const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            description: description
        })
    });

    if (!response.ok) {
        throw new Error(
            `Backend error: ${response.status} ${response.statusText}`
        );
    }

    // Read JSON response
    const data = await response.json();

    console.log("Backend response:", data);

    // Get DOT code from backend
    const dotCode = data.dot;

    if (!dotCode) {
        throw new Error("No DOT code received from backend.");
    }

    // Remove Markdown code fences
    const cleanedDot = dotCode
        .replace(/```plaintext/g, "")
        .replace(/```/g, "")
        .trim();

    console.log("Clean DOT:", cleanedDot);

    // Send DOT code to Graphviz renderer
    const graphvizResponse = await fetch(
        "https://quickchart.io/graphviz",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                graph: cleanedDot,
                format: "svg"
            })
        }
    );

    if (!graphvizResponse.ok) {
        throw new Error("Could not generate the architecture diagram.");
    }

    // Get generated SVG
    const svg = await graphvizResponse.text();

    // Display diagram
    diagramContainer.innerHTML = svg;

    status.textContent = "Architecture generated successfully.";

} catch (error) {
    console.error("Error:", error);

    status.textContent = "Error: " + error.message;

    diagramContainer.innerHTML = `
        <p class="placeholder">
            Failed to generate architecture diagram.
        </p>
    `;
} finally {
    button.disabled = false;
    button.textContent = "Generate Architecture";
}

}
