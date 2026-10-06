// Text2Arch - Frontend + Interactive Architecture Editor
let diagramZoom = 1;
let selectedNode = null;
let selectedEdge = null;
let edgeSourceNode = null;
let editorMode = "select";
let architectureData = {
    nodes: [],
    edges: []
};
const $ = id => document.getElementById(id);
// GENERATE ARCHITECTURE
async function generateArchitecture() {
    const description = $("description").value.trim();
    const button = $("generateBtn");
    const status = $("status");
    const container = $("diagramContainer");
    if (!description) {
        status.textContent = "Please enter an architecture description.";
        return;
    }
    button.disabled = true;
    button.textContent = "Generating...";
    status.textContent = "Generating architecture...";
    container.innerHTML =
        '<p class="placeholder">Generating architecture diagram...</p>';
    try {
        const response = await fetch(
            "http://127.0.0.1:8000/generate",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    description: description
                })
            }
        );
        if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
        }
        const data = await response.json();
        if (!data.dot) {
            throw new Error("No DOT code received from backend.");
        }
        const dot = cleanDOT(data.dot);
        console.log("Original DOT:", dot);
        const graphvizResponse = await fetch(
            "https://quickchart.io/graphviz",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    graph: makeHorizontalDOT(dot),
                    format: "svg"
                })
            }
        );
        if (!graphvizResponse.ok) {
            throw new Error(
                "Could not generate the architecture diagram."
            );
        }
        container.innerHTML =
            await graphvizResponse.text();
        extractArchitectureData(dot);
        prepareDiagram();
        selectedNode = null;
        selectedEdge = null;
        edgeSourceNode = null;
        editorMode = "select";
        diagramZoom = 1;
        updateStatistics();
        updateSelectedNodeDisplay();
        updateEditorStatus(
            "Architecture generated. Select a node to edit it."
        );
        status.textContent =
            "Architecture generated successfully.";
    } catch (error) {
        console.error(error);
        status.textContent =
            "Error: " + error.message;
        container.innerHTML =
            '<p class="placeholder">Failed to generate architecture diagram.</p>';
        architectureData = {
            nodes: [],
            edges: []
        };
        updateStatistics();
    } finally {
        button.disabled = false;
        button.textContent = "Generate Architecture";
    }
}
function cleanDOT(dot) {
    return dot
        .replace(/```(?:plaintext|dot)?/gi, "")
        .trim();
}
function makeHorizontalDOT(dot) {
    if (/rankdir\s*=/i.test(dot)) {
        return dot.replace(
            /rankdir\s*=\s*\w+\s*;/i,
            "rankdir=LR;"
        );
    }
    return dot.replace(
        /^(\s*digraph[^{]*\{)/i,
        "$1\nrankdir=LR;"
    );
}
// DIAGRAM PREPARATION
function prepareDiagram() {
    const svg =
        $("diagramContainer").querySelector("svg");
    if (!svg) return;
    svg.style.maxWidth = "100%";
    svg.style.height = "auto";
    svg.style.display = "block";
    svg.style.margin = "auto";
    svg.style.transformOrigin = "center center";
    // Prepare nodes
    svg.querySelectorAll("g.node").forEach(
        (node, index) => {
            node.style.cursor = "pointer";
            // Give nodes an ID if they don't already have one
            if (!node.id) {
                node.id = "diagram_node_" + index;
            }
            node.addEventListener(
                "click",
                event => {
                    event.stopPropagation();
                    if (editorMode === "addEdge") {
                        handleEdgeNodeClick(node);
                        return;
                    }
                    selectSvgNode(node);
                }
            );
        }
    );
    // Prepare existing Graphviz edges
    svg.querySelectorAll("g.edge").forEach(
        edge => {
            edge.style.cursor = "pointer";
            edge.addEventListener(
                "click",
                event => {
                    event.stopPropagation();
                    if (editorMode === "deleteEdge") {
                        selectEdge(edge);
                    }
                }
            );
        }
    );
    updateDiagramZoom();
}
// NODE SELECTION
function selectSvgNode(node) {
    clearNodeSelection();
    clearEdgeSelection();
    selectedNode = node;
    node.classList.add(
        "text2arch-selected-node"
    );
    const shape =
        node.querySelector(
            "polygon, rect, ellipse, path"
        );
    if (shape) {
        shape.dataset.originalStroke =
            shape.getAttribute("stroke") || "";
        shape.dataset.originalStrokeWidth =
            shape.getAttribute("stroke-width") || "";
        shape.setAttribute(
            "stroke",
            "#2563eb"
        );
        shape.setAttribute(
            "stroke-width",
            "3"
        );
    }
    const label =
        getNodeLabel(node);
    $("selectedNode").textContent =
        label || "Selected";
    updateEditorStatus(
        `Selected node: ${label || "Unknown"}`
    );
}
function clearNodeSelection() {
    $("diagramContainer")
        .querySelectorAll(
            ".text2arch-selected-node"
        )
        .forEach(node => {
            const shape =
                node.querySelector(
                    "polygon, rect, ellipse, path"
                );
            if (shape) {
                if (
                    shape.dataset.originalStroke
                ) {
                    shape.setAttribute(
                        "stroke",
                        shape.dataset.originalStroke
                    );
                } else {
                    shape.removeAttribute(
                        "stroke"
                    );
                }
                if (
                    shape.dataset.originalStrokeWidth
                ) {
                    shape.setAttribute(
                        "stroke-width",
                        shape.dataset.originalStrokeWidth
                    );
                }
            }
            node.classList.remove(
                "text2arch-selected-node"
            );
        });
    selectedNode = null;
    updateSelectedNodeDisplay();
}
function getNodeLabel(node) {
    return [
        ...node.querySelectorAll("text")
    ]
        .map(text => text.textContent.trim())
        .join(" ");
}
// EDIT NODE
function editSelectedNode() {
    if (!selectedNode) {
        updateEditorStatus(
            "Please select a node first."
        );
        return;
    }
    const oldLabel =
        getNodeLabel(selectedNode);
    const newLabel =
        window.prompt(
            "Enter new node label:",
            oldLabel
        );
    if (
        newLabel === null ||
        !newLabel.trim()
    ) {
        updateEditorStatus(
            "Node editing cancelled."
        );
        return;
    }
    const label =
        newLabel.trim();
    const text =
        selectedNode.querySelectorAll("text");
    if (text.length) {
        text[0].textContent = label;
    }
    $("selectedNode").textContent =
        label;
    const nodeData =
        architectureData.nodes.find(
            node => node.label === oldLabel
        );
    if (nodeData) {
        nodeData.label = label;
    }
    updateEditorStatus(
        `Node renamed to: ${label}`
    );
}
// ADD NODE
function addNode() {
    const name = window.prompt("Enter node name:");
    if (!name || !name.trim()) return;

    const nodeName = name.trim();

    const svg = $("diagramContainer").querySelector("svg");
    if (!svg) {
        alert("Generate an architecture diagram first.");
        return;
    }

    const graph = svg.querySelector("g.graph");
    if (!graph) {
        alert("Unable to access the diagram.");
        return;
    }

    const SVG_NS = "http://www.w3.org/2000/svg";

    // Get existing nodes
    const nodes = [...graph.querySelectorAll("g.node")];

    // Find a safe position from the existing diagram
    let maxX = 0;
    let minY = Infinity;
    let maxY = -Infinity;

    nodes.forEach(node => {
        const box = node.getBBox();

        maxX = Math.max(maxX, box.x + box.width);
        minY = Math.min(minY, box.y);
        maxY = Math.max(maxY, box.y + box.height);
    });

    const nodeWidth = 150;
    const nodeHeight = 50;

    // Put new node to the right of existing nodes
    const x = maxX + 30;
    const y = (minY + maxY) / 2 - nodeHeight / 2;

    // Create node group
    const group = document.createElementNS(SVG_NS, "g");

    const nodeId = "custom_" + Date.now();

    group.setAttribute("id", nodeId);
    group.classList.add("node", "custom-node");

    // Rectangle
    const rect = document.createElementNS(SVG_NS, "rect");

    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", nodeWidth);
    rect.setAttribute("height", nodeHeight);
    rect.setAttribute("rx", "8");

    rect.setAttribute("fill", "white");
    rect.setAttribute("stroke", "#2563eb");
    rect.setAttribute("stroke-width", "3");

    // Text
    const text = document.createElementNS(SVG_NS, "text");

    text.setAttribute("x", x + nodeWidth / 2);
    text.setAttribute("y", y + 31);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "14");
    text.setAttribute("font-family", "Arial");
    text.setAttribute("fill", "black");

    text.textContent = nodeName;

    group.appendChild(rect);
    group.appendChild(text);

    graph.appendChild(group);

    // IMPORTANT:
    // Expand the existing viewBox without replacing the original layout
    const oldViewBox = svg.getAttribute("viewBox");

    if (oldViewBox) {
        const parts = oldViewBox.split(/\s+/).map(Number);

        if (parts.length === 4) {
            let [vx, vy, vw, vh] = parts;

            const requiredRight = x + nodeWidth + 30;

            if (requiredRight > vx + vw) {
                vw = requiredRight - vx;
            }

            svg.setAttribute(
                "viewBox",
                `${vx} ${vy} ${vw} ${vh}`
            );
        }
    }

    // Store node
    architectureData.nodes.push({
        id: nodeId,
        label: nodeName,
        type: detectNodeType(nodeName)
    });

    // Make node selectable
    group.addEventListener("click", function(event) {
        event.stopPropagation();

        if (editorMode === "addEdge") {
            handleEdgeNodeClick(group);
            return;
        }
        selectSvgNode(group);
    });
    selectSvgNode(group);
    updateStatistics();
    updateEditorStatus(
        `"${nodeName}" added successfully.`
    );
}
// DELETE NODE
function deleteSelectedNode() {
    if (!selectedNode) {
        updateEditorStatus(
            "Please select a node first."
        );
        return;
    }
    const label =
        getNodeLabel(selectedNode);
    if (
        !confirm(
            `Delete "${label}"?`
        )
    ) {
        return;
    }
    // Remove connected custom edges
    const nodeId =
        getNodeIdentifier(
            selectedNode
        );
    $("diagramContainer")
        .querySelectorAll(
            "g.custom-edge"
        )
        .forEach(edge => {
            const source =
                edge.dataset.source;
            const target =
                edge.dataset.target;
            if (
                source === nodeId ||
                target === nodeId
            ) {
                edge.remove();
            }
        });
    selectedNode.remove();
    architectureData.nodes =
        architectureData.nodes.filter(
            node =>
                node.label !== label
        );
    architectureData.edges =
        architectureData.edges.filter(
            edge =>
                edge.source !== nodeId &&
                edge.target !== nodeId
        );
    selectedNode = null;
    updateSelectedNodeDisplay();
    updateStatistics();
    updateEditorStatus(
        `Deleted node: ${label}`
    );
}
// ADD EDGE
function addEdge() {
    edgeSourceNode = null;
    editorMode = "addEdge";
    clearNodeSelection();
    clearEdgeSelection();
    updateEditorStatus(
        "Add Edge: click the source node first, then the target node."
    );
}
function handleEdgeNodeClick(node) {
    const label =
        getNodeLabel(node);
    if (!edgeSourceNode) {
        edgeSourceNode = node;
        node.classList.add(
            "edge-source-node"
        );
        updateEditorStatus(
            `Source selected: ${label}. Now click the target node.`
        );
        return;
    }
    if (
        edgeSourceNode === node
    ) {
        updateEditorStatus(
            "Source and target cannot be the same node."
        );
        return;
    }
    const sourceLabel =
        getNodeLabel(
            edgeSourceNode
        );
    const targetLabel =
        getNodeLabel(node);
    createInteractiveEdge(
        edgeSourceNode,
        node,
        sourceLabel,
        targetLabel
    );
    edgeSourceNode.classList.remove(
        "edge-source-node"
    );
    edgeSourceNode = null;
    editorMode = "select";
    updateEditorStatus(
        `Connection added: ${sourceLabel} → ${targetLabel}`
    );
}
// CREATE EDGE
function createInteractiveEdge(
    sourceNode,
    targetNode,
    sourceLabel,
    targetLabel
) {
    const svg =
        $("diagramContainer").querySelector("svg");
    if (!svg) return;
    const graph =
        svg.querySelector("g.graph");
    if (!graph) return;
    const sourceBox =
        sourceNode.getBBox();
    const targetBox =
        targetNode.getBBox();
    // Center points
    const x1 =
        sourceBox.x +
        sourceBox.width / 2;
    const y1 =
        sourceBox.y +
        sourceBox.height / 2;
    const x2 =
        targetBox.x +
        targetBox.width / 2;
    const y2 =
        targetBox.y +
        targetBox.height / 2;
    const SVG_NS =
        "http://www.w3.org/2000/svg";
    const edgeGroup =
        document.createElementNS(
            SVG_NS,
            "g"
        );
    edgeGroup.classList.add(
        "edge",
        "custom-edge"
    );
    const sourceId =
        getNodeIdentifier(
            sourceNode
        );
    const targetId =
        getNodeIdentifier(
            targetNode
        );
    edgeGroup.dataset.source =
        sourceId;
    edgeGroup.dataset.target =
        targetId;
    edgeGroup.dataset.sourceLabel =
        sourceLabel;
    edgeGroup.dataset.targetLabel =
        targetLabel;
    // Line
    const line =
        document.createElementNS(
            SVG_NS,
            "line"
        );
    line.setAttribute(
        "x1",
        x1
    );
    line.setAttribute(
        "y1",
        y1
    );
    line.setAttribute(
        "x2",
        x2
    );
    line.setAttribute(
        "y2",
        y2
    );
    line.setAttribute(
        "stroke",
        "#555"
    );
    line.setAttribute(
        "stroke-width",
        "2"
    );
    // Arrow
    const arrow =
        document.createElementNS(
            SVG_NS,
            "polygon"
        );
    const angle =
        Math.atan2(
            y2 - y1,
            x2 - x1
        );
    const size = 8;
    const p1x =
        x2 -
        size *
        Math.cos(
            angle - Math.PI / 6
        );
    const p1y =
        y2 -
        size *
        Math.sin(
            angle - Math.PI / 6
        );
    const p2x =
        x2 -
        size *
        Math.cos(
            angle + Math.PI / 6
        );
    const p2y =
        y2 -
        size *
        Math.sin(
            angle + Math.PI / 6
        );
    arrow.setAttribute(
        "points",
        `${x2},${y2} ${p1x},${p1y} ${p2x},${p2y}`
    );
    arrow.setAttribute(
        "fill",
        "#555"
    );
    edgeGroup.appendChild(
        line
    );
    edgeGroup.appendChild(
        arrow
    );
    // Put edge behind nodes
    graph.insertBefore(
        edgeGroup,
        graph.firstChild
    );
    // Add to architecture data
    architectureData.edges.push({
        source: sourceId,
        target: targetId,
        sourceLabel: sourceLabel,
        targetLabel: targetLabel
    });
    // Make custom edge selectable
    edgeGroup.addEventListener(
        "click",
        event => {
            event.stopPropagation();
            if (
                editorMode === "deleteEdge"
            ) {
                selectEdge(
                    edgeGroup
                );
            }
        }
    );
    updateStatistics();
}
// GET NODE IDENTIFIER
function getNodeIdentifier(node) {
    if (node.id) {
        return node.id;
    }
    const title =
        node.querySelector("title");
    if (
        title &&
        title.textContent.trim()
    ) {
        return title.textContent.trim();
    }
    return getNodeLabel(node);
}
// SELECT EDGE
function selectEdge(edge) {
    clearEdgeSelection();
    selectedEdge = edge;
    edge.classList.add(
        "text2arch-selected-edge"
    );
    const path =
        edge.querySelector(
            "path, line"
        );
    if (path) {
        path.dataset.originalStroke =
            path.getAttribute("stroke") || "";
        path.dataset.originalStrokeWidth =
            path.getAttribute(
                "stroke-width"
            ) || "";
        path.setAttribute(
            "stroke",
            "#2563eb"
        );
        path.setAttribute(
            "stroke-width",
            "4"
        );
    }
    updateEditorStatus(
        "Edge selected. Click Delete Edge again to remove it."
    );
}
function clearEdgeSelection() {
    $("diagramContainer")
        .querySelectorAll(
            ".text2arch-selected-edge"
        )
        .forEach(edge => {
            const path =
                edge.querySelector(
                    "path, line"
                );
            if (path) {
                if (
                    path.dataset.originalStroke
                ) {
                    path.setAttribute(
                        "stroke",
                        path.dataset.originalStroke
                    );
                }
                if (
                    path.dataset.originalStrokeWidth
                ) {
                    path.setAttribute(
                        "stroke-width",
                        path.dataset.originalStrokeWidth
                    );
                }
            }
            edge.classList.remove(
                "text2arch-selected-edge"
            );
        });
    selectedEdge = null;
}
// DELETE EDGE
function deleteEdge() {
    if (!selectedEdge) {
        editorMode = "deleteEdge";
        clearNodeSelection();
        clearEdgeSelection();
        updateEditorStatus(
            "Delete Edge: click the edge you want to delete."
        );
        return;
    }
    // Find selected edge
    const edge =
        selectedEdge;
    if (
        !confirm(
            "Delete this connection?"
        )
    ) {
        return;
    }
    const source =
        edge.dataset.source;
    const target =
        edge.dataset.target;
    // Remove visual edge
    edge.remove();
    // Remove from architecture data
    architectureData.edges =
        architectureData.edges.filter(
            item =>
                !(
                    item.source === source &&
                    item.target === target
                )
        );
    selectedEdge = null;
    editorMode = "select";
    updateStatistics();
    updateEditorStatus(
        "Edge deleted successfully."
    );
}
// ZOOM
function updateDiagramZoom() {
    const svg =
        $("diagramContainer").querySelector(
            "svg"
        );
    if (svg) {
        svg.style.transform =
            `scale(${diagramZoom})`;
    }
}
function zoomIn() {
    diagramZoom =
        Math.min(
            2,
            diagramZoom + 0.1
        );
    updateDiagramZoom();
    updateEditorStatus(
        `Zoom: ${Math.round(
            diagramZoom * 100
        )}%`
    );
}
function zoomOut() {
    diagramZoom =
        Math.max(
            0.5,
            diagramZoom - 0.1
        );
    updateDiagramZoom();
    updateEditorStatus(
        `Zoom: ${Math.round(
            diagramZoom * 100
        )}%`
    );
}
function resetZoom() {
    diagramZoom = 1;
    updateDiagramZoom();
    updateEditorStatus(
        "Zoom reset to 100%."
    );
}
// DATA EXTRACTION
function extractArchitectureData(dot) {
    const nodes = [];
    const edges = [];
    const nodeIds =
        new Set();
    const nodeRegex =
        /([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*[^]]*label\s*=\s*(?:"([^"]*)"|([^,\]]+))/gi;
    let match;
    while (
        (match = nodeRegex.exec(dot)) !== null
    ) {
        const id =
            match[1];
        const label =
            (
                match[2] ||
                match[3] ||
                ""
            )
                .trim()
                .replace(
                    /^["']|["']$/g,
                    ""
                );
        if (
            id !== "node" &&
            id !== "edge" &&
            label &&
            label.toLowerCase() !== "null"
        ) {
            nodeIds.add(id);
            nodes.push({
                id: id,
                label: label,
                type: detectNodeType(
                    label
                )
            });
        }
    }
    // Extract edges
    const edgeRegex =
        /([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)/g;
    while (
        (match = edgeRegex.exec(dot)) !== null
    ) {
        const source =
            match[1];
        const target =
            match[2];
        edges.push({
            source: source,
            target: target
        });
        [source, target].forEach(
            id => {
                if (
                    !nodeIds.has(id)
                ) {
                    nodeIds.add(id);
                    nodes.push({
                        id: id,
                        label:
                            id.replace(
                                /_/g,
                                " "
                            ),
                        type:
                            detectNodeType(id)
                    });
                }
            }
        );
    }
    architectureData = {
        nodes: nodes,
        edges: edges
    };
    console.log(
        "Architecture data:",
        architectureData
    );
}
// NODE TYPE
function detectNodeType(label) {
    const text =
        label.toLowerCase();
    if (
        [
            "frontend",
            "react",
            "angular",
            "vue",
            "client"
        ].some(
            word =>
                text.includes(word)
        )
    ) {
        return "frontend";
    }
    if (
        [
            "backend",
            "server",
            "spring",
            "flask",
            "fastapi",
            "node"
        ].some(
            word =>
                text.includes(word)
        )
    ) {
        return "backend";
    }
    if (
        [
            "database",
            "mysql",
            "postgres",
            "mongodb",
            "sql"
        ].some(
            word =>
                text.includes(word)
        )
    ) {
        return "database";
    }
    if (
        [
            "service",
            "api",
            "cloud",
            "payment"
        ].some(
            word =>
                text.includes(word)
        )
    ) {
        return "service";
    }
    return "other";
}
// UI
function updateStatistics() {
    $("componentCount").textContent =
        architectureData.nodes.length;
    $("connectionCount").textContent =
        architectureData.edges.length;
    $("architectureStatus").textContent =
        architectureData.nodes.length
            ? "Generated"
            : "Not Generated";
}
function updateSelectedNodeDisplay() {
    const element =
        $("selectedNode");
    if (
        element &&
        !selectedNode
    ) {
        element.textContent =
            "None";
    }
}
function updateEditorStatus(message) {
    const element =
        $("editorStatus");
    if (element) {
        element.textContent =
            message;
    }
}
// EXPORT SVG
function downloadSVG() {
    const svg =
        $("diagramContainer").querySelector(
            "svg"
        );
    if (!svg) {
        alert(
            "Please generate a diagram first."
        );
        return;
    }
    const svgData =
        new XMLSerializer()
            .serializeToString(svg);
    const blob =
        new Blob(
            [svgData],
            {
                type:
                    "image/svg+xml"
            }
        );
    const url =
        URL.createObjectURL(blob);
    const link =
        document.createElement("a");
    link.href = url;
    link.download =
        "text2arch-diagram.svg";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    updateEditorStatus(
        "SVG diagram exported successfully."
    );
}
// EXPORT JSON
function exportJSON() {
    if (
        !architectureData.nodes.length
    ) {
        alert(
            "Please generate a diagram first."
        );
        return;
    }
    const blob =
        new Blob(
            [
                JSON.stringify(
                    architectureData,
                    null,
                    2
                )
            ],
            {
                type:
                    "application/json"
            }
        );
    const url =
        URL.createObjectURL(blob);
    const link =
        document.createElement("a");
    link.href = url;
    link.download =
        "text2arch-architecture.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    updateEditorStatus(
        "Architecture JSON exported successfully."
    );
}
// BUTTONS
document.addEventListener(
    "click",
    event => {
        const button =
            event.target.closest(
                "button"
            );
        if (!button) return;
        switch (button.id) {
            case "zoomInBtn":
                zoomIn();
                break;
            case "zoomOutBtn":
                zoomOut();
                break;
            case "resetZoomBtn":
                resetZoom();
                break;
            case "selectNodeBtn":
                editorMode =
                    "select";
                edgeSourceNode =
                    null;
                clearEdgeSelection();
                updateEditorStatus(
                    "Select mode enabled. Click a node in the diagram."
                );
                break;
            case "editNodeBtn":
                editorMode =
                    "select";
                editSelectedNode();
                break;
            case "addNodeBtn":
                editorMode =
                    "select";
                addNode();
                break;
            case "deleteNodeBtn":
                editorMode =
                    "select";
                deleteSelectedNode();
                break;
            case "addEdgeBtn":
                addEdge();
                break;
            case "deleteEdgeBtn":
                deleteEdge();
                break;
            case "downloadSvgBtn":
                downloadSVG();
                break;
            case "exportJsonBtn":
                exportJSON();
                break;
        }
    }
);
// INITIAL STATE
document.addEventListener(
    "DOMContentLoaded",
    () => {
        updateStatistics();
        updateSelectedNodeDisplay();
        updateEditorStatus(
            "Ready"
        );
    }
);