// Text2Arch - Frontend + Interactive Architecture Editor
// Keeps the existing FastAPI endpoint and Graphviz/QuickChart SVG renderer.

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
const SVG_NS = "http://www.w3.org/2000/svg";

// =====================================================
// GENERATE ARCHITECTURE
// =====================================================

async function generateArchitecture() {
    const descriptionElement = $("description");
    const button = $("generateBtn");
    const status = $("status");
    const container = $("diagramContainer");

    if (!descriptionElement || !button || !status || !container) {
        console.error("Text2Arch is missing a required HTML element.");
        return;
    }

    const description = descriptionElement.value.trim();

    if (!description) {
        status.textContent = "Please enter an architecture description.";
        return;
    }

    button.disabled = true;
    button.textContent = "Generating...";
    status.textContent = "Generating architecture...";
    container.innerHTML =
        '<p class="placeholder">Generating architecture diagram...</p>';

    architectureData = { nodes: [], edges: [] };
    selectedNode = null;
    selectedEdge = null;
    edgeSourceNode = null;
    editorMode = "select";
    diagramZoom = 1;
    updateStatistics();
    updateSelectedNodeDisplay();

    try {
        let response;

        try {
            response = await fetch("http://127.0.0.1:8000/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ description })
            });
        } catch (networkError) {
            throw new Error(
                "Backend is not reachable. Make sure the FastAPI server is running."
            );
        }

        if (!response.ok) {
            let message = "";
            try {
                const errorData = await response.json();
                message = errorData.detail || errorData.message || "";
            } catch (_) {
                // Use the HTTP status below if the response is not JSON.
            }
            throw new Error(message || `Backend returned HTTP ${response.status}.`);
        }

        const data = await response.json();

        if (!data || typeof data.dot !== "string" || !data.dot.trim()) {
            throw new Error("Backend response does not contain valid DOT code.");
        }

        const dot = cleanDOT(data.dot);

        if (!isValidDOT(dot)) {
            throw new Error("The AI returned invalid or incomplete DOT code.");
        }

        console.log("Clean DOT:", dot);

        let graphvizResponse;

        try {
            graphvizResponse = await fetch("https://quickchart.io/graphviz", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    graph: makeHorizontalDOT(dot),
                    format: "svg"
                })
            });
        } catch (renderError) {
            throw new Error("Unable to connect to the diagram rendering service.");
        }

        if (!graphvizResponse.ok) {
            throw new Error(
                `Diagram rendering failed (HTTP ${graphvizResponse.status}).`
            );
        }

        const svgText = await graphvizResponse.text();

        if (!svgText || !/<svg[\s>]/i.test(svgText)) {
            throw new Error("Diagram renderer returned an invalid SVG.");
        }

        // The SVG is returned by the trusted diagram renderer.
        container.innerHTML = svgText;

        const svg = container.querySelector("svg");
        if (!svg) {
            throw new Error("Generated diagram SVG could not be loaded.");
        }

        prepareDiagram();
        syncArchitectureDataFromSVG();

        selectedNode = null;
        selectedEdge = null;
        edgeSourceNode = null;
        editorMode = "select";
        diagramZoom = 1;

        updateDiagramZoom();
        updateStatistics();
        updateSelectedNodeDisplay();
        updateEditorStatus("Architecture generated. Select a node to edit it.");
        status.textContent = "Architecture generated successfully.";
    } catch (error) {
        console.error("Text2Arch generation error:", error);
        status.textContent = "Error: " + error.message;
        container.innerHTML =
            `<p class="placeholder">${escapeHTML(error.message)}</p>`;

        architectureData = { nodes: [], edges: [] };
        selectedNode = null;
        selectedEdge = null;
        edgeSourceNode = null;
        editorMode = "select";

        updateStatistics();
        updateSelectedNodeDisplay();
    } finally {
        button.disabled = false;
        button.textContent = "Generate Architecture";
    }
}

// =====================================================
// DOT CLEANING AND VALIDATION
// =====================================================

function cleanDOT(dot) {
    return String(dot)
        .replace(/```(?:plaintext|dot|graphviz)?/gi, "")
        .replace(/```/g, "")
        .replace(/-\|->/g, "->")
        .replace(/\|->/g, "->")
        .trim();
}

function isValidDOT(dot) {
    if (!dot || !/^\s*(strict\s+)?(di)?graph\b/i.test(dot)) {
        return false;
    }

    if (!dot.includes("{") || !dot.includes("}")) {
        return false;
    }

    return true;
}

function makeHorizontalDOT(dot) {
    if (/rankdir\s*=/i.test(dot)) {
        return dot.replace(
            /rankdir\s*=\s*\w+\s*;?/i,
            "rankdir=LR;"
        );
    }

    return dot.replace(
        /^(\s*(?:strict\s+)?digraph\b[^{]*\{)/i,
        "$1\nrankdir=LR;"
    );
}

// =====================================================
// DIAGRAM PREPARATION
// =====================================================

function prepareDiagram() {
    const svg = $("diagramContainer")?.querySelector("svg");
    if (!svg) return;

    svg.style.maxWidth = "100%";
    svg.style.height = "auto";
    svg.style.display = "block";
    svg.style.margin = "auto";
    svg.style.transformOrigin = "center center";

    svg.querySelectorAll("g.node").forEach((node, index) => {
        node.style.cursor = "pointer";

        if (!node.id) {
            node.id = `diagram_node_${index}`;
        }

        // Avoid attaching duplicate handlers if preparation is called again.
        if (node.dataset.text2archNodeHandler === "true") return;
        node.dataset.text2archNodeHandler = "true";

        node.addEventListener("click", event => {
            event.stopPropagation();

            if (editorMode === "addEdge") {
                handleEdgeNodeClick(node);
                return;
            }

            if (editorMode === "deleteEdge") {
                updateEditorStatus("Delete Connection mode: click a connection, not a node.");
                return;
            }

            selectSvgNode(node);
        });
    });

    svg.querySelectorAll("g.edge").forEach(edge => {
        edge.style.cursor = "pointer";

        if (edge.dataset.text2archEdgeHandler === "true") return;
        edge.dataset.text2archEdgeHandler = "true";

        edge.addEventListener("click", event => {
            event.stopPropagation();
            if (editorMode === "deleteEdge") {
                selectEdge(edge);
            }
        });
    });

    updateDiagramZoom();
}

// =====================================================
// NODE HELPERS AND SELECTION
// =====================================================

function getNodeLabel(node) {
    if (!node) return "";

    return [...node.querySelectorAll("text")]
        .map(text => text.textContent.trim())
        .filter(Boolean)
        .join(" ");
}

function getNodeIdentifier(node) {
    if (!node) return "";

    // Custom nodes have IDs created by this editor.
    if (node.classList.contains("custom-node")) {
        return node.id;
    }

    // Graphviz SVG stores the original DOT node name in a child <title>.
    const title = node.querySelector(":scope > title");
    if (title && title.textContent.trim()) {
        return title.textContent.trim();
    }

    return node.id || getNodeLabel(node);
}

function selectSvgNode(node) {
    clearNodeSelection();
    clearEdgeSelection();

    selectedNode = node;
    node.classList.add("text2arch-selected-node");

    const shape = node.querySelector("polygon, rect, ellipse, path");
    if (shape) {
        if (shape.dataset.originalStroke === undefined) {
            shape.dataset.originalStroke = shape.getAttribute("stroke") || "";
        }
        if (shape.dataset.originalStrokeWidth === undefined) {
            shape.dataset.originalStrokeWidth =
                shape.getAttribute("stroke-width") || "";
        }
        shape.setAttribute("stroke", "#2563eb");
        shape.setAttribute("stroke-width", "3");
    }

    const label = getNodeLabel(node);
    const selectedDisplay = $("selectedNode");
    if (selectedDisplay) selectedDisplay.textContent = label || "Selected";

    updateEditorStatus(`Selected node: ${label || "Unknown"}`);
}

function clearNodeSelection() {
    $("diagramContainer")
        ?.querySelectorAll(".text2arch-selected-node")
        .forEach(node => {
            const shape = node.querySelector("polygon, rect, ellipse, path");

            if (shape) {
                const originalStroke = shape.dataset.originalStroke;
                const originalWidth = shape.dataset.originalStrokeWidth;

                if (originalStroke) {
                    shape.setAttribute("stroke", originalStroke);
                } else {
                    shape.removeAttribute("stroke");
                }

                if (originalWidth) {
                    shape.setAttribute("stroke-width", originalWidth);
                } else {
                    shape.removeAttribute("stroke-width");
                }

                delete shape.dataset.originalStroke;
                delete shape.dataset.originalStrokeWidth;
            }

            node.classList.remove("text2arch-selected-node");
        });

    selectedNode = null;
    updateSelectedNodeDisplay();
}

// =====================================================
// EDIT NODE
// =====================================================

function editSelectedNode() {
    if (!selectedNode) {
        updateEditorStatus("Please select a node first.");
        return;
    }

    const oldLabel = getNodeLabel(selectedNode);
    const newLabel = window.prompt("Enter new node label:", oldLabel);

    if (newLabel === null || !newLabel.trim()) {
        updateEditorStatus("Node editing cancelled.");
        return;
    }

    const label = newLabel.trim();
    const nodeId = getNodeIdentifier(selectedNode);

    const duplicate = architectureData.nodes.some(
        node => node.id !== nodeId &&
            node.label.toLowerCase() === label.toLowerCase()
    );

    if (duplicate) {
        updateEditorStatus("A node with this label already exists.");
        return;
    }

    const textElements = selectedNode.querySelectorAll("text");
    if (textElements.length) {
        textElements[0].textContent = label;
        // Remove extra label text elements when renaming a multi-line Graphviz label.
        for (let i = 1; i < textElements.length; i++) {
            textElements[i].remove();
        }
    }

    const nodeData = architectureData.nodes.find(node => node.id === nodeId);
    if (nodeData) {
        nodeData.label = label;
        nodeData.type = detectNodeType(label);
    }

    architectureData.edges.forEach(edge => {
        if (edge.source === nodeId) edge.sourceLabel = label;
        if (edge.target === nodeId) edge.targetLabel = label;
    });

    $("selectedNode").textContent = label;
    updateEditorStatus(`Node renamed to: ${label}`);
    updateStatistics();
}

// =====================================================
// SVG COORDINATE HELPERS
// =====================================================

function getNodeBoundsInGraph(node, svg, graph) {
    const box = node.getBBox();
    const nodeMatrix = node.getScreenCTM();
    const graphMatrix = graph.getScreenCTM();

    if (!nodeMatrix || !graphMatrix) {
        return { x: box.x, y: box.y, width: box.width, height: box.height };
    }

    const corners = [
        [box.x, box.y],
        [box.x + box.width, box.y],
        [box.x, box.y + box.height],
        [box.x + box.width, box.y + box.height]
    ].map(([x, y]) => {
        const point = svg.createSVGPoint();
        point.x = x;
        point.y = y;
        const screenPoint = point.matrixTransform(nodeMatrix);
        return screenPoint.matrixTransform(graphMatrix.inverse());
    });

    const xs = corners.map(point => point.x);
    const ys = corners.map(point => point.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    return {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY
    };
}

function getCenterInGraph(node, svg, graph) {
    const bounds = getNodeBoundsInGraph(node, svg, graph);
    return {
        x: bounds.x + bounds.width / 2,
        y: bounds.y + bounds.height / 2
    };
}
// =====================================================
// ADD NODE
// =====================================================

function addNode() {
    const name = window.prompt("Enter node name:");
    if (!name || !name.trim()) return;

    const nodeName = name.trim();
    const svg = $("diagramContainer")?.querySelector("svg");

    if (!svg) {
        alert("Generate an architecture diagram first.");
        return;
    }

    const graph = svg.querySelector("g.graph");
    if (!graph) {
        alert("Unable to access the diagram.");
        return;
    }

    if (architectureData.nodes.some(
        node => node.label.toLowerCase() === nodeName.toLowerCase()
    )) {
        updateEditorStatus("A node with this name already exists.");
        return;
    }

    const existingNodes = [...graph.querySelectorAll("g.node")];
    let maxX = 0;
    let minY = Infinity;
    let maxY = -Infinity;

    existingNodes.forEach(node => {
        try {
            const bounds = getNodeBoundsInGraph(node, svg, graph);
            maxX = Math.max(maxX, bounds.x + bounds.width);
            minY = Math.min(minY, bounds.y);
            maxY = Math.max(maxY, bounds.y + bounds.height);
        } catch (error) {
            console.warn("Could not measure a node:", error);
        }
    });

    if (!Number.isFinite(minY)) minY = 0;
    if (!Number.isFinite(maxY)) maxY = 100;

    const nodeWidth = Math.max(150, Math.min(260, nodeName.length * 9 + 30));
    const nodeHeight = 50;
    const x = maxX + 35;
    const y = (minY + maxY) / 2 - nodeHeight / 2;

    const group = document.createElementNS(SVG_NS, "g");
    const nodeId = `custom_${Date.now()}_${Math.floor(Math.random() * 10000)}`;

    group.setAttribute("id", nodeId);
    group.classList.add("node", "custom-node");
    group.style.cursor = "pointer";

    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", nodeWidth);
    rect.setAttribute("height", nodeHeight);
    rect.setAttribute("rx", "8");
    rect.setAttribute("fill", "white");
    rect.setAttribute("stroke", "#2563eb");
    rect.setAttribute("stroke-width", "2");

    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", x + nodeWidth / 2);
    text.setAttribute("y", y + 30);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "14");
    text.setAttribute("font-family", "Arial, sans-serif");
    text.setAttribute("fill", "black");
    text.textContent = nodeName;

    group.append(rect, text);
    graph.appendChild(group);

    // Expand the SVG viewBox to include the new node.
    const viewBox = svg.getAttribute("viewBox");
    if (viewBox) {
        const parts = viewBox.trim().split(/[,\s]+/).map(Number);
        if (parts.length === 4 && parts.every(Number.isFinite)) {
            let [vx, vy, vw, vh] = parts;
            const right = x + nodeWidth + 25;
            const bottom = y + nodeHeight + 25;
            const left = x - 25;
            const top = y - 25;

            if (right > vx + vw) vw = right - vx;
            if (bottom > vy + vh) vh = bottom - vy;
            if (left < vx) {
                vw += vx - left;
                vx = left;
            }
            if (top < vy) {
                vh += vy - top;
                vy = top;
            }

            svg.setAttribute("viewBox", `${vx} ${vy} ${vw} ${vh}`);
        }
    }

    group.addEventListener("click", event => {
        event.stopPropagation();
        if (editorMode === "addEdge") {
            handleEdgeNodeClick(group);
        } else if (editorMode !== "deleteEdge") {
            selectSvgNode(group);
        }
    });

    syncArchitectureDataFromSVG();
    selectSvgNode(group);
    updateStatistics();
    updateEditorStatus(`"${nodeName}" added successfully.`);
}

// =====================================================
// DELETE NODE
// =====================================================

function deleteSelectedNode() {
    if (!selectedNode) {
        updateEditorStatus("Please select a node first.");
        return;
    }

    const label = getNodeLabel(selectedNode);
    if (!confirm(`Delete "${label}"?`)) return;

    const nodeId = getNodeIdentifier(selectedNode);
    const svg = $("diagramContainer").querySelector("svg");

    // Remove every SVG edge connected to this node, including original Graphviz edges.
    if (svg) {
        svg.querySelectorAll("g.edge").forEach(edge => {
            const endpoints = getEdgeEndpoints(edge);
            const source = edge.dataset.source || endpoints.source;
            const target = edge.dataset.target || endpoints.target;

            if (source === nodeId || target === nodeId) {
                edge.remove();
            }
        });
    }

    selectedNode.remove();
    selectedNode = null;
    selectedEdge = null;
    edgeSourceNode = null;
    editorMode = "select";

    syncArchitectureDataFromSVG();
    updateSelectedNodeDisplay();
    updateStatistics();
    updateEditorStatus(`Deleted node: ${label}`);
}

// =====================================================
// ADD CONNECTION
// =====================================================

function addEdge() {
    const svg = $("diagramContainer")?.querySelector("svg");
    if (!svg) {
        updateEditorStatus("Generate an architecture diagram first.");
        return;
    }

    syncArchitectureDataFromSVG();

    if (architectureData.nodes.length < 2) {
        updateEditorStatus("At least two nodes are required to create a connection.");
        return;
    }

    edgeSourceNode = null;
    editorMode = "addEdge";
    clearNodeSelection();
    clearEdgeSelection();

    updateEditorStatus(
        "Add Connection: click the source node first, then the target node."
    );
}

function handleEdgeNodeClick(node) {
    if (!node || !node.isConnected) return;

    const label = getNodeLabel(node);

    if (!edgeSourceNode) {
        edgeSourceNode = node;
        node.classList.add("edge-source-node");
        updateEditorStatus(`Source selected: ${label}. Now click the target node.`);
        return;
    }

    if (edgeSourceNode === node) {
        updateEditorStatus("Source and target cannot be the same node.");
        return;
    }

    const sourceNode = edgeSourceNode;
    const sourceId = getNodeIdentifier(sourceNode);
    const targetId = getNodeIdentifier(node);
    const sourceLabel = getNodeLabel(sourceNode);
    const targetLabel = getNodeLabel(node);

    syncArchitectureDataFromSVG();

    const sourceExists = architectureData.nodes.some(item => item.id === sourceId);
    const targetExists = architectureData.nodes.some(item => item.id === targetId);

    if (!sourceExists || !targetExists) {
        sourceNode.classList.remove("edge-source-node");
        edgeSourceNode = null;
        editorMode = "select";
        updateEditorStatus("Invalid connection. Both nodes must exist.");
        return;
    }

    const duplicate = architectureData.edges.some(
        edge => edge.source === sourceId && edge.target === targetId
    );

    if (duplicate) {
        sourceNode.classList.remove("edge-source-node");
        edgeSourceNode = null;
        editorMode = "select";
        updateEditorStatus("This connection already exists.");
        return;
    }

    const created = createInteractiveEdge(
        sourceNode,
        node,
        sourceLabel,
        targetLabel
    );

    sourceNode.classList.remove("edge-source-node");
    edgeSourceNode = null;
    editorMode = "select";

    if (created) {
        syncArchitectureDataFromSVG();
        updateEditorStatus(`Connection added: ${sourceLabel} → ${targetLabel}`);
    }
}

// =====================================================
// CREATE CONNECTION
// =====================================================

function createInteractiveEdge(sourceNode, targetNode, sourceLabel, targetLabel) {
    const svg = $("diagramContainer")?.querySelector("svg");
    const graph = svg?.querySelector("g.graph");

    if (!svg || !graph) {
        updateEditorStatus("Unable to access the diagram graph.");
        return false;
    }

    const sourceId = getNodeIdentifier(sourceNode);
    const targetId = getNodeIdentifier(targetNode);

    if (!sourceId || !targetId || sourceId === targetId) return false;

    const source = getCenterInGraph(sourceNode, svg, graph);
    const target = getCenterInGraph(targetNode, svg, graph);

    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const length = Math.hypot(dx, dy);

    if (!Number.isFinite(length) || length === 0) {
        updateEditorStatus("Could not calculate the connection position.");
        return false;
    }

    // Shorten the line so it ends close to the target node rather than its centre.
    const targetBounds = getNodeBoundsInGraph(targetNode, svg, graph);
    const sourceBounds = getNodeBoundsInGraph(sourceNode, svg, graph);
    const ux = dx / length;
    const uy = dy / length;

    const sourceRadius = Math.min(
        Math.abs(ux) > 0.001 ? sourceBounds.width / 2 / Math.abs(ux) : Infinity,
        Math.abs(uy) > 0.001 ? sourceBounds.height / 2 / Math.abs(uy) : Infinity
    );
    const targetRadius = Math.min(
        Math.abs(ux) > 0.001 ? targetBounds.width / 2 / Math.abs(ux) : Infinity,
        Math.abs(uy) > 0.001 ? targetBounds.height / 2 / Math.abs(uy) : Infinity
    );

    const startX = source.x + ux * (Number.isFinite(sourceRadius) ? sourceRadius : 0);
    const startY = source.y + uy * (Number.isFinite(sourceRadius) ? sourceRadius : 0);
    const endX = target.x - ux * (Number.isFinite(targetRadius) ? targetRadius : 0);
    const endY = target.y - uy * (Number.isFinite(targetRadius) ? targetRadius : 0);

    const edgeGroup = document.createElementNS(SVG_NS, "g");
    edgeGroup.classList.add("edge", "custom-edge");
    edgeGroup.dataset.source = sourceId;
    edgeGroup.dataset.target = targetId;
    edgeGroup.dataset.sourceLabel = sourceLabel;
    edgeGroup.dataset.targetLabel = targetLabel;

    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${sourceId}->${targetId}`;
    edgeGroup.appendChild(title);

    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", startX);
    line.setAttribute("y1", startY);
    line.setAttribute("x2", endX);
    line.setAttribute("y2", endY);
    line.setAttribute("stroke", "#555");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("pointer-events", "stroke");
    line.style.cursor = "pointer";

    const angle = Math.atan2(endY - startY, endX - startX);
    const arrowSize = 9;
    const p1x = endX - arrowSize * Math.cos(angle - Math.PI / 6);
    const p1y = endY - arrowSize * Math.sin(angle - Math.PI / 6);
    const p2x = endX - arrowSize * Math.cos(angle + Math.PI / 6);
    const p2y = endY - arrowSize * Math.sin(angle + Math.PI / 6);

    const arrow = document.createElementNS(SVG_NS, "polygon");
    arrow.setAttribute("points", `${endX},${endY} ${p1x},${p1y} ${p2x},${p2y}`);
    arrow.setAttribute("fill", "#555");
    arrow.setAttribute("pointer-events", "none");

    edgeGroup.append(line, arrow);

    // Put the new connection behind the nodes, but above the graph background.
    const firstNode = graph.querySelector("g.node");
    if (firstNode) {
        graph.insertBefore(edgeGroup, firstNode);
    } else {
        graph.appendChild(edgeGroup);
    }

    edgeGroup.style.cursor = "pointer";
    edgeGroup.addEventListener("click", event => {
        event.stopPropagation();
        if (editorMode === "deleteEdge") selectEdge(edgeGroup);
    });

    updateStatistics();
    return true;
}
// =====================================================
// EDGE HELPERS, SELECTION, AND DELETION
// =====================================================

function getEdgeEndpoints(edge) {
    const title = edge?.querySelector("title");
    if (!title) return { source: "", target: "" };

    const match = title.textContent.match(/^\s*(.*?)\s*->\s*(.*?)\s*$/);
    if (!match) return { source: "", target: "" };

    return {
        source: match[1].trim(),
        target: match[2].trim()
    };
}

function selectEdge(edge) {
    clearEdgeSelection();
    selectedEdge = edge;
    edge.classList.add("text2arch-selected-edge");

    const shape = edge.querySelector("path, line");
    if (shape) {
        if (shape.dataset.originalStroke === undefined) {
            shape.dataset.originalStroke = shape.getAttribute("stroke") || "";
        }
        if (shape.dataset.originalStrokeWidth === undefined) {
            shape.dataset.originalStrokeWidth =
                shape.getAttribute("stroke-width") || "";
        }
        shape.setAttribute("stroke", "#2563eb");
        shape.setAttribute("stroke-width", "4");
    }

    updateEditorStatus("Connection selected. Click Delete Connection to remove it.");
}

function clearEdgeSelection() {
    $("diagramContainer")
        ?.querySelectorAll(".text2arch-selected-edge")
        .forEach(edge => {
            const shape = edge.querySelector("path, line");

            if (shape) {
                const originalStroke = shape.dataset.originalStroke;
                const originalWidth = shape.dataset.originalStrokeWidth;

                if (originalStroke) {
                    shape.setAttribute("stroke", originalStroke);
                } else {
                    shape.removeAttribute("stroke");
                }

                if (originalWidth) {
                    shape.setAttribute("stroke-width", originalWidth);
                } else {
                    shape.removeAttribute("stroke-width");
                }

                delete shape.dataset.originalStroke;
                delete shape.dataset.originalStrokeWidth;
            }

            edge.classList.remove("text2arch-selected-edge");
        });

    selectedEdge = null;
}

function deleteEdge() {
    if (!selectedEdge) {
        const svg = $("diagramContainer")?.querySelector("svg");
        if (!svg) {
            updateEditorStatus("Generate an architecture diagram first.");
            return;
        }

        editorMode = "deleteEdge";
        edgeSourceNode = null;
        clearNodeSelection();
        clearEdgeSelection();
        updateEditorStatus("Delete Connection: click the connection you want to delete.");
        return;
    }

    if (!confirm("Delete this connection?")) return;

    selectedEdge.remove();
    selectedEdge = null;
    editorMode = "select";

    syncArchitectureDataFromSVG();
    updateStatistics();
    updateEditorStatus("Connection deleted successfully.");
}

// =====================================================
// ZOOM
// =====================================================

function updateDiagramZoom() {
    const svg = $("diagramContainer")?.querySelector("svg");
    if (svg) {
        svg.style.transform = `scale(${diagramZoom})`;
    }
}

function zoomIn() {
    diagramZoom = Math.min(2, diagramZoom + 0.1);
    updateDiagramZoom();
    updateEditorStatus(`Zoom: ${Math.round(diagramZoom * 100)}%`);
}

function zoomOut() {
    diagramZoom = Math.max(0.5, diagramZoom - 0.1);
    updateDiagramZoom();
    updateEditorStatus(`Zoom: ${Math.round(diagramZoom * 100)}%`);
}

function resetZoom() {
    diagramZoom = 1;
    updateDiagramZoom();
    updateEditorStatus("Zoom reset to 100%.");
}

// =====================================================
// SYNCHRONIZE DATA FROM THE RENDERED SVG
// =====================================================

function syncArchitectureDataFromSVG() {
    const svg = $("diagramContainer")?.querySelector("svg");

    if (!svg) {
        architectureData = { nodes: [], edges: [] };
        updateStatistics();
        return;
    }

    const nodes = [];
    const edges = [];
    const nodeIds = new Set();

    svg.querySelectorAll("g.node").forEach(node => {
        const id = getNodeIdentifier(node);
        const label = getNodeLabel(node);

        if (!id || nodeIds.has(id)) return;

        nodeIds.add(id);
        nodes.push({
            id,
            label: label || id,
            type: detectNodeType(label || id)
        });
    });

    svg.querySelectorAll("g.edge").forEach(edge => {
        const titleEndpoints = getEdgeEndpoints(edge);
        const source = edge.dataset.source || titleEndpoints.source;
        const target = edge.dataset.target || titleEndpoints.target;

        if (!source || !target || !nodeIds.has(source) || !nodeIds.has(target)) {
            return;
        }

        if (edges.some(item => item.source === source && item.target === target)) {
            return;
        }

        const sourceNode = nodes.find(node => node.id === source);
        const targetNode = nodes.find(node => node.id === target);

        edges.push({
            source,
            target,
            sourceLabel: sourceNode?.label || source,
            targetLabel: targetNode?.label || target
        });
    });

    architectureData = { nodes, edges };
    updateStatistics();

    console.log("Synchronized architecture:", architectureData);
}

// =====================================================
// NODE TYPE
// =====================================================

function detectNodeType(label) {
    const text = String(label).toLowerCase();

    if (["frontend", "react", "angular", "vue", "client"].some(word => text.includes(word))) {
        return "frontend";
    }

    if (["backend", "server", "spring", "flask", "fastapi", "node"].some(word => text.includes(word))) {
        return "backend";
    }

    if (["database", "mysql", "postgres", "mongodb", "sql"].some(word => text.includes(word))) {
        return "database";
    }

    if (["service", "api", "cloud", "payment"].some(word => text.includes(word))) {
        return "service";
    }

    return "other";
}

// =====================================================
// UI STATISTICS AND STATUS
// =====================================================

function updateStatistics() {
    if ($("componentCount")) {
        $("componentCount").textContent = architectureData.nodes.length;
    }

    if ($("connectionCount")) {
        $("connectionCount").textContent = architectureData.edges.length;
    }

    if ($("architectureStatus")) {
        $("architectureStatus").textContent =
            architectureData.nodes.length ? "Generated" : "Not Generated";
    }
}

function updateSelectedNodeDisplay() {
    const element = $("selectedNode");
    if (element && !selectedNode) element.textContent = "None";
}

function updateEditorStatus(message) {
    const element = $("editorStatus");
    if (element) element.textContent = message;
}
// =====================================================
// EXPORT SVG
// =====================================================

function downloadSVG() {
    const svg = $("diagramContainer")?.querySelector("svg");

    if (!svg) {
        alert("Please generate a diagram first.");
        return;
    }

    try {
        const svgData = new XMLSerializer().serializeToString(svg);
        const blob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = "text2arch-diagram.svg";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);

        updateEditorStatus("SVG diagram exported successfully.");
    } catch (error) {
        console.error("SVG export error:", error);
        updateEditorStatus("SVG export failed.");
    }
}

// =====================================================
// EXPORT JSON
// =====================================================

function exportJSON() {
    syncArchitectureDataFromSVG();

    if (!architectureData.nodes.length) {
        alert("Please generate a diagram first.");
        return;
    }

    try {
        const latestData = {
            nodes: architectureData.nodes.map(node => ({ ...node })),
            edges: architectureData.edges.map(edge => ({ ...edge }))
        };

        const blob = new Blob(
            [JSON.stringify(latestData, null, 2)],
            { type: "application/json" }
        );

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = "text2arch-architecture.json";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);

        updateEditorStatus("Architecture JSON exported successfully.");
    } catch (error) {
        console.error("JSON export error:", error);
        updateEditorStatus("JSON export failed.");
    }
}

// =====================================================
// ESCAPE HTML
// =====================================================

function escapeHTML(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// =====================================================
// BUTTON HANDLERS AND INITIAL STATE
// =====================================================

document.addEventListener("click", event => {
    const button = event.target.closest("button");
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
            editorMode = "select";
            edgeSourceNode = null;
            $("diagramContainer")
                ?.querySelectorAll(".edge-source-node")
                .forEach(node => node.classList.remove("edge-source-node"));
            clearEdgeSelection();
            updateEditorStatus("Select mode enabled. Click a node in the diagram.");
            break;

        case "editNodeBtn":
            editorMode = "select";
            editSelectedNode();
            break;

        case "addNodeBtn":
            editorMode = "select";
            addNode();
            break;

        case "deleteNodeBtn":
            editorMode = "select";
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
});

document.addEventListener("DOMContentLoaded", () => {
    const description = $("description");
    const charCount = $("charCount");
    const exampleBtn = $("exampleBtn");

    if (description && charCount) {
        charCount.textContent = `${description.value.length} / 2000`;

        description.addEventListener("input", () => {
            charCount.textContent = `${description.value.length} / 2000`;
        });
    }

    if (exampleBtn && description) {
        exampleBtn.addEventListener("click", () => {
            description.value =
                "A React frontend communicates with a Spring Boot backend. The backend connects to a MySQL database.";

            if (charCount) {
                charCount.textContent = `${description.value.length} / 2000`;
            }

            description.focus();
        });
    }

    updateStatistics();
    updateSelectedNodeDisplay();
    updateEditorStatus("Ready");
});