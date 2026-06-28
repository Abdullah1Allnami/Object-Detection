document.addEventListener("DOMContentLoaded", () => {
    // Canvas Setup
    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d");
    const vizCanvas = document.getElementById("viz-canvas");
    const vizCtx = vizCanvas.getContext("2d");
    const overlay = document.getElementById("detection-overlay");

    // Controls & Stats
    const minConfSlider = document.getElementById("slider-min-conf");
    const minConfVal = document.getElementById("val-min-conf");
    const iouSlider = document.getElementById("slider-iou-thresh");
    const iouVal = document.getElementById("val-iou-thresh");
    
    const btnClear = document.getElementById("btn-clear-canvas");
    const btnDetect = document.getElementById("btn-detect");
    
    const statProposals = document.getElementById("stat-proposals");
    const statLatency = document.getElementById("stat-latency");
    const statRcnnLatency = document.getElementById("stat-rcnn-latency");
    const statSpeedup = document.getElementById("stat-speedup");
    
    const detectionsList = document.getElementById("detections-list");

    // Tabs & Steps Descriptions
    const tabButtons = document.querySelectorAll(".tab-btn");
    const stepTitle = document.getElementById("step-title");
    const stepDesc = document.getElementById("step-desc");

    const stepInfo = {
        proposals: {
            title: "Step 1: Region Proposal Network (RPN)",
            desc: "The RPN processes the convolutional feature map in a sliding-window fashion, evaluating 3 anchors at each position. It filters them down to the top region proposals (blue dashed boxes) using NMS on RPN objectness scores. These form the dynamic input proposals to the second stage."
        },
        regression: {
            title: "Step 2: Bounding Box Regression",
            desc: "For each proposal, the detection head predicts class probabilities and bounding box regression adjustments. The original RPN proposal (blue) is aligned and shifted to the refined bounding box (yellow). Dashed arrows trace the regression offsets."
        },
        detections: {
            title: "Step 3: Final Non-Maximum Suppression (NMS)",
            desc: "Proposals predicted as background class are removed. A score threshold filters out low-confidence predictions, and class-agnostic NMS resolves duplicates, yielding the final tight green bounding boxes with predicted digit labels."
        }
    };

    let activeStep = "proposals";
    let lastDetectionResponse = null;

    // Initialize Canvas (White stroke on Black background)
    function clearCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // Clear viz canvas
        vizCtx.fillStyle = "#000000";
        vizCtx.fillRect(0, 0, vizCanvas.width, vizCanvas.height);
        // Clear overlays
        overlay.innerHTML = "";
        lastDetectionResponse = null;
        
        // Reset Stats
        statProposals.textContent = "-";
        statLatency.textContent = "-";
        statRcnnLatency.textContent = "-";
        statSpeedup.textContent = "-";
        
        // Reset detections list
        detectionsList.innerHTML = '<div class="empty-list-msg">No objects detected. Draw digits and click detect.</div>';
    }
    clearCanvas();

    // Drawing Logic
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function getCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: ((clientX - rect.left) / rect.width) * canvas.width,
            y: ((clientY - rect.top) / rect.height) * canvas.height
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        const coords = getCoords(e);
        lastX = coords.x;
        lastY = coords.y;
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        
        const coords = getCoords(e);
        
        ctx.beginPath();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 14;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        
        lastX = coords.x;
        lastY = coords.y;
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Event Listeners for Drawing Canvas
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    canvas.addEventListener("touchstart", startDrawing, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDrawing);

    btnClear.addEventListener("click", clearCanvas);

    // Hyperparameter Slider updates
    minConfSlider.addEventListener("input", (e) => {
        minConfVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    iouSlider.addEventListener("input", (e) => {
        iouVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Mirror main canvas to visualization canvas
    function mirrorCanvas() {
        vizCtx.drawImage(canvas, 0, 0);
    }

    // Call detect API
    async function runDetection() {
        mirrorCanvas();
        overlay.innerHTML = "";
        
        const base64Image = canvas.toDataURL("image/png");
        const payload = {
            image: base64Image,
            min_conf: parseFloat(minConfSlider.value),
            iou_threshold: parseFloat(iouSlider.value)
        };

        btnDetect.disabled = true;
        btnDetect.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

        try {
            const response = await fetch("/api/detect", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                alert(`Error: ${errData.error || "Inference failed"}`);
                return;
            }

            const data = await response.json();
            lastDetectionResponse = data;
            
            // Update performance stats
            statProposals.textContent = data.metrics.num_proposals;
            statLatency.textContent = `${data.metrics.faster_rcnn_latency_ms} ms`;
            statRcnnLatency.textContent = `${data.metrics.est_fast_rcnn_total_latency_ms} ms`;
            statSpeedup.textContent = `${data.metrics.speedup_ratio}x`;

            // Draw normalized image returned by the server on the visualization canvas
            const imgObj = new Image();
            imgObj.onload = () => {
                vizCtx.drawImage(imgObj, 0, 0);
                // Render active visual overlays
                renderOverlays();
            };
            imgObj.src = data.normalized_image;

            // Populate detections list
            populateDetectionsList(data.final_detections);

        } catch (error) {
            console.error(error);
            alert("Network connection error. Ensure Flask server is running.");
        } finally {
            btnDetect.disabled = false;
            btnDetect.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run Faster R-CNN Detection';
        }
    }

    btnDetect.addEventListener("click", runDetection);

    // List rendering
    function populateDetectionsList(detections) {
        detectionsList.innerHTML = "";
        
        if (detections.length === 0) {
            detectionsList.innerHTML = '<div class="empty-list-msg">No digits detected above threshold.</div>';
            return;
        }

        detections.forEach(det => {
            const item = document.createElement("div");
            item.className = "det-item";
            
            const [x1, y1, x2, y2] = det.box;
            const w = x2 - x1;
            const h = y2 - y1;

            item.innerHTML = `
                <div class="det-icon">${det.class}</div>
                <div class="det-info">
                    <span class="det-name">Digit Class: ${det.class}</span>
                    <span class="det-coords">Box: [x:${x1}, y:${y1}, w:${w}, h:${h}]</span>
                </div>
                <div class="det-score">${Math.round(det.score * 100)}%</div>
            `;
            detectionsList.appendChild(item);
        });
    }

    // Toggle Tab Steps
    tabButtons.forEach(btn => {
        btn.addEventListener("click", (e) => {
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            activeStep = btn.dataset.step;
            
            // Update description panel
            const info = stepInfo[activeStep];
            stepTitle.textContent = info.title;
            stepDesc.textContent = info.desc;
            
            // Rerender overlays
            renderOverlays();
        });
    });

    // Overlay Drawing Engine
    function renderOverlays() {
        // Always clear overlays first
        overlay.innerHTML = "";
        
        if (!lastDetectionResponse) return;
        
        const proposals = lastDetectionResponse.all_proposals;
        const detections = lastDetectionResponse.final_detections;

        if (activeStep === "proposals") {
            // Draw all initial generated region proposals (blue dashed boxes)
            proposals.forEach(prop => {
                const box = prop.original_xywh;
                const div = document.createElement("div");
                div.className = "roi-proposal";
                div.style.left = `${box[0]}px`;
                div.style.top = `${box[1]}px`;
                div.style.width = `${box[2]}px`;
                div.style.height = `${box[3]}px`;
                overlay.appendChild(div);
            });
        } 
        else if (activeStep === "regression") {
            // Draw arrows showing the box adjustments and the new regressed coordinates
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("class", "roi-arrow-svg");
            svg.setAttribute("width", "100%");
            svg.setAttribute("height", "100%");
            
            // Add arrow marker definition
            const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
            defs.innerHTML = `
                <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                    <polygon points="0 0, 6 3, 0 6" fill="#fee440" />
                </marker>
            `;
            svg.appendChild(defs);

            proposals.forEach(prop => {
                // Ignore background class for clearer regression visuals
                if (prop.is_background) return;
                
                const orig = prop.original_xywh;
                const ref = prop.refined_xywh;
                
                // 1. Draw original box (blue)
                const origDiv = document.createElement("div");
                origDiv.className = "roi-proposal";
                origDiv.style.left = `${orig[0]}px`;
                origDiv.style.top = `${orig[1]}px`;
                origDiv.style.width = `${orig[2]}px`;
                origDiv.style.height = `${orig[3]}px`;
                overlay.appendChild(origDiv);

                // 2. Draw refined box (yellow)
                const refDiv = document.createElement("div");
                refDiv.className = "roi-regressed";
                refDiv.style.left = `${ref[0]}px`;
                refDiv.style.top = `${ref[1]}px`;
                refDiv.style.width = `${ref[2]}px`;
                refDiv.style.height = `${ref[3]}px`;
                overlay.appendChild(refDiv);

                // 3. Draw arrow from original center to refined center
                const cx1 = orig[0] + orig[2] / 2.0;
                const cy1 = orig[1] + orig[3] / 2.0;
                const cx2 = ref[0] + ref[2] / 2.0;
                const cy2 = ref[1] + ref[3] / 2.0;

                // Only draw arrow if center has moved significantly
                const dist = Math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2);
                if (dist > 1.5) {
                    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                    line.setAttribute("x1", cx1);
                    line.setAttribute("y1", cy1);
                    line.setAttribute("x2", cx2);
                    line.setAttribute("y2", cy2);
                    line.setAttribute("class", "roi-arrow-line");
                    svg.appendChild(line);
                }
            });
            overlay.appendChild(svg);
        } 
        else if (activeStep === "detections") {
            // Draw NMS filtered final detections with labels (green boxes)
            detections.forEach(det => {
                const box = det.xywh;
                
                const div = document.createElement("div");
                div.className = "roi-detection";
                div.style.left = `${box[0]}px`;
                div.style.top = `${box[1]}px`;
                div.style.width = `${box[2]}px`;
                div.style.height = `${box[3]}px`;
                
                const label = document.createElement("div");
                label.className = "roi-label";
                label.textContent = `${det.class} (${Math.round(det.score * 100)}%)`;
                
                div.appendChild(label);
                overlay.appendChild(div);
            });
        }
    }
});
