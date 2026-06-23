document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const btnModelDigit = document.getElementById("btn-model-digit");
    const btnModelResNet50 = document.getElementById("btn-model-resnet50");
    const digitWorkspace = document.getElementById("digit-workspace");
    const resnet50Workspace = document.getElementById("resnet50-workspace");
    const workspaceHint = document.getElementById("workspace-hint");
    
    const sliderMinConf = document.getElementById("slider-min-conf");
    const valMinConf = document.getElementById("val-min-conf");
    const sliderMaxProps = document.getElementById("slider-max-props");
    const valMaxProps = document.getElementById("val-max-props");
    
    const btnTrainModel = document.getElementById("btn-train-model");
    const modelStatusBadge = document.getElementById("model-status-badge");
    const modelStatusText = document.getElementById("model-status-text");
    const trainingProgressContainer = document.getElementById("training-progress-container");
    const trainingProgressBar = document.getElementById("training-progress-bar");
    const trainingProgressMsg = document.getElementById("training-progress-msg");
    const trainingProgressPct = document.getElementById("training-progress-pct");
    const trainingMetrics = document.getElementById("training-metrics");
    const metricLoss = document.getElementById("metric-loss");
    const metricAcc = document.getElementById("metric-acc");
    
    const drawingCanvas = document.getElementById("drawing-canvas");
    const btnClearCanvas = document.getElementById("btn-clear-canvas");
    
    const uploadDropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("file-input");
    
    const vizCanvas = document.getElementById("viz-canvas");
    const heatmapCanvas = document.getElementById("heatmap-canvas");
    const proposalInteractiveLayer = document.getElementById("proposal-interactive-layer");
    const detectionOverlay = document.getElementById("detection-overlay");
    
    const btnDetectInstant = document.getElementById("btn-detect-instant");
    
    const btnViewProposals = document.getElementById("btn-view-proposals");
    const btnViewFeatures = document.getElementById("btn-view-features");
    const btnViewDetections = document.getElementById("btn-view-detections");
    
    const statTotalProps = document.getElementById("stat-total-props");
    const statRawHits = document.getElementById("stat-raw-hits");
    const statFinalDetections = document.getElementById("stat-final-detections");
    
    // Inspector elements
    const inspectorContent = document.getElementById("inspector-content");
    const inspectorEmptyState = document.querySelector(".inspector-empty-state");
    const inspectorFullState = document.querySelector(".inspector-full-state");
    const inspectorActiveBadge = document.getElementById("inspector-active-badge");
    
    const roiImgCoords = document.getElementById("roi-img-coords");
    const roiDownsampleRatio = document.getElementById("roi-downsample-ratio");
    const roiConvCoords = document.getElementById("roi-conv-coords");
    const roiPooledGrid = document.getElementById("roi-pooled-grid");
    const clsProbabilityList = document.getElementById("cls-probability-list");
    
    const offDx = document.getElementById("off-dx");
    const offDy = document.getElementById("off-dy");
    const offDw = document.getElementById("off-dw");
    const offDh = document.getElementById("off-dh");
    const compPropBox = document.getElementById("comp-prop-box");
    const compRefBox = document.getElementById("comp-ref-box");
    
    // FLOPs Calculator elements
    const calcClassicFlops = document.getElementById("calc-classic-flops");
    const calcFastFlops = document.getElementById("calc-fast-flops");
    const calcSavingsBadge = document.getElementById("calc-savings-badge");

    // Application State
    let activeModel = "digit"; // "digit" or "resnet50"
    let activeView = "proposals"; // "proposals", "features", "detections"
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let trainingPollInterval = null;
    let loadedImage = null; // Stores uploaded / generated image object
    let detectionData = null; // Stores last API response data
    
    // Canvas Contexts
    const drawCtx = drawingCanvas.getContext("2d");
    const vizCtx = vizCanvas.getContext("2d");
    const heatmapCtx = heatmapCanvas.getContext("2d");
    
    // Initialize Drawing Canvas (White Background, Black Ink)
    function clearDrawCanvas() {
        drawCtx.fillStyle = "#ffffff";
        drawCtx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        syncVizCanvas();
    }
    clearDrawCanvas();
    
    // Synchronize the visualization canvas
    function syncVizCanvas() {
        vizCtx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);
        
        if (activeModel === "digit") {
            vizCtx.drawImage(drawingCanvas, 0, 0, vizCanvas.width, vizCanvas.height);
        } else if (loadedImage) {
            vizCtx.drawImage(loadedImage, 0, 0, vizCanvas.width, vizCanvas.height);
        } else {
            // Draw a dark empty placeholder
            vizCtx.fillStyle = "#0d1117";
            vizCtx.fillRect(0, 0, vizCanvas.width, vizCanvas.height);
            vizCtx.fillStyle = "#8b949e";
            vizCtx.font = "12px Outfit";
            vizCtx.textAlign = "center";
            vizCtx.fillText("Upload an image or load sample", vizCanvas.width / 2, vizCanvas.height / 2);
        }
        
        // Reset overlay visual elements
        clearVisualOverlays();
    }
    
    function clearVisualOverlays() {
        proposalInteractiveLayer.innerHTML = "";
        detectionOverlay.innerHTML = "";
        heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        heatmapCanvas.classList.add("hidden");
        
        // Reset stats
        statTotalProps.textContent = "-";
        statRawHits.textContent = "-";
        statFinalDetections.textContent = "-";
        
        // Reset inspector
        resetInspector();
        
        // Reset FLOPs
        calcClassicFlops.textContent = "- GigaFLOPs";
        calcFastFlops.textContent = "- GigaFLOPs";
        calcSavingsBadge.textContent = "-x Faster";
    }
    
    function resetInspector() {
        inspectorEmptyState.classList.remove("hidden");
        inspectorFullState.classList.add("hidden");
        inspectorActiveBadge.classList.add("hidden");
    }

    // --- Sliders Listener ---
    sliderMinConf.addEventListener("input", (e) => {
        valMinConf.textContent = parseFloat(e.target.value).toFixed(2);
    });
    sliderMaxProps.addEventListener("input", (e) => {
        valMaxProps.textContent = e.target.value;
        if (detectionData) renderActiveView();
    });

    // --- Model Toggle Switch ---
    btnModelDigit.addEventListener("click", () => {
        if (activeModel === "digit") return;
        activeModel = "digit";
        btnModelDigit.classList.add("active");
        btnModelResNet50.classList.remove("active");
        digitWorkspace.classList.add("active");
        resnet50Workspace.classList.remove("active");
        workspaceHint.textContent = "Draw multiple separate digits";
        
        // Show training section
        document.getElementById("digit-training-section").classList.remove("hidden");
        
        detectionData = null;
        syncVizCanvas();
    });

    btnModelResNet50.addEventListener("click", () => {
        if (activeModel === "resnet50") return;
        activeModel = "resnet50";
        btnModelResNet50.classList.add("active");
        btnModelDigit.classList.remove("active");
        resnet50Workspace.classList.add("active");
        digitWorkspace.classList.remove("active");
        workspaceHint.textContent = "Upload or select a sample image";
        
        // Hide training section
        document.getElementById("digit-training-section").classList.add("hidden");
        
        detectionData = null;
        syncVizCanvas();
    });

    // --- View Tabs ---
    btnViewProposals.addEventListener("click", () => setViewTab("proposals"));
    btnViewFeatures.addEventListener("click", () => setViewTab("features"));
    btnViewDetections.addEventListener("click", () => setViewTab("detections"));

    function setViewTab(tab) {
        activeView = tab;
        btnViewProposals.classList.toggle("active", tab === "proposals");
        btnViewFeatures.classList.toggle("active", tab === "features");
        btnViewDetections.classList.toggle("active", tab === "detections");
        
        if (detectionData) {
            renderActiveView();
        }
    }

    // --- Drawing Canvas Listeners ---
    drawingCanvas.addEventListener("mousedown", startDrawing);
    drawingCanvas.addEventListener("mousemove", draw);
    drawingCanvas.addEventListener("mouseup", stopDrawing);
    drawingCanvas.addEventListener("mouseout", stopDrawing);
    
    // Touch support
    drawingCanvas.addEventListener("touchstart", (e) => {
        const touch = e.touches[0];
        const rect = drawingCanvas.getBoundingClientRect();
        startDrawing({
            clientX: touch.clientX,
            clientY: touch.clientY,
            preventDefault: () => e.preventDefault()
        });
    });
    drawingCanvas.addEventListener("touchmove", (e) => {
        const touch = e.touches[0];
        draw({
            clientX: touch.clientX,
            clientY: touch.clientY,
            preventDefault: () => e.preventDefault()
        });
    });
    drawingCanvas.addEventListener("touchend", stopDrawing);

    function startDrawing(e) {
        if (e.preventDefault) e.preventDefault();
        isDrawing = true;
        const rect = drawingCanvas.getBoundingClientRect();
        lastX = ((e.clientX - rect.left) / rect.width) * drawingCanvas.width;
        lastY = ((e.clientY - rect.top) / rect.height) * drawingCanvas.height;
    }

    function draw(e) {
        if (!isDrawing) return;
        if (e.preventDefault) e.preventDefault();
        
        const rect = drawingCanvas.getBoundingClientRect();
        const currX = ((e.clientX - rect.left) / rect.width) * drawingCanvas.width;
        const currY = ((e.clientY - rect.top) / rect.height) * drawingCanvas.height;
        
        drawCtx.beginPath();
        drawCtx.moveTo(lastX, lastY);
        drawCtx.lineTo(currX, currY);
        drawCtx.strokeStyle = "#000000";
        drawCtx.lineWidth = 10;
        drawCtx.lineCap = "round";
        drawCtx.lineJoin = "round";
        drawCtx.stroke();
        
        lastX = currX;
        lastY = currY;
        
        syncVizCanvas();
    }

    function stopDrawing() {
        isDrawing = false;
    }

    btnClearCanvas.addEventListener("click", clearDrawCanvas);

    // --- Image Upload Dropzone ---
    uploadDropzone.addEventListener("click", () => fileInput.click());
    uploadDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadDropzone.classList.add("dragover");
    });
    uploadDropzone.addEventListener("dragleave", () => {
        uploadDropzone.classList.remove("dragover");
    });
    uploadDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    function handleUploadedFile(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                loadedImage = img;
                syncVizCanvas();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    // --- Programmatic Sample Images Generation ---
    const sampleButtons = document.querySelectorAll(".sample-btn");
    sampleButtons.forEach(btn => {
        btn.addEventListener("click", (e) => {
            const type = e.target.getAttribute("data-sample");
            drawSampleImage(type);
        });
    });

    function drawSampleImage(type) {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 448;
        tempCanvas.height = 448;
        const ctx = tempCanvas.getContext("2d");
        
        if (type === "mug") {
            ctx.fillStyle = "#1e293b"; // Background desk
            ctx.fillRect(0, 0, 448, 448);
            
            // Wooden panel lines
            ctx.strokeStyle = "#334155";
            ctx.lineWidth = 4;
            for (let i = 50; i < 448; i += 80) {
                ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(448, i); ctx.stroke();
            }
            
            // Mug handle
            ctx.strokeStyle = "#ef476f"; // Coral Red Mug
            ctx.lineWidth = 24;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.arc(310, 224, 45, -Math.PI/2, Math.PI/2);
            ctx.stroke();
            
            // Mug body
            ctx.fillStyle = "#ef476f";
            ctx.beginPath();
            ctx.roundRect(150, 140, 130, 160, [10, 10, 40, 40]);
            ctx.fill();
            
            // Mug opening
            ctx.fillStyle = "#ff7096";
            ctx.beginPath();
            ctx.ellipse(215, 140, 65, 18, 0, 0, 2 * Math.PI);
            ctx.fill();
            
            // Coffee contents
            ctx.fillStyle = "#5c3d2e";
            ctx.beginPath();
            ctx.ellipse(215, 143, 58, 14, 0, 0, 2 * Math.PI);
            ctx.fill();
            
            // Steam curls
            ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
            ctx.lineWidth = 4;
            ctx.lineCap = "round";
            ctx.beginPath(); ctx.moveTo(190, 110); ctx.bezierCurveTo(180, 95, 200, 85, 190, 65); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(240, 110); ctx.bezierCurveTo(230, 95, 250, 85, 240, 65); ctx.stroke();
            
        } else if (type === "keyboard") {
            ctx.fillStyle = "#0f172a"; // Dark desk mat
            ctx.fillRect(0, 0, 448, 448);
            
            // Keyboard body
            ctx.fillStyle = "#1e293b";
            ctx.strokeStyle = "#475569";
            ctx.lineWidth = 8;
            ctx.beginPath();
            ctx.roundRect(40, 160, 368, 130, 12);
            ctx.fill();
            ctx.stroke();
            
            // Draw rows of keys
            ctx.fillStyle = "#334155";
            const rowY = [180, 210, 240, 270];
            const cols = 12;
            const keyW = 23;
            const keyH = 18;
            const gap = 5;
            
            rowY.forEach((ry, rIdx) => {
                for (let c = 0; c < cols; c++) {
                    let kw = keyW;
                    let xOffset = 58 + c * (keyW + gap);
                    
                    if (rIdx === 3 && c === 4) {
                        kw = keyW * 4 + gap * 3;
                        ctx.fillStyle = "#06d6a0"; // Green neon spacebar
                    } else if (rIdx === 3 && (c > 4 && c < 8)) {
                        continue;
                    } else {
                        ctx.fillStyle = "#334155";
                    }
                    
                    if (xOffset + kw < 400) {
                        ctx.beginPath();
                        ctx.roundRect(xOffset, ry, kw, keyH, 3);
                        ctx.fill();
                    }
                }
            });
            
        } else if (type === "mouse") {
            ctx.fillStyle = "#180f2a"; // Purple mat
            ctx.fillRect(0, 0, 448, 448);
            
            // Wire
            ctx.strokeStyle = "#8b5cf6";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(224, 0);
            ctx.bezierCurveTo(200, 60, 250, 110, 224, 150);
            ctx.stroke();
            
            // Body Shadow
            ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
            ctx.beginPath();
            ctx.ellipse(228, 260, 68, 105, 0, 0, 2 * Math.PI);
            ctx.fill();
            
            // Mouse Body
            ctx.fillStyle = "#2e1065";
            ctx.strokeStyle = "#6d28d9";
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.ellipse(224, 254, 62, 98, 0, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Split Line
            ctx.strokeStyle = "#6d28d9";
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(224, 156); ctx.lineTo(224, 220); ctx.stroke();
            
            // Scroll wheel
            ctx.fillStyle = "#eab308"; // Neon scroll wheel
            ctx.beginPath();
            ctx.roundRect(220, 180, 8, 26, 3);
            ctx.fill();
        }
        
        const img = new Image();
        img.onload = () => {
            loadedImage = img;
            syncVizCanvas();
        };
        img.src = tempCanvas.toDataURL();
    }

    // --- Digit Fast R-CNN Model Training ---
    function checkModelStatus() {
        fetch("/api/train_status")
            .then(res => res.json())
            .then(data => {
                updateModelStatusUI(data);
                
                if (data.status === "training") {
                    btnTrainModel.disabled = true;
                    trainingProgressContainer.classList.remove("hidden");
                    trainingProgressBar.style.width = `${data.progress}%`;
                    trainingProgressMsg.textContent = data.message;
                    trainingProgressPct.textContent = `${data.progress}%`;
                    
                    if (data.loss !== null && data.accuracy !== null) {
                        trainingMetrics.classList.remove("hidden");
                        metricLoss.textContent = data.loss;
                        metricAcc.textContent = `${data.accuracy}%`;
                    }
                    
                    if (!trainingPollInterval) {
                        trainingPollInterval = setInterval(checkModelStatus, 1000);
                    }
                } else {
                    btnTrainModel.disabled = false;
                    if (data.status === "completed" || data.status === "failed") {
                        trainingProgressBar.style.width = `${data.progress}%`;
                        trainingProgressMsg.textContent = data.message;
                        trainingProgressPct.textContent = `${data.progress}%`;
                        
                        if (data.status === "completed" && data.loss !== null) {
                            trainingMetrics.classList.remove("hidden");
                            metricLoss.textContent = data.loss;
                            metricAcc.textContent = `${data.accuracy}%`;
                        }
                    }
                    
                    if (trainingPollInterval) {
                        clearInterval(trainingPollInterval);
                        trainingPollInterval = null;
                    }
                }
            })
            .catch(err => console.error("Error checking model status:", err));
    }
    
    function updateModelStatusUI(data) {
        if (data.model_available) {
            modelStatusBadge.textContent = "Trained";
            modelStatusBadge.className = "badge badge-success";
            modelStatusText.textContent = "Ready to run Fast R-CNN detector!";
        } else if (data.status === "training") {
            modelStatusBadge.textContent = "Training";
            modelStatusBadge.className = "badge badge-warning";
            modelStatusText.textContent = "Fast R-CNN learning digits & offsets in background...";
        } else {
            modelStatusBadge.textContent = "Untrained";
            modelStatusBadge.className = "badge badge-danger";
            modelStatusText.textContent = "Train the Fast R-CNN detector model (1-2 mins) to enable digit detection.";
        }
    }
    
    checkModelStatus();
    
    btnTrainModel.addEventListener("click", () => {
        fetch("/api/train", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    checkModelStatus();
                }
            })
            .catch(err => console.error("Error starting training:", err));
    });

    // --- Fast R-CNN Detection Inference ---
    function getActiveImageBase64() {
        if (activeModel === "digit") {
            return drawingCanvas.toDataURL("image/png");
        } else if (loadedImage) {
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = 448;
            tempCanvas.height = 448;
            const ctx = tempCanvas.getContext("2d");
            ctx.drawImage(loadedImage, 0, 0, 448, 448);
            return tempCanvas.toDataURL("image/png");
        }
        return null;
    }

    btnDetectInstant.addEventListener("click", () => {
        const base64Img = getActiveImageBase64();
        if (!base64Img) {
            alert("Provide an input drawing or image first!");
            return;
        }
        
        btnDetectInstant.disabled = true;
        const originalText = btnDetectInstant.innerHTML;
        btnDetectInstant.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Fast R-CNN...';
        
        const payload = {
            image: base64Img,
            model_type: activeModel,
            min_conf: parseFloat(sliderMinConf.value)
        };
        
        fetch("/api/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw err; });
            }
            return res.json();
        })
        .then(data => {
            detectionData = data;
            
            // Update stats
            statTotalProps.textContent = data.all_steps.length;
            const rawHits = data.all_steps.filter(s => s.is_detection).length;
            statRawHits.textContent = rawHits;
            statFinalDetections.textContent = data.final_detections.length;
            
            // Set tab to show proposals or detections automatically
            if (activeView === "features") {
                setViewTab("features");
            } else if (data.final_detections.length > 0) {
                setViewTab("detections");
            } else {
                setViewTab("proposals");
            }
            
            // Compute complexity comparison GFLOPs
            updateFLOPsCalculator(data.all_steps.length);
        })
        .catch(err => {
            console.error("Detection failed:", err);
            clearVisualOverlays();
            if (err.code === "MODEL_NOT_TRAINED") {
                alert("The custom Digit Fast R-CNN model must be trained first. Click 'Train Fast R-CNN' in settings!");
            } else {
                alert(err.error || "An error occurred during detection.");
            }
        })
        .finally(() => {
            btnDetectInstant.disabled = false;
            btnDetectInstant.innerHTML = originalText;
        });
    });

    // --- Render Overlays Based on Active View Tab ---
    function renderActiveView() {
        if (!detectionData) return;
        
        // Clean everything first
        proposalInteractiveLayer.innerHTML = "";
        detectionOverlay.innerHTML = "";
        heatmapCanvas.classList.add("hidden");
        
        const maxDisplayCount = parseInt(sliderMaxProps.value);
        
        if (activeView === "proposals") {
            // Render region proposals dynamically
            // Sort proposals so that high confidence ones are easy to select
            const steps = detectionData.all_steps.slice(0, maxDisplayCount);
            
            steps.forEach((step) => {
                const [x, y, w, h] = step.box;
                
                const pctX = (x / (activeModel === "digit" ? 256 : 448)) * 100;
                const pctY = (y / (activeModel === "digit" ? 256 : 448)) * 100;
                const pctW = (w / (activeModel === "digit" ? 256 : 448)) * 100;
                const pctH = (h / (activeModel === "digit" ? 256 : 448)) * 100;
                
                const propRect = document.createElement("div");
                propRect.className = "prop-rect";
                propRect.style.left = `${pctX}%`;
                propRect.style.top = `${pctY}%`;
                propRect.style.width = `${pctW}%`;
                propRect.style.height = `${pctH}%`;
                
                // Active Hover/Click Inspector trigger
                propRect.addEventListener("mouseenter", () => {
                    selectProposalForInspector(step);
                    propRect.classList.add("selected");
                });
                propRect.addEventListener("mouseleave", () => {
                    propRect.classList.remove("selected");
                    // Keep selected outline on canvas if desired, or clear
                    detectionOverlay.innerHTML = "";
                });
                
                proposalInteractiveLayer.appendChild(propRect);
            });
            
        } else if (activeView === "features") {
            // Renders base64 heatmap
            if (detectionData.heatmap) {
                const img = new Image();
                img.onload = () => {
                    const ctx = heatmapCanvas.getContext("2d");
                    ctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
                    ctx.drawImage(img, 0, 0, heatmapCanvas.width, heatmapCanvas.height);
                    heatmapCanvas.classList.remove("hidden");
                };
                img.src = "data:image/png;base64," + detectionData.heatmap;
            }
        } else if (activeView === "detections") {
            // Renders final bounding boxes
            detectionData.final_detections.forEach((det) => {
                const [x, y, w, h] = det.refined_box;
                const canvasSize = activeModel === "digit" ? 256 : 448;
                
                const pctX = (x / canvasSize) * 100;
                const pctY = (y / canvasSize) * 100;
                const pctW = (w / canvasSize) * 100;
                const pctH = (h / canvasSize) * 100;
                
                const detBox = document.createElement("div");
                detBox.className = "det-box";
                detBox.style.left = `${pctX}%`;
                detBox.style.top = `${pctY}%`;
                detBox.style.width = `${pctW}%`;
                detBox.style.height = `${pctH}%`;
                
                const detLabel = document.createElement("div");
                detLabel.className = "det-label";
                detLabel.innerHTML = `<i class="fa-solid fa-tag"></i> ${det.class} ${(det.score * 100).toFixed(0)}%`;
                
                detBox.appendChild(detLabel);
                detectionOverlay.appendChild(detBox);
            });
        }
    }

    // --- Interactive Proposal Inspector populator ---
    function selectProposalForInspector(step) {
        inspectorEmptyState.classList.add("hidden");
        inspectorFullState.classList.remove("hidden");
        inspectorActiveBadge.classList.remove("hidden");
        
        // 1. Coordinates Projection info (both Digit and ImageNet models now use the ResNet50 backbone with 1/32 stride)
        const scale = 0.03125;
        const scaleLabel = "1/32 (0.03125)";
        
        const [x, y, w, h] = step.box;
        roiImgCoords.textContent = `[x:${x}, y:${y}, w:${w}, h:${h}]`;
        roiDownsampleRatio.textContent = scaleLabel;
        
        const px1 = Math.round(x * scale);
        const py1 = Math.round(y * scale);
        const pw = Math.round(w * scale);
        const ph = Math.round(h * scale);
        roiConvCoords.textContent = `[x:${px1}, y:${py1}, w:${pw}, h:${ph}]`;
        
        // 2. Render 7x7 Pooled grid cells
        roiPooledGrid.innerHTML = "";
        const gridData = step.grid || new Array(49).fill(0);
        gridData.forEach((val) => {
            const cell = document.createElement("div");
            cell.className = "roi-cell";
            // Set opacity based on pooled activation value
            cell.style.opacity = Math.max(0.1, val);
            roiPooledGrid.appendChild(cell);
        });
        
        // 3. Classification Head softmax probabilities list
        clsProbabilityList.innerHTML = "";
        
        if (activeModel === "digit") {
            const logits = step.logits || new Array(11).fill(0);
            logits.forEach((prob, classIdx) => {
                const className = classIdx < 10 ? `Digit ${classIdx}` : "Background";
                const row = createProbabilityRow(className, prob);
                clsProbabilityList.appendChild(row);
            });
        } else {
            // ImageNet top predictions
            const topLogits = step.top_logits || [];
            topLogits.forEach((item) => {
                const row = createProbabilityRow(item.class, item.score);
                clsProbabilityList.appendChild(row);
            });
        }
        
        // 4. Bounding Box Regression Head offsets
        const offsets = step.offsets || [0, 0, 0, 0];
        offDx.textContent = offsets[0].toFixed(3);
        offDy.textContent = offsets[1].toFixed(3);
        offDw.textContent = offsets[2].toFixed(3);
        offDh.textContent = offsets[3].toFixed(3);
        
        const [rx, ry, rw, rh] = step.refined_box;
        compPropBox.textContent = `[${x}, ${y}, ${w}, ${h}]`;
        
        if (activeModel === "digit") {
            compRefBox.textContent = `[${rx}, ${ry}, ${rw}, ${rh}]`;
            
            // Draw comparison box outline overlay on main visualizer canvas
            drawInspectorRegressionOutline(step);
        } else {
            compRefBox.textContent = `N/A (ImageNet Disabled)`;
        }
    }
    
    function createProbabilityRow(name, val) {
        const row = document.createElement("div");
        row.className = "cls-row";
        
        const lbl = document.createElement("span");
        lbl.className = "cls-lbl";
        lbl.textContent = name;
        lbl.title = name;
        
        const wrapper = document.createElement("div");
        wrapper.className = "cls-bar-wrapper";
        
        const bar = document.createElement("div");
        bar.className = "cls-bar";
        bar.style.width = `${(val * 100).toFixed(0)}%`;
        
        const pct = document.createElement("span");
        pct.className = "cls-pct";
        pct.textContent = `${(val * 100).toFixed(0)}%`;
        
        wrapper.appendChild(bar);
        row.appendChild(lbl);
        row.appendChild(wrapper);
        row.appendChild(pct);
        return row;
    }
    
    // Draw regression comparison on visual overlay
    function drawInspectorRegressionOutline(step) {
        detectionOverlay.innerHTML = "";
        
        const canvasSize = 256;
        const [x, y, w, h] = step.box;
        const [rx, ry, rw, rh] = step.refined_box;
        
        // Draw dotted original box (red)
        const origOutline = document.createElement("div");
        origOutline.className = "original-proposal-outline";
        origOutline.style.left = `${(x / canvasSize) * 100}%`;
        origOutline.style.top = `${(y / canvasSize) * 100}%`;
        origOutline.style.width = `${(w / canvasSize) * 100}%`;
        origOutline.style.height = `${(h / canvasSize) * 100}%`;
        detectionOverlay.appendChild(origOutline);
        
        // Draw solid refined box (green)
        const refOutline = document.createElement("div");
        refOutline.className = "det-box";
        refOutline.style.borderColor = "var(--green)";
        refOutline.style.boxShadow = "0 0 4px var(--green)";
        refOutline.style.left = `${(rx / canvasSize) * 100}%`;
        refOutline.style.top = `${(ry / canvasSize) * 100}%`;
        refOutline.style.width = `${(rw / canvasSize) * 100}%`;
        refOutline.style.height = `${(rh / canvasSize) * 100}%`;
        
        const refLbl = document.createElement("div");
        refLbl.className = "det-label";
        refLbl.style.background = "var(--green)";
        refLbl.textContent = "Refined Box Regress Output";
        refOutline.appendChild(refLbl);
        
        detectionOverlay.appendChild(refOutline);
    }

    // --- Computational Efficiency FLOPs Calculator ---
    function updateFLOPsCalculator(numProposals) {
        // Assume backbone is ResNet50: ~8.0 GigaFLOPs (8x10^9 FLOPs)
        // Sibling Heads for classification & regression: ~0.02 GigaFLOPs per crop
        
        const backboneFlops = activeModel === "digit" ? 0.05 : 8.0; // MNIST backbone is small
        const headFlops = activeModel === "digit" ? 0.0005 : 0.02;  // Head FC layers
        
        // Classic R-CNN: full backbone run for every proposal crop
        const classicFlopsVal = numProposals * backboneFlops;
        
        // Fast R-CNN: backbone run once, lightweight heads run for each proposal
        const fastFlopsVal = backboneFlops + (numProposals * headFlops);
        
        // Savings multiplier
        const ratio = classicFlopsVal / fastFlopsVal;
        
        // Render
        calcClassicFlops.textContent = `${classicFlopsVal.toFixed(2)} GigaFLOPs`;
        calcFastFlops.textContent = `${fastFlopsVal.toFixed(3)} GigaFLOPs`;
        calcSavingsBadge.textContent = `${ratio.toFixed(1)}x Faster`;
    }
});
