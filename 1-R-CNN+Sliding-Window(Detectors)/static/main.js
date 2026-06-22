document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const btnModelDigit = document.getElementById("btn-model-digit");
    const btnModelResNet50 = document.getElementById("btn-model-resnet50");
    const digitWorkspace = document.getElementById("digit-workspace");
    const resnet50Workspace = document.getElementById("resnet50-workspace");
    const workspaceHint = document.getElementById("workspace-hint");
    
    const sliderWindowSize = document.getElementById("slider-window-size");
    const sliderStride = document.getElementById("slider-stride");
    const sliderMinConf = document.getElementById("slider-min-conf");
    const valWindowSize = document.getElementById("val-window-size");
    const valStride = document.getElementById("val-stride");
    const valMinConf = document.getElementById("val-min-conf");
    
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
    const detectionOverlay = document.getElementById("detection-overlay");
    const slidingBoxGuide = document.getElementById("sliding-box-guide");
    const cropCanvas = document.getElementById("crop-canvas");
    const predictionsBars = document.getElementById("predictions-bars");
    
    const btnDetectInstant = document.getElementById("btn-detect-instant");
    const btnDetectAnimate = document.getElementById("btn-detect-animate");
    const btnStop = document.getElementById("btn-stop");
    
    const statTotalSteps = document.getElementById("stat-total-steps");
    const statRawHits = document.getElementById("stat-raw-hits");
    const statFinalDetections = document.getElementById("stat-final-detections");

    const selectMethod = document.getElementById("select-method");
    const containerWindowSize = document.getElementById("container-window-size");
    const containerStride = document.getElementById("container-stride");
    const detectMethodBadge = document.getElementById("detect-method-badge");

    // Application State
    let activeModel = "digit"; // "digit" or "resnet50"
    let animationFrameId = null;
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let trainingPollInterval = null;
    let loadedImage = null; // Store uploaded / generated image object
    
    // Canvas Contexts
    const drawCtx = drawingCanvas.getContext("2d");
    const vizCtx = vizCanvas.getContext("2d");
    const heatmapCtx = heatmapCanvas.getContext("2d");
    const cropCtx = cropCanvas.getContext("2d");
    
    // Initialize Drawing Canvas (White Background, Black Ink)
    function clearDrawCanvas() {
        drawCtx.fillStyle = "#ffffff";
        drawCtx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        syncVizCanvas();
    }
    clearDrawCanvas();
    
    // Synchronize the main visualization canvas with drawing canvas or uploaded image
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
            vizCtx.font = "14px Outfit";
            vizCtx.textAlign = "center";
            vizCtx.fillText("Upload an image to start", vizCanvas.width / 2, vizCanvas.height / 2);
        }
        
        // Reset overlay and heatmap
        detectionOverlay.innerHTML = "";
        heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        heatmapCanvas.classList.add("hidden");
        slidingBoxGuide.classList.add("hidden");
    }

    // --- Hyperparameter Sliders ---
    sliderWindowSize.addEventListener("input", (e) => {
        valWindowSize.textContent = e.target.value;
    });
    sliderStride.addEventListener("input", (e) => {
        valStride.textContent = e.target.value;
    });
    sliderMinConf.addEventListener("input", (e) => {
        valMinConf.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // --- Search Method Dropdown & Prompt Logic ---
    function handleMethodChange() {
        if (selectMethod.value === "selective_search") {
            containerWindowSize.classList.add("hidden");
            containerStride.classList.add("hidden");
            updateMethodBadge("selective_search");
        } else {
            containerWindowSize.classList.remove("hidden");
            containerStride.classList.remove("hidden");
            updateMethodBadge("sliding_window");
        }
    }
    
    selectMethod.addEventListener("change", handleMethodChange);



    function updateMethodBadge(method) {
        if (method === "selective_search") {
            detectMethodBadge.textContent = "Selective Search";
            detectMethodBadge.className = "badge badge-success";
        } else {
            detectMethodBadge.textContent = "Sliding Window";
            detectMethodBadge.className = "badge";
        }
    }

    // --- Model Toggle Switch ---
    btnModelDigit.addEventListener("click", () => {
        if (activeModel === "digit") return;
        activeModel = "digit";
        btnModelDigit.classList.add("active");
        btnModelResNet50.classList.remove("active");
        digitWorkspace.classList.add("active");
        resnet50Workspace.classList.remove("active");
        workspaceHint.textContent = "Draw multiple digits here";
        
        // Show/hide relevant settings
        document.getElementById("digit-training-section").classList.remove("hidden");
        
        // Reset and sync
        stopEvaluationAnimation();
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
        
        // Show/hide relevant settings
        document.getElementById("digit-training-section").classList.add("hidden");
        
        // Reset and sync
        stopEvaluationAnimation();
        syncVizCanvas();
    });

    // --- Drawing Functionality ---
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
        // Scale coordinate space to actual canvas width/height
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
        drawCtx.lineWidth = 14;
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

    // --- Image Upload & Drag-Drop ---
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

    // --- Draw Stylized Samples directly to Canvas for ResNet50 ---
    const sampleButtons = document.querySelectorAll(".sample-btn");
    sampleButtons.forEach(btn => {
        btn.addEventListener("click", (e) => {
            const type = e.target.getAttribute("data-sample");
            drawSampleImage(type);
        });
    });

    function drawSampleImage(type) {
        // Create an offscreen canvas to generate sample images
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 400;
        tempCanvas.height = 400;
        const ctx = tempCanvas.getContext("2d");
        
        if (type === "mug") {
            // Draw Coffee Mug scene
            ctx.fillStyle = "#1e293b"; // Background table
            ctx.fillRect(0, 0, 400, 400);
            
            // Draw wooden pattern
            ctx.strokeStyle = "#334155";
            ctx.lineWidth = 4;
            for (let i = 50; i < 400; i += 70) {
                ctx.beginPath();
                ctx.moveTo(0, i);
                ctx.lineTo(400, i);
                ctx.stroke();
            }
            
            // Mug handle
            ctx.strokeStyle = "#ef476f"; // Coral Red Mug
            ctx.lineWidth = 20;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.arc(280, 200, 40, -Math.PI/2, Math.PI/2);
            ctx.stroke();
            
            // Mug body
            ctx.fillStyle = "#ef476f";
            ctx.beginPath();
            ctx.roundRect(140, 130, 110, 140, [10, 10, 40, 40]);
            ctx.fill();
            
            // Mug top opening ellipse
            ctx.fillStyle = "#ff7096";
            ctx.beginPath();
            ctx.ellipse(195, 130, 55, 15, 0, 0, 2 * Math.PI);
            ctx.fill();
            
            // Coffee inside
            ctx.fillStyle = "#5c3d2e";
            ctx.beginPath();
            ctx.ellipse(195, 133, 50, 12, 0, 0, 2 * Math.PI);
            ctx.fill();
            
            // Steaming lines
            ctx.strokeStyle = "rgba(255,255,255,0.4)";
            ctx.lineWidth = 4;
            ctx.lineCap = "round";
            // Steam 1
            ctx.beginPath();
            ctx.moveTo(170, 105);
            ctx.bezierCurveTo(165, 95, 175, 85, 170, 70);
            ctx.stroke();
            // Steam 2
            ctx.beginPath();
            ctx.moveTo(210, 105);
            ctx.bezierCurveTo(205, 95, 215, 85, 210, 70);
            ctx.stroke();
            
        } else if (type === "keyboard") {
            // Draw Keyboard scene
            ctx.fillStyle = "#0f172a"; // Dark desk mat
            ctx.fillRect(0, 0, 400, 400);
            
            // Keyboard Base
            ctx.fillStyle = "#1e293b";
            ctx.strokeStyle = "#475569";
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.roundRect(40, 150, 320, 120, 12);
            ctx.fill();
            ctx.stroke();
            
            // Key Grid rows
            ctx.fillStyle = "#64748b";
            const rowY = [165, 190, 215, 240];
            const cols = 12;
            const keyW = 20;
            const keyH = 16;
            const gap = 4;
            
            rowY.forEach((ry, rIdx) => {
                for (let c = 0; c < cols; c++) {
                    let kw = keyW;
                    let xOffset = 55 + c * (keyW + gap);
                    
                    // Spacebar and special key sizing
                    if (rIdx === 3 && c === 4) {
                        kw = keyW * 4 + gap * 3;
                        ctx.fillStyle = "#06d6a0"; // Neon spacebar
                    } else if (rIdx === 3 && (c > 4 && c < 8)) {
                        continue; // skip occupied spacebar columns
                    } else {
                        ctx.fillStyle = "#334155";
                    }
                    
                    if (xOffset + kw < 345) {
                        ctx.beginPath();
                        ctx.roundRect(xOffset, ry, kw, keyH, 3);
                        ctx.fill();
                    }
                }
            });
            
        } else if (type === "mouse") {
            // Draw Mouse scene
            ctx.fillStyle = "#1e1b4b"; // Purple-ish desk mat
            ctx.fillRect(0, 0, 400, 400);
            
            // Mouse Wire
            ctx.strokeStyle = "#6366f1";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(200, 0);
            ctx.bezierCurveTo(180, 50, 220, 100, 200, 140);
            ctx.stroke();
            
            // Mouse Body shadow
            ctx.fillStyle = "rgba(0,0,0,0.4)";
            ctx.beginPath();
            ctx.ellipse(204, 235, 60, 95, 0, 0, 2 * Math.PI);
            ctx.fill();
            
            // Mouse Body
            ctx.fillStyle = "#312e81";
            ctx.strokeStyle = "#4f46e5";
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.ellipse(200, 230, 55, 90, 0, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Left/Right split line
            ctx.strokeStyle = "#4f46e5";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(200, 140);
            ctx.lineTo(200, 200);
            ctx.stroke();
            
            // Scroll Wheel
            ctx.fillStyle = "#06d6a0"; // Neon scroll wheel
            ctx.beginPath();
            ctx.roundRect(196, 160, 8, 24, 3);
            ctx.fill();
            
            // Side LED light strips
            ctx.strokeStyle = "#00b4d8";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(200, 230, 52, 0.2*Math.PI, 0.8*Math.PI);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(200, 230, 52, 1.2*Math.PI, 1.8*Math.PI);
            ctx.stroke();
        }
        
        // Export offscreen image to image element
        const img = new Image();
        img.onload = () => {
            loadedImage = img;
            syncVizCanvas();
        };
        img.src = tempCanvas.toDataURL();
    }

    // --- Digit CNN Model Training Polling ---
    function checkModelStatus() {
        fetch("/api/train_status")
            .then(res => res.json())
            .then(data => {
                updateModelStatusUI(data);
                
                if (data.status === "training") {
                    // Show progress bars and lock train button
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
                    
                    // Poll again if not started
                    if (!trainingPollInterval) {
                        trainingPollInterval = setInterval(checkModelStatus, 800);
                    }
                } else {
                    // Training complete or idle
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
            modelStatusText.textContent = "Ready to run sliding window digit detection!";
        } else if (data.status === "training") {
            modelStatusBadge.textContent = "Training";
            modelStatusBadge.className = "badge badge-warning";
            modelStatusText.textContent = "CNN is learning on MNIST in background...";
        } else {
            modelStatusBadge.textContent = "Untrained";
            modelStatusBadge.className = "badge badge-danger";
            modelStatusText.textContent = "Please train the CNN model on digits (1-2 mins) to enable digit detection.";
        }
    }
    
    // Start initial check
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

    // --- Core Object Detection Logic ---
    
    // Get Base64 image depending on active workspace
    function getActiveImageBase64() {
        if (activeModel === "digit") {
            return drawingCanvas.toDataURL("image/png");
        } else if (loadedImage) {
            // Draw image on a temp canvas to convert to base64
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = 400;
            tempCanvas.height = 400;
            const ctx = tempCanvas.getContext("2d");
            ctx.drawImage(loadedImage, 0, 0, 400, 400);
            return tempCanvas.toDataURL("image/png");
        }
        return null;
    }

    btnDetectInstant.addEventListener("click", () => {
        const base64Img = getActiveImageBase64();
        if (!base64Img) {
            alert("Please draw on the canvas or upload an image first!");
            return;
        }
        
        stopEvaluationAnimation();
        
        // Show loading state
        btnDetectInstant.disabled = true;
        btnDetectAnimate.disabled = true;
        const originalText = btnDetectInstant.innerHTML;
        btnDetectInstant.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        
        const payload = {
            image: base64Img,
            model_type: activeModel,
            window_size: parseInt(sliderWindowSize.value),
            stride: parseInt(sliderStride.value),
            min_conf: parseFloat(sliderMinConf.value),
            method: selectMethod.value
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
            // Update active method badge
            updateMethodBadge(data.method);
            
            // Render instant results
            statTotalSteps.textContent = data.all_steps.length;
            
            const rawHits = data.all_steps.filter(s => s.is_detection).length;
            statRawHits.textContent = rawHits;
            statFinalDetections.textContent = data.final_detections.length;
            
            // Draw final bounding boxes
            drawFinalBoundingBoxes(data.final_detections);
            
            // Draw a quick static mock preview of predictions
            updatePredictionDashboardStatic(data.final_detections);
        })
        .catch(err => {
            console.error("Detection failed:", err);
            if (err.code === "MODEL_NOT_TRAINED") {
                alert("The custom digit CNN must be trained first. Click 'Train Classifier' under the model selector!");
            } else {
                alert(err.error || "An error occurred during detection.");
            }
        })
        .finally(() => {
            btnDetectInstant.disabled = false;
            btnDetectAnimate.disabled = false;
            btnDetectInstant.innerHTML = originalText;
        });
    });

    // --- Interactive Evaluation Animation ---
    btnDetectAnimate.addEventListener("click", () => {
        const base64Img = getActiveImageBase64();
        if (!base64Img) {
            alert("Please draw on the canvas or upload an image first!");
            return;
        }
        
        stopEvaluationAnimation();
        
        // Disable controls
        btnDetectInstant.disabled = true;
        btnDetectAnimate.disabled = true;
        btnStop.classList.remove("hidden");
        
        const payload = {
            image: base64Img,
            model_type: activeModel,
            window_size: parseInt(sliderWindowSize.value),
            stride: parseInt(sliderStride.value),
            min_conf: parseFloat(sliderMinConf.value),
            method: selectMethod.value
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
            // Update active method badge
            updateMethodBadge(data.method);
            
            runSlidingWindowAnimation(data.all_steps, data.final_detections);
        })
        .catch(err => {
            console.error("Detection failed:", err);
            resetAnimationButtons();
            if (err.code === "MODEL_NOT_TRAINED") {
                alert("The custom digit CNN must be trained first. Click 'Train Classifier' under the model selector!");
            } else {
                alert(err.error || "An error occurred during detection.");
            }
        });
    });

    btnStop.addEventListener("click", stopEvaluationAnimation);

    function stopEvaluationAnimation() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        resetAnimationButtons();
        slidingBoxGuide.classList.add("hidden");
    }

    function resetAnimationButtons() {
        btnDetectInstant.disabled = false;
        btnDetectAnimate.disabled = false;
        btnStop.classList.add("hidden");
    }

    function drawFinalBoundingBoxes(detections) {
        detectionOverlay.innerHTML = "";
        detections.forEach(det => {
            const [x, y, w, h] = det.box;
            
            // Map coordinates from 400x400 source space to rendering container percent
            const pctX = (x / 400) * 100;
            const pctY = (y / 400) * 100;
            const pctW = (w / 400) * 100;
            const pctH = (h / 400) * 100;
            
            const boxDiv = document.createElement("div");
            boxDiv.className = "detection-box";
            boxDiv.style.left = `${pctX}%`;
            boxDiv.style.top = `${pctY}%`;
            boxDiv.style.width = `${pctW}%`;
            boxDiv.style.height = `${pctH}%`;
            
            const labelDiv = document.createElement("div");
            labelDiv.className = "detection-label";
            labelDiv.innerHTML = `<i class="fa-solid fa-tag"></i> ${det.class} (${(det.score * 100).toFixed(0)}%)`;
            
            boxDiv.appendChild(labelDiv);
            detectionOverlay.appendChild(boxDiv);
        });
    }

    // Run animation step-by-step
    function runSlidingWindowAnimation(steps, finalDetections) {
        // Clear previous overlays and heatmaps
        detectionOverlay.innerHTML = "";
        heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        heatmapCanvas.classList.remove("hidden");
        slidingBoxGuide.classList.remove("hidden");
        
        let stepIdx = 0;
        const totalStepsCount = steps.length;
        statTotalSteps.textContent = totalStepsCount;
        
        let rawHitsCount = 0;
        statRawHits.textContent = 0;
        statFinalDetections.textContent = "-";
        
        // Speed scaling based on number of steps
        // If we have 1000 steps, we process 5 steps per frame.
        // If we have 100 steps, we process 1 step per frame.
        const stepsPerFrame = Math.max(1, Math.ceil(totalStepsCount / 200));
        
        function animate() {
            for (let i = 0; i < stepsPerFrame; i++) {
                if (stepIdx >= totalStepsCount) {
                    // Animation complete
                    slidingBoxGuide.classList.add("hidden");
                    heatmapCanvas.classList.add("hidden"); // hide heatmap to show final clean boxes
                    drawFinalBoundingBoxes(finalDetections);
                    statFinalDetections.textContent = finalDetections.length;
                    resetAnimationButtons();
                    animationFrameId = null;
                    return;
                }
                
                const step = steps[stepIdx];
                const [x, y, w, h] = step.box;
                
                // 1. Update the HTML sliding window bounding box guide
                const pctX = (x / 400) * 100;
                const pctY = (y / 400) * 100;
                const pctW = (w / 400) * 100;
                const pctH = (h / 400) * 100;
                
                slidingBoxGuide.style.left = `${pctX}%`;
                slidingBoxGuide.style.top = `${pctY}%`;
                slidingBoxGuide.style.width = `${pctW}%`;
                slidingBoxGuide.style.height = `${pctH}%`;
                
                // 2. Crop the image in real-time and draw to crop preview
                cropCtx.fillStyle = "#ffffff";
                cropCtx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
                
                // Draw crop from vizCanvas which holds current source image
                cropCtx.drawImage(
                    vizCanvas,
                    x, y, w, h,            // Source crop region
                    0, 0, 80, 80           // Destination crop preview
                );
                
                // 3. Update the classification confidence graph
                updateLivePredictionUI(step);
                
                // 4. Update stats and heatmaps if hit detected
                if (step.is_detection) {
                    rawHitsCount++;
                    statRawHits.textContent = rawHitsCount;
                    
                    // Draw semi-transparent hit overlay on heatmap canvas
                    heatmapCtx.fillStyle = "rgba(239, 71, 111, 0.15)";
                    heatmapCtx.fillRect(x, y, w, h);
                    
                    // Draw temporary light green border on viz overlay
                    const tempHit = document.createElement("div");
                    tempHit.className = "detection-box";
                    tempHit.style.borderColor = "rgba(6, 214, 160, 0.4)";
                    tempHit.style.boxShadow = "none";
                    tempHit.style.left = `${pctX}%`;
                    tempHit.style.top = `${pctY}%`;
                    tempHit.style.width = `${pctW}%`;
                    tempHit.style.height = `${pctH}%`;
                    detectionOverlay.appendChild(tempHit);
                }
                
                stepIdx++;
            }
            
            // Keep stats current
            statTotalSteps.textContent = `${stepIdx} / ${totalStepsCount}`;
            
            animationFrameId = requestAnimationFrame(animate);
        }
        
        animate();
    }

    function updateLivePredictionUI(step) {
        predictionsBars.innerHTML = "";
        
        if (step.class === "Background") {
            predictionsBars.innerHTML = `
                <div class="pred-row">
                    <span class="pred-label">Background</span>
                    <div class="pred-bar-container">
                        <div class="pred-bar" style="width: 100%; background: #9ba4b0;"></div>
                    </div>
                    <span class="pred-val">100%</span>
                </div>
            `;
            return;
        }
        
        // Show the top prediction
        const label = activeModel === "digit" ? `Digit ${step.class}` : step.class;
        const scorePct = (step.score * 100).toFixed(0);
        
        const isHit = step.is_detection;
        const color = isHit ? "var(--color-success)" : "var(--color-primary)";
        
        predictionsBars.innerHTML = `
            <div class="pred-row">
                <span class="pred-label">${label}</span>
                <div class="pred-bar-container">
                    <div class="pred-bar" style="width: ${scorePct}%; background: ${color};"></div>
                </div>
                <span class="pred-val" style="color: ${color};">${scorePct}%</span>
            </div>
            <div class="pred-row" style="font-size: 0.7rem; color: var(--text-muted);">
                <span>Threshold: ${(parseFloat(sliderMinConf.value)*100).toFixed(0)}%</span>
                <span style="margin-left: auto;">${isHit ? 'TRIGGERED HIT' : 'BELOW THRESHOLD'}</span>
            </div>
        `;
    }

    function updatePredictionDashboardStatic(detections) {
        predictionsBars.innerHTML = "";
        if (detections.length === 0) {
            predictionsBars.innerHTML = '<div class="predictions-empty">No objects detected above threshold.</div>';
            return;
        }
        
        // Display top 3 final detections
        const displayDets = detections.slice(0, 3);
        displayDets.forEach(det => {
            const label = activeModel === "digit" ? `Digit ${det.class}` : det.class;
            const scorePct = (det.score * 100).toFixed(0);
            
            const row = document.createElement("div");
            row.className = "pred-row";
            row.innerHTML = `
                <span class="pred-label">${label}</span>
                <div class="pred-bar-container">
                    <div class="pred-bar" style="width: ${scorePct}%; background: var(--color-success);"></div>
                </div>
                <span class="pred-val" style="color: var(--color-success);">${scorePct}%</span>
            `;
            predictionsBars.appendChild(row);
        });
    }
});
