/**
 * VisionID AI - Frontend Web Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    if (window.lucide) {
        lucide.createIcons();
    }

    // State Variables
    let currentTab = 'tab-webcam';
    let webcamStream = null;
    let webcamInterval = null;
    let isProcessingFrame = false;

    // Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPages = document.querySelectorAll('.tab-page');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const selectDetector = document.getElementById('select-detector');
    const hudActiveModel = document.getElementById('hud-active-model');

    // Tab Descriptions
    const tabMeta = {
        'tab-webcam': { title: 'Live Webcam Stream', sub: 'Real-time face detection and deep learning recognition' },
        'tab-image': { title: 'Image Inspector', sub: 'Inspect single or batch static images for faces' },
        'tab-video': { title: 'Video Processor', sub: 'Process video files with bounding box tracking overlay' },
        'tab-enroll': { title: 'Face Enrollment', sub: 'Register new face identity embeddings into gallery' },
        'tab-gallery': { title: 'Identity Database', sub: 'View and manage registered biometric identities' }
    };

    // Tab Switching
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            navButtons.forEach(b => b.classList.remove('active'));
            tabPages.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(target).classList.add('active');

            if (tabMeta[target]) {
                pageTitle.textContent = tabMeta[target].title;
                pageSubtitle.textContent = tabMeta[target].sub;
            }

            currentTab = target;
            if (target === 'tab-gallery') {
                loadIdentities();
            }
        });
    });

    // Detector Selection
    selectDetector.addEventListener('change', async () => {
        const chosen = selectDetector.value;
        try {
            const resp = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detector_type: chosen })
            });
            const data = await resp.json();
            if (data.status === 'success') {
                hudActiveModel.textContent = chosen.toUpperCase();
                showToast(`Detector switched to ${chosen.toUpperCase()}`);
            }
        } catch (e) {
            console.error(e);
            showToast('Failed to switch detector model');
        }
    });

    // ----------------------------------------------------
    // WEBCAM LIVE STREAMING LOGIC
    // ----------------------------------------------------
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const processedStreamView = document.getElementById('processed-stream-view');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const btnStartWebcam = document.getElementById('btn-start-webcam');
    const btnStopWebcam = document.getElementById('btn-stop-webcam');
    const btnSnapshotEnroll = document.getElementById('btn-snapshot-enroll');

    const statFps = document.getElementById('stat-fps');
    const statFaces = document.getElementById('stat-faces');
    const statLatency = document.getElementById('stat-latency');
    const liveIdentitiesList = document.getElementById('live-identities-list');

    btnStartWebcam.addEventListener('click', startWebcam);
    btnStopWebcam.addEventListener('click', stopWebcam);

    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            webcamVideo.srcObject = webcamStream;
            await webcamVideo.play();

            cameraPlaceholder.classList.add('hidden');
            btnStartWebcam.classList.add('hidden');
            btnStopWebcam.classList.remove('hidden');
            btnSnapshotEnroll.classList.remove('hidden');

            // Start capture loop
            webcamInterval = setInterval(processWebcamFrame, 100);
            showToast('Webcam initialized');
        } catch (e) {
            console.error(e);
            showToast('Could not access webcam camera device');
        }
    }

    function stopWebcam() {
        if (webcamInterval) clearInterval(webcamInterval);
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
        }

        processedStreamView.src = '';
        cameraPlaceholder.classList.remove('hidden');
        btnStartWebcam.classList.remove('hidden');
        btnStopWebcam.classList.add('hidden');
        btnSnapshotEnroll.classList.add('hidden');

        statFps.textContent = '0.0';
        statFaces.textContent = '0';
        statLatency.textContent = '0 ms';
        liveIdentitiesList.innerHTML = '<li class="empty-msg">No faces currently in view</li>';
        showToast('Webcam stopped');
    }

    async function processWebcamFrame() {
        if (isProcessingFrame || !webcamStream) return;
        isProcessingFrame = true;

        try {
            webcamCanvas.width = webcamVideo.videoWidth || 640;
            webcamCanvas.height = webcamVideo.videoHeight || 480;
            const ctx = webcamCanvas.getContext('2d');
            ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

            const frameB64 = webcamCanvas.toDataURL('image/jpeg', 0.75);

            const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameB64 })
            });

            const data = await resp.json();
            if (data.status === 'success') {
                processedStreamView.src = data.processed_image;

                statFps.textContent = data.telemetry.fps.toFixed(1);
                statFaces.textContent = data.telemetry.face_count;
                statLatency.textContent = `${data.telemetry.latency_ms.toFixed(0)} ms`;

                // Update live identity list
                if (data.detections.length === 0) {
                    liveIdentitiesList.innerHTML = '<li class="empty-msg">No faces in frame</li>';
                } else {
                    liveIdentitiesList.innerHTML = data.detections.map(d => `
                        <li class="live-item ${d.is_known ? '' : 'unknown'}">
                            <span class="identity">${d.identity}</span>
                            <span class="conf">${(d.match_confidence * 100).toFixed(0)}%</span>
                        </li>
                    `).join('');
                }
            }
        } catch (e) {
            console.error('Frame processing error:', e);
        } finally {
            isProcessingFrame = false;
        }
    }

    // Snapshot enroll button handler
    btnSnapshotEnroll.addEventListener('click', () => {
        const name = prompt("Enter full name for this face identity:");
        if (!name || !name.trim()) return;

        webcamCanvas.width = webcamVideo.videoWidth || 640;
        webcamCanvas.height = webcamVideo.videoHeight || 480;
        const ctx = webcamCanvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
        const frameB64 = webcamCanvas.toDataURL('image/jpeg', 0.9);

        enrollIdentity(name.trim(), frameB64);
    });

    // ----------------------------------------------------
    // IMAGE INSPECTOR LOGIC
    // ----------------------------------------------------
    const imageDropzone = document.getElementById('image-dropzone');
    const imageFileInput = document.getElementById('image-file-input');
    const imageResultView = document.getElementById('image-result-view');
    const imageResultPlaceholder = document.getElementById('image-result-placeholder');

    imageDropzone.addEventListener('click', () => imageFileInput.click());
    imageFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });

    async function handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            const b64 = e.target.result;
            showToast('Detecting faces in image...');
            try {
                const resp = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: b64 })
                });

                const data = await resp.json();
                if (data.status === 'success') {
                    imageResultView.src = data.annotated_image;
                    imageResultView.classList.remove('hidden');
                    imageResultPlaceholder.classList.add('hidden');
                    showToast(`Found ${data.telemetry.face_count} faces in ${data.telemetry.latency_ms.toFixed(0)}ms`);
                }
            } catch (err) {
                console.error(err);
                showToast('Failed to process image');
            }
        };
        reader.readAsDataURL(file);
    }

    // ----------------------------------------------------
    // FACE ENROLLMENT LOGIC
    // ----------------------------------------------------
    const enrollForm = document.getElementById('enroll-form');
    const enrollNameInput = document.getElementById('enroll-name');
    const enrollDropzone = document.getElementById('enroll-dropzone');
    const enrollFileInput = document.getElementById('enroll-file-input');
    const enrollPreviewImg = document.getElementById('enroll-preview-img');
    const enrollDropPrompt = document.getElementById('enroll-drop-prompt');
    let selectedEnrollB64 = null;

    enrollDropzone.addEventListener('click', () => enrollFileInput.click());
    enrollFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (evt) => {
                selectedEnrollB64 = evt.target.result;
                enrollPreviewImg.src = selectedEnrollB64;
                enrollPreviewImg.classList.remove('hidden');
                enrollDropPrompt.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    enrollForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = enrollNameInput.value.trim();
        if (!name || !selectedEnrollB64) {
            showToast('Please provide both a name and an image photo');
            return;
        }
        enrollIdentity(name, selectedEnrollB64);
    });

    async function enrollIdentity(name, imageB64) {
        showToast(`Enrolling '${name}' into database...`);
        try {
            const resp = await fetch('/api/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, image: imageB64 })
            });
            const data = await resp.json();
            if (data.status === 'success') {
                showToast(`Successfully enrolled identity: ${name}`);
                enrollNameInput.value = '';
                enrollPreviewImg.classList.add('hidden');
                enrollDropPrompt.classList.remove('hidden');
                selectedEnrollB64 = null;
            } else {
                showToast(`Enrollment failed: ${data.message}`);
            }
        } catch (e) {
            console.error(e);
            showToast('Error sending enrollment request');
        }
    }

    // ----------------------------------------------------
    // IDENTITY DATABASE GALLERY LOGIC
    // ----------------------------------------------------
    const identityCardsGrid = document.getElementById('identity-cards-grid');
    const btnRefreshGallery = document.getElementById('btn-refresh-gallery');

    btnRefreshGallery.addEventListener('click', loadIdentities);

    async function loadIdentities() {
        try {
            const resp = await fetch('/api/identities');
            const data = await resp.json();
            if (data.status === 'success') {
                renderIdentityCards(data.identities);
            }
        } catch (e) {
            console.error(e);
        }
    }

    function renderIdentityCards(identities) {
        if (identities.length === 0) {
            identityCardsGrid.innerHTML = '<p class="empty-msg">No identities enrolled yet.</p>';
            return;
        }

        identityCardsGrid.innerHTML = identities.map(id => `
            <div class="identity-card glass-panel">
                <div class="identity-avatar">${id.name.charAt(0).toUpperCase()}</div>
                <h4>${id.name}</h4>
                <p class="stat-lbl">${id.sample_count} enrolled sample vectors</p>
                <button class="btn btn-danger btn-sm" onclick="deleteIdentity('${id.name}')">
                    Delete
                </button>
            </div>
        `).join('');
    }

    window.deleteIdentity = async function(name) {
        if (!confirm(`Are you sure you want to delete identity '${name}'?`)) return;

        try {
            const resp = await fetch(`/api/identities/${encodeURIComponent(name)}`, { method: 'DELETE' });
            const data = await resp.json();
            if (data.status === 'success') {
                showToast(`Deleted identity: ${name}`);
                loadIdentities();
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Helper Toast Notification
    function showToast(msg) {
        const toast = document.getElementById('toast');
        toast.textContent = msg;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3500);
    }
});
