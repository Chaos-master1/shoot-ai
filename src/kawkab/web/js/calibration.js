// Kawkab AI - Homography Calibration UI
// Allows the coach to click 4 pitch corners for camera-to-pitch transformation

(function() {
    'use strict';

    function escapeHtml(str) {
        if (typeof str !== 'string') str = String(str || '');
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    const canvas = document.getElementById('calibration-canvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const clicksCountEl = document.getElementById('clicks-count');
    const statusEl = document.getElementById('calibration-status');
    const resetBtn = document.getElementById('calibration-reset');
    const saveBtn = document.getElementById('calibration-save');
    const skipBtn = document.getElementById('calibration-skip');
    const lightglueBtn = document.getElementById('calibration-lightglue');

    const cornerLabels = [
        'Top-Left of Pitch',
        'Top-Right of Pitch',
        'Bottom-Right of Pitch',
        'Bottom-Left of Pitch',
    ];

    const cornerColors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'];

    let clicks = [];
    let videoFrame = null;
    let videoFrameWidth = 1280;
    let videoFrameHeight = 720;
    let cameraSegments = [];
    let selectedSegment = 0;
    let segmentCalibrations = {};

    function init() {
        if (!canvas) return;

        canvas.addEventListener('click', handleCanvasClick);
        if (resetBtn) resetBtn.addEventListener('click', resetCalibration);
        if (saveBtn) saveBtn.addEventListener('click', saveCalibration);
        if (skipBtn) skipBtn.addEventListener('click', skipCalibration);
        if (lightglueBtn) lightglueBtn.addEventListener('click', lightglueCalibration);

        if (typeof bridge !== 'undefined' && bridge) {
            loadFirstFrame();
        }
    }

    async function loadSegments() {
        const listEl = document.getElementById('segment-list');
        if (!listEl) return;
        try {
            if (typeof bridge !== 'undefined' && bridge.get_tactical_periods) {
                const periods = await bridge.get_tactical_periods(currentMatchId);
                if (periods && periods.segments) {
                    cameraSegments = periods.segments;
                }
            }
        } catch (e) {
            console.warn('Could not load segments:', e);
        }
        if (!cameraSegments.length) {
            cameraSegments = [{id: 0, label: 'Main Camera', frame_start: 0, frame_end: 0}];
        }
        try {
            if (typeof bridge !== 'undefined' && bridge.get_segment_homographies) {
                const result = await bridge.get_segment_homographies(currentMatchId);
                if (result && result.segments) {
                    segmentCalibrations = {};
                    Object.keys(result.segments).forEach(function(k) {
                        segmentCalibrations[k] = {method: result.segments[k].source || 'manual', confidence: result.segments[k].confidence || 0.5};
                    });
                }
            }
        } catch (e) {
            console.warn('Could not load segment calibrations:', e);
        }
        renderSegmentList(listEl);
        renderSummary();
    }

    function renderSegmentList(listEl) {
        listEl.innerHTML = '';
        cameraSegments.forEach((seg, idx) => {
            const item = document.createElement('div');
            item.className = 'segment-item' + (idx === selectedSegment ? ' active' : '');
            item.innerHTML =
                '<span class="segment-label">' + escapeHtml(seg.label || 'Segment ' + (idx + 1)) + '</span>' +
                '<span class="segment-status">' + (segmentCalibrations[idx] ? 'Calibrated' : 'Not set') + '</span>' +
                '<button class="btn btn-secondary segment-calibrate" data-idx="' + idx + '">Calibrate</button>';
            item.querySelector('.segment-calibrate').addEventListener('click', function(e) {
                e.stopPropagation();
                selectedSegment = parseInt(this.dataset.idx);
                renderSegmentList(listEl);
                if (statusEl) statusEl.textContent = 'Calibrating: ' + cameraSegments[selectedSegment].label;
                resetCalibration();
                loadFirstFrame();
            });
            item.addEventListener('click', function() {
                selectedSegment = idx;
                renderSegmentList(listEl);
                resetCalibration();
            });
            listEl.appendChild(item);
        });
    }

    async function loadFirstFrame() {
        try {
            const frameData = await bridge.get_first_frame(currentMatchId);
            if (frameData && frameData.path) {
                const img = new Image();
                img.onload = () => {
                    videoFrame = img;
                    videoFrameWidth = img.naturalWidth;
                    videoFrameHeight = img.naturalHeight;
                    canvas.width = videoFrameWidth;
                    canvas.height = videoFrameHeight;
                    redraw();
                };
                img.src = 'file:///' + frameData.path;
            }
        } catch (e) {
            console.error('Failed to load first frame:', e);
            drawPlaceholder();
        }
    }

    function drawPlaceholder() {
        if (!ctx) return;
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#666';
        ctx.font = '20px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click 4 pitch corners in order:', canvas.width / 2, canvas.height / 2 - 20);
        ctx.fillText('1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left',
                    canvas.width / 2, canvas.height / 2 + 10);
        drawGrid();
    }

    function drawGrid() {
        if (!ctx) return;
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        for (let i = 1; i < 10; i++) {
            ctx.beginPath();
            ctx.moveTo(i * canvas.width / 10, 0);
            ctx.lineTo(i * canvas.width / 10, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * canvas.height / 10);
            ctx.lineTo(canvas.width, i * canvas.height / 10);
            ctx.stroke();
        }
    }

    function handleCanvasClick(e) {
        if (clicks.length >= 4) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        clicks.push({x: Math.round(x), y: Math.round(y)});
        updateUI();
        redraw();
    }

    function redraw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (videoFrame) {
            ctx.drawImage(videoFrame, 0, 0, canvas.width, canvas.height);
        } else {
            drawPlaceholder();
        }

        drawGrid();

        clicks.forEach((pt, i) => {
            ctx.fillStyle = cornerColors[i];
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(String(i + 1), pt.x, pt.y - 20);
        });

        if (clicks.length === 4) {
            ctx.strokeStyle = '#50c878';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(clicks[0].x, clicks[0].y);
            for (let i = 1; i < 4; i++) {
                ctx.lineTo(clicks[i].x, clicks[i].y);
            }
            ctx.closePath();
            ctx.stroke();
        }
    }

    function updateUI() {
        if (clicksCountEl) clicksCountEl.textContent = clicks.length;

        if (clicks.length < 4) {
            if (statusEl) statusEl.textContent = `Click corner ${clicks.length + 1}: ${cornerLabels[clicks.length]}`;
            if (saveBtn) saveBtn.disabled = true;
        } else {
            if (statusEl) statusEl.textContent = 'All 4 corners set! Click Save to apply.';
            if (saveBtn) saveBtn.disabled = false;
        }
    }

    function resetCalibration() {
        clicks = [];
        updateUI();
        redraw();
    }

    function saveToBackend(cornersValue, method) {
        if (typeof bridge !== 'undefined' && bridge.save_segment_homography) {
            return bridge.save_segment_homography(
                currentMatchId,
                selectedSegment,
                cornersValue,
                105.0,
                68.0
            );
        }
        return bridge.save_homography(currentMatchId, cornersValue, 105.0, 68.0);
    }

    async function saveCalibration() {
        if (clicks.length !== 4) return;
        try {
            const result = await saveToBackend(JSON.stringify(clicks), 'manual');
            if (result && result.success) {
                segmentCalibrations[selectedSegment] = {method: 'manual', confidence: result.confidence || 0.5};
                renderSegmentList(document.getElementById('segment-list'));
                renderSummary();
                if (statusEl) {
                    statusEl.textContent = 'Saved! Confidence: ' + ((result.confidence || 0.5) * 100).toFixed(0) + '%';
                }
                if (window.showNotification) {
                    window.showNotification('Camera calibration saved', 'success');
                }
            }
        } catch (e) {
            console.error('Failed to save calibration:', e);
            if (statusEl) statusEl.textContent = 'Error saving calibration';
        }
    }

    async function skipCalibration() {
        try {
            const result = await saveToBackend('auto', 'auto');
            if (result && result.success) {
                segmentCalibrations[selectedSegment] = {method: 'auto', confidence: result.confidence || 0.3};
                renderSegmentList(document.getElementById('segment-list'));
                renderSummary();
                if (statusEl) {
                    statusEl.textContent = 'Using estimated calibration (no clicks)';
                }
            }
        } catch (e) {
            console.error('Failed to save estimated calibration:', e);
        }
    }

    async function lightglueCalibration() {
        if (lightglueBtn) {
            lightglueBtn.disabled = true;
            lightglueBtn.textContent = 'Running AI matching...';
        }
        try {
            const result = await saveToBackend('lightglue', 'lightglue');
            if (result && result.success) {
                segmentCalibrations[selectedSegment] = {method: 'lightglue', confidence: result.confidence || 0.85};
                renderSegmentList(document.getElementById('segment-list'));
                renderSummary();
                if (statusEl) {
                    statusEl.textContent = 'LightGlue calibration saved! Confidence: ' +
                        (result.confidence * 100).toFixed(0) + '%';
                }
                if (window.showNotification) {
                    window.showNotification('AI camera calibration saved', 'success');
                }
            } else if (result && result.error) {
                if (statusEl) statusEl.textContent = 'LightGlue failed: ' + result.error;
                if (window.showNotification) {
                    window.showNotification('AI calibration failed: ' + result.error, 'error');
                }
            }
        } catch (e) {
            console.error('LightGlue calibration failed:', e);
            if (statusEl) statusEl.textContent = 'Error running AI calibration';
        } finally {
            if (lightglueBtn) {
                lightglueBtn.disabled = false;
                lightglueBtn.textContent = 'Auto via AI Matching';
            }
        }
    }

    function renderSummary() {
        const container = document.getElementById('calibration-segment-cards');
        const summary = document.getElementById('calibration-summary');
        if (!container || !summary) return;
        const keys = Object.keys(segmentCalibrations);
        if (!keys.length) {
            summary.classList.add('hidden');
            return;
        }
        summary.classList.remove('hidden');
        container.innerHTML = '';
        keys.forEach(function(k) {
            const cal = segmentCalibrations[k];
            const seg = cameraSegments[parseInt(k)];
            if (!seg) return;
            const card = document.createElement('div');
            card.className = 'segment-card';
            card.innerHTML =
                '<span class="card-label">' + escapeHtml(seg.label || 'Segment ' + (parseInt(k) + 1)) + '</span>' +
                '<span class="card-detail">Method: ' + escapeHtml(cal.method || 'manual') + '</span>' +
                '<span class="card-detail">Confidence: ' + (cal.confidence ? (cal.confidence * 100).toFixed(0) + '%' : 'N/A') + '</span>';
            container.appendChild(card);
        });
    }

    function showCalibrationSection(matchId) {
        currentMatchId = matchId;
        const section = document.getElementById('calibration-section');
        if (section) {
            section.classList.remove('hidden');
            section.scrollIntoView({ behavior: 'smooth' });
            resetCalibration();
            loadFirstFrame();
            loadSegments();
            renderSummary();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.showCalibrationSection = showCalibrationSection;
})();
