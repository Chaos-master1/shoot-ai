// Kawkab AI - First-run onboarding flow.
// Replaces the toast-only stub: a 4-step wizard that (2) actually checks the
// system (GPU tier, model cache) and offers the model download, (3) offers
// real import vs sample-data paths, (4) orients the coach in the app.

(function() {
    'use strict';

    var _step = 1;
    var _stepsTotal = 4;

    function resolveBridge() {
        // Shared helper lives in utils.js (KawkabUtils.getBridge).
        return (window.KawkabUtils && KawkabUtils.getBridge) ? KawkabUtils.getBridge() : null;
    }

    function _parse(result) {
        var data = typeof result === 'string' ? JSON.parse(result) : result;
        if (data && data.error) throw new Error(data.error);
        return data;
    }

    function showPane(n) {
        _step = n;
        for (var i = 1; i <= _stepsTotal; i++) {
            var pane = document.getElementById('onboarding-pane-' + i);
            if (pane) pane.classList.toggle('hidden', i !== n);
            var dot = document.querySelector('.onboarding-step-dot[data-step="' + i + '"]');
            if (dot) dot.classList.toggle('active', i === n);
        }
        var back = document.getElementById('onboarding-back');
        if (back) back.classList.toggle('hidden', n === 1);
        var next = document.getElementById('first-run-start');
        if (next) {
            next.textContent = n === _stepsTotal ? "Let's Go!" : 'Next →';
            // Steps 1 and 4 have no async work: Next closes on 4.
            next.dataset.action = n === _stepsTotal ? 'finish' : 'advance';
        }
        if (n === 2) runSystemCheck();
    }

    function runSystemCheck() {
        var bridge = resolveBridge();
        var el = document.getElementById('onboarding-system');
        var actionEl = document.getElementById('onboarding-model-action');
        if (!el) return;
        if (!bridge) {
            el.innerHTML = '<p class="hint">Bridge not connected yet — you can continue and check later in Settings.</p>';
            if (actionEl) actionEl.innerHTML = '';
            return;
        }

        el.innerHTML = '<p class="hint">Checking GPU and model cache…</p>';
        bridge.get_settings_overview(function(result) {
            try {
                var d = _parse(result);
                var rows = [];
                rows.push('<div><span class="settings-k">GPU</span><span class="settings-v">' +
                    _esc(d.gpu.backend || 'cpu') + ' (tier ' + _esc(String(d.gpu.tier != null ? d.gpu.tier : '--')) + ')</span></div>');
                rows.push('<div><span class="settings-k">Detection model</span><span class="settings-v">' +
                    _esc(d.model.current || '--') + ' / recommended ' + _esc(d.model.recommended || '--') + '</span></div>');
                rows.push('<div><span class="settings-k">Model cache</span><span class="settings-v">' +
                    Number(d.cache_size_mb || 0).toLocaleString() + ' MB</span></div>');
                el.innerHTML = rows.join('');
                if (actionEl) {
                    actionEl.innerHTML = '<button id="onboarding-model-btn" class="btn btn-primary">⬇ Download recommended model (' +
                        _esc(d.model.recommended || 'yolo11n') + ')</button>' +
                        '<p class="hint" style="margin-top:6px">Or continue — models also download on first analysis.</p>';
                    var btn = document.getElementById('onboarding-model-btn');
                    btn.addEventListener('click', function() {
                        btn.disabled = true;
                        btn.textContent = 'Downloading…';
                        bridge.download_model(d.model.recommended || 'yolo11n', function(dlResult) {
                            try {
                                _parse(dlResult);
                                btn.textContent = '✓ Downloaded';
                                showToast('Model downloaded.', 'success');
                            } catch (e) {
                                btn.disabled = false;
                                btn.textContent = 'Download failed — retry';
                                showToast('Download failed: ' + e.message, 'error');
                            }
                        });
                    });
                }
            } catch (e) {
                el.innerHTML = '<p class="hint">System info unavailable: ' + _esc(e.message) + '</p>';
            }
        });
    }

    function _esc(s) {
        if (typeof window.escapeHtml === 'function') return window.escapeHtml(s);
        return String(s == null ? '' : s).replace(/[&<>"']/g, function(c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    function loadSample() {
        var bridge = resolveBridge();
        var statusEl = document.getElementById('onboarding-sample-status');
        if (!bridge) {
            if (statusEl) statusEl.innerHTML = '<p class="hint">Bridge not connected yet.</p>';
            return;
        }
        if (statusEl) statusEl.innerHTML = '<p class="hint">Generating sample match…</p>';
        bridge.load_sample_data(function(result) {
            if (!statusEl) return;
            try {
                var d = _parse(result);
                if (d && d.ok) {
                    statusEl.innerHTML = '<p class="hint">✓ Sample match loaded: ' + _esc(d.match) +
                        ' (' + Number(d.events_count || 0).toLocaleString() + ' events). Check the Dashboard!</p>';
                    if (window.__kawkab && window.__kawkab.loadMatchHistory) window.__kawkab.loadMatchHistory();
                } else {
                    statusEl.innerHTML = '<p class="hint">Sample data failed: ' + _esc((d && d.error) || 'unknown') + '</p>';
                }
            } catch (e) {
                statusEl.innerHTML = '<p class="hint">Sample data failed: ' + _esc(e.message) + '</p>';
            }
        });
    }

    function finish() {
        try { localStorage.setItem('kawkab_first_run_done', 'true'); } catch (e) { /* private mode */ }
        if (window.KawkabUI && window.KawkabUI.closeModal) {
            window.KawkabUI.closeModal('first-run-modal');
            var modal = document.getElementById('first-run-modal');
            if (modal) modal.style.display = '';
        } else {
            var modal = document.getElementById('first-run-modal');
            if (modal) {
                modal.classList.add('hidden');
                modal.style.display = '';
            }
        }
    }

    function showFirstRunWizard() {
        var done = false;
        try { done = localStorage.getItem('kawkab_first_run_done') === 'true'; } catch (e) { /* private mode */ }
        if (done) return;
        var modal = document.getElementById('first-run-modal');
        if (!modal) return;

        // Open via the shared modal helper: wires the focus trap (Tab
        // cycles inside, Escape closes) and remembers the opener for
        // focus restore. The title heading is the dialog's labelled target.
        if (window.KawkabUI && window.KawkabUI.openModal) {
            window.KawkabUI.openModal('first-run-modal');
        } else {
            modal.classList.remove('hidden');
            modal.setAttribute('tabindex', '-1');
            modal.focus();
        }
        modal.style.display = 'flex';
        showPane(1);

        var next = document.getElementById('first-run-start');
        var back = document.getElementById('onboarding-back');
        var skip = document.getElementById('first-run-dismiss');
        if (next) {
            next.addEventListener('click', function() {
                if (this.dataset.action === 'finish') {
                    finish();
                    showToast('Welcome to Kawkab AI!', 'info');
                } else {
                    showPane(Math.min(_step + 1, _stepsTotal));
                }
            });
        }
        if (back) {
            back.addEventListener('click', function() {
                showPane(Math.max(_step - 1, 1));
            });
        }
        if (skip) skip.addEventListener('click', finish);

        var importBtn = document.getElementById('onboarding-import-btn');
        if (importBtn) {
            importBtn.addEventListener('click', function() {
                finish();
                var hash = window.location.hash;
                window.location.hash = 'upload';
                document.querySelectorAll('.section').forEach(function(s) {
                    s.classList.toggle('hidden', s.id !== 'upload-section');
                });
            });
        }
        var sampleBtn = document.getElementById('onboarding-sample-btn');
        if (sampleBtn) sampleBtn.addEventListener('click', loadSample);

        // Escape-to-close is handled by the shared modal focus trap when
        // available; keep a fallback for bare-open mode.
        modal.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && !(window.KawkabUI && window.KawkabUI.closeModal)) finish();
        });
    }

    window.KawkabOnboarding = { showFirstRunWizard: showFirstRunWizard };
})();
