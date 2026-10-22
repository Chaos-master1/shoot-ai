// Kawkab AI - Settings (system overview + model cache manager + contract alerts)
// Wires the Settings bridge slots that had no UI: get_settings_overview,
// get_model_cache_info, download_model, delete_cached_model,
// get_contract_alerts — plus honest checksum-pinning status (Phase 3.1).

(function() {
    'use strict';

    var _models = [];

    function resolveBridge() {
        // Shared helper lives in utils.js (KawkabUtils.getBridge).
        return (window.KawkabUtils && KawkabUtils.getBridge) ? KawkabUtils.getBridge() : null;
    }

    function escapeHtml(s) {
        if (typeof window.escapeHtml === 'function') return window.escapeHtml(s);
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function _parse(result) {
        var data = typeof result === 'string' ? JSON.parse(result) : result;
        if (data && data.error) throw new Error(data.error);
        return data;
    }

    function initSettingsWorkspace() {
        if (document.getElementById('settings-overview') === null) return;
        if (document.getElementById('settings-overview').dataset.settingsInit === '1') return;
        document.getElementById('settings-overview').dataset.settingsInit = '1';

        loadOverview();
        loadModelCache();
        loadContractAlerts();
    }

    function loadOverview() {
        var bridge = resolveBridge();
        if (!bridge) return;
        bridge.get_settings_overview(function(result) {
            var el = document.getElementById('settings-overview');
            if (!el) return;
            try {
                var d = _parse(result);
                var pinBadge = d.checksum_pinned
                    ? '<span class="settings-pin ok">✓ checksums pinned</span>'
                    : '<span class="settings-pin warn" title="Models are downloaded without verified checksums in this build">⚠ checksums not pinned</span>';
                el.innerHTML =
                    '<div class="settings-grid">' +
                    '<div><span class="settings-k">App</span><span class="settings-v">' + escapeHtml(d.app.name) + ' v' + escapeHtml(d.app.version) + '</span></div>' +
                    '<div><span class="settings-k">Platform</span><span class="settings-v">' + escapeHtml(d.app.platform || '--') + '</span></div>' +
                    '<div><span class="settings-k">GPU</span><span class="settings-v">' + escapeHtml(d.gpu.backend || '--') + ' (tier ' + escapeHtml(String(d.gpu.tier != null ? d.gpu.tier : '--')) + ')</span></div>' +
                    '<div><span class="settings-k">Model</span><span class="settings-v">' + escapeHtml(d.model.current || '--') +
                    (d.model.recommended && d.model.recommended !== d.model.current
                        ? ' <em>(recommended: ' + escapeHtml(d.model.recommended) + ')</em>' : '') + '</span></div>' +
                    '<div><span class="settings-k">Model cache</span><span class="settings-v">' + escapeHtml(String(d.cache_size_mb)) + ' MB</span></div>' +
                    '<div>' + pinBadge + '</div>' +
                    '</div>';
            } catch (e) {
                el.innerHTML = '<p class="hint">System info unavailable: ' + escapeHtml(e.message) + '</p>';
            }
        });
    }

    function loadModelCache() {
        var bridge = resolveBridge();
        if (!bridge || !bridge.get_model_cache_info) return;
        bridge.get_model_cache_info(function(result) {
            var el = document.getElementById('settings-model-list');
            if (!el) return;
            try {
                var d = _parse(result);
                _models = d.models || [];
                if (!_models.length) {
                    el.innerHTML = '<p class="hint">No models known to the manager.</p>';
                    return;
                }
                var html = '';
                _models.forEach(function(m) {
                    var sizeLabel = m.cached
                        ? Number(m.size_mb).toLocaleString() + ' MB on disk'
                        : '~' + Number(m.size_mb).toLocaleString() + ' MB download';
                    html += '<div class="settings-model-row">' +
                        '<span class="settings-model-name">' + escapeHtml(m.name) +
                        (m.current_variant ? ' <em>(active)</em>' : '') + '</span>' +
                        '<span class="settings-model-size">' + sizeLabel + '</span>' +
                        (m.cached
                            ? '<button class="btn btn-secondary settings-model-btn" data-action="delete" data-model="' + escapeHtml(m.name) + '">Remove</button>'
                            : '<button class="btn btn-primary settings-model-btn" data-action="download" data-model="' + escapeHtml(m.name) + '">Download</button>') +
                        '</div>';
                });
                el.innerHTML = html;
                var sizeEl = document.getElementById('settings-cache-size');
                if (sizeEl) sizeEl.textContent = 'Total cache: ' + Number(d.cache_size_mb || 0).toLocaleString() + ' MB';

                el.querySelectorAll('.settings-model-btn').forEach(function(btn) {
                    btn.addEventListener('click', function() {
                        if (this.dataset.action === 'download') downloadModel(this.dataset.model, this);
                        else deleteModel(this.dataset.model, this);
                    });
                });
            } catch (e) {
                el.innerHTML = '<p class="hint">Model cache unavailable: ' + escapeHtml(e.message) + '</p>';
            }
        });
    }

    function downloadModel(name, btn) {
        var bridge = resolveBridge();
        if (!bridge) return;
        btn.disabled = true;
        btn.textContent = 'Downloading…';
        bridge.download_model(name, function(result) {
            btn.disabled = false;
            try {
                var d = _parse(result);
                if (d && d.success) {
                    showToast('Model ' + name + ' downloaded.', 'success');
                    loadModelCache();
                } else {
                    btn.textContent = 'Download';
                    showToast('Download failed: ' + (d && d.error ? d.error : 'unknown'), 'error');
                }
            } catch (e) {
                btn.textContent = 'Download';
                showToast('Download failed: ' + e.message, 'error');
            }
        });
    }

    function deleteModel(name, btn) {
        var bridge = resolveBridge();
        if (!bridge) return;
        btn.disabled = true;
        bridge.delete_cached_model(name, function(result) {
            btn.disabled = false;
            try {
                var d = _parse(result);
                if (d && d.success) {
                    showToast('Model ' + name + ' removed.', 'success');
                    loadModelCache();
                } else {
                    showToast('Remove failed.', 'error');
                }
            } catch (e) {
                showToast('Remove failed: ' + e.message, 'error');
            }
        });
    }

    function loadContractAlerts() {
        var bridge = resolveBridge();
        if (!bridge || !bridge.get_contract_alerts) return;
        bridge.get_contract_alerts(function(result) {
            var el = document.getElementById('settings-contract-alerts');
            if (!el) return;
            try {
                var d = _parse(result);
                var alerts = d.alerts || [];
                if (!alerts.length) {
                    el.innerHTML = '<p class="hint">No contract alerts — all squad contracts are current.</p>';
                    return;
                }
                var html = '<ul class="settings-alert-list">';
                alerts.forEach(function(a) {
                    html += '<li class="settings-alert ' + (a.level === 'critical' ? 'critical' : 'warning') + '">' +
                        '<strong>' + escapeHtml(a.player) + '</strong> — ' + escapeHtml(a.message) + '</li>';
                });
                el.innerHTML = html + '</ul>';
            } catch (e) {
                el.innerHTML = '<p class="hint">Alerts unavailable: ' + escapeHtml(e.message) + '</p>';
            }
        });
    }

    window.KawkabSettings = { initSettingsWorkspace: initSettingsWorkspace };
    window.initSettingsWorkspace = initSettingsWorkspace;
})();
