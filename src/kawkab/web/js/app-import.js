// Kawkab AI - Vendor Data Import (season / tracking / events)
// Panel logic for the vendor-import section in the Data Providers tab.
// Follows the app-data-providers.js IIFE + window.__kawkab seam pattern.
(function() {
    'use strict';

    var escapeHtml = function(v) {
        return window.__kawkab && window.__kawkab.escapeHtml
            ? window.__kawkab.escapeHtml(String(v == null ? '' : v))
            : String(v == null ? '' : v);
    };

    function $(id) { return document.getElementById(id); }

    function setResult(id, text, ok) {
        var el = $(id);
        if (!el) return;
        el.textContent = text;
        el.className = 'feedback-result ' + (ok === true ? 'success' : ok === false ? 'error' : '');
    }

    async function importSeason(bridge) {
        var dir = ($('season-dir') || {}).value || '';
        if (!dir) { setResult('season-import-result', 'Enter a folder path first.', false); return; }
        var comp = ($('season-competition') || {}).value || '';
        var sidRaw = ($('season-id') || {}).value || '';
        var sid = parseInt(sidRaw, 10);
        setResult('season-import-result', 'Importing season…', null);
        try {
            var res = JSON.parse(await bridge.import_season_directory(
                dir, comp, isFinite(sid) ? sid : 0, 0
            ));
            if (!res.success) {
                setResult('season-import-result', '❌ ' + (res.error || 'import failed'), false);
                return;
            }
            setResult('season-import-result',
                '✅ imported ' + res.imported +
                ', already present ' + res.skipped_already +
                ', not event files ' + res.skipped_not_events +
                ', failed ' + res.failed +
                ' (' + res.total_files + ' files scanned)', true);
        } catch (e) {
            setResult('season-import-result', '❌ ' + e, false);
        }
    }

    async function importTracking(bridge) {
        var path = ($('tracking-path') || {}).value || '';
        if (!path) { setResult('tracking-import-result', 'Enter a tracking file path first.', false); return; }
        var vendor = ($('tracking-vendor') || {}).value || '';
        var away = ($('tracking-away-csv') || {}).value || '';
        setResult('tracking-import-result', 'Importing tracking feed…', null);
        try {
            var res = JSON.parse(await bridge.import_tracking_file(path, vendor, away, '', '', ''));
            if (!res.success) {
                setResult('tracking-import-result', '❌ ' + (res.error || 'import failed'), false);
                return;
            }
            var q = res.quality || {};
            setResult('tracking-import-result',
                '✅ match #' + res.match_id + ': ' + res.frames_imported + ' frames, ' +
                res.players_registered + ' players @ ' + res.fps + ' fps' +
                ' | quality: gaps ' + (q.frame_gaps || 0) +
                ', out-of-bounds ' + (q.out_of_bounds_pct || 0) + '%' +
                ', ball missing ' + (q.missing_ball_pct || 0) + '%', true);
        } catch (e) {
            setResult('tracking-import-result', '❌ ' + e, false);
        }
    }

    async function importEvents(bridge) {
        var path = ($('event-path') || {}).value || '';
        if (!path) { setResult('event-import-result', 'Enter an event file path first.', false); return; }
        var f7 = ($('event-f7-path') || {}).value || '';
        setResult('event-import-result', 'Importing events…', null);
        try {
            var res = JSON.parse(await bridge.import_event_file(path, f7, '', '', ''));
            if (!res.success) {
                setResult('event-import-result', '❌ ' + (res.error || 'import failed'), false);
                return;
            }
            setResult('event-import-result',
                '✅ match #' + res.match_id + ': ' +
                (res.events_imported != null ? res.events_imported + ' events' : 'imported') +
                ' (' + (res.provider || 'vendor') + ')', true);
        } catch (e) {
            setResult('event-import-result', '❌ ' + e, false);
        }
    }

    async function importKloppyStatsbomb(bridge) {
        var path = ($('kloppy-sb-path') || {}).value || '';
        if (!path) { setResult('kloppy-sb-result', 'Enter a StatsBomb events file path first.', false); return; }
        var lineup = ($('kloppy-sb-lineup') || {}).value || '';
        setResult('kloppy-sb-result', 'Importing events via kloppy…', null);
        try {
            var res = JSON.parse(await bridge.import_kloppy_statsbomb(path, lineup, '', '', ''));
            if (!res.success) {
                var msg = res.error || 'import failed';
                if (res.provider_unavailable) {
                    msg = '⛔ provider unavailable — ' + msg;
                } else {
                    msg = '❌ ' + msg;
                }
                setResult('kloppy-sb-result', msg, false);
                return;
            }
            setResult('kloppy-sb-result',
                '✅ match #' + res.match_id + ': ' + res.events_imported + ' events' +
                ' (skipped ' + res.events_skipped + '), ' + res.players_registered + ' players' +
                ' | shots ' + res.shots + ', goals ' + res.goals + ', vendor xG ' + res.xg_total +
                ' (kloppy ' + res.kloppy_version + ')', true);
        } catch (e) {
            setResult('kloppy-sb-result', '❌ ' + e, false);
        }
    }

    async function importKloppySkillcorner(bridge) {
        var meta = ($('kloppy-sc-meta') || {}).value || '';
        var raw = ($('kloppy-sc-raw') || {}).value || '';
        if (!meta || !raw) { setResult('kloppy-sc-result', 'Enter both SkillCorner meta and raw paths first.', false); return; }
        setResult('kloppy-sc-result', 'Importing tracking via kloppy…', null);
        try {
            var res = JSON.parse(await bridge.import_kloppy_skillcorner(meta, raw, '', '', ''));
            if (!res.success) {
                var msg = res.error || 'import failed';
                if (res.provider_unavailable) {
                    msg = '⛔ provider unavailable — ' + msg;
                } else {
                    msg = '❌ ' + msg;
                }
                setResult('kloppy-sc-result', msg, false);
                return;
            }
            var q = res.quality || {};
            setResult('kloppy-sc-result',
                '✅ match #' + res.match_id + ': ' + res.frames_imported + ' frames @ ' +
                res.fps + ' fps (kloppy ' + res.kloppy_version + ')' +
                ' | quality: gaps ' + (q.frame_gaps || 0) +
                ', ball missing ' + (q.missing_ball_pct || 0) + '%', true);
        } catch (e) {
            setResult('kloppy-sc-result', '❌ ' + e, false);
        }
    }

    // ── Init hook (called by app.js after QWebChannel connects) ────────
    window.__kawkab._initImportBridge = function(bridge) {
        if (!bridge) return;
        // Live season-import progress from the bridge signal
        if (bridge.importProgress) {
            bridge.importProgress.connect(function(n, file) {
                setResult('season-progress', '⏳ ' + Math.round(n) + ' files done — ' + file, null);
            });
        }
        var bind = function(btnId, fn) {
            var b = $(btnId);
            if (b) b.addEventListener('click', function() { fn(bridge); });
        };
        bind('season-import-btn', importSeason);
        bind('tracking-import-btn', importTracking);
        bind('event-import-btn', importEvents);
        bind('kloppy-sb-btn', importKloppyStatsbomb);
        bind('kloppy-sc-btn', importKloppySkillcorner);
    };
})();
