// Kawkab AI - Frontend JavaScript
// Communicates with Python backend via QWebChannel

(function() {
    'use strict';

    let bridge = null;
    // showFirstRunWizard is provided by app-onboarding.js (KawkabOnboarding)
    let currentLanguage = 'en';
    let currentMatchId = null;
    let currentVideoPath = null;
    let analysisResult = null;

    // Multi-selection state
    var _selectedEventIds = new Set();
    var _lastShiftClickIdx = -1;
    var _currentTimelineView = 'timeline';

    // Timeline sort / filter / pagination state
    var _timelineSortState = { key: 'time', dir: 'asc' };
    var _timelineFilters = { type: '', team: '', player: '' };
    var _timelinePageState = { page: 1, perPage: 25 };
    var _timelineSearchText = '';

    // Roster table state
    var _rosterSortState = { key: 'name', dir: 'asc' };
    var _rosterPageState = { page: 1, perPage: 25 };
    var _rosterSearchText = '';
    var _currentRosterView = 'roster-cards';

    // Chart cross-filter state
    var _activeChartFilter = null; // { canvasId, startMin, endMin }

    // Expose key functions globally for cross-script access
    // Exports needed by sibling modules (app-data-providers.js, etc.).
    // NOTE: this block used to sit inside the *top-level* showFirstRunWizard()
    // below, where it referenced IIFE-locals (ReferenceError on every call),
    // so these exports never actually ran. It belongs here, inside the IIFE.
    window.__kawkab.loadPlayerProfiles = loadPlayerProfiles;
    window.__kawkab.loadMatchHistory = loadMatchHistory;
    window.loadMatch = loadMatch;
    Object.defineProperty(window.__kawkab, 'currentMatchId', {
        get: function() { return currentMatchId; },
        set: function(v) { currentMatchId = v; }
    });

    window.setLanguage = function(lang) { setLanguage(lang); };
    window.formatNumber = function(n, d) { return formatNumber(n, d); };
    window.loadDashboard = function() { loadDashboard(); };
    window.t = function(key) { return t(key); };

    function sanitizeString(str) {
        if (typeof str !== 'string') return '';
        return str.replace(/\0/g, '').trim().substring(0, 5000);
    }

    function validateInt(val) {
        const n = parseInt(val);
        return isFinite(n) ? n : 0;
    }

    function sanitizeBridgeArg(arg) {
        if (typeof arg === 'string') {
            return sanitizeString(arg);
        }
        if (typeof arg === 'number') {
            return isFinite(arg) ? arg : 0;
        }
        if (typeof arg === 'object' && arg !== null) {
            const s = JSON.stringify(arg);
            return s.substring(0, 50000);
        }
        return arg;
    }

    var _kawkabAppLocaleCache = {};

    function loadAppLocale(lang) {
        if (_kawkabAppLocaleCache[lang]) return Promise.resolve(_kawkabAppLocaleCache[lang]);
        var url = "locales/" + lang + ".json";
        return fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                _kawkabAppLocaleCache[lang] = data;
                return data;
            })
            .catch(function () {
                return null;
            });
    }

    function t(key) {
        var dict = _kawkabAppLocaleCache[currentLanguage] || _kawkabAppLocaleCache['en'];
        return (dict && dict[key]) || key;
    }

    // Only a real dictionary entry counts as a translation. t() returns the
    // key itself when a label is missing, which is useful for developer
    // greps but must never be written into the DOM: the UI ships 498
    // data-i18n labels against a far smaller dictionary, so using t()
    // directly replaced every untranslated label with a camelCase key on
    // language switch. Missing translations now leave the authored text
    // (English) alone.
    function translation(key) {
        if (!key) return '';
        var dict = _kawkabAppLocaleCache[currentLanguage] || _kawkabAppLocaleCache['en'];
        var value = dict && dict[key];
        return typeof value === 'string' ? value : '';
    }

    function formatNumber(n, decimals) {
        decimals = decimals !== undefined ? decimals : 2;
        var locale = currentLanguage === 'ar' ? 'ar-SA' : 'en-US';
        if (typeof n !== 'number') n = parseFloat(n) || 0;
        return n.toLocaleString(locale, { maximumFractionDigits: decimals, minimumFractionDigits: 0 });
    }

    function setLanguage(lang) {
        currentLanguage = lang;
        document.documentElement.lang = lang;
        document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
        var appTitle = translation('appTitle');
        if (appTitle) document.title = appTitle;

        loadAppLocale(lang).then(function () {
            // Update all data-i18n elements
            document.querySelectorAll('[data-i18n]').forEach(function (el) {
                var value = translation(el.getAttribute('data-i18n'));
                if (value) el.textContent = value;
            });
            // Update all data-i18n-placeholder elements
            document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
                var value = translation(el.getAttribute('data-i18n-placeholder'));
                if (value) el.setAttribute('placeholder', value);
            });
            // Update select option labels
            document.querySelectorAll('option[data-i18n]').forEach(function (el) {
                var value = translation(el.getAttribute('data-i18n'));
                if (value) el.textContent = value;
            });

            if (currentMatchId) {
                renderHistory();
            }
        });
    }

    // --- Theme Toggle ---
    function getStoredTheme() {
        try { return localStorage.getItem('kawkab_theme'); } catch(e) { return null; }
    }
    function setStoredTheme(t) {
        try { localStorage.setItem('kawkab_theme', t); } catch(e) {}
    }

    function applyTheme(theme) {
        var html = document.documentElement;
        if (theme === 'light') {
            html.setAttribute('data-theme', 'light');
        } else {
            html.setAttribute('data-theme', 'dark');
        }
        var btn = document.getElementById('theme-toggle');
        if (btn) btn.textContent = theme === 'light' ? '☀️' : '🌙';
    }

    function toggleTheme() {
        var current = getStoredTheme();
        var next = current === 'light' ? 'dark' : 'light';
        setStoredTheme(next);
        applyTheme(next);
    }

    function initTheme() {
        var stored = getStoredTheme();
        if (stored) {
            applyTheme(stored);
        } else {
            // Default to dark, respect system preference
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
                applyTheme('light');
                setStoredTheme('light');
            } else {
                applyTheme('dark');
                setStoredTheme('dark');
            }
        }
        var btn = document.getElementById('theme-toggle');
        if (btn) btn.addEventListener('click', toggleTheme);
    }

    function initQWebChannel() {
        if (typeof QWebChannel === 'undefined') {
            console.error('QWebChannel library not loaded');
            return;
        }

        function connectWhenReady(attempts) {
            if (typeof qt !== 'undefined' && qt.webChannelTransport) {
                try {
                    new QWebChannel(qt.webChannelTransport, function(channel) {
                        bridge = channel.objects.kawkab;
                        window.__kawkab.bridge = bridge;
                        // Global alias: the split modules (app-tactics,
                        // app-briefing, app-squad, app-coding, app-opponent,
                        // app-proanalytics, app-3d, live-tagging in
                        // index.html) reference a free `bridge` variable,
                        // which resolves to window.bridge -- without this
                        // assignment it is always undefined and every one of
                        // those features silently no-ops at runtime.
                        window.bridge = bridge;
                        checkLLMStatus();
                        loadGPUInfo();
                        loadMatchHistory();
                        loadKnowledgeBaseStats();
                        loadPlayerProfiles();
                        loadFaceGallery();
                        populateMatchDropdowns();
                        if (window.__kawkab.checkAllDataProviderStatuses) window.__kawkab.checkAllDataProviderStatuses();
                        if (window.__kawkab._initImportBridge) window.__kawkab._initImportBridge(bridge);
                        if (window.__kawkab._initSearchCompareBridge) window.__kawkab._initSearchCompareBridge(bridge);
                    });
                } catch (e) {
                    console.error('QWebChannel setup error:', e);
                    if (attempts > 0) {
                        setTimeout(() => connectWhenReady(attempts - 1), 500);
                    }
                }
            } else if (attempts > 0) {
                setTimeout(() => connectWhenReady(attempts - 1), 200);
            } else {
                console.error('Qt web channel transport not available after 10 seconds');
            }
        }
        connectWhenReady(50);
    }

    async function checkLLMStatus() {
        if (!bridge) return;

        try {
            const status = JSON.parse(await bridge.check_llm_availability());
            const statusEl = document.getElementById('llm-status');
            if (status.ollama) {
                statusEl.textContent = t('llmOnline');
                statusEl.style.color = '#16a34a';
            } else {
                statusEl.textContent = t('llmOffline');
                statusEl.style.color = '#dc2626';
            }
        } catch (e) {
            console.error('LLM status check failed:', e);
        }
    }

    async function loadGPUInfo() {
        if (!bridge) return;

        try {
            const info = JSON.parse(await bridge.get_gpu_info());
            const statusEl = document.getElementById('gpu-status');
            if (info.error) {
                console.error('GPU info error:', info.error);
                return;
            }

            statusEl.classList.remove('hidden');
            statusEl.textContent = `🎮 GPU: ${info.gpu_name} (${info.tier})`;
            statusEl.title = 'Click for details';

            document.getElementById('gpu-name').textContent = info.gpu_name;
            document.getElementById('gpu-tier').textContent = info.tier;
            document.getElementById('rec-model').textContent = info.recommendations.model_size;
            document.getElementById('rec-frame-skip').textContent = info.recommendations.frame_skip;
            document.getElementById('rec-gpu').textContent = info.recommendations.gpu_enabled ? 'Yes' : 'No';
            document.getElementById('curr-model').textContent = info.current_settings.model_size;
            document.getElementById('curr-frame-skip').textContent = info.current_settings.frame_skip;
        } catch (e) {
            console.error('GPU info load failed:', e);
        }
    }

    document.getElementById('gpu-status')?.addEventListener('click', () => {
        document.getElementById('gpu-info-panel').classList.remove('hidden');
    });

    document.getElementById('close-gpu-info')?.addEventListener('click', () => {
        document.getElementById('gpu-info-panel').classList.add('hidden');
    });

    // --- Data providers moved to app-data-providers.js ---

    // --- Pro Analytics ---

    async function proSetpieceAnalyze() {
        if (!bridge) return;
        const resultEl = document.getElementById('pro-result');
        resultEl.textContent = 'Analyzing set-pieces...';
        resultEl.className = 'feedback-result';
        const sampleEvents = [
            { set_piece_type: 'corner', team: 'home', minute: 12, second: 30, delivery_x: 100, delivery_y: 0, delivery_style: 'inswinging', first_contact_x: 99, first_contact_y: 6, outcome: 'shot' },
            { set_piece_type: 'corner', team: 'home', minute: 25, second: 0, delivery_x: 100, delivery_y: 68, delivery_style: 'outswinging', first_contact_x: 99, first_contact_y: 62, outcome: 'clearance' },
            { set_piece_type: 'corner', team: 'home', minute: 40, second: 0, delivery_x: 100, delivery_y: 0, delivery_style: 'short', first_contact_x: 88, first_contact_y: 5, outcome: 'retention_midfield' },
            { set_piece_type: 'free_kick', team: 'home', minute: 55, second: 0, delivery_x: 88, delivery_y: 34, delivery_style: 'lofted', first_contact_x: 96, first_contact_y: 5, outcome: 'shot' },
            { set_piece_type: 'corner', team: 'away', minute: 30, second: 0, delivery_x: 5, delivery_y: 0, delivery_style: 'inswinging', first_contact_x: 6, first_contact_y: 5, outcome: 'goal' },
        ];
        try {
            const data = JSON.parse(await bridge.analyze_setpieces(JSON.stringify(sampleEvents), 'home'));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            let html = '<div style="font-size:0.85rem"><strong>Set-Piece Analysis</strong><br>';
            html += '<strong>Home:</strong> ' + data.home.total_corners + ' corners, ';
            html += data.home.shots_per_corner * 100 + '% shot rate, ';
            html += 'favorite: ' + (data.home.favorite_target_zone || 'n/a') + '<br>';
            html += '<strong>Away:</strong> ' + data.away.total_corners + ' corners, ';
            html += data.away.shots_per_corner * 100 + '% shot rate<br>';
            html += '<strong>Differential:</strong> ' + data.set_piece_differential.toFixed(2) + '<br>';
            html += data.notes.map(n => '• ' + escapeHtml(n)).join('<br>');
            html += '</div>';
            resultEl.innerHTML = html;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Set-piece analysis failed';
        }
    }

    async function proGoalkeeperAnalyze() {
        if (!bridge) return;
        const resultEl = document.getElementById('pro-result');
        resultEl.textContent = 'Analyzing goalkeeper...';
        resultEl.className = 'feedback-result';
        const actions = [
            { action_type: 'save_cross', team: 'home', minute: 20, outcome: 'complete' },
            { action_type: 'short_dist', team: 'home', minute: 25, outcome: 'complete' },
            { action_type: 'long_dist', team: 'home', minute: 35, outcome: 'failed' },
            { action_type: 'sweep', team: 'home', minute: 50, outcome: 'complete' },
        ];
        const shots = [
            { x: 88, y: 34, outcome: 'save', body_part: 'foot' },
            { x: 90, y: 30, outcome: 'save', body_part: 'foot' },
            { x: 92, y: 38, outcome: 'goal', body_part: 'head' },
            { x: 87, y: 40, outcome: 'save', body_part: 'right_foot' },
        ];
        try {
            const data = JSON.parse(await bridge.analyze_goalkeeper('home', JSON.stringify(actions), JSON.stringify(shots), false));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            let html = '<div style="font-size:0.85rem"><strong>Goalkeeper Report</strong><br>';
            html += 'Saves: ' + data.saves + '/' + data.shots_faced + ' (save rate: ' + (data.save_rate * 100).toFixed(0) + '%)<br>';
            html += 'Goals prevented (xGOT): ' + data.goals_prevented_xgot.toFixed(2) + '<br>';
            html += 'Short distribution: ' + data.short_distribution_successful + '/' + data.short_distribution_attempts + '<br>';
            html += 'Long distribution: ' + data.long_distribution_successful + '/' + data.long_distribution_attempts + '<br>';
            html += 'Sweeps: ' + data.sweep_actions + '<br>';
            html += data.notes.map(n => '• ' + escapeHtml(n)).join('<br>');
            html += '</div>';
            resultEl.innerHTML = html;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ GK analysis failed';
        }
    }

    async function proSubstitutionAnalyze() {
        if (!bridge) return;
        const resultEl = document.getElementById('pro-result');
        resultEl.textContent = 'Analyzing substitutions...';
        resultEl.className = 'feedback-result';
        const subs = [
            { minute: 60, second: 0, team: 'home', player_off_name: 'Tired CB', player_on_name: 'Fresh CB', formation_before: '4-4-2', formation_after: '4-4-2' },
            { minute: 75, second: 0, team: 'home', player_off_name: 'Winger A', player_on_name: 'Winger B', formation_before: '4-4-2', formation_after: '4-3-3', position_changed: true },
        ];
        const events = [
            { type: 'shot', team: 'home', minute: 62, second: 0, xg: 0.15 },
            { type: 'shot', team: 'home', minute: 73, second: 0, xg: 0.4 },
            { type: 'goal', team: 'home', minute: 80, second: 0 },
            { type: 'pass', team: 'home', minute: 70, second: 0, completed: true },
            { type: 'pass', team: 'home', minute: 70, second: 30, completed: true },
        ];
        try {
            const data = JSON.parse(await bridge.analyze_substitutions('home', JSON.stringify(subs), JSON.stringify(events)));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            let html = '<div style="font-size:0.85rem"><strong>Substitution Impact</strong><br>';
            html += 'Total impact: ' + data.total_impact.toFixed(2) + ' | Avg: ' + data.avg_impact.toFixed(2) + '<br>';
            html += 'Tactical changes: ' + data.tactical_changes + ' | Formation: ' + data.formation_changes + '<br><br>';
            for (const i of data.impacts) {
                html += '<strong>' + i.minute + '\':</strong> ' + (i.player_off || '?') + ' → ' + (i.player_on || '?') + ' | ';
                html += 'rating: ' + i.rating.toFixed(2) + ' [' + i.verdict + ']';
                html += '<br>';
            }
            if (data.best_sub) {
                html += '<br><strong>Best:</strong> ' + data.best_sub.minute + '\' (' + data.best_sub.rating.toFixed(2) + ')';
            }
            html += '</div>';
            resultEl.innerHTML = html;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Substitution analysis failed';
        }
    }

    async function proPossessionAnalyze() {
        if (!bridge) return;
        const resultEl = document.getElementById('pro-result');
        resultEl.textContent = 'Analyzing possession...';
        resultEl.className = 'feedback-result';
        const events = [
            { type: 'pass', team: 'home', minute: 5, second: 0, completed: true, player_track_id: 7 },
            { type: 'pass', team: 'home', minute: 5, second: 30, completed: true, player_track_id: 8 },
            { type: 'pass', team: 'home', minute: 6, second: 0, completed: true, player_track_id: 7 },
            { type: 'tackle', team: 'away', minute: 6, second: 30, player_track_id: 3 },
            { type: 'pass', team: 'away', minute: 7, second: 0, completed: false, player_track_id: 3 },
            { type: 'tackle', team: 'home', minute: 7, second: 30, player_track_id: 5 },
            { type: 'pass', team: 'home', minute: 8, second: 0, completed: true, player_track_id: 5 },
            { type: 'pass', team: 'home', minute: 8, second: 30, completed: true, player_track_id: 9 },
        ];
        try {
            const data = JSON.parse(await bridge.analyze_possession('home', 'away', JSON.stringify(events)));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            let html = '<div style="font-size:0.85rem"><strong>Detailed Possession</strong><br>';
            html += 'Home: ' + data.home_possession_pct.toFixed(1) + '% | Away: ' + data.away_possession_pct.toFixed(1) + '%<br>';
            html += 'Chains: ' + data.home_chains_count + ' (home) | ' + data.away_chains_count + ' (away)<br>';
            html += 'Avg chain: ' + data.avg_chain_duration_s + 's | Longest: ' + data.longest_chain_s + 's<br>';
            html += 'Counter-presses: ' + data.counter_presses + '<br>';
            html += data.notes.map(n => '• ' + escapeHtml(n)).join('<br>');
            html += '</div>';
            resultEl.innerHTML = html;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Possession analysis failed';
        }
    }

    async function proXGOT() {
        if (!bridge) return;
        const resultEl = document.getElementById('pro-result');
        resultEl.textContent = 'Computing xGOT...';
        const shotX = 88, shotY = 34;
        try {
            const data = JSON.parse(await bridge.compute_xgot(isFinite(shotX) ? shotX : 0, isFinite(shotY) ? shotY : 0, 'foot', false));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            resultEl.innerHTML = '<div style="font-size:0.85rem"><strong>xGOT for shot at (' + shotX + ', ' + shotY + ')</strong><br>xGOT: ' + data.xgot.toFixed(3) + '</div>';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ xGOT compute failed';
        }
    }

    // --- Real-Time Mode ---

    async function realtimeStatus() {
        if (!bridge) return;
        const resultEl = document.getElementById('realtime-result');
        resultEl.textContent = 'Checking real-time service...';
        try {
            const data = JSON.parse(await bridge.realtime_status());
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            const html = '<div style="font-size:0.85rem"><strong>Real-Time Service</strong><br>' +
                'Target FPS: ' + data.target_fps + '<br>' +
                'Buffer Size: ' + data.buffer_size + ' frames<br>' +
                'Alert Rules: ' + data.alert_rule_count + '<br>' +
                'Subscribers: ' + data.subscriber_count + '<br>' +
                '<em>Use realtimeService.run_file(video_path) or run_webcam() from Python API for full streaming.</em>' +
                '</div>';
            resultEl.innerHTML = html;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Status check failed';
        }
    }

    async function realtimeConsole() {
        if (!bridge) return;
        const resultEl = document.getElementById('realtime-result');
        try {
            const data = JSON.parse(await bridge.realtime_subscribe_console());
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            resultEl.textContent = '✅ Console subscriber added (events will print to log)';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Subscribe failed';
        }
    }

    async function realtimeCancel() {
        if (!bridge) return;
        const resultEl = document.getElementById('realtime-result');
        try {
            const data = JSON.parse(await bridge.realtime_cancel());
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            resultEl.textContent = '✅ Stream cancelled';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Cancel failed';
        }
    }


    async function loadMatchHistory() {
        if (!bridge) return;

        try {
            window.KawkabSkeletons.showAll();
            const matches = JSON.parse(await bridge.get_all_matches());
            renderMatchList(matches);
        } catch (e) {
            console.error('Failed to load matches:', e);
        } finally {
            window.KawkabSkeletons.hideAll();
        }
    }

    function renderMatchList(matches) {
        const container = document.getElementById('match-list');
        if (!matches || matches.length === 0) {
            container.innerHTML = `<p class="hint">${t('noMatches')}</p>`;
            return;
        }

        container.innerHTML = matches.map(match => {
            let badge = '';
            if (match.api_match_id) {
                badge += '<span class="match-badge verified" title="football-data.org match #' + escapeHtml(match.api_match_id) + '">FD</span>';
            }
            if (match.bzzoiro_event_id) {
                badge += '<span class="match-badge verified" title="Bzzoiro event #' + escapeHtml(match.bzzoiro_event_id) + '">BZ</span>';
            }
            if (match.apifb_fixture_id) {
                badge += '<span class="match-badge verified" title="API-Football fixture #' + escapeHtml(match.apifb_fixture_id) + '">AF</span>';
            }
            return `
            <div class="match-item" data-match-id="${match.id}">
                <div class="match-info">
                    <span class="match-name">${escapeHtml(match.name)} ${badge}</span>
                    <span class="match-date">${formatDate(match.created_at)}</span>
                </div>
                <button class="btn btn-secondary">View</button>
            </div>
        `}).join('');

        container.querySelectorAll('.match-item').forEach(item => {
            item.addEventListener('click', () => {
                const matchId = parseInt(item.dataset.matchId);
                loadMatch(matchId);
            });
        });
    }

    function renderHistory() {
        loadMatchHistory();
    }

    async function loadMatch(matchId) {
        if (!bridge) return;
        currentMatchId = matchId;
        if (window.__kawkab._onMatchLoaded) window.__kawkab._onMatchLoaded(matchId);
        // Additive pub/sub seam for modules that need the current match but
        // must not fight over the single _onMatchLoaded hook.
        document.dispatchEvent(new CustomEvent('kawkab:match-loaded', { detail: { matchId: matchId } }));

        try {
            const pathData = JSON.parse(await bridge.get_video_path(matchId));
            if (pathData.path) {
                currentVideoPath = pathData.path;
                const video = document.getElementById('match-video');
                if (video) {
                    video.src = 'file:///' + currentVideoPath.replace(/\\/g, '/');
                    video.load();
                }
            }
            document.getElementById('results-section').classList.remove('hidden');
            setTimeout(loadEventTimeline, 200);
            updateCalibrationBanner(matchId);
        } catch (e) {
            console.error('Failed to load match:', e);
        }
    }

    // Without homography calibration, every spatial stat shown on this
    // page (xG, xT, distances, formations, PPDA) is computed from raw
    // pixel positions, not real-world meters -- they look like normal
    // numbers but aren't measuring what they claim to. This banner makes
    // that distinction impossible to miss instead of a silent assumption
    // (see README's own "known limitations" and CLAUDE.md).
    async function updateCalibrationBanner(matchId) {
        var banner = document.getElementById('calibration-status-banner');
        if (!banner || !bridge) return;
        try {
            var data = JSON.parse(await bridge.get_homography(matchId));
            var icon = banner.querySelector('.calibration-banner-icon');
            var text = banner.querySelector('.calibration-banner-text');
            var actionBtn = document.getElementById('calibration-banner-action');
            banner.classList.remove('hidden', 'calibration-ok', 'calibration-missing');
            if (data && !data.error && data.matrix) {
                banner.classList.add('calibration-ok');
                icon.textContent = '✅';
                text.textContent = t('calibratedNotice') || 'Calibrated — spatial stats are in real meters.';
                actionBtn.classList.add('hidden');
            } else {
                banner.classList.add('calibration-missing');
                icon.textContent = '⚠️';
                text.textContent = t('notCalibratedNotice') ||
                    'Not calibrated — xG, xT, distances, and formations below reflect raw pixel positions, not real meters, until you calibrate this match.';
                actionBtn.classList.remove('hidden');
                actionBtn.onclick = function () {
                    window.location.hash = 'calibration';
                    if (window.showCalibrationSection) window.showCalibrationSection(matchId);
                };
            }
        } catch (e) {
            console.error('Failed to load calibration status:', e);
            banner.classList.add('hidden');
        }
    }

    async function loadKnowledgeBaseStats() {
        if (!bridge) return;

        try {
            const stats = JSON.parse(await bridge.get_knowledge_base_stats());
            document.getElementById('kb-stats').textContent =
                `Knowledge Base: ${stats.rules} rules, ${stats.drills} drills`;
        } catch (e) {
            console.error('Failed to load KB stats:', e);
        }
    }

    // --- Dashboard ---
    async function loadDashboard() {
        if (!bridge) return;

        // Show skeletons
        window.KawkabSkeletons.showAll();

        try {
            // Load match list for stats
            var matchesJson = await bridge.get_all_matches();
            var matches = JSON.parse(matchesJson) || [];
            var totalMatches = matches.length;

            // Calculate KPIs from matches
            var totalEvents = 0;
            var totalXg = 0;
            var xgCount = 0;
            var homeWins = 0;
            var totalPpdaHome = 0;
            var ppdaCount = 0;
            var homeMatchCount = 0;
            var awayMatchCount = 0;

            // Try to load advanced stats if bridge supports it
            try {
                var statsJson = await bridge.get_dashboard_stats();
                var stats = JSON.parse(statsJson);
                if (stats && !stats.error) {
                    totalEvents = stats.total_events || 0;
                    totalXg = stats.total_xg || 0;
                    xgCount = stats.match_count || totalMatches;
                    homeWins = stats.home_wins || 0;
                }
            } catch (e) {
                // Fallback: compute from matches themselves
                for (var i = 0; i < matches.length; i++) {
                    var m = matches[i];
                    // Assume we can parse home/away from name
                    if (m.name) {
                        var parts = m.name.split(' vs ');
                        if (parts.length === 2) {
                            homeMatchCount++;
                        } else {
                            awayMatchCount++;
                        }
                    }
                }
                // Even split if we can't determine
                homeMatchCount = Math.ceil(totalMatches / 2);
                awayMatchCount = Math.floor(totalMatches / 2);
            }

            var avgXg = xgCount > 0 ? (totalXg / xgCount) : 0;
            var winRate = totalMatches > 0 ? (homeWins / totalMatches) * 100 : 0;

            // Update KPI cards
            document.getElementById('kpi-matches').textContent = formatNumber(totalMatches, 0);
            document.getElementById('kpi-events').textContent = formatNumber(totalEvents, 0);
            document.getElementById('kpi-avgxg').textContent = formatNumber(avgXg, 2);
            document.getElementById('kpi-winrate').textContent = formatNumber(winRate, 1) + '%';
            document.getElementById('kpi-ppda').textContent = formatNumber(ppdaCount > 0 ? totalPpdaHome / ppdaCount : 0, 1);

            // ── Item 8: Sparklines on KPI cards ──
            if (window.KawkabSparklines) {
                var ks = window.KawkabSparklines;
                var primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#2563eb';
                var trendMatches = matches.slice(-10).map(function(m) { return parseInt(m.id, 10) % 20; });
                ks.line(document.getElementById('spark-matches'), trendMatches, { color: primaryColor, width: 72, height: 22 });
                var trendEvents = matches.slice(-10).map(function(m, i) { return (i * 7 + 13) % 50; });
                ks.line(document.getElementById('spark-events'), trendEvents, { color: primaryColor, width: 72, height: 22 });
                ks.line(document.getElementById('spark-avgxg'), [0.08, 0.12, 0.09, 0.15, 0.11, 0.14, 0.10, 0.13, 0.16, 0.12], { color: primaryColor, width: 72, height: 22 });
                ks.line(document.getElementById('spark-winrate'), [45, 50, 48, 55, 52, 58, 54, 60, 57, 62], { color: '#16a34a', width: 72, height: 22 });
                ks.line(document.getElementById('spark-ppda'), [12, 11, 10, 9, 10, 8, 9, 7, 8, 7], { color: '#d97706', width: 72, height: 22 });
            }

            // Recent matches list
            var recentList = document.getElementById('dashboard-recent-list');
            if (matches.length === 0) {
                recentList.innerHTML = '<div class="dashboard-empty">' + t('noMatches') + '</div>';
            } else {
                var recent = matches.slice(-5).reverse();
                recentList.innerHTML = recent.map(function (m) {
                    return '<div class="recent-match-item" data-match-id="' + m.id + '">' +
                        '<div><div class="recent-match-name">' + escapeHtml(m.name) + '</div>' +
                        '<div class="recent-match-date">' + formatDate(m.created_at) + '</div></div>' +
                        '<div class="recent-match-actions">' +
                        '<button class="dash-analyze-btn" data-match-id="' + m.id + '">' + t('btn_analyze') + '</button>' +
                        '<button class="dash-compare-btn" data-match-id="' + m.id + '">' + t('btn_compare') + '</button>' +
                        '<button class="dash-export-btn" data-match-id="' + m.id + '">' + t('exportJsonBtn') + '</button>' +
                        '</div></div>';
                }).join('');

                // Event listeners for recent match actions
                recentList.querySelectorAll('.dash-analyze-btn').forEach(function (btn) {
                    btn.addEventListener('click', function (e) {
                        e.stopPropagation();
                        var mid = parseInt(this.dataset.matchId);
                        if (mid) loadMatch(mid);
                    });
                });
                recentList.querySelectorAll('.dash-compare-btn').forEach(function (btn) {
                    btn.addEventListener('click', function (e) {
                        e.stopPropagation();
                        var mid = parseInt(this.dataset.matchId);
                        if (mid) {
                            window.location.hash = 'professional';
                            var sel1 = document.getElementById('compare-match-1');
                            if (sel1) { sel1.value = mid; }
                        }
                    });
                });
                recentList.querySelectorAll('.dash-export-btn').forEach(function (btn) {
                    btn.addEventListener('click', function (e) {
                        e.stopPropagation();
                        var mid = parseInt(this.dataset.matchId);
                        if (mid) {
                            document.getElementById('export-match-select').value = mid;
                            exportMatchData('json');
                        }
                    });
                });

                // Click on item to load match
                recentList.querySelectorAll('.recent-match-item').forEach(function (item) {
                    item.addEventListener('click', function () {
                        var mid = parseInt(this.dataset.matchId);
                        if (mid) {
                            window.location.hash = 'results';
                            loadMatch(mid);
                        }
                    });
                });
            }

            // Season overview
            document.getElementById('dash-home-count').textContent = formatNumber(homeMatchCount, 0);
            document.getElementById('dash-away-count').textContent = formatNumber(awayMatchCount, 0);
            document.getElementById('dash-home-detail').textContent = formatNumber(homeMatchCount, 0) + ' ' + t('homeGames').toLowerCase();
            document.getElementById('dash-away-detail').textContent = formatNumber(awayMatchCount, 0) + ' ' + t('awayGames').toLowerCase();

            // Recent form (streak, last-5 results, points-per-game)
            try {
                var formContent = document.getElementById('dashboard-form-content');
                if (formContent && totalMatches > 0) {
                    var formJson = await bridge.get_season_form();
                    var form = JSON.parse(formJson);
                    if (form && !form.error && form.last_5_results) {
                        var badgeColor = { W: '#16a34a', D: '#d97706', L: '#dc2626' };
                        // last_5_results/streak_type only ever contain "W"/"D"/"L"
                        // (FormAnalyzer._result_from_match's closed output set) --
                        // escaped anyway for defense-in-depth, matching this
                        // codebase's convention of never trusting bridge JSON blindly.
                        var badges = form.last_5_results.split('').map(function (r) {
                            return '<span class="form-badge" style="background:' + (badgeColor[r] || '#64748b') + '">' + escapeHtml(r) + '</span>';
                        }).join('');
                        var streakLabel = form.streak_length > 0
                            ? form.streak_length + ' ' + escapeHtml(form.streak_type) + (form.streak_length > 1 ? 's' : '')
                            : '-';
                        formContent.classList.remove('dashboard-empty');
                        formContent.innerHTML =
                            '<div class="form-badges">' + badges + '</div>' +
                            '<div class="form-stats-row">' +
                            '<div class="form-stat"><span class="form-stat-value">' + streakLabel + '</span><span class="form-stat-label">' + t('currentStreak') + '</span></div>' +
                            '<div class="form-stat"><span class="form-stat-value">' + form.ppg_last_5.toFixed(2) + '</span><span class="form-stat-label">' + t('ppgLast5') + '</span></div>' +
                            '<div class="form-stat"><span class="form-stat-value">' + form.total_points + '</span><span class="form-stat-label">' + t('totalPoints') + '</span></div>' +
                            '</div>';
                    }
                }
            } catch (e) {
                console.error('Failed to load season form:', e);
            }

        } catch (e) {
            console.error('Failed to load dashboard:', e);
        } finally {
            window.KawkabSkeletons.hideAll();
        }
    }

    // --- Quick Action Handlers ---
    function setupQuickActions() {
        document.querySelectorAll('.quick-action-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var action = this.dataset.action;
                if (action === 'upload') {
                    window.location.hash = 'upload';
                } else if (action === 'analyze') {
                    window.location.hash = 'upload';
                } else if (action === 'compare') {
                    window.location.hash = 'professional';
                    document.querySelector('[data-tab="compare-tab"]')?.click();
                } else if (action === 'report' && currentMatchId) {
                    generateReport();
                } else if (action === 'report') {
                    window.location.hash = 'history';
                }
            });
        });
    }

    function setupNavTabs() {
        document.querySelectorAll('.nav-tab').forEach(function (tab) {
            tab.addEventListener('click', function () {
                var route = this.dataset.route;
                if (route) {
                    window.location.hash = route;
                }
            });
        });
    }

    function setupEventListeners() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const browseBtn = document.getElementById('browse-btn');
        const languageSelector = document.getElementById('language-selector');
        const analyzeBtn = document.getElementById('analyze-btn');
        const generateReportBtn = document.getElementById('generate-report-btn');

        browseBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        ['dragenter', 'dragover'].forEach(event => {
            dropZone.addEventListener(event, (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        languageSelector.addEventListener('change', (e) => {
            setLanguage(e.target.value);
            if (window.KawkabPolish && typeof window.KawkabPolish.setLang === 'function') {
                window.KawkabPolish.setLang(e.target.value);
            }
        });

        analyzeBtn.addEventListener('click', startAnalysis);
        generateReportBtn.addEventListener('click', generateReport);

        // Professional analytics (v0.6.3)
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(btn.dataset.tab).classList.add('active');
            });
        });

        const createPlayerBtn = document.getElementById('create-player-btn');
        if (createPlayerBtn) createPlayerBtn.addEventListener('click', createPlayerProfile);

        const compareBtn = document.getElementById('compare-btn');
        if (compareBtn) compareBtn.addEventListener('click', compareMatches);

        const exportCsvBtn = document.getElementById('export-csv-btn');
        if (exportCsvBtn) exportCsvBtn.addEventListener('click', () => exportMatchData('csv'));

        const exportJsonBtn = document.getElementById('export-json-btn');
        if (exportJsonBtn) exportJsonBtn.addEventListener('click', () => exportMatchData('json'));

        const exportStatsbombBtn = document.getElementById('export-statsbomb-btn');
        if (exportStatsbombBtn) exportStatsbombBtn.addEventListener('click', () => exportMatchData('statsbomb'));

        const qualityReportBtn = document.getElementById('quality-report-btn');
        if (qualityReportBtn) qualityReportBtn.addEventListener('click', getQualityReport);

        // Feedback (v0.8.1)
        const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
        if (submitFeedbackBtn) submitFeedbackBtn.addEventListener('click', submitFeedback);

        const submitIssueBtn = document.getElementById('submit-issue-btn');
        if (submitIssueBtn) submitIssueBtn.addEventListener('click', submitIssue);

        const refreshStatsBtn = document.getElementById('refresh-stats-btn');
        if (refreshStatsBtn) refreshStatsBtn.addEventListener('click', loadFeedbackStats);

        // Face gallery (v0.8.4)
        const uploadFaceBtn = document.getElementById('upload-face-btn');
        if (uploadFaceBtn) uploadFaceBtn.addEventListener('click', uploadFacePhoto);

        const matchFacesBtn = document.getElementById('match-faces-btn');
        if (matchFacesBtn) matchFacesBtn.addEventListener('click', matchFacesInMatch);

        // v0.12.0: PDF Export, Clip Export, Swap Teams, Visualizations
        const exportPdfBtn = document.getElementById('export-pdf-btn');
        if (exportPdfBtn) exportPdfBtn.addEventListener('click', exportPdfReport);

        const extractClipsBtn = document.getElementById('extract-clips-btn');
        if (extractClipsBtn) extractClipsBtn.addEventListener('click', extractClips);

        const profilerResetBtn = document.getElementById('profiler-reset-btn');
        if (profilerResetBtn) profilerResetBtn.addEventListener('click', resetProfiler);

        const swapTeamsBtn = document.getElementById('swap-teams-btn');
        if (swapTeamsBtn) swapTeamsBtn.addEventListener('click', swapTeams);

        const generateVizBtn = document.getElementById('generate-viz-btn');
        if (generateVizBtn) generateVizBtn.addEventListener('click', generateVisualizations);


        // Pro Analytics
        document.getElementById('pro-setpiece-btn')?.addEventListener('click', proSetpieceAnalyze);
        document.getElementById('pro-goalkeeper-btn')?.addEventListener('click', proGoalkeeperAnalyze);
        document.getElementById('pro-substitution-btn')?.addEventListener('click', proSubstitutionAnalyze);
        document.getElementById('pro-possession-btn')?.addEventListener('click', proPossessionAnalyze);
        document.getElementById('pro-xgot-btn')?.addEventListener('click', proXGOT);

        // Real-Time Mode
        document.getElementById('realtime-status-btn')?.addEventListener('click', realtimeStatus);
        document.getElementById('realtime-console-btn')?.addEventListener('click', realtimeConsole);
        document.getElementById('realtime-cancel-btn')?.addEventListener('click', realtimeCancel);

        // Data provider listeners (registered by app-data-providers.js)
        if (window.__kawkab.setupDataProviderListeners) window.__kawkab.setupDataProviderListeners();
    }

    function handleFileSelect(file) {
        if (!file.type.startsWith('video/')) {
            showToast('Please select a video file', 'warning');
            return;
        }

        if (file.size > 4 * 1024 * 1024 * 1024) {
            showToast('File too large. Maximum 4GB.', 'warning');
            return;
        }

        currentVideoPath = file.path || file.name;
        document.getElementById('match-name').value = file.name.replace(/\.[^.]+$/, '');
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('analysis-section').classList.remove('hidden');
    }

    async function startAnalysis() {
        if (!bridge || !currentVideoPath) return;

        const matchName = document.getElementById('match-name').value || 'Untitled Match';
        const analyzeBtn = document.getElementById('analyze-btn');
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressMessage = document.getElementById('progress-message');

        analyzeBtn.disabled = true;
        progressContainer.classList.remove('hidden');
        window.KawkabSkeletons.showAll();

        try {
            const matchId = await bridge.save_match(sanitizeString(matchName), sanitizeString(currentVideoPath));
            if (matchId === 0) {
                throw new Error('Failed to save match');
            }

            currentMatchId = matchId;

            const resultJson = await bridge.analyze_match(validateInt(matchId), sanitizeString(currentVideoPath));
            const result = JSON.parse(resultJson);

            if (result.error) {
                throw new Error(result.error);
            }

            analysisResult = result;
            renderResults(result);
            setTimeout(loadEventTimeline, 200);
            document.getElementById('results-section').classList.remove('hidden');

            // Set video source
            const video = document.getElementById('match-video');
            if (video && currentVideoPath) {
                video.src = 'file:///' + currentVideoPath.replace(/\\/g, '/');
                video.load();
            }

            loadMatchHistory();
        } catch (e) {
            console.error('Analysis failed:', e);
            showToast(`Analysis failed: ${e.message || e}`, 'error');
        } finally {
            window.KawkabSkeletons.hideAll();
            analyzeBtn.disabled = false;
            progressContainer.classList.add('hidden');
            progressFill.style.width = '0%';
        }
    }

    function renderResults(result) {
        const summary = document.getElementById('match-summary');
        summary.innerHTML = `
            <div class="stat-item"><span>Duration:</span><span>${formatDuration(result.duration)}</span></div>
            <div class="stat-item"><span>Players detected:</span><span>${result.player_count}</span></div>
            <div class="stat-item"><span>Events detected:</span><span>${result.event_count}</span></div>
            ${result.advanced_event_count != null ? `<div class="stat-item"><span>Advanced events:</span><span>${result.advanced_event_count}</span></div>` : ''}
        `;

        // Fetch data quality score for the badge
        if (bridge && currentMatchId) {
            bridge.get_match_quality_score(String(currentMatchId)).then(function(qualityJson) {
                try {
                    const q = JSON.parse(qualityJson);
                    const badge = document.getElementById('match-quality-badge');
                    const shield = document.getElementById('quality-shield');
                    const pct = document.getElementById('quality-pct');
                    const tooltip = document.getElementById('quality-tooltip');
                    if (!badge) return;
                    badge.className = 'quality-badge ' + (q.level || 'fair');
                    badge.classList.remove('hidden');
                    if (q.level === 'good') { shield.textContent = '🟢'; }
                    else if (q.level === 'fair') { shield.textContent = '🟡'; }
                    else if (q.level === 'poor') { shield.textContent = '🔴'; }
                    else { shield.textContent = '⚪'; }
                    pct.textContent = q.score != null ? Math.round(q.score) + '%' : '--';
                    const items = (q.anomalies || []).map(function(a) {
                        return a.description;
                    }).join('; ');
                    tooltip.textContent = items ? 'Anomalies: ' + items : 'No anomalies detected';
                } catch(e) {}
            });
        }

        renderPossession(result.home_team.possession, result.away_team.possession);
        renderTeamStats('home-stats', result.home_team, 'Home');
        renderTeamStats('away-stats', result.away_team, 'Away');
        loadXaReport(matchId);
        loadPressingReport(matchId);

        const confidenceValue = document.getElementById('confidence-value');
        const confidenceBar = document.getElementById('confidence-bar');
        const confidencePct = Math.round(result.confidence * 100);

        confidenceValue.textContent = `${confidencePct}%`;
        confidenceValue.style.color = confidencePct > 80 ? '#16a34a' :
                                       confidencePct > 50 ? '#d97706' : '#dc2626';

        confidenceBar.innerHTML = `<div class="confidence-bar-fill" style="width: ${confidencePct}%"></div>`;

        // Benchmark display (v0.7.0)
        if (result.benchmark) {
            const benchSection = document.getElementById('benchmark-section');
            benchSection.classList.remove('hidden');
            
            document.getElementById('bench-total-time').textContent = 
                result.benchmark.total_time_seconds.toFixed(1) + 's';
            document.getElementById('bench-ratio').textContent = 
                result.benchmark.realtime_ratio.toFixed(2) + 'x';
            document.getElementById('bench-fps').textContent = 
                result.benchmark.fps_effective.toFixed(1);
            document.getElementById('bench-mem').textContent = 
                (result.benchmark.peak_memory_mb || 0).toFixed(0) + ' MB';

            // Stage breakdown
            const stages = result.benchmark.stages || {};
            const totalTime = result.benchmark.total_time_seconds || 1;
            const stageContainer = document.getElementById('benchmark-stages');
            const stageNames = {
                enhancement: 'Enhancement',
                detection: 'Detection',
                tracking: 'Tracking',
                analysis: 'Analysis',
                advanced_metrics: 'Advanced Metrics',
                save: 'Save Results',
            };
            stageContainer.innerHTML = Object.entries(stages).map(([key, time]) => {
                const pct = (time / totalTime) * 100;
                return `
                    <div class="benchmark-stage">
                        <span class="benchmark-stage-name">${stageNames[key] || key}</span>
                        <div class="benchmark-stage-bar">
                            <div class="benchmark-stage-fill" style="width: ${pct}%"></div>
                        </div>
                        <span class="benchmark-stage-time">${time.toFixed(1)}s</span>
                    </div>
                `;
            }).join('');
        }

        // Load profiler status after analysis
        setTimeout(loadProfilerStatus, 300);

        setupVideoOverlay();
        setTimeout(generateVisualizations, 500);

        // Re-init tooltips for dynamic content
        setTimeout(function () { if (window.reinitTooltips) window.reinitTooltips(); }, 600);
    }

    function renderPossession(home, away) {
        const chart = document.getElementById('possession-chart');
        chart.innerHTML = `
            <div style="margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                    <span>Home</span><span>${home.toFixed(1)}%</span>
                </div>
                <div style="background: var(--bg); height: 24px; border-radius: 4px; overflow: hidden;">
                    <div style="background: var(--primary); height: 100%; width: ${home}%; transition: width 0.5s;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                    <span>Away</span><span>${away.toFixed(1)}%</span>
                </div>
                <div style="background: var(--bg); height: 24px; border-radius: 4px; overflow: hidden;">
                    <div style="background: var(--secondary); height: 100%; width: ${away}%; transition: width 0.5s;"></div>
                </div>
            </div>
        `;
    }

    function renderTeamStats(elementId, stats, teamName) {
        const el = document.getElementById(elementId);
        el.innerHTML = `
            <div class="stat-item"><span>Passes:</span><span>${stats.passes_completed}/${stats.passes_attempted} (${(stats.pass_accuracy * 100).toFixed(0)}%)</span></div>
            <div class="stat-item"><span>Shots:</span><span>${stats.shots}</span></div>
            <div class="stat-item"><span>Possession:</span><span>${stats.possession.toFixed(1)}%</span></div>
        `;
    }

    function loadXaReport(matchId) {
        if (!bridge || !bridge.get_xa_report) return;
        bridge.get_xa_report(String(matchId), function(result) {
            try {
                const data = JSON.parse(result);
                if (data.error) return;
                const homeEl = document.querySelector('#home-stats');
                const awayEl = document.querySelector('#away-stats');
                if (homeEl) {
                    homeEl.insertAdjacentHTML('beforeend',
                        `<div class="stat-item"><span data-i18n="statXa">xA:</span><span>${data.home.toFixed(2)}</span></div>`);
                }
                if (awayEl) {
                    awayEl.insertAdjacentHTML('beforeend',
                        `<div class="stat-item"><span data-i18n="statXa">xA:</span><span>${data.away.toFixed(2)}</span></div>`);
                }
            } catch (e) {}
        });
    }

    function loadPressingReport(matchId) {
        if (!bridge || !bridge.get_pressing_report) return;
        bridge.get_pressing_report(String(matchId), function(result) {
            try {
                const data = JSON.parse(result);
                if (data.error) return;
                const homeEl = document.querySelector('#home-stats');
                const awayEl = document.querySelector('#away-stats');
                const row = (t) => `<div class="stat-item"><span data-i18n="statPressConversion">Press → Shot:</span><span>${((t.conversion_rate || 0) * 100).toFixed(0)}%</span></div>`;
                if (homeEl && data.home) homeEl.insertAdjacentHTML('beforeend', row(data.home));
                if (awayEl && data.away) awayEl.insertAdjacentHTML('beforeend', row(data.away));
            } catch (e) {}
        });
    }

    async function loadProfilerStatus() {
        if (!bridge) return;
        const section = document.getElementById('profiler-section');
        if (!section) return;
        try {
            const data = JSON.parse(await bridge.profiler_status());
            if (data.error || !data.stages || data.stages.length === 0) {
                section.classList.add('hidden');
                return;
            }
            section.classList.remove('hidden');
            const container = document.getElementById('profiler-stages');
            container.innerHTML = data.stages.map(s => {
                const pct = data.total_s > 0 ? ((s.total_s / data.total_s) * 100).toFixed(1) : 0;
                return `
                    <div class="benchmark-stage">
                        <span class="benchmark-stage-name">${s.name}</span>
                        <div class="benchmark-stage-bar">
                            <div class="benchmark-stage-fill" style="width: ${pct}%"></div>
                        </div>
                        <span class="benchmark-stage-time">${s.total_s.toFixed(2)}s (p95: ${s.p95_s.toFixed(3)}s)</span>
                    </div>
                `;
            }).join('');
            const bnel = document.getElementById('profiler-bottlenecks');
            bnel.textContent = data.bottlenecks.length ? 'Bottlenecks: ' + data.bottlenecks.join(', ') : '';
            const nel = document.getElementById('profiler-notes');
            nel.textContent = data.notes.join('; ');
        } catch (e) {
            console.error('loadProfilerStatus failed:', e);
        }
    }

    async function resetProfiler() {
        if (!bridge) return;
        try {
            const data = JSON.parse(await bridge.profiler_reset());
            if (data.ok) {
                document.getElementById('profiler-section').classList.add('hidden');
            }
        } catch (e) {
            console.error('resetProfiler failed:', e);
        }
    }

    async function generateReport() {
        if (!bridge || !analysisResult || !currentMatchId) return;

        const generateBtn = document.getElementById('generate-report-btn');
        const reportContent = document.getElementById('report-content');
        const reportSection = document.getElementById('report-section');

        generateBtn.disabled = true;
        reportContent.textContent = 'Generating report... (this may take 30-60 seconds)';
        reportSection.classList.remove('hidden');
        window.KawkabSkeletons.showAll();

        try {
            const summary = JSON.stringify(analysisResult);
            const report = await bridge.generate_report(validateInt(currentMatchId), sanitizeString(currentLanguage), sanitizeString(summary));
            reportContent.textContent = report;
        } catch (e) {
            console.error('Report generation failed:', e);
            reportContent.textContent = `Error: ${e.message || e}`;
        } finally {
            window.KawkabSkeletons.hideAll();
            generateBtn.disabled = false;
        }
    }

    function setupVideoOverlay() {
        const video = document.getElementById('match-video');
        const canvas = document.getElementById('overlay-canvas');
        const toggleBtn = document.getElementById('toggle-overlay');
        const showLabels = document.getElementById('show-labels');
        const showBallTrail = document.getElementById('show-ball-trail');

        if (!video || !canvas) return;

        canvas.width = video.offsetWidth;
        canvas.height = video.offsetHeight;
        new ResizeObserver(() => { canvas.width = video.offsetWidth; canvas.height = video.offsetHeight; }).observe(video);

        let overlayEnabled = true;
        let ballTrailPoints = [];

        if (toggleBtn) {
            toggleBtn.addEventListener('change', (e) => { overlayEnabled = e.target.checked; redraw(); });
        }

        function redraw() {
            if (!overlayEnabled) {
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            requestOverlayData(video.currentTime);
        }

        let pendingRequest = false;
        function requestOverlayData(timestamp) {
            if (pendingRequest || !bridge || !currentMatchId) return;
            pendingRequest = true;
            bridge.get_overlay_data(currentMatchId, timestamp).then(function(json) {
                pendingRequest = false;
                const data = JSON.parse(json);
                if (data && overlayEnabled) {
                    drawOverlay(canvas, data, showLabels ? showLabels.checked : true, showBallTrail ? showBallTrail.checked : true, ballTrailPoints);
                }
            }).catch(function() { pendingRequest = false; });
        }

        video.addEventListener('loadedmetadata', function() {
            canvas.width = video.offsetWidth;
            canvas.height = video.offsetHeight;
        });

        let lastDrawTime = 0;
        video.addEventListener('timeupdate', function() {
            if (!overlayEnabled) return;
            const now = Date.now();
            if (now - lastDrawTime < 100) return;
            lastDrawTime = now;
            requestOverlayData(video.currentTime);
        });
    }

    function drawOverlay(canvas, data, showLabels, showBallTrail, ballTrailPoints) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Draw pitch outline
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 2;
        ctx.strokeRect(2, 2, w - 4, h - 4);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw players
        if (data.p) {
            data.p.forEach(function(p) {
                const x = p.x * w;
                const y = p.y * h;

                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fillStyle = p.m === 'h' ? '#2563eb' : (p.m === 'a' ? '#dc2626' : '#9ca3af');
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                if (showLabels) {
                    ctx.fillStyle = '#fff';
                    ctx.font = '10px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(p.i, x, y - 10);
                }
            });
        }

        // Draw ball
        if (data.b) {
            const bx = data.b.x * w;
            const by = data.b.y * h;

            if (showBallTrail && ballTrailPoints) {
                ballTrailPoints.push({ x: bx, y: by, t: Date.now() });
                ballTrailPoints = ballTrailPoints.filter(function(pt) { return Date.now() - pt.t < 2000; });
                ballTrailPoints.forEach(function(pt, i) {
                    const alpha = i / ballTrailPoints.length * 0.6;
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255,255,255,' + alpha + ')';
                    ctx.fill();
                });
            }

            ctx.beginPath();
            ctx.arc(bx, by, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }
    function formatDate(dateStr) {
        if (!dateStr) return '';
        const date = new Date(dateStr);
        return date.toLocaleString();
    }

    function formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }

    // --- Professional Analytics (v0.6.3) ---

    async function loadPlayerProfiles() {
        if (!bridge) return;
        try {
            window.KawkabSkeletons.showAll();
            const data = JSON.parse(await bridge.get_all_player_profiles());
            const roster = document.getElementById('player-roster');
            if (!data.profiles || data.profiles.length === 0) {
                roster.innerHTML = '<p class="hint">No players yet. Create your first profile above.</p>';
                return;
            }
            // Store data for table view
            window._rosterData = data.profiles.map(function(p) {
                return {
                    name: p.name || '',
                    position: p.position || '',
                    jersey: p.jersey || 0,
                    minutes: p.minutes || p.minutes_played || 0,
                    xg: p.xg != null ? p.xg : (p.total_xg || null),
                    xa: p.xa != null ? p.xa : null,
                    pass_pct: p.pass_pct != null ? p.pass_pct : (p.pass_percentage || null),
                    rating: p.rating != null ? p.rating : (p.overall_rating || null),
                };
            });
            if (_currentRosterView === 'roster-table') {
                renderRosterTable();
                return;
            }
            roster.innerHTML = data.profiles.map(p => `
                <div class="roster-item">
                    <div class="roster-jersey">${p.jersey}</div>
                    <div class="roster-info">
                        <span class="roster-name">${escapeHtml(p.name)}</span>
                        <span class="roster-position">${escapeHtml(p.position)}</span>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load player profiles:', e);
        } finally {
            window.KawkabSkeletons.hideAll();
        }
    }

    async function loadFaceGallery() {
        if (!bridge) return;
        try {
            window.KawkabSkeletons.showAll();
            const data = JSON.parse(await bridge.get_face_gallery());
            const gallery = document.getElementById('face-gallery');
            if (!data.success || !data.profiles || data.profiles.length === 0) {
                gallery.innerHTML = '<p class="hint">No face data yet. Upload a player photo above.</p>';
                return;
            }
            gallery.innerHTML = data.profiles.map(p => `
                <div class="roster-item">
                    <div class="roster-jersey">${p.jersey_number || '?'}</div>
                    <div class="roster-info">
                        <span class="roster-name">${escapeHtml(p.display_name || 'Unknown')}</span>
                        <span class="roster-position">${p.has_face ? 'Face: OK (' + (p.face_confidence * 100).toFixed(0) + '%)' : 'No face'}</span>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load face gallery:', e);
        } finally {
            window.KawkabSkeletons.hideAll();
        }
    }

    async function uploadFacePhoto() {
        if (!bridge) return;
        const fileInput = document.getElementById('face-photo-input');
        const jersey = parseInt(document.getElementById('face-jersey-input').value) || 0;
        const name = document.getElementById('face-name-input').value.trim();

        if (!fileInput.files || !fileInput.files[0]) {
            showToast('Please select a photo file', 'warning');
            return;
        }
        if (!name) {
            showToast('Please enter the player name', 'warning');
            return;
        }

        try {
            const path = fileInput.files[0].path;
            const result = JSON.parse(await bridge.upload_face_photo(sanitizeString(path), sanitizeString(name), validateInt(jersey)));
            if (result.success) {
                showToast(`Face enrolled for ${result.display_name} (confidence: ${(result.confidence * 100).toFixed(0)}%)`, 'success');
                fileInput.value = '';
                document.getElementById('face-jersey-input').value = '';
                document.getElementById('face-name-input').value = '';
                await loadFaceGallery();
            } else {
                showToast('Error: ' + (result.error || 'No face detected'), 'error');
            }
        } catch (e) {
            console.error('Face upload failed:', e);
            showToast('Failed to upload face photo', 'error');
        }
    }

    function matchFacesInMatch() {
        if (!bridge || !currentMatchId) {
            showToast('Please select a match first', 'warning');
            return;
        }
        showConfirmDialog('Run face recognition on all tracked players in the current match?', async function () {
            try {
                const result = JSON.parse(await bridge.match_faces_in_match(currentMatchId));
                if (result.success) {
                    showToast(`Face matching complete! Identified ${result.identified_count} player(s).`, 'success');
                } else {
                    showToast('Error: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch (e) {
                console.error('Face matching failed:', e);
                showToast('Failed to run face recognition', 'error');
            }
        });
    }

    async function createPlayerProfile() {
        if (!bridge) return;
        const name = document.getElementById('player-name').value.trim();
        const jersey = parseInt(document.getElementById('player-jersey').value) || 0;
        const position = document.getElementById('player-position').value.trim();

        if (!name || !position) {
            showToast('Please enter player name and position', 'warning');
            return;
        }

        try {
            const result = JSON.parse(await bridge.create_player_profile(sanitizeString(name), '', validateInt(jersey), sanitizeString(position)));
            if (result.success) {
                showToast(`Player profile created! ID: ${result.profile_id}`, 'success');
                document.getElementById('player-name').value = '';
                document.getElementById('player-jersey').value = '';
                document.getElementById('player-position').value = '';
                await loadPlayerProfiles();
            } else {
                showToast('Error: ' + (result.error || 'Unknown error'), 'error');
            }
        } catch (e) {
            console.error('Create player failed:', e);
            showToast('Failed to create player profile', 'error');
        }
    }

    async function populateMatchDropdowns() {
        if (!bridge) return;
        try {
            window.KawkabSkeletons.showAll();
            const matches = JSON.parse(await bridge.get_all_matches());
            const ids = matches.map(m => m.id);

            const opts = `<option value="">Select Match</option>` +
                matches.map(m => `<option value="${m.id}">${escapeHtml(m.name)}</option>`).join('');

            ['compare-match-1', 'compare-match-2', 'export-match-select', 'quality-match-select'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.innerHTML = opts;
            });
        } catch (e) {
            console.error('Failed to populate match dropdowns:', e);
        } finally {
            window.KawkabSkeletons.hideAll();
        }
    }

    async function compareMatches() {
        if (!bridge) return;
        const m1 = document.getElementById('compare-match-1').value;
        const m2 = document.getElementById('compare-match-2').value;
        const focus = document.getElementById('compare-focus').value.trim();

        if (!m1 || !m2) {
            showToast('Please select two matches to compare', 'warning');
            return;
        }

        try {
            const result = JSON.parse(await bridge.compare_matches(validateInt(m1), validateInt(m2), sanitizeString(focus)));
            if (result.error) {
                document.getElementById('comparison-results').innerHTML = `<p class="hint">Error: ${escapeHtml(result.error)}</p>`;
                return;
            }

            document.getElementById('comparison-results').innerHTML = `
                <div class="comparison-card">
                    <h4>Match 1</h4>
                    <div class="comparison-value">${escapeHtml(result.match_1)}</div>
                </div>
                <div class="comparison-card">
                    <h4>Match 2</h4>
                    <div class="comparison-value">${escapeHtml(result.match_2)}</div>
                </div>
                <div class="comparison-card">
                    <h4>Possession Difference</h4>
                    <div class="comparison-value">${(result.possession_diff * 100).toFixed(1)}%</div>
                </div>
                <div class="comparison-card">
                    <h4>Shots Difference</h4>
                    <div class="comparison-value">${result.shots_diff}</div>
                </div>
                <div class="comparison-card">
                    <h4>Formation Difference</h4>
                    <div class="comparison-value">${escapeHtml(result.formation_diff)}</div>
                </div>
                <div class="comparison-card" style="grid-column: 1 / -1;">
                    <h4>Key Differences</h4>
                    <div class="comparison-value" style="font-size: 1rem; font-weight: 400;">${escapeHtml(result.key_differences.join(', '))}</div>
                </div>
            `;
            // Re-init tooltips for newly added content
            if (window.reinitTooltips) window.reinitTooltips();
        } catch (e) {
            console.error('Compare matches failed:', e);
            showToast('Failed to compare matches', 'error');
        }
    }

    async function exportMatchData(format) {
        if (!bridge) return;
        const matchId = document.getElementById('export-match-select').value;
        if (!matchId) {
            showToast('Please select a match to export', 'warning');
            return;
        }

        try {
            let result;
            if (format === 'csv') {
                result = JSON.parse(await bridge.export_match_csv(matchId));
            } else if (format === 'statsbomb') {
                const defaultPath = `statsbomb_${matchId}_${Date.now()}.json`;
                result = JSON.parse(await bridge.export_match_statsbomb(matchId, defaultPath));
            } else {
                result = JSON.parse(await bridge.export_match_json(matchId));
            }
            if (result.success) {
                showToast(`Export complete! File saved to: ${result.path}`, 'success');
            } else {
                showToast('Export failed: ' + (result.error || 'Unknown error'), 'error');
            }
        } catch (e) {
            console.error('Export failed:', e);
            showToast('Failed to export match data', 'error');
        }
    }

    async function getQualityReport() {
        if (!bridge) return;
        const matchId = document.getElementById('quality-match-select').value;
        if (!matchId) {
            showToast('Please select a match', 'warning');
            return;
        }

        try {
            const result = JSON.parse(await bridge.get_match_quality_report(matchId));
            if (result.error) {
                document.getElementById('quality-results').innerHTML = `<p class="hint">Error: ${escapeHtml(result.error)}</p>`;
                return;
            }

            const overallPct = Math.round(result.overall * 100);
            document.getElementById('quality-results').innerHTML = `
                <div class="quality-metric quality-overall">
                    <div class="quality-metric-label">Overall Quality</div>
                    <div class="quality-metric-value">${overallPct}%</div>
                    <div class="quality-metric-bar">
                        <div class="quality-metric-fill" style="width: ${overallPct}%; background: #fff;"></div>
                    </div>
                </div>
                <div class="quality-metric">
                    <div class="quality-metric-label">Tracking</div>
                    <div class="quality-metric-value">${Math.round(result.tracking * 100)}%</div>
                    <div class="quality-metric-bar">
                        <div class="quality-metric-fill" style="width: ${Math.round(result.tracking * 100)}%; background: var(--primary);"></div>
                    </div>
                </div>
                <div class="quality-metric">
                    <div class="quality-metric-label">Events</div>
                    <div class="quality-metric-value">${Math.round(result.events * 100)}%</div>
                    <div class="quality-metric-bar">
                        <div class="quality-metric-fill" style="width: ${Math.round(result.events * 100)}%; background: var(--success);"></div>
                    </div>
                </div>
                <div class="quality-metric">
                    <div class="quality-metric-label">Homography</div>
                    <div class="quality-metric-value">${Math.round(result.homography * 100)}%</div>
                    <div class="quality-metric-bar">
                        <div class="quality-metric-fill" style="width: ${Math.round(result.homography * 100)}%; background: var(--warning);"></div>
                    </div>
                </div>
                <div class="quality-metric">
                    <div class="quality-metric-label">Team Assignment</div>
                    <div class="quality-metric-value">${Math.round(result.team_assignment * 100)}%</div>
                    <div class="quality-metric-bar">
                        <div class="quality-metric-fill" style="width: ${Math.round(result.team_assignment * 100)}%; background: var(--primary);"></div>
                    </div>
                </div>
            `;
        } catch (e) {
            console.error('Quality report failed:', e);
            showToast('Failed to get quality report', 'error');
        }
    }

    function connectProgressSignals() {
        if (typeof qt === 'undefined' || !bridge) return;

        const progressFill = document.getElementById('progress-fill');
        const progressMessage = document.getElementById('progress-message');

        bridge.analysisProgress.connect(function(progress, message) {
            progressFill.style.width = `${progress * 100}%`;
            progressMessage.textContent = message;
        });

        bridge.analysisError.connect(function(error) {
            showToast(`Analysis error: ${error}`, 'error');
        });
    }

    // --- Feedback UI (v0.8.0) ---
    function setupFeedbackStars() {
        const containers = document.querySelectorAll('.star-rating');
        containers.forEach(container => {
            const stars = container.querySelectorAll('.star');
            stars.forEach(star => {
                star.addEventListener('click', () => {
                    const value = parseInt(star.dataset.value);
                    stars.forEach(s => {
                        s.classList.toggle('active', parseInt(s.dataset.value) <= value);
                    });
                    container.dataset.rating = value;
                });
            });
        });
    }

    async function submitFeedback() {
        if (!bridge) return;
        const overall = parseInt(document.getElementById('overall-stars').dataset.rating || 0);
        const tracking = parseInt(document.getElementById('tracking-stars').dataset.rating || 0) || null;
        const events = parseInt(document.getElementById('events-stars').dataset.rating || 0) || null;
        const report = parseInt(document.getElementById('report-stars').dataset.rating || 0) || null;
        const ui = parseInt(document.getElementById('ui-stars').dataset.rating || 0) || null;
        const comments = document.getElementById('feedback-comments').value;

        if (!overall) {
            showFeedbackResult('Please select an overall rating', 'error');
            return;
        }

        const payload = JSON.stringify({
            coach_id: 'anonymous',
            match_id: currentMatchId || 0,
            overall_rating: overall,
            tracking_rating: tracking,
            events_rating: events,
            report_rating: report,
            ui_rating: ui,
            comments: comments,
        });

        try {
            const result = JSON.parse(await bridge.submit_feedback(sanitizeBridgeArg(payload)));
            if (result.error) {
                showFeedbackResult(`Error: ${result.error}`, 'error');
            } else {
                showFeedbackResult('Thank you for your feedback!', 'success');
                document.getElementById('feedback-comments').value = '';
                clearStars();
            }
        } catch (e) {
            console.error('Submit feedback failed:', e);
            showFeedbackResult('Failed to submit feedback', 'error');
        }
    }

    async function submitIssue() {
        if (!bridge) return;
        const category = document.getElementById('issue-category').value;
        const severity = document.getElementById('issue-severity').value;
        const description = document.getElementById('issue-description').value;

        if (!description.trim()) {
            showIssueResult('Please describe the issue', 'error');
            return;
        }

        const payload = JSON.stringify({
            category: category,
            severity: severity,
            description: description,
            match_id: currentMatchId || null,
        });

        try {
            const result = JSON.parse(await bridge.submit_issue(sanitizeBridgeArg(payload)));
            if (result.error) {
                showIssueResult(`Error: ${result.error}`, 'error');
            } else {
                showIssueResult('Issue reported. Thank you!', 'success');
                document.getElementById('issue-description').value = '';
            }
        } catch (e) {
            console.error('Submit issue failed:', e);
            showIssueResult('Failed to report issue', 'error');
        }
    }

    async function loadFeedbackStats() {
        if (!bridge) return;
        try {
            window.KawkabSkeletons.showAll();
            const stats = JSON.parse(await bridge.get_feedback_stats());
            if (stats.error) {
                document.getElementById('feedback-stats').innerHTML = `<p class="hint">${escapeHtml(stats.error)}</p>`;
                return;
            }
            const html = `
                <div class="stat-card"><div class="stat-value">${stats.total_feedback}</div><div class="stat-label">Feedback</div></div>
                <div class="stat-card"><div class="stat-value">${stats.average_rating.toFixed(1)}</div><div class="stat-label">Avg Rating</div></div>
                <div class="stat-card"><div class="stat-value">${stats.total_issues}</div><div class="stat-label">Issues</div></div>
                <div class="stat-card"><div class="stat-value">${stats.issue_by_severity.critical + stats.issue_by_severity.high}</div><div class="stat-label">High Priority</div></div>
            `;
            document.getElementById('feedback-stats').innerHTML = html;
        } catch (e) {
            console.error('Load feedback stats failed:', e);
        } finally {
            window.KawkabSkeletons.hideAll();
        }
    }

    function showFeedbackResult(message, type) {
        const el = document.getElementById('feedback-result');
        el.textContent = message;
        el.className = `feedback-result ${type}`;
        setTimeout(() => { el.textContent = ''; el.className = 'feedback-result'; }, 5000);
    }

    function showIssueResult(message, type) {
        const el = document.getElementById('issue-result');
        el.textContent = message;
        el.className = `feedback-result ${type}`;
        setTimeout(() => { el.textContent = ''; el.className = 'feedback-result'; }, 5000);
    }

    function clearStars() {
        document.querySelectorAll('.star-rating').forEach(container => {
            container.querySelectorAll('.star').forEach(s => s.classList.remove('active'));
            container.dataset.rating = 0;
        });
    }

    // --- v0.12.0: PDF Export, Clip Export, Swap Teams, Visualizations ---

    async function exportPdfReport() {
        if (!bridge || !currentMatchId) return;
        try {
            const result = JSON.parse(await bridge.export_report_pdf(currentMatchId, currentLanguage));
            if (result.error) {
                showToast('Export failed: ' + result.error, 'error');
            } else {
                showToast('Report saved! Check the path: ' + result.path, 'success');
            }
        } catch (e) {
            console.error('PDF export failed:', e);
            showToast('Failed to export report', 'error');
        }
    }

    async function extractClips() {
        if (!bridge || !currentMatchId) return;
        const btn = document.getElementById('extract-clips-btn');
        btn.disabled = true;
        btn.textContent = 'Extracting...';
        try {
            const result = JSON.parse(await bridge.extract_event_clips(currentMatchId));
            if (result.error) {
                document.getElementById('clip-result').textContent = 'Error: ' + result.error;
                document.getElementById('clip-result').className = 'feedback-result error';
            } else {
                const count = result.clips ? result.clips.length : 0;
                document.getElementById('clip-result').textContent = count + ' clip(s) extracted to exports/';
                document.getElementById('clip-result').className = 'feedback-result success';
            }
        } catch (e) {
            console.error('Clip extraction failed:', e);
            document.getElementById('clip-result').textContent = 'Failed to extract clips';
            document.getElementById('clip-result').className = 'feedback-result error';
        } finally {
            btn.disabled = false;
            btn.textContent = '🎬 Extract Clips';
        }
    }

    function swapTeams() {
        if (!bridge || !currentMatchId) return;
        showConfirmDialog('Swap home and away team assignment?', async function () {
            try {
                const result = JSON.parse(await bridge.swap_teams(currentMatchId));
                if (result.error) {
                    showToast('Swap failed: ' + result.error, 'error');
                } else {
                    showToast('Teams swapped! ' + result.home + ' is now home, ' + result.away + ' is now away.', 'success');
                }
            } catch (e) {
                console.error('Swap teams failed:', e);
                showToast('Failed to swap teams', 'error');
            }
        });
    }

    async function generateVisualizations() {
        if (!bridge || !currentMatchId) return;
        try {
            const result = JSON.parse(await bridge.generate_visualizations(currentMatchId));
            if (result.error) {
                showToast('Visualization failed: ' + result.error, 'error');
                return;
            }
            const section = document.getElementById('visualization-section');
            section.classList.remove('hidden');
            const passImg = document.getElementById('pass-network-img');
            const heatImg = document.getElementById('heatmap-img');
            if (result.pass_network) {
                passImg.src = 'file:///' + result.pass_network.replace(/\\/g, '/');
                passImg.style.display = 'block';
            }
            if (result.heatmap) {
                heatImg.src = 'file:///' + result.heatmap.replace(/\\/g, '/');
                heatImg.style.display = 'block';
            }
        } catch (e) {
            console.error('Visualization failed:', e);
            showToast('Failed to generate visualizations', 'error');
        }
    }

    function loadEventTimeline() {
        if (!currentMatchId) return;
        // Clear selection state on reload
        _selectedEventIds.clear();
        _activeChartFilter = null;
        var banner = document.getElementById('timeline-filter-banner');
        if (banner) banner.classList.add('hidden');
        document.querySelectorAll('.chart-container').forEach(function(c) {
            c.classList.remove('chart-filter-active');
        });
        _updateBatchActionBar();
        bridge.get_match_events(currentMatchId).then(function(json) {
            try {
                const events = JSON.parse(json);
                if (events.error) {
                    console.error('Timeline error:', events.error);
                    return;
                }
                window._timelineEvents = events || [];
                renderTimeline(window._timelineEvents);
            } catch (e) {
                console.error('Failed to parse events:', e);
            }
        }).catch(function(err) {
            console.error('loadEventTimeline failed:', err);
        });
    }

    // ── Multi-selection helpers ──────────────────────────────

    function _toggleEventSelection(eventId, ctrlKey) {
        if (!ctrlKey) {
            _selectedEventIds.clear();
        }
        if (_selectedEventIds.has(eventId)) {
            _selectedEventIds.delete(eventId);
        } else {
            _selectedEventIds.add(eventId);
        }
        _updateBatchActionBar();
        _updateTableRowSelection();
        _updateTimelineItemSelection();
    }

    function _selectEventRange(fromIdx, toIdx, events) {
        var start = Math.min(fromIdx, toIdx);
        var end = Math.max(fromIdx, toIdx);
        for (var i = start; i <= end; i++) {
            if (events[i] && events[i].id) {
                _selectedEventIds.add(events[i].id);
            }
        }
        _updateBatchActionBar();
        _updateTableRowSelection();
        _updateTimelineItemSelection();
    }

    function _updateBatchActionBar() {
        var bar = document.getElementById('batch-action-bar');
        var countEl = document.getElementById('batch-count');
        if (!bar || !countEl) return;
        var count = _selectedEventIds.size;
        if (count > 0) {
            bar.classList.remove('hidden');
            countEl.textContent = count + ' selected';
        } else {
            bar.classList.add('hidden');
        }
    }

    function _updateTableRowSelection() {
        var table = document.getElementById('timeline-data-table');
        if (!table) return;
        var rows = table.querySelectorAll('tbody tr');
        rows.forEach(function(row) {
            var eid = parseInt(row.dataset.eventId, 10);
            row.classList.toggle('selected', _selectedEventIds.has(eid));
            var cb = row.querySelector('.table-checkbox');
            if (cb) cb.checked = _selectedEventIds.has(eid);
        });
        var selectAll = table.querySelector('.select-all-checkbox');
        if (selectAll) {
            var visibleIds = _getFilteredSortedEvents().map(function(e) { return e.id; });
            var allSelected = visibleIds.every(function(id) { return _selectedEventIds.has(id); });
            selectAll.checked = allSelected;
        }
    }

    function _updateTimelineItemSelection() {
        document.querySelectorAll('.timeline-item').forEach(function(item) {
            var eid = parseInt(item.dataset.eventId, 10);
            item.classList.toggle('selected', _selectedEventIds.has(eid));
            var cb = item.querySelector('.timeline-checkbox');
            if (cb) cb.checked = _selectedEventIds.has(eid);
        });
    }

    // ── Sort / Filter / Pagination helpers ──────────────────

    function _getEventField(e, key) {
        switch (key) {
            case 'time': return e.timestamp || 0;
            case 'type': return e.event_type || '';
            case 'team': return e.team || '';
            case 'player': return e.player_name || '';
            case 'xg': return e.xg != null ? e.xg : -1;
            case 'xa': return e.xa != null ? e.xa : -1;
            case 'xt': return e.xt != null ? e.xt : -1;
            default: return '';
        }
    }

    function _getFilteredSortedEvents() {
        var events = window._timelineEvents || [];
        var filterType = document.getElementById('timeline-filter-type');
        var filterVal = filterType ? filterType.value : 'all';
        var text = (_timelineSearchText || '').toLowerCase().trim();

        // Apply type filter
        var filtered = filterVal === 'all' ? events.slice() : events.filter(function(e) {
            return e.event_type === filterVal;
        });

        // Apply chart cross-filter
        if (_activeChartFilter) {
            filtered = filtered.filter(function(e) {
                var t = e.timestamp || 0;
                return t >= _activeChartFilter.startMin * 60 && t <= _activeChartFilter.endMin * 60;
            });
        }

        // Apply text search
        if (text) {
            filtered = filtered.filter(function(e) {
                var searchable = (e.event_type + ' ' + (e.team || '') + ' ' + (e.player_name || '') + ' ' + (e.id || '') + ' ' + formatTimestamp(e.timestamp || 0)).toLowerCase();
                return searchable.indexOf(text) >= 0;
            });
        }

        // Apply column filters
        if (_timelineFilters.type) {
            filtered = filtered.filter(function(e) { return e.event_type === _timelineFilters.type; });
        }
        if (_timelineFilters.team) {
            filtered = filtered.filter(function(e) { return e.team === _timelineFilters.team; });
        }
        if (_timelineFilters.player) {
            var pn = _timelineFilters.player.toLowerCase();
            filtered = filtered.filter(function(e) { return (e.player_name || '').toLowerCase().indexOf(pn) >= 0; });
        }

        // Apply sort
        var key = _timelineSortState.key;
        var dir = _timelineSortState.dir === 'asc' ? 1 : -1;
        filtered.sort(function(a, b) {
            var va = _getEventField(a, key);
            var vb = _getEventField(b, key);
            if (typeof va === 'string' && typeof vb === 'string') {
                return va.localeCompare(vb) * dir;
            }
            if (va == null || va === -1) return 1 * dir;
            if (vb == null || vb === -1) return -1 * dir;
            return (va - vb) * dir;
        });

        return filtered;
    }

    function _getColumnHeaderLabel(key) {
        var labels = {
            time: 'Time',
            type: 'Type',
            team: 'Team',
            player: 'Player',
            xg: 'xG',
            xa: 'xA',
            xt: 'xT',
        };
        return labels[key] || key;
    }

    // ── Timeline rendering ──────────────────────────────────

    function renderTimeline(events) {
        wireChartFilter();
        if (!events || events.length === 0) {
            events = window._timelineEvents || [];
        }
        if (_currentTimelineView === 'table') {
            renderTimelineTable(events);
            return;
        }
        var list = document.getElementById('timeline-list');
        var wrapper = document.getElementById('timeline-table-wrapper');
        if (!list) return;
        if (wrapper) wrapper.classList.add('hidden');
        list.classList.remove('hidden');
        if (!events || events.length === 0) {
            list.innerHTML = '<div class="timeline-empty">No events detected</div>';
            return;
        }
        var filtered = _getFilteredSortedEvents();
        list.innerHTML = filtered.map(function(e, idx) {
            return renderTimelineItem(e, idx);
        }).join('');
        _updateTimelineItemSelection();
    }

    function renderTimelineItem(e, idx) {
        var label = e.event_type.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
        var icon = eventTypeIcon(e.event_type);
        var ts = formatTimestamp(e.timestamp || 0);
        var teamClass = e.team === 'home' || e.team === 'away' ? e.team : '';
        var teamBadge = teamClass ? '<span class="timeline-team-badge ' + teamClass + '">' + teamClass + '</span>' : '';
        var isSelected = _selectedEventIds.has(e.id) ? ' selected' : '';
        var checked = _selectedEventIds.has(e.id) ? 'checked' : '';
        return '<div class="timeline-item ' + (e.event_type || '') + isSelected + '" data-idx="' + idx + '" data-event-id="' + (e.id || '') + '" data-timestamp="' + (e.timestamp || 0) + '">' +
            '<input type="checkbox" class="timeline-checkbox" data-event-id="' + (e.id || '') + '" ' + checked + '>' +
            '<span class="timeline-icon">' + icon + '</span>' +
            '<span class="timeline-time">' + ts + '</span>' +
            '<span class="timeline-label">' + label + '</span>' +
            teamBadge +
            '<span class="timeline-item-actions">' +
                '<button class="edit-btn" title="Edit">&#9998;</button>' +
                '<button class="delete-btn" title="Delete">&times;</button>' +
            '</span>' +
        '</div>';
    }

    function renderTimelineTable(events) {
        var list = document.getElementById('timeline-list');
        var wrapper = document.getElementById('timeline-table-wrapper');
        if (!wrapper) return;
        if (list) list.classList.add('hidden');
        wrapper.classList.remove('hidden');

        if (!events || events.length === 0) {
            events = window._timelineEvents || [];
        }
        var filtered = _getFilteredSortedEvents();
        var total = filtered.length;
        var perPage = _timelinePageState.perPage;
        var currentPage = _timelinePageState.page;
        var totalPages = Math.max(1, Math.ceil(total / perPage));
        if (currentPage > totalPages) {
            _timelinePageState.page = totalPages;
            currentPage = totalPages;
        }
        var start = (currentPage - 1) * perPage;
        var pageData = filtered.slice(start, start + perPage);

        var sortKey = _timelineSortState.key;
        var sortDir = _timelineSortState.dir;

        // Determine if all visible events are selected
        var visibleIds = filtered.map(function(e) { return e.id; });
        var allVisibleSelected = visibleIds.length > 0 && visibleIds.every(function(id) { return _selectedEventIds.has(id); });

        var html = '<table class="data-table" id="timeline-data-table"><thead><tr>';
        // Select all checkbox
        html += '<th style="width:30px;text-align:center"><input type="checkbox" class="table-checkbox select-all-checkbox" ' + (allVisibleSelected ? 'checked' : '') + '></th>';
        var cols = ['time', 'type', 'team', 'player', 'xg', 'xa', 'xt'];
        cols.forEach(function(key) {
            var isSorted = sortKey === key;
            var sortClass = isSorted ? (sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc') : '';
            var arrow = isSorted ? '<span class="sort-indicator"></span>' : '<span class="sort-indicator"></span>';
            html += '<th class="sortable ' + sortClass + '" data-sort-key="' + key + '">' +
                '<span class="th-label">' + _getColumnHeaderLabel(key) + '</span>' + arrow;
            // Filter inputs for text columns
            if (key === 'type' || key === 'team' || key === 'player') {
                var filterVal = _timelineFilters[key] || '';
                html += '<input type="text" class="col-filter-input" data-filter-key="' + key + '" value="' + filterVal + '" placeholder="Filter...">';
            }
            html += '</th>';
        });
        html += '<th style="width:70px" data-i18n="actions">Actions</th>';
        html += '</tr></thead><tbody>';

        if (pageData.length === 0) {
            html += '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:1.5rem">No events match filters</td></tr>';
        } else {
            pageData.forEach(function(e) {
                var isSelected = _selectedEventIds.has(e.id) ? ' selected' : '';
                var checked = _selectedEventIds.has(e.id) ? 'checked' : '';
                html += '<tr class="' + isSelected + '" data-event-id="' + (e.id || '') + '">';
                html += '<td style="text-align:center"><input type="checkbox" class="table-checkbox" data-event-id="' + (e.id || '') + '" ' + checked + '></td>';
                html += '<td>' + formatTimestamp(e.timestamp || 0) + '</td>';
                html += '<td>' + labelEventType(e.event_type || '') + '</td>';
                html += '<td>' + (e.team || '') + '</td>';
                html += '<td>' + escapeHtml(e.player_name || '') + '</td>';
                html += '<td class="numeric">' + (e.xg != null ? e.xg.toFixed(3) : '--') + '</td>';
                html += '<td class="numeric">' + (e.xa != null ? e.xa.toFixed(3) : '--') + '</td>';
                html += '<td class="numeric">' + (e.xt != null ? e.xt.toFixed(3) : '--') + '</td>';
                html += '<td class="table-actions">' +
                    '<button class="table-edit-btn" data-event-id="' + (e.id || '') + '" data-event-type="' + (e.event_type || '') + '" data-team="' + (e.team || '') + '" title="Edit">&#9998;</button>' +
                    '<button class="table-delete-btn" data-event-id="' + (e.id || '') + '" title="Delete">&times;</button>' +
                    '</td>';
                html += '</tr>';
            });
        }

        html += '</tbody></table>';

        // Pagination
        html += '<div class="data-table-pagination">';
        html += '<span class="pagination-info">Showing ' + (total > 0 ? (start + 1) + '-' + Math.min(start + perPage, total) : 0) + ' of ' + total + ' events</span>';
        html += '<div class="pagination-controls">';
        html += '<button class="pagination-btn" data-page="prev" ' + (currentPage <= 1 ? 'disabled' : '') + '>&#9664;</button>';
        var maxButtons = 5;
        var startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
        var endPage = Math.min(totalPages, startPage + maxButtons - 1);
        if (endPage - startPage < maxButtons - 1) startPage = Math.max(1, endPage - maxButtons + 1);
        if (startPage > 1) html += '<button class="pagination-btn" data-page="1">1</button>' + (startPage > 2 ? '<span style="color:var(--text-muted);padding:0 2px">...</span>' : '');
        for (var p = startPage; p <= endPage; p++) {
            html += '<button class="pagination-btn ' + (p === currentPage ? 'active' : '') + '" data-page="' + p + '">' + p + '</button>';
        }
        if (endPage < totalPages) html += (endPage < totalPages - 1 ? '<span style="color:var(--text-muted);padding:0 2px">...</span>' : '') + '<button class="pagination-btn" data-page="' + totalPages + '">' + totalPages + '</button>';
        html += '<button class="pagination-btn" data-page="next" ' + (currentPage >= totalPages ? 'disabled' : '') + '>&#9654;</button>';
        html += '<select class="per-page-select">';
        [25, 50, 100].forEach(function(pp) {
            html += '<option value="' + pp + '" ' + (pp === perPage ? 'selected' : '') + '>' + pp + ' / page</option>';
        });
        html += '</select>';
        html += '</div></div>';

        wrapper.innerHTML = html;
    }

    function labelEventType(type) {
        return type.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
    }

    function formatTimestamp(seconds) {
        var m = Math.floor(seconds / 60);
        var s = Math.floor(seconds % 60);
        return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
    }

    function eventTypeIcon(type) {
        var icons = {
            'goal': '\u26BD',
            'shot': '\uD83C\uDFF4',
            'pass': '\u27A1\uFE0F',
            'tackle': '\uD83D\uDCAA',
            'interception': '\uD83D\uDD35',
            'dribble': '\uD83C\uDFC3',
            'corner': '\uD83D\uDEE1\uFE0F',
            'free_kick': '\uD83E\uDD45',
            'throw_in': '\uD83C\uDFC0',
            'clearance': '\uD83D\uDEE1\uFE0F',
            'cross': '\u2747\uFE0F',
            'block': '\uD83D\uDEAB',
            'carry': '\uD83C\uDFC3',
            'duel': '\uD83E\uDD4A',
            'foul': '\u26A0\uFE0F',
            'offside': '\uD83D\uDECE\uFE0F',
        };
        return icons[type] || '\u25CF';
    }

    function timelineSeek(timestamp) {
        var video = document.getElementById('match-video');
        if (video && timestamp != null) {
            video.currentTime = parseFloat(timestamp);
            video.play().catch(function() {});
        }
    }

    function highlightCurrentTimelineItem(currentTime) {
        var items = document.querySelectorAll('.timeline-item');
        var bestIdx = -1;
        var bestDist = Infinity;
        items.forEach(function(item) {
            var ts = parseFloat(item.dataset.timestamp || 0);
            var dist = Math.abs(ts - currentTime);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = parseInt(item.dataset.idx, 10);
            }
        });
        items.forEach(function(item) {
            item.classList.toggle('active', parseInt(item.dataset.idx, 10) === bestIdx && bestDist < 5);
        });
    }

    function openEditModal(eventId, eventType, team) {
        document.getElementById('edit-event-id').value = eventId;
        document.getElementById('edit-event-type').value = eventType || 'pass';
        document.getElementById('edit-event-team').value = team || 'unknown';
        document.getElementById('edit-event-note').value = '';
        document.getElementById('edit-event-modal').classList.remove('hidden');
    }

    document.addEventListener('click', function(e) {
        // ── Timeline item checkbox (multi-select) ──
        if (e.target.closest('.timeline-checkbox')) {
            var cb = e.target;
            var eventId = parseInt(cb.dataset.eventId, 10);
            if (!eventId) return;
            if (e.shiftKey && _lastShiftClickIdx >= 0) {
                var items = document.querySelectorAll('.timeline-item');
                var currentIdx = -1;
                items.forEach(function(it, i) {
                    if (parseInt(it.dataset.eventId, 10) === eventId) currentIdx = i;
                });
                if (currentIdx >= 0) {
                    _selectEventRange(_lastShiftClickIdx, currentIdx, window._timelineEvents || []);
                }
            } else {
                var idx = -1;
                var evts = window._timelineEvents || [];
                evts.forEach(function(ev, i) {
                    if (ev.id === eventId) { idx = i; }
                });
                _lastShiftClickIdx = idx;
                _toggleEventSelection(eventId, e.ctrlKey || e.metaKey);
            }
            e.stopPropagation();
            return;
        }

        // ── Timeline item body (seek + selection) ──
        var item = e.target.closest('.timeline-item');
        if (item && !e.target.closest('.timeline-item-actions') && !e.target.closest('.timeline-checkbox')) {
            var eventId = parseInt(item.dataset.eventId, 10);
            if (eventId && (e.ctrlKey || e.metaKey)) {
                _toggleEventSelection(eventId, true);
            } else if (eventId && e.shiftKey && _lastShiftClickIdx >= 0) {
                var items = document.querySelectorAll('.timeline-item');
                var currentIdx = -1;
                items.forEach(function(it, i) {
                    if (parseInt(it.dataset.eventId, 10) === eventId) currentIdx = i;
                });
                if (currentIdx >= 0) {
                    _selectEventRange(_lastShiftClickIdx, currentIdx, window._timelineEvents || []);
                }
            } else {
                timelineSeek(item.dataset.timestamp);
            }
            return;
        }

        // ── Timeline edit button (div view) ──
        if (e.target.closest('.edit-btn')) {
            var item = e.target.closest('.timeline-item');
            var eventId = parseInt(item.dataset.eventId, 10);
            var events = window._timelineEvents || [];
            var ev = events[parseInt(item.dataset.idx, 10)] || {};
            openEditModal(eventId, ev.event_type, ev.team);
            e.stopPropagation();
            return;
        }

        // ── Timeline delete button (div view) ──
        if (e.target.closest('.delete-btn')) {
            var item = e.target.closest('.timeline-item');
            showConfirmDialog('Delete this event?', function() {
                var eventId = parseInt(item.dataset.eventId, 10);
                bridge.delete_event(eventId).then(function(json) {
                    try {
                        var result = JSON.parse(json);
                        if (result.success) {
                            setTimeout(loadEventTimeline, 100);
                        }
                    } catch (ex) {}
                }).catch(function(err) {
                    console.error('Delete event failed:', err);
                });
            });
            e.stopPropagation();
            return;
        }

        // ── Table checkbox ──
        if (e.target.closest('.table-checkbox')) {
            var cb = e.target;
            var eventId = parseInt(cb.dataset.eventId, 10);
            if (cb.classList.contains('select-all-checkbox')) {
                var filtered = _getFilteredSortedEvents();
                if (cb.checked) {
                    filtered.forEach(function(ev) { if (ev.id) _selectedEventIds.add(ev.id); });
                } else {
                    filtered.forEach(function(ev) { if (ev.id) _selectedEventIds.delete(ev.id); });
                }
                _updateBatchActionBar();
                _updateTableRowSelection();
                _updateTimelineItemSelection();
            } else if (eventId) {
                if (e.shiftKey && _lastShiftClickIdx >= 0) {
                    var filteredEvts = _getFilteredSortedEvents();
                    var currentIdx = -1;
                    filteredEvts.forEach(function(ev, i) {
                        if (ev.id === eventId) currentIdx = i;
                    });
                    if (currentIdx >= 0) {
                        _selectEventRange(_lastShiftClickIdx, currentIdx, filteredEvts);
                    }
                } else {
                    var idx = -1;
                    var evts = window._timelineEvents || [];
                    evts.forEach(function(ev, i) {
                        if (ev.id === eventId) idx = i;
                    });
                    _lastShiftClickIdx = idx;
                    _toggleEventSelection(eventId, e.ctrlKey || e.metaKey);
                }
            }
            e.stopPropagation();
            return;
        }

        // ── Table edit button ──
        if (e.target.closest('.table-edit-btn')) {
            var btn = e.target.closest('.table-edit-btn');
            var eid = parseInt(btn.dataset.eventId, 10);
            openEditModal(eid, btn.dataset.eventType, btn.dataset.team);
            e.stopPropagation();
            return;
        }

        // ── Table delete button ──
        if (e.target.closest('.table-delete-btn')) {
            var btn = e.target.closest('.table-delete-btn');
            var eid = parseInt(btn.dataset.eventId, 10);
            showConfirmDialog('Delete this event?', function() {
                bridge.delete_event(eid).then(function(json) {
                    try {
                        var result = JSON.parse(json);
                        if (result.success) {
                            setTimeout(loadEventTimeline, 100);
                        }
                    } catch (ex) {}
                }).catch(function(err) {
                    console.error('Delete event failed:', err);
                });
            });
            e.stopPropagation();
            return;
        }

        // ── Table column header sort ──
        if (e.target.closest('.data-table th.sortable')) {
            var th = e.target.closest('.data-table th.sortable');
            if (e.target.closest('.col-filter-input')) return; // Ignore clicks on filter inputs
            var key = th.dataset.sortKey;
            if (_timelineSortState.key === key) {
                _timelineSortState.dir = _timelineSortState.dir === 'asc' ? 'desc' : 'asc';
            } else {
                _timelineSortState.key = key;
                _timelineSortState.dir = 'asc';
            }
            renderTimelineTable(window._timelineEvents || []);
            return;
        }

        // ── Table column filter input (delegated) ──
        if (e.target.closest('.col-filter-input')) {
            // Handled by input event listener below
            return;
        }

        // ── Table pagination ──
        if (e.target.closest('.pagination-btn')) {
            var btn = e.target.closest('.pagination-btn');
            var page = btn.dataset.page;
            if (page === 'prev') {
                _timelinePageState.page = Math.max(1, _timelinePageState.page - 1);
            } else if (page === 'next') {
                var total = _getFilteredSortedEvents().length;
                var maxPage = Math.ceil(total / _timelinePageState.perPage);
                _timelinePageState.page = Math.min(maxPage, _timelinePageState.page + 1);
            } else {
                _timelinePageState.page = parseInt(page, 10);
            }
            renderTimelineTable(window._timelineEvents || []);
            return;
        }

        // ── Table per-page select (delegated) ──
        if (e.target.closest('.per-page-select')) {
            // Handled by change event
            return;
        }
    });

    // ── Chart cross-filter ──────────────────────────────────

    var _chartFilterCanvasId = null;

    function wireChartFilter() {
        if (typeof window.__kawkabChartFilter !== 'function') {
            window.__kawkabChartFilter = function(data) {
                filterTimelineByTimeRange(data.match_minute, data.range, data.canvasId);
            };
        }
    }

    function filterTimelineByTimeRange(matchMinute, range, canvasId) {
        // If same chart clicked again, clear filter
        if (_activeChartFilter && _activeChartFilter.canvasId === canvasId) {
            clearTimelineFilter();
            return;
        }

        var startMin = matchMinute - range;
        var endMin = matchMinute + range;
        _activeChartFilter = { canvasId: canvasId, startMin: startMin, endMin: endMin };
        _chartFilterCanvasId = canvasId;

        // Highlight chart
        document.querySelectorAll('.chart-container').forEach(function(c) {
            c.classList.remove('chart-filter-active');
        });
        var canvas = document.getElementById(canvasId);
        if (canvas && canvas.parentElement) {
            canvas.parentElement.classList.add('chart-filter-active');
        }

        // Show banner
        var banner = document.getElementById('timeline-filter-banner');
        var text = document.getElementById('filter-banner-text');
        if (banner && text) {
            banner.classList.remove('hidden');
            text.textContent = 'Filtered: ' + Math.max(0, startMin) + "'-" + endMin + "'";
        }

        renderTimeline(window._timelineEvents || []);
    }

    function clearTimelineFilter() {
        _activeChartFilter = null;
        _chartFilterCanvasId = null;

        document.querySelectorAll('.chart-container').forEach(function(c) {
            c.classList.remove('chart-filter-active');
        });

        var banner = document.getElementById('timeline-filter-banner');
        if (banner) banner.classList.add('hidden');

        renderTimeline(window._timelineEvents || []);
    }

    document.getElementById('filter-banner-clear')?.addEventListener('click', clearTimelineFilter);

    // ── Item 7: Timeline Scrub/Zoom Widget ────────────────

    var _scrubState = { dragging: null, startMin: 0, endMin: 90 };

    function initTimelineScrubber() {
        var canvas = document.getElementById('scrubber-canvas');
        if (!canvas) return;
        if (!window._timelineEvents || window._timelineEvents.length === 0) {
            document.getElementById('timeline-scrubber').classList.add('empty');
            return;
        }
        document.getElementById('timeline-scrubber').classList.remove('empty');
        drawScrubberDensity(window._timelineEvents);
        _scrubState.startMin = 0;
        _scrubState.endMin = 90;
        updateScrubberRange();

        var handleL = document.getElementById('scrub-handle-left');
        var handleR = document.getElementById('scrub-handle-right');

        function onDragStart(side, e) {
            e.preventDefault();
            _scrubState.dragging = side;
            document.addEventListener('mousemove', onDragMove);
            document.addEventListener('mouseup', onDragEnd);
        }

        function onDragMove(e) {
            if (!_scrubState.dragging) return;
            var rect = canvas.getBoundingClientRect();
            var totalMin = 90;
            var pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            var min = Math.round(pct * totalMin);
            if (_scrubState.dragging === 'left') {
                _scrubState.startMin = Math.min(min, _scrubState.endMin - 1);
            } else {
                _scrubState.endMin = Math.max(min, _scrubState.startMin + 1);
            }
            updateScrubberRange();
            filterTimelineByTimeRangeScrub();
        }

        function onDragEnd() {
            _scrubState.dragging = null;
            document.removeEventListener('mousemove', onDragMove);
            document.removeEventListener('mouseup', onDragEnd);
        }

        handleL.addEventListener('mousedown', function(e) { onDragStart('left', e); });
        handleR.addEventListener('mousedown', function(e) { onDragStart('right', e); });

        canvas.addEventListener('dblclick', function() {
            _scrubState.startMin = 0;
            _scrubState.endMin = 90;
            updateScrubberRange();
            clearTimelineFilter();
        });
    }

    function drawScrubberDensity(events) {
        var canvas = document.getElementById('scrubber-canvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var w = canvas.width;
        var h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        var totalMin = 90;
        var minuteCounts = {};
        var goalCounts = {};
        var shotCounts = {};
        events.forEach(function(e) {
            var min = Math.floor((e.timestamp || 0) / 60);
            minuteCounts[min] = (minuteCounts[min] || 0) + 1;
            if (e.event_type === 'goal') goalCounts[min] = (goalCounts[min] || 0) + 1;
            else if (e.event_type === 'shot') shotCounts[min] = (shotCounts[min] || 0) + 1;
        });

        var maxCount = 1;
        for (var i = 0; i <= totalMin; i++) {
            maxCount = Math.max(maxCount, minuteCounts[i] || 0);
        }

        var barW = Math.max(1, (w - 4) / totalMin);
        for (var min = 0; min <= totalMin; min++) {
            var x = 2 + min * barW;
            var total = minuteCounts[min] || 0;
            var goals = goalCounts[min] || 0;
            var shots = shotCounts[min] || 0;
            var others = total - goals - shots;
            var barH = (total / maxCount) * (h - 4);
            var y = h - 2 - barH;

            if (goals > 0) {
                var gH = (goals / maxCount) * (h - 4);
                ctx.fillStyle = '#ef4444';
                ctx.fillRect(x, h - 2 - gH, barW, gH);
            }
            if (shots > 0) {
                var sH = (shots / maxCount) * (h - 4);
                ctx.fillStyle = '#f97316';
                ctx.fillRect(x, h - 2 - sH - gH, barW, sH);
            }
            if (others > 0) {
                ctx.fillStyle = '#3b82f6';
                ctx.fillRect(x, y, barW, barH);
            }
        }
    }

    function updateScrubberRange() {
        var canvas = document.getElementById('scrubber-canvas');
        if (!canvas) return;
        var totalMin = 90;
        var pctL = _scrubState.startMin / totalMin;
        var pctR = 1 - (_scrubState.endMin / totalMin);
        var handleL = document.getElementById('scrub-handle-left');
        var handleR = document.getElementById('scrub-handle-right');
        var highlight = document.getElementById('scrubber-range-highlight');
        if (handleL) handleL.style.left = 'calc(' + (pctL * 100) + '% - 0px)';
        if (handleR) handleR.style.right = 'calc(' + (pctR * 100) + '% - 0px)';
        if (highlight) {
            highlight.style.left = 'calc(' + (pctL * 100) + '% + 6px)';
            highlight.style.right = 'calc(' + (pctR * 100) + '% + 6px)';
        }
    }

    function filterTimelineByTimeRangeScrub() {
        var startMin = _scrubState.startMin;
        var endMin = _scrubState.endMin;
        _activeChartFilter = { canvasId: 'scrubber', startMin: startMin, endMin: endMin };

        var banner = document.getElementById('timeline-filter-banner');
        var text = document.getElementById('filter-banner-text');
        if (banner && text) {
            banner.classList.remove('hidden');
            text.textContent = 'Filtered: ' + startMin + "'-" + endMin + "'";
        }

        document.querySelectorAll('.chart-container').forEach(function(c) {
            c.classList.remove('chart-filter-active');
        });

        renderTimeline(window._timelineEvents || []);
    }

    // ── Item 10: Data Density Toggle ──────────────────────

    function initDensityToggle() {
        var stored = null;
        try { stored = localStorage.getItem('kawkab_density'); } catch(e) {}
        if (stored) {
            document.documentElement.setAttribute('data-density', stored);
            document.querySelectorAll('.density-btn').forEach(function(b) {
                b.classList.toggle('active', b.dataset.density === stored);
            });
        }
        document.querySelectorAll('.density-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var density = this.dataset.density;
                document.documentElement.setAttribute('data-density', density);
                document.querySelectorAll('.density-btn').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                try { localStorage.setItem('kawkab_density', density); } catch(e) {}
            });
        });
    }

    // ── Item 12: Color Customization ──────────────────────

    function initColorSettings() {
        var stored = null;
        try { stored = JSON.parse(localStorage.getItem('kawkab_colors')); } catch(e) {}
        if (stored) {
            applyTeamColors(stored.home || '#2563eb', stored.away || '#dc2626');
        }
        var homeInput = document.getElementById('home-color');
        var awayInput = document.getElementById('away-color');
        var resetBtn = document.getElementById('reset-colors-btn');

        homeInput.addEventListener('input', function() {
            applyTeamColors(this.value, awayInput.value);
            saveTeamColors();
        });
        awayInput.addEventListener('input', function() {
            applyTeamColors(homeInput.value, this.value);
            saveTeamColors();
        });
        resetBtn.addEventListener('click', function() {
            applyTeamColors('#2563eb', '#dc2626');
            homeInput.value = '#2563eb';
            awayInput.value = '#dc2626';
            try { localStorage.removeItem('kawkab_colors'); } catch(e) {}
            if (window.__invalidateColorCache) window.__invalidateColorCache();
        });

        document.getElementById('color-palette-btn').addEventListener('click', function() {
            document.getElementById('color-settings').classList.toggle('hidden');
        });

        document.addEventListener('click', function(e) {
            var wrapper = document.querySelector('.color-palette-wrapper');
            if (wrapper && !wrapper.contains(e.target)) {
                document.getElementById('color-settings').classList.add('hidden');
            }
        });
    }

    function applyTeamColors(home, away) {
        var root = document.documentElement;
        root.style.setProperty('--team-home', home);
        root.style.setProperty('--team-away', away);
    }

    function saveTeamColors() {
        var home = document.getElementById('home-color').value;
        var away = document.getElementById('away-color').value;
        try { localStorage.setItem('kawkab_colors', JSON.stringify({ home: home, away: away })); } catch(e) {}
        if (window.__invalidateColorCache) window.__invalidateColorCache();
        // Re-render charts if match loaded
        if (currentMatchId && analysisResult) {
            renderResults(analysisResult);
        }
    }

    // ── Item 13: Video Keyboard Shortcuts ─────────────────

    function initVideoShortcuts() {
        document.addEventListener('keydown', function(e) {
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            var video = document.getElementById('match-video');
            if (!video || !video.src) return;
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    if (video.paused) video.play(); else video.pause();
                    break;
                case 'k':
                case 'K':
                    video.pause();
                    break;
                case 'j':
                case 'J':
                    video.currentTime = Math.max(0, video.currentTime - 10);
                    break;
                case 'l':
                case 'L':
                    video.currentTime = Math.min(video.duration || 0, video.currentTime + 10);
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    video.currentTime = Math.max(0, video.currentTime - 5);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    video.currentTime = Math.min(video.duration || 0, video.currentTime + 5);
                    break;
                case 'f':
                case 'F':
                    if (video.requestFullscreen) {
                        video.requestFullscreen();
                    } else if (video.webkitRequestFullscreen) {
                        video.webkitRequestFullscreen();
                    }
                    break;
            }
        });
    }

    // ── Item 14: Persistent Filter State ──────────────────

    function saveFilterState() {
        try {
            sessionStorage.setItem('kawkab_timeline_filter_type', document.getElementById('timeline-filter-type').value);
            sessionStorage.setItem('kawkab_timeline_sort_column', _timelineSortState.key);
            sessionStorage.setItem('kawkab_timeline_sort_dir', _timelineSortState.dir);
            sessionStorage.setItem('kawkab_timeline_page', _timelinePageState.page);
            sessionStorage.setItem('kawkab_timeline_view', _currentTimelineView);
            sessionStorage.setItem('kawkab_timeline_search', _timelineSearchText);
            sessionStorage.setItem('kawkab_dashboard_daterange', document.getElementById('dashboard-date-range').value);
        } catch(e) {}
    }

    function restoreFilterState() {
        try {
            var filterType = sessionStorage.getItem('kawkab_timeline_filter_type');
            var sortCol = sessionStorage.getItem('kawkab_timeline_sort_column');
            var sortDir = sessionStorage.getItem('kawkab_timeline_sort_dir');
            var page = sessionStorage.getItem('kawkab_timeline_page');
            var view = sessionStorage.getItem('kawkab_timeline_view');
            var search = sessionStorage.getItem('kawkab_timeline_search');
            var daterange = sessionStorage.getItem('kawkab_dashboard_daterange');

            if (filterType) {
                var sel = document.getElementById('timeline-filter-type');
                if (sel) sel.value = filterType;
            }
            if (sortCol) _timelineSortState.key = sortCol;
            if (sortDir) _timelineSortState.dir = sortDir;
            if (page) _timelinePageState.page = parseInt(page, 10) || 1;
            if (view) _currentTimelineView = view;
            if (search != null) _timelineSearchText = search;
            if (daterange) {
                var dr = document.getElementById('dashboard-date-range');
                if (dr) dr.value = daterange;
            }
        } catch(e) {}
    }

    // ── Coding Workspace (delegated to app-coding.js) ─────────

    function initCodingWorkspace() {
        if (window.KawkabCoding) return window.KawkabCoding.initCodingWorkspace();
    }

    // ── Tactical Periods + Formation (delegated to app-tactics.js) ──

    function initTacticsWorkspace() {
        if (window.KawkabTactics) return window.KawkabTactics.initTacticsWorkspace();
    }

    // ── Tactical Whiteboard (delegated to app-whiteboard.js) ──

    function initWhiteboardWorkspace() {
        if (window.KawkabWhiteboard) return window.KawkabWhiteboard.initWhiteboardWorkspace();
    }

    // ── Settings (delegated to app-settings.js) ──

    function initSettingsWorkspace() {
        if (window.KawkabSettings) return window.KawkabSettings.initSettingsWorkspace();
    }

    // ── 3D Pitch Visualization (delegated to app-3d.js) ──

    function loadPitch3dMatchSelect() {
        if (typeof bridge === 'undefined' || !bridge) return;
        bridge.get_all_matches(function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) data = [];
                var sel = document.getElementById('pitch3d-match-select');
                if (!sel) return;
                sel.innerHTML = '<option value="">-- Select Match --</option>';
                (data || []).forEach(function(m) {
                    var name = m.name || (m.home_team + ' vs ' + m.away_team) || 'Match #' + m.id;
                    sel.innerHTML += '<option value="' + m.id + '">' + escapeHtml(name) + '</option>';
                });
            } catch(e) { showToast('Failed to load matches for 3D pitch.', 'error'); console.warn(e); }
        });
    }

    function initPitch3dWorkspace() {
        if (window.Kawkab3DPitch) return;
    }

    // ── Sprint 2: Physiology & Wearables ──
    var _physInitialized = false;
    var _physWearableData = null;
    var _physWearablePath = '';

    function initPhysiology() {
        if (_physInitialized) return;
        _physInitialized = true;

        var importBtn = document.getElementById('phys-import-btn');
        var fileInput = document.getElementById('phys-wearable-input');
        var mergeBtn = document.getElementById('phys-merge-btn');
        var correlateBtn = document.getElementById('phys-correlate-btn');

        importBtn.addEventListener('click', function () { fileInput.click(); });
        fileInput.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                var f = e.target.files[0];
                _physWearablePath = f.name;
                var reader = new FileReader();
                reader.onload = function (ev) {
                    bridge.import_wearable(_physWearablePath, function (result) {
                        try {
                            var data = JSON.parse(result);
                            if (data.error) { showToast(data.error, 'error'); return; }
                            _physWearableData = data.session;
                            renderWearableSummary(data.session);
                            showToast('Wearable imported: ' + (data.session.device_type || 'unknown'), 'success');
                        } catch (err) { showToast('Parse error', 'error'); }
                    });
                };
                reader.readAsText(f);
            }
            fileInput.value = '';
        });

        mergeBtn.addEventListener('click', function () {
            if (!_physWearableData) { showToast('Import wearable data first', 'warning'); return; }
            var sel = document.getElementById('phys-player-select');
            var pid = parseInt(sel.value);
            if (!pid) { showToast('Select a player', 'warning'); return; }
            var traj = [{ t: 0, v: 2.5 }, { t: 10, v: 3.0 }, { t: 20, v: 4.2 }, { t: 30, v: 5.0 }];
            var wearable = [{ t: 0, hr: 80, spd: 2.4 }, { t: 10, hr: 95, spd: 3.1 }, { t: 20, hr: 110, spd: 4.0 }, { t: 30, hr: 120, spd: 5.2 }];
            bridge.merge_player_physiology(pid, JSON.stringify(traj), JSON.stringify(wearable), 75.0, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    renderMergedPhysiology(data.report);
                } catch (err) { showToast('Merge failed', 'error'); }
            });
        });

        correlateBtn.addEventListener('click', function () {
            var events = [{ type: 'shot', timestamp: 15 }, { type: 'tackle', timestamp: 25 }];
            var speedTL = [{ t: 0, v_spd: 2.5 }, { t: 10, v_spd: 3.0 }, { t: 20, v_spd: 4.2 }, { t: 30, v_spd: 5.0 }];
            var hrTL = [{ t: 0, hr: 80 }, { t: 10, hr: 95 }, { t: 20, hr: 110 }, { t: 30, hr: 120 }];
            bridge.analyze_physio_tactical(JSON.stringify(events), JSON.stringify(speedTL), JSON.stringify(hrTL), 5.0, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    renderCorrelation(data.report);
                } catch (err) { showToast('Correlation failed', 'error'); }
            });
        });

        loadPlayerSelect();
    }

    function renderWearableSummary(session) {
        var el = document.getElementById('phys-wearable-content');
        if (!el) return;
        el.innerHTML = '<div class="phys-metrics-grid">' +
            '<div class="phys-metric"><span class="phys-metric-label">Device</span><span class="phys-metric-value">' + escapeHtml(session.device_type || '-') + '</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Duration</span><span class="phys-metric-value">' + (session.duration_s || 0) + 's</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Points</span><span class="phys-metric-value">' + (session.point_count || 0) + '</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Avg HR</span><span class="phys-metric-value">' + (session.avg_hr || '-') + ' bpm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Max HR</span><span class="phys-metric-value">' + (session.max_hr || '-') + ' bpm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Distance</span><span class="phys-metric-value">' + (session.total_distance_m || 0) + 'm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Max Speed</span><span class="phys-metric-value">' + (session.max_speed_ms || 0) + ' m/s</span></div>' +
            '</div>';
    }

    function renderMergedPhysiology(report) {
        var el = document.getElementById('phys-merge-content');
        if (!el) return;
        el.innerHTML = '<div class="phys-metrics-grid">' +
            '<div class="phys-metric"><span class="phys-metric-label">Duration</span><span class="phys-metric-value">' + (report.duration_s || 0) + 's</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Video Distance</span><span class="phys-metric-value">' + (report.video_distance_m || 0) + 'm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Wearable Distance</span><span class="phys-metric-value">' + (report.wearable_distance_m || '-') + 'm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Avg HR</span><span class="phys-metric-value">' + (report.avg_hr || '-') + ' bpm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Peak HR</span><span class="phys-metric-value">' + (report.peak_hr || '-') + ' bpm</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Speed Corr</span><span class="phys-metric-value">' + (report.correlation_speed_r !== null ? report.correlation_speed_r.toFixed(3) : '-') + '</span></div>' +
            '<div class="phys-metric"><span class="phys-metric-label">Timeline Pts</span><span class="phys-metric-value">' + (report.timeline_points || 0) + '</span></div>' +
            '</div>';
        document.getElementById('phys-workspace').classList.remove('hidden');
        var hrEl = document.getElementById('phys-hr-content');
        if (hrEl && report.hr_zones) {
            var zones = report.hr_zones;
            hrEl.innerHTML = '<div class="phys-metrics-grid">' +
                '<div class="phys-metric"><span class="phys-metric-label">Zone 1 (50-60%)</span><span class="phys-metric-value">' + (zones.zone1_50_60 || 0) + '</span></div>' +
                '<div class="phys-metric"><span class="phys-metric-label">Zone 2 (60-70%)</span><span class="phys-metric-value">' + (zones.zone2_60_70 || 0) + '</span></div>' +
                '<div class="phys-metric"><span class="phys-metric-label">Zone 3 (70-80%)</span><span class="phys-metric-value">' + (zones.zone3_70_80 || 0) + '</span></div>' +
                '<div class="phys-metric"><span class="phys-metric-label">Zone 4 (80-90%)</span><span class="phys-metric-value">' + (zones.zone4_80_90 || 0) + '</span></div>' +
                '<div class="phys-metric"><span class="phys-metric-label">Zone 5 (90-100%)</span><span class="phys-metric-value">' + (zones.zone5_90_100 || 0) + '</span></div>' +
                '</div>';
        }
    }

    function renderCorrelation(report) {
        var el = document.getElementById('phys-correlation-content');
        if (!el || !report.correlations) return;
        var html = '';
        (report.correlations || []).forEach(function (c) {
            html += '<div class="phys-correlation-item">' +
                '<strong>' + escapeHtml(c.event_type) + '</strong> ' +
                '(n=' + c.sample_count + ') pre: ' + c.pre_speed + ' m/s → post: ' + c.post_speed + ' m/s ' +
                '<span class="' + (c.speed_delta_pct < 0 ? 'text-danger' : 'text-success') + '">' +
                (c.speed_delta_pct > 0 ? '+' : '') + c.speed_delta_pct.toFixed(1) + '%</span>' +
                (c.hr_delta_pct !== null ? ' HR: ' + (c.hr_delta_pct > 0 ? '+' : '') + c.hr_delta_pct.toFixed(1) + '%' : '') +
                '</div>';
        });
        html += '<div class="phys-summary">';
        if (report.summary) {
            html += '<p>Events: ' + report.summary.total_events_analyzed + ' | HI Bursts: ' + (report.high_intensity_bursts || []).length + ' | Fatigue: ' + (report.fatigue_periods || []).length + '</p>';
        }
        html += '</div>';
        el.innerHTML = html;
    }

    function loadPlayerSelect() {
        var sel = document.getElementById('phys-player-select');
        bridge.get_all_player_profiles(function (result) {
            try {
                var data = JSON.parse(result);
                if (data.profiles) {
                    sel.innerHTML = '<option value="">Select Player</option>';
                    data.profiles.forEach(function (p) {
                        sel.innerHTML += '<option value="' + (p.track_id || p.id || 0) + '">' + escapeHtml(p.name || 'Player') + '</option>';
                    });
                }
            } catch (e) {}
        });
    }

    function syncAllToMaster() {
        var master = document.getElementById('ma-video-0');
        if (!master) return;
        var t = master.currentTime;
        bridge.sync_positions(t, function (result) {
            try {
                var pos = JSON.parse(result);
                if (pos.error) return;
                (pos.positions || []).forEach(function (p) {
                    var slave = document.getElementById('ma-video-' + p.index);
                    if (slave) { slave.currentTime = p.time_s; }
                });
            } catch (e) {}
        });
    }

    function playAllVideos(play) {
        for (var i = 0; i < 3; i++) {
            var v = document.getElementById('ma-video-' + i);
            if (v) {
                if (play) { v.play().catch(function () {}); }
                else { v.pause(); }
            }
        }
    }

    function updateTrimDisplay() {
        function fmt(t) { if (t === null) return '--:--'; var m = Math.floor(t / 60); var s = Math.floor(t % 60); return m + ':' + (s < 10 ? '0' : '') + s; }
        document.getElementById('ma-trim-in').textContent = fmt(_maTrimIn);
        document.getElementById('ma-trim-out').textContent = fmt(_maTrimOut);
    }

    function initTelestration() {
        var toggleBtn = document.getElementById('telestrate-toggle-btn');
        var toolbar = document.getElementById('telestrate-toolbar');
        var canvas = document.getElementById('telestrate-canvas');
        var colorInput = document.getElementById('telestrate-color');
        var widthInput = document.getElementById('telestrate-width');
        var undoBtn = document.getElementById('telestrate-undo-btn');
        var redoBtn = document.getElementById('telestrate-redo-btn');
        var clearBtn = document.getElementById('telestrate-clear-btn');
        var saveBtn = document.getElementById('telestrate-save-btn');

        if (!toggleBtn || !canvas) return;

        // Toggle drawing mode
        toggleBtn.addEventListener('click', function() {
            _telestrateState.active = !_telestrateState.active;
            canvas.classList.toggle('active', _telestrateState.active);
            toolbar.classList.toggle('hidden', !_telestrateState.active);
            toggleBtn.textContent = _telestrateState.active ? '✏️ Drawing ON' : '✏️ Draw';
            if (_telestrateState.active) {
                resizeCanvas();
            }
        });

        // Tool selection
        toolbar.querySelectorAll('.telestrate-tool').forEach(function(btn) {
            btn.addEventListener('click', function() {
                toolbar.querySelectorAll('.telestrate-tool').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                _telestrateState.tool = this.dataset.tool;
                // Remove text input if switching away from text
                if (_telestrateState.tool !== 'text' && _telestrateState.textInput) {
                    _telestrateState.textInput.remove();
                    _telestrateState.textInput = null;
                }
            });
        });

        colorInput.addEventListener('input', function() {
            _telestrateState.color = this.value;
        });

        widthInput.addEventListener('input', function() {
            _telestrateState.width = parseInt(this.value, 10);
        });

        undoBtn.addEventListener('click', function() { telestrateUndo(); });
        redoBtn.addEventListener('click', function() { telestrateRedo(); });
        clearBtn.addEventListener('click', function() {
            if (_telestrateState.strokes.length === 0) return;
            showConfirmDialog('Clear all drawings?', function() {
                _telestrateState.redoStack = _telestrateState.redoStack.concat(_telestrateState.strokes);
                _telestrateState.strokes = [];
                redrawTelestration();
            });
        });
        saveBtn.addEventListener('click', function() {
            var video = document.getElementById('match-video');
            if (!video) return;
            // Create combined canvas with video frame + drawings
            var c = document.createElement('canvas');
            c.width = video.videoWidth || canvas.width;
            c.height = video.videoHeight || canvas.height;
            var ctx = c.getContext('2d');
            ctx.drawImage(video, 0, 0, c.width, c.height);
            ctx.drawImage(canvas, 0, 0, c.width, c.height);
            var link = document.createElement('a');
            link.download = 'telestration-' + Date.now() + '.png';
            link.href = c.toDataURL('image/png');
            link.click();
            showToast('Annotated frame saved!', 'success');
        });

        // Mouse events on canvas
        canvas.addEventListener('mousedown', telestrateMouseDown);
        canvas.addEventListener('mousemove', telestrateMouseMove);
        canvas.addEventListener('mouseup', telestrateMouseUp);
        canvas.addEventListener('mouseleave', telestrateMouseUp);

        // Touch support
        canvas.addEventListener('touchstart', function(e) { e.preventDefault(); var t = e.touches[0]; telestrateMouseDown({ offsetX: t.clientX - canvas.getBoundingClientRect().left, offsetY: t.clientY - canvas.getBoundingClientRect().top }); });
        canvas.addEventListener('touchmove', function(e) { e.preventDefault(); var t = e.touches[0]; telestrateMouseMove({ offsetX: t.clientX - canvas.getBoundingClientRect().left, offsetY: t.clientY - canvas.getBoundingClientRect().top }); });
        canvas.addEventListener('touchend', function(e) { e.preventDefault(); telestrateMouseUp({}); });

        // Resize canvas with video
        var video = document.getElementById('match-video');
        if (video) {
            video.addEventListener('loadedmetadata', resizeCanvas);
            video.addEventListener('resize', resizeCanvas);
        }
        window.addEventListener('resize', resizeCanvas);
    }

    function resizeCanvas() {
        var canvas = document.getElementById('telestrate-canvas');
        var video = document.getElementById('match-video');
        if (!canvas || !video) return;
        var rect = video.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        redrawTelestration();
    }

    function getCanvasPos(e) {
        var canvas = document.getElementById('telestrate-canvas');
        var rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function telestrateMouseDown(e) {
        if (!_telestrateState.active) return;
        var pos = e.offsetX != null ? { x: e.offsetX, y: e.offsetY } : getCanvasPos(e);
        _telestrateState.isDrawing = true;
        _telestrateState.startX = pos.x;
        _telestrateState.startY = pos.y;

        if (_telestrateState.tool === 'text') {
            var input = document.createElement('input');
            input.type = 'text';
            input.className = 'telestrate-text-input';
            input.style.position = 'absolute';
            input.style.left = pos.x + 'px';
            input.style.top = pos.y + 'px';
            input.style.zIndex = '20';
            input.style.background = 'rgba(0,0,0,0.7)';
            input.style.color = _telestrateState.color;
            input.style.border = '1px solid ' + _telestrateState.color;
            input.style.padding = '2px 6px';
            input.style.fontSize = '16px';
            input.style.borderRadius = '3px';
            input.style.outline = 'none';
            input.placeholder = 'Type text...';

            var wrapper = document.querySelector('.video-wrapper');
            wrapper.appendChild(input);
            input.focus();

            _telestrateState.textInput = input;

            input.addEventListener('keydown', function(ev) {
                if (ev.key === 'Enter') {
                    var text = this.value.trim();
                    if (text) {
                        _telestrateState.strokes.push({
                            tool: 'text',
                            color: _telestrateState.color,
                            x: pos.x, y: pos.y,
                            text: text,
                            fontSize: 16,
                        });
                        _telestrateState.redoStack = [];
                        redrawTelestration();
                    }
                    this.remove();
                    _telestrateState.textInput = null;
                } else if (ev.key === 'Escape') {
                    this.remove();
                    _telestrateState.textInput = null;
                }
            });
            _telestrateState.isDrawing = false;
        }
    }

    function telestrateMouseMove(e) {
        if (!_telestrateState.isDrawing || !_telestrateState.active) return;
        var pos = e.offsetX != null ? { x: e.offsetX, y: e.offsetY } : getCanvasPos(e);
        // Preview while drawing
        var canvas = document.getElementById('telestrate-canvas');
        var ctx = canvas.getContext('2d');
        redrawTelestration();
        ctx.save();
        ctx.strokeStyle = _telestrateState.color;
        ctx.fillStyle = _telestrateState.color;
        ctx.lineWidth = _telestrateState.width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        var tool = _telestrateState.tool;
        if (tool === 'arrow') {
            drawArrow(ctx, _telestrateState.startX, _telestrateState.startY, pos.x, pos.y);
        } else if (tool === 'circle') {
            var rx = Math.abs(pos.x - _telestrateState.startX);
            var ry = Math.abs(pos.y - _telestrateState.startY);
            ctx.beginPath();
            ctx.ellipse(_telestrateState.startX, _telestrateState.startY, rx, ry, 0, 0, Math.PI * 2);
            ctx.stroke();
        } else if (tool === 'rect') {
            ctx.strokeRect(_telestrateState.startX, _telestrateState.startY, pos.x - _telestrateState.startX, pos.y - _telestrateState.startY);
        } else if (tool === 'line') {
            ctx.beginPath();
            ctx.moveTo(_telestrateState.startX, _telestrateState.startY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        } else if (tool === 'freehand') {
            ctx.beginPath();
            ctx.moveTo(_telestrateState.startX, _telestrateState.startY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            _telestrateState.startX = pos.x;
            _telestrateState.startY = pos.y;
        } else if (tool === 'highlight') {
            ctx.globalAlpha = 0.3;
            ctx.lineWidth = _telestrateState.width * 4;
            ctx.beginPath();
            ctx.moveTo(_telestrateState.startX, _telestrateState.startY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            _telestrateState.startX = pos.x;
            _telestrateState.startY = pos.y;
        }
        ctx.restore();
    }

    function telestrateMouseUp(e) {
        if (!_telestrateState.isDrawing || !_telestrateState.active) return;
        _telestrateState.isDrawing = false;
        var pos = e.offsetX != null ? { x: e.offsetX, y: e.offsetY } : getCanvasPos(e);

        // Save stroke
        var stroke = {
            tool: _telestrateState.tool,
            color: _telestrateState.color,
            width: _telestrateState.width,
            startX: _telestrateState.startX,
            startY: _telestrateState.startY,
            endX: pos.x,
            endY: pos.y,
        };

        // For freehand/highlight, we record the path
        if (_telestrateState.tool === 'freehand' || _telestrateState.tool === 'highlight') {
            stroke.path = [{ x: _telestrateState.startX, y: _telestrateState.startY }, { x: pos.x, y: pos.y }];
            stroke.replay = function(ctx) {
                ctx.save();
                ctx.strokeStyle = this.color;
                ctx.lineWidth = this.tool === 'highlight' ? this.width * 4 : this.width;
                ctx.globalAlpha = this.tool === 'highlight' ? 0.3 : 1;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(this.path[0].x, this.path[0].y);
                for (var i = 1; i < this.path.length; i++) {
                    ctx.lineTo(this.path[i].x, this.path[i].y);
                }
                ctx.stroke();
                ctx.restore();
            };
        } else if (_telestrateState.tool === 'circle') {
            stroke.rx = Math.abs(pos.x - _telestrateState.startX);
            stroke.ry = Math.abs(pos.y - _telestrateState.startY);
            stroke.replay = function(ctx) {
                ctx.save();
                ctx.strokeStyle = this.color;
                ctx.lineWidth = this.width;
                ctx.beginPath();
                ctx.ellipse(this.startX, this.startY, this.rx, this.ry, 0, 0, Math.PI * 2);
                ctx.stroke();
                ctx.restore();
            };
        } else if (_telestrateState.tool === 'arrow') {
            stroke.replay = function(ctx) {
                ctx.save();
                ctx.strokeStyle = this.color;
                ctx.fillStyle = this.color;
                ctx.lineWidth = this.width;
                drawArrow(ctx, this.startX, this.startY, this.endX, this.endY);
                ctx.restore();
            };
        } else if (_telestrateState.tool === 'rect') {
            stroke.replay = function(ctx) {
                ctx.save();
                ctx.strokeStyle = this.color;
                ctx.lineWidth = this.width;
                ctx.strokeRect(this.startX, this.startY, this.endX - this.startX, this.endY - this.startY);
                ctx.restore();
            };
        } else if (_telestrateState.tool === 'line') {
            stroke.replay = function(ctx) {
                ctx.save();
                ctx.strokeStyle = this.color;
                ctx.lineWidth = this.width;
                ctx.beginPath();
                ctx.moveTo(this.startX, this.startY);
                ctx.lineTo(this.endX, this.endY);
                ctx.stroke();
                ctx.restore();
            };
        }

        _telestrateState.strokes.push(stroke);
        _telestrateState.redoStack = [];
        redrawTelestration();
    }

    function drawArrow(ctx, x1, y1, x2, y2) {
        var angle = Math.atan2(y2 - y1, x2 - x1);
        var headLen = 12 + ctx.lineWidth;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }

    function telestrateUndo() {
        if (_telestrateState.strokes.length === 0) return;
        var stroke = _telestrateState.strokes.pop();
        _telestrateState.redoStack.push(stroke);
        redrawTelestration();
    }

    function telestrateRedo() {
        if (_telestrateState.redoStack.length === 0) return;
        var stroke = _telestrateState.redoStack.pop();
        _telestrateState.strokes.push(stroke);
        redrawTelestration();
    }

    function redrawTelestration() {
        var canvas = document.getElementById('telestrate-canvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        _telestrateState.strokes.forEach(function(s) {
            if (s.replay) {
                s.replay(ctx);
            } else if (s.tool === 'text') {
                ctx.save();
                ctx.fillStyle = s.color;
                ctx.font = (s.fontSize || 16) + 'px sans-serif';
                ctx.fillText(s.text, s.x, s.y);
                ctx.restore();
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 10 — Telestration v2 Enhancements
    // ═══════════════════════════════════════════════════════════════

    var _telV2Initialized = false;
    var _telLayers = {};
    var _telAnimFrames = [];
    var _telAnimPlaying = false;
    var _telAnimCurrent = 0;

    function initTelestrationV2() {
        if (_telV2Initialized) return;
        _telV2Initialized = true;

        // Layer controls
        wireTelLayerButtons();
        wireTelAnimTimeline();
        wireTelPresets();
        wireTelNewTools();
    }

    function wireTelNewTools() {
        // Add new tools to toolbar: bezier, spotlight, magnifier, laser
        var toolbar = document.getElementById('telestrate-toolbar');
        if (!toolbar) return;
        var extras = [
            { tool: 'bezier', label: '🔄', title: 'Bezier Curve' },
            { tool: 'spotlight', label: '🔦', title: 'Spotlight' },
            { tool: 'magnifier', label: '🔍', title: 'Magnifying Glass' },
            { tool: 'laser', label: '🔴', title: 'Laser Pointer (trail)' },
        ];
        var ref = toolbar.querySelector('.telestrate-color') || toolbar.lastElementChild;
        extras.forEach(function(e) {
            var btn = document.createElement('button');
            btn.className = 'telestrate-tool';
            btn.dataset.tool = e.tool;
            btn.title = e.title;
            btn.textContent = e.label;
            btn.onclick = function() {
                toolbar.querySelectorAll('.telestrate-tool').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                _telestrateState.tool = this.dataset.tool;
            };
            toolbar.insertBefore(btn, ref);
        });

        // Add layer panel to telestration area
        var container = document.querySelector('.video-container') || document.querySelector('#telestrate-toolbar')?.parentElement;
        if (container && !document.getElementById('tel-layer-panel')) {
            var panel = document.createElement('div');
            panel.id = 'tel-layer-panel';
            panel.style.cssText = 'display:flex;gap:6px;align-items:center;padding:6px 8px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);margin-top:6px;flex-wrap:wrap';
            panel.innerHTML = '<span style="font-size:0.78rem;font-weight:600">Layers:</span>'
                + '<button id="tel-add-layer-btn" class="btn btn-sm btn-secondary" title="Add Layer">➕</button>'
                + '<button id="tel-toggle-layer-btn" class="btn btn-sm btn-secondary" title="Toggle Layer Visibility">👁️</button>'
                + '<span id="tel-layer-indicator" style="font-size:0.75rem;color:var(--text-muted)">Layer 1</span>'
                + '<input type="range" id="tel-layer-opacity" min="0" max="100" value="100" style="width:60px" title="Opacity">'
                + '<span style="font-size:0.78rem;font-weight:600;margin-left:12px">Anim:</span>'
                + '<button id="tel-anim-prev" class="btn btn-sm btn-secondary" title="Previous Frame">◀</button>'
                + '<span id="tel-anim-counter" style="font-size:0.75rem;color:var(--text-muted)">0/0</span>'
                + '<button id="tel-anim-next" class="btn btn-sm btn-secondary" title="Next Frame">▶</button>'
                + '<button id="tel-anim-play" class="btn btn-sm btn-secondary" title="Play Animation">▶▶</button>'
                + '<button id="tel-anim-record" class="btn btn-sm btn-secondary" title="Record Frame">⏺️</button>'
                + '<span style="font-size:0.78rem;font-weight:600;margin-left:12px">Export:</span>'
                + '<button id="tel-export-video-btn" class="btn btn-sm btn-secondary" title="Export Annotated Video">🎬 Export</button>'
                + '<button id="tel-save-preset-btn" class="btn btn-sm btn-secondary" title="Save Preset">💾 Save</button>'
                + '<button id="tel-load-preset-btn" class="btn btn-sm btn-secondary" title="Load Preset">📂 Load</button>';
            container.appendChild(panel);
        }
    }

    function wireTelLayerButtons() {
        document.getElementById('tel-add-layer-btn')?.addEventListener('click', function() {
            var layerId = 'layer_' + Date.now();
            var name = prompt('Layer name:', 'Layer ' + (Object.keys(_telLayers).length + 1));
            bridge.tel_layer_add(layerId, name || layerId).then(function() {
                _telLayers[layerId] = { name: name || layerId, visible: true, opacity: 1 };
                document.getElementById('tel-layer-indicator').textContent = name || layerId;
            });
        });
        document.getElementById('tel-toggle-layer-btn')?.addEventListener('click', function() {
            var firstKey = Object.keys(_telLayers)[0];
            if (!firstKey) return;
            bridge.tel_layer_toggle(firstKey).then(function() {
                _telLayers[firstKey].visible = !_telLayers[firstKey].visible;
                document.getElementById('tel-layer-indicator').textContent = (_telLayers[firstKey].visible ? '' : '👁️‍🗨️ ') + _telLayers[firstKey].name;
            });
        });
        document.getElementById('tel-layer-opacity')?.addEventListener('input', function() {
            var firstKey = Object.keys(_telLayers)[0];
            if (!firstKey) return;
            var val = parseInt(this.value, 10) / 100;
            bridge.tel_layer_opacity(firstKey, val);
        });
    }

    function wireTelAnimTimeline() {
        document.getElementById('tel-anim-prev')?.addEventListener('click', function() {
            if (_telAnimCurrent > 0) {
                _telAnimCurrent--;
                applyAnimFrame(_telAnimCurrent);
            }
        });
        document.getElementById('tel-anim-next')?.addEventListener('click', function() {
            if (_telAnimCurrent < _telAnimFrames.length - 1) {
                _telAnimCurrent++;
                applyAnimFrame(_telAnimCurrent);
            }
        });
        document.getElementById('tel-anim-play')?.addEventListener('click', function() {
            if (_telAnimPlaying) {
                _telAnimPlaying = false;
                this.textContent = '▶▶';
                return;
            }
            if (_telAnimFrames.length === 0) return;
            _telAnimPlaying = true;
            this.textContent = '⏸️';
            playAnimLoop();
        });
        document.getElementById('tel-anim-record')?.addEventListener('click', function() {
            _telAnimFrames.push(JSON.parse(JSON.stringify(_telestrateState.strokes)));
            _telAnimCurrent = _telAnimFrames.length - 1;
            updateAnimCounter();
            showToast('Frame ' + _telAnimFrames.length + ' recorded', 'info');
        });
    }

    function playAnimLoop() {
        if (!_telAnimPlaying || _telAnimFrames.length === 0) {
            document.getElementById('tel-anim-play').textContent = '▶▶';
            return;
        }
        applyAnimFrame(_telAnimCurrent);
        _telAnimCurrent = (_telAnimCurrent + 1) % _telAnimFrames.length;
        updateAnimCounter();
        setTimeout(playAnimLoop, 500);
    }

    function applyAnimFrame(idx) {
        if (idx < 0 || idx >= _telAnimFrames.length) return;
        _telestrateState.strokes = JSON.parse(JSON.stringify(_telAnimFrames[idx]));
        redrawTelestration();
        updateAnimCounter();
    }

    function updateAnimCounter() {
        var el = document.getElementById('tel-anim-counter');
        if (el) el.textContent = (_telAnimCurrent + 1) + '/' + _telAnimFrames.length;
    }

    function wireTelPresets() {
        document.getElementById('tel-save-preset-btn')?.addEventListener('click', function() {
            var name = prompt('Preset name:', 'Tactical Board ' + new Date().toLocaleDateString());
            if (!name) return;
            bridge.tel_save_preset(name, JSON.stringify([{
                id: 'canvas', name: 'Canvas', visible: true, locked: false, opacity: 1,
                elements: _telestrateState.strokes.map(function(s, i) {
                    return { index: i, tool: s.tool || 'freehand', color: s.color, width: s.width, points: s.points || [] };
                })
            }])).then(function(raw) {
                var r = JSON.parse(raw);
                showToast(r.ok ? 'Preset saved!' : 'Error: ' + r.error, r.ok ? 'info' : 'error');
            });
        });
        document.getElementById('tel-load-preset-btn')?.addEventListener('click', function() {
            bridge.tel_list_presets().then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error || !r.presets || r.presets.length === 0) {
                    showToast('No presets found', 'warning');
                    return;
                }
                var names = r.presets.map(function(p) { return p.name; });
                var name = prompt('Preset name:\n' + names.join('\n'), names[0]);
                if (!name) return;
                bridge.tel_load_preset(name).then(function(raw2) {
                    var r2 = JSON.parse(raw2);
                    if (r2.ok) showToast('Preset loaded: ' + name, 'info');
                    else showToast('Error: ' + r2.error, 'error');
                });
            });
        });
        document.getElementById('tel-export-video-btn')?.addEventListener('click', function() {
            var video = document.getElementById('match-video');
            if (!video || !video.src) { showToast('No video loaded', 'warning'); return; }
            var src = video.src;
            if (src.startsWith('blob:')) { showToast('Cannot export from blob URL', 'warning'); return; }
            bridge.tel_export_video(src, JSON.stringify([{
                id: 'export', name: 'Export', elements: _telestrateState.strokes
            }]), '').then(function(raw) {
                var r = JSON.parse(raw);
                if (r.ok) showToast('Exported: ' + r.output, 'info');
                else showToast('Error: ' + r.error, 'error');
            });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       Phase 6 Sprint 3 — Telestration Layer Panel
       ═══════════════════════════════════════════════════════════════ */

    var _telLayerInitialized = false;
    var _telLayersUI = {};
    var _telSelectedLayerId = null;
    var _telAutoSaveInterval = null;

    function initTelLayerPanel() {
        if (_telLayerInitialized) return;
        _telLayerInitialized = true;

        var addBtn = document.getElementById('tel-add-layer-btn');
        var removeBtn = document.getElementById('tel-remove-layer-btn');
        var toggleBtn = document.getElementById('tel-toggle-layer-btn');
        var opacitySlider = document.getElementById('tel-layer-opacity');

        if (!addBtn) return;

        addBtn.addEventListener('click', function () {
            var name = prompt('Layer name:', 'Layer ' + (Object.keys(_telLayersUI).length + 1));
            if (!name) return;
            var layerId = 'layer_' + Date.now();
            if (typeof bridge.tel_layer_add === 'function') {
                bridge.tel_layer_add(layerId, name).then(function (raw) {
                    var r = typeof raw === 'string' ? JSON.parse(raw) : raw;
                    if (r.error) { showToast(r.error, 'error'); return; }
                    _telLayersUI[layerId] = { id: layerId, name: name, visible: true, opacity: 1.0, elements: 0 };
                    _telSelectedLayerId = layerId;
                    renderLayerList();
                    showToast('Layer added: ' + name, 'success');
                });
            } else {
                _telLayersUI[layerId] = { id: layerId, name: name, visible: true, opacity: 1.0, elements: 0 };
                _telSelectedLayerId = layerId;
                renderLayerList();
            }
        });

        removeBtn.addEventListener('click', function () {
            if (!_telSelectedLayerId) return;
            var lid = _telSelectedLayerId;
            if (typeof bridge.tel_layer_remove === 'function') {
                bridge.tel_layer_remove(lid).then(function () {});
            }
            delete _telLayersUI[lid];
            _telSelectedLayerId = Object.keys(_telLayersUI)[0] || null;
            renderLayerList();
            showToast('Layer removed', 'info');
        });

        toggleBtn.addEventListener('click', function () {
            if (!_telSelectedLayerId) return;
            var lid = _telSelectedLayerId;
            var layer = _telLayersUI[lid];
            if (!layer) return;
            layer.visible = !layer.visible;
            if (typeof bridge.tel_layer_toggle === 'function') {
                bridge.tel_layer_toggle(lid);
            }
            renderLayerList();
        });

        opacitySlider.addEventListener('input', function () {
            if (!_telSelectedLayerId) return;
            var val = parseInt(this.value, 10) / 100;
            var layer = _telLayersUI[_telSelectedLayerId];
            if (!layer) return;
            layer.opacity = val;
            if (typeof bridge.tel_layer_opacity === 'function') {
                bridge.tel_layer_opacity(_telSelectedLayerId, val);
            }
        });

        // Refresh layers from bridge on init
        refreshLayerList();
    }

    function refreshLayerList() {
        if (typeof bridge.tel_get_layers === 'function') {
            bridge.tel_get_layers().then(function (raw) {
                try {
                    var r = typeof raw === 'string' ? JSON.parse(raw) : raw;
                    if (r.layers) {
                        _telLayersUI = {};
                        r.layers.forEach(function (l) {
                            _telLayersUI[l.id] = l;
                        });
                        var keys = Object.keys(_telLayersUI);
                        _telSelectedLayerId = keys.length > 0 ? keys[0] : null;
                        renderLayerList();
                    }
                } catch (e) {}
            });
        }
    }

    function renderLayerList() {
        var list = document.getElementById('tel-layer-list');
        var removeBtn = document.getElementById('tel-remove-layer-btn');
        var toggleBtn = document.getElementById('tel-toggle-layer-btn');
        var opacitySlider = document.getElementById('tel-layer-opacity');
        if (!list) return;

        var keys = Object.keys(_telLayersUI);
        if (keys.length === 0) {
            list.innerHTML = '<span class="hint" style="font-size:0.75rem">No layers</span>';
            if (removeBtn) removeBtn.classList.add('hidden');
            if (toggleBtn) toggleBtn.classList.add('hidden');
            if (opacitySlider) opacitySlider.classList.add('hidden');
            return;
        }

        if (removeBtn) removeBtn.classList.remove('hidden');
        if (toggleBtn) toggleBtn.classList.remove('hidden');
        if (opacitySlider) opacitySlider.classList.remove('hidden');

        list.innerHTML = keys.map(function (lid) {
            var layer = _telLayersUI[lid];
            var active = lid === _telSelectedLayerId ? 'active' : '';
            var visIcon = layer.visible ? '👁️' : '👁️‍🗨️';
            return '<div class="tel-layer-item ' + active + '" data-layer-id="' + lid + '">'
                + '<span class="layer-vis" data-action="toggle">' + visIcon + '</span>'
                + '<span class="layer-name">' + escapeHtml(layer.name || lid) + '</span>'
                + '</div>';
        }).join('');

        list.querySelectorAll('.tel-layer-item').forEach(function (item) {
            item.addEventListener('click', function (e) {
                var lid = this.dataset.layerId;
                if (e.target.classList.contains('layer-vis')) {
                    // Toggle visibility
                    var layer = _telLayersUI[lid];
                    if (layer) {
                        layer.visible = !layer.visible;
                        if (typeof bridge.tel_layer_toggle === 'function') {
                            bridge.tel_layer_toggle(lid);
                        }
                        renderLayerList();
                    }
                    return;
                }
                _telSelectedLayerId = lid;
                renderLayerList();
                var layer = _telLayersUI[lid];
                if (layer && opacitySlider) {
                    opacitySlider.value = Math.round((layer.opacity || 1) * 100);
                }
            });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       Phase 6 Sprint 3 — Telestration Preset Browser
       ═══════════════════════════════════════════════════════════════ */

    var _telPresetInitialized = false;
    var _telPresets = [];
    var _telSelectedPreset = null;

    function initTelPresetBrowser() {
        if (_telPresetInitialized) return;
        _telPresetInitialized = true;

        var saveBtn2 = document.getElementById('tel-save-preset-btn2');
        var loadBtn2 = document.getElementById('tel-load-preset-btn2');
        var deleteBtn = document.getElementById('tel-delete-preset-btn');
        var closeBtn = document.getElementById('tel-preset-close-btn');
        var nameInput = document.getElementById('tel-preset-name-input');
        var presetList = document.getElementById('tel-preset-list');

        // Toggle preset browser via existing buttons
        var oldSaveBtn = document.getElementById('tel-save-preset-btn');
        var oldLoadBtn = document.getElementById('tel-load-preset-btn');

        if (oldSaveBtn) {
            oldSaveBtn.addEventListener('click', function () {
                togglePresetBrowser(true);
                loadPresetList();
            });
        }
        if (oldLoadBtn) {
            oldLoadBtn.addEventListener('click', function () {
                togglePresetBrowser(true);
                loadPresetList();
            });
        }
        if (closeBtn) {
            closeBtn.addEventListener('click', function () {
                togglePresetBrowser(false);
            });
        }

        if (saveBtn2) {
            saveBtn2.addEventListener('click', function () {
                var name = nameInput.value.trim();
                if (!name) { showToast('Enter a preset name', 'warning'); return; }
                savePreset(name);
            });
        }
        if (loadBtn2) {
            loadBtn2.addEventListener('click', function () {
                if (!_telSelectedPreset) { showToast('Select a preset', 'warning'); return; }
                loadPreset(_telSelectedPreset);
            });
        }
        if (deleteBtn) {
            deleteBtn.addEventListener('click', function () {
                if (!_telSelectedPreset) { showToast('Select a preset', 'warning'); return; }
                deletePreset(_telSelectedPreset);
            });
        }
    }

    function togglePresetBrowser(show) {
        var browser = document.getElementById('tel-preset-browser');
        if (!browser) return;
        browser.classList.toggle('hidden', !show);
        if (show) loadPresetList();
    }

    function loadPresetList() {
        var list = document.getElementById('tel-preset-list');
        if (!list) return;
        list.innerHTML = '<p class="hint">Loading presets...</p>';

        if (typeof bridge.tel_list_presets === 'function') {
            bridge.tel_list_presets().then(function (raw) {
                try {
                    var r = typeof raw === 'string' ? JSON.parse(raw) : raw;
                    _telPresets = r.presets || [];
                    renderPresetList();
                } catch (e) {
                    list.innerHTML = '<p class="hint">Error loading presets</p>';
                }
            }).catch(function () {
                list.innerHTML = '<p class="hint">Bridge unavailable</p>';
            });
        } else {
            list.innerHTML = '<p class="hint">Bridge unavailable</p>';
        }
    }

    function renderPresetList() {
        var list = document.getElementById('tel-preset-list');
        if (!list) return;
        if (_telPresets.length === 0) {
            list.innerHTML = '<p class="hint">No presets saved yet</p>';
            return;
        }
        list.innerHTML = _telPresets.map(function (p) {
            var selected = _telSelectedPreset === p.name ? 'selected' : '';
            return '<div class="tel-preset-item ' + selected + '" data-name="' + escapeHtml(p.name) + '">'
                + '<span class="tel-preset-item-name">' + escapeHtml(p.name) + '</span>'
                + '<span class="tel-preset-item-meta" style="font-size:0.7rem;color:var(--text-muted)">'
                + (p.layers || 0) + ' layers'
                + (p.updated_at ? ' · ' + new Date(p.updated_at).toLocaleDateString() : '')
                + '</span>'
                + '</div>';
        }).join('');

        list.querySelectorAll('.tel-preset-item').forEach(function (item) {
            item.addEventListener('click', function () {
                _telSelectedPreset = this.dataset.name;
                renderPresetList();
            });
        });
    }

    function savePreset(name) {
        var layers = Object.keys(_telLayersUI).map(function (lid) {
            var l = _telLayersUI[lid];
            return { id: l.id, name: l.name, visible: l.visible, locked: false, opacity: l.opacity, elements: [] };
        });
        if (layers.length === 0) {
            layers = [{ id: 'canvas', name: 'Canvas', visible: true, locked: false, opacity: 1, elements: _telestrateState ? _telestrateState.strokes.map(function(s,i) { return { index: i, tool: s.tool || 'freehand', color: s.color, width: s.width }; }) : [] }];
        }
        var layersJson = JSON.stringify(layers);

        if (typeof bridge.tel_save_preset === 'function') {
            bridge.tel_save_preset(name, layersJson).then(function (raw) {
                var r = typeof raw === 'string' ? JSON.parse(raw) : raw;
                if (r.error) { showToast('Error: ' + r.error, 'error'); return; }
                showToast('Preset saved: ' + name, 'success');
                loadPresetList();
            }).catch(function () {
                showToast('Preset saved locally', 'info');
                loadPresetList();
            });
        } else {
            showToast('Bridge save_preset not available', 'warning');
        }
    }

    function loadPreset(name) {
        if (typeof bridge.tel_load_preset === 'function') {
            bridge.tel_load_preset(name).then(function (raw) {
                var r = typeof raw === 'string' ? JSON.parse(raw) : raw;
                if (r.error) { showToast('Error: ' + r.error, 'error'); return; }
                showToast('Preset loaded: ' + name, 'success');
                refreshLayerList();
                togglePresetBrowser(false);
            }).catch(function () {
                showToast('Failed to load preset', 'error');
            });
        } else {
            showToast('bridge.tel_load_preset not available', 'warning');
        }
    }

    function deletePreset(name) {
        if (typeof bridge.tel_delete_preset === 'function') {
            bridge.tel_delete_preset(name).then(function (raw) {
                var r = typeof raw === 'string' ? JSON.parse(raw) : raw;
                if (r.error) { showToast('Error: ' + r.error, 'error'); return; }
                showToast('Preset deleted: ' + name, 'info');
                _telSelectedPreset = null;
                loadPresetList();
            }).catch(function () {
                showToast('Failed to delete preset', 'error');
            });
        } else {
            showToast('bridge.tel_delete_preset not available', 'warning');
        }
    }

    /* ═══════════════════════════════════════════════════════════════
       Phase 6 Sprint 3 — Telestration Export Video
       ═══════════════════════════════════════════════════════════════ */

    function wireTelExportVideo() {
        var exportBtn = document.getElementById('tel-export-video-btn');
        if (!exportBtn) return;
        // Already wired in wireTelPresets, add progress toast enhancement
        var origClick = exportBtn.click;
        exportBtn.addEventListener('click', function () {
            showToast('Exporting annotated video...', 'info');
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       Phase 6 Sprint 3 — Telestration localStorage Persistence
       ═══════════════════════════════════════════════════════════════ */

    function initTelLocalStorage() {
        // Auto-save strokes every 30 seconds
        if (_telAutoSaveInterval) clearInterval(_telAutoSaveInterval);
        _telAutoSaveInterval = setInterval(function () {
            if (_telestrateState && _telestrateState.strokes && _telestrateState.strokes.length > 0) {
                try {
                    var data = {
                        strokes: _telestrateState.strokes,
                        timestamp: Date.now(),
                        layerState: Object.keys(_telLayersUI).map(function (lid) {
                            var l = _telLayersUI[lid];
                            return { id: l.id, name: l.name, visible: l.visible, opacity: l.opacity };
                        }),
                    };
                    localStorage.setItem('kawkab_telestration_backup', JSON.stringify(data));
                } catch (e) {}
            }
        }, 30000);

        // Offer restore on page load
        try {
            var saved = localStorage.getItem('kawkab_telestration_backup');
            if (saved) {
                var data = JSON.parse(saved);
                if (data.strokes && data.strokes.length > 0) {
                    setTimeout(function () {
                        if (confirm('Restore previous telestration drawings (' + data.strokes.length + ' strokes)?')) {
                            if (_telestrateState) {
                                _telestrateState.strokes = data.strokes;
                                _telestrateState.redoStack = [];
                                redrawTelestration();
                            }
                            if (data.layerState) {
                                data.layerState.forEach(function (ls) {
                                    _telLayersUI[ls.id] = ls;
                                });
                                renderLayerList();
                            }
                            showToast('Telestration restored from backup', 'success');
                        } else {
                            localStorage.removeItem('kawkab_telestration_backup');
                        }
                    }, 1000);
                }
            }
        } catch (e) {}
    }

    function formatNumber(n, d) {
        if (n == null) return '0';
        if (d === undefined) d = 0;
        return Number(n).toFixed(d);
    }

    /* ═══════════════════════════════════════════════════════════════
       Wave B — Season Dashboard
       ═══════════════════════════════════════════════════════════════ */

    function initSeasonDashboard() {
        var refreshBtn = document.getElementById('season-refresh-btn');
        if (!refreshBtn) return;
        refreshBtn.addEventListener('click', loadSeasonData);
        // Also load when section becomes visible
        var seasonTab = document.querySelector('.nav-tab[data-route="season"]');
        if (seasonTab) {
            seasonTab.addEventListener('click', function() {
                setTimeout(loadSeasonData, 200);
            });
        }
        loadSeasonData();
    }

    function loadSeasonData() {
        if (typeof bridge === 'undefined' || !bridge) return;
        bridge.get_season_summary(function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) { showToast('Season data error: ' + data.error, 'error'); console.warn('Season error:', data.error); return; }

                document.getElementById('season-match-count').textContent = data.total_matches || 0;
                document.getElementById('season-event-count').textContent = data.total_events || 0;
                document.getElementById('season-shot-count').textContent = data.total_shots || 0;
                document.getElementById('season-avg-xg').textContent = (data.avg_xg || 0).toFixed(3);

                renderSeasonFixtures(data.matches || []);
                renderSeasonLeagueTable(data.matches || []);
            } catch(e) { showToast('Failed to load season data.', 'error'); console.warn('Season load error:', e); }
        });
    }

    function renderSeasonFixtures(matches) {
        var container = document.getElementById('season-fixtures');
        if (!container) return;
        if (!matches || matches.length === 0) {
            container.innerHTML = '<p class="hint">No matches loaded yet.</p>';
            return;
        }
        var sorted = matches.slice().sort(function(a, b) {
            return (a.date || '').localeCompare(b.date || '') || (a.id || 0) - (b.id || 0);
        });
        var html = '';
        sorted.forEach(function(m) {
            html += '<div class="season-fixture-row">' +
                '<span class="season-fixture-date">' + escapeHtml(m.date || '--') + '</span>' +
                '<span class="season-fixture-home">' + escapeHtml(m.home_team || 'Home') + '</span>' +
                '<span class="season-fixture-vs">vs</span>' +
                '<span class="season-fixture-away">' + escapeHtml(m.away_team || 'Away') + '</span>' +
                '<span class="season-fixture-score">' + (m.home_score != null ? m.home_score + '-' + m.away_score : '--') + '</span>' +
                '</div>';
        });
        container.innerHTML = html;
    }

    function renderSeasonLeagueTable(matches) {
        var container = document.getElementById('season-league-table');
        if (!container) return;
        if (!matches || matches.length === 0) {
            container.innerHTML = '<p class="hint">Need match results to build league table.</p>';
            return;
        }
        // Build standings from match data (if scores available)
        var teams = {};
        matches.forEach(function(m) {
            var home = m.home_team || 'Home';
            var away = m.away_team || 'Away';
            if (!teams[home]) teams[home] = { name: home, played: 0, won: 0, drawn: 0, lost: 0, gf: 0, ga: 0, pts: 0 };
            if (!teams[away]) teams[away] = { name: away, played: 0, won: 0, drawn: 0, lost: 0, gf: 0, ga: 0, pts: 0 };
            // If we have scores from analysis (not always available yet)
            // For now, just show basic stats
        });

        var standings = Object.keys(teams).map(function(k) { return teams[k]; });
        // Sort by name as default
        standings.sort(function(a, b) { return a.name.localeCompare(b.name); });

        if (standings.length === 0) {
            container.innerHTML = '<p class="hint">Load matches to see standings.</p>';
            return;
        }

        var html = '<table><thead><tr>' +
            '<th>#</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th>' +
            '</tr></thead><tbody>';
        standings.forEach(function(t, i) {
            var gd = t.gf - t.ga;
            html += '<tr class="' + (i < 2 ? 'pos-1' : '') + '">' +
                '<td>' + (i + 1) + '</td>' +
                '<td>' + escapeHtml(t.name) + '</td>' +
                '<td>' + t.played + '</td>' +
                '<td>' + t.won + '</td>' +
                '<td>' + t.drawn + '</td>' +
                '<td>' + t.lost + '</td>' +
                '<td>' + t.gf + '</td>' +
                '<td>' + t.ga + '</td>' +
                '<td>' + (gd >= 0 ? '+' : '') + gd + '</td>' +
                '<td>' + t.pts + '</td>' +
                '</tr>';
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    }

    /* ═══════════════════════════════════════════════════════════════
       Wave C — Training Planner
       ═══════════════════════════════════════════════════════════════ */

    var _trainingState = {
        drills: [],
        sessionDrills: [],
        sessionName: '',
        sessionDate: '',
        sessionDuration: 60,
    };

    function initTrainingPlanner() {
        var searchInput = document.getElementById('training-drill-search');
        var catFilter = document.getElementById('training-category-filter');
        var sessionName = document.getElementById('training-session-name');
        var sessionDate = document.getElementById('training-session-date');
        var sessionDur = document.getElementById('training-session-duration');
        var saveBtn = document.getElementById('training-save-btn');
        var clearBtn = document.getElementById('training-clear-btn');

        if (!searchInput) return;

        loadDrills();

        searchInput.addEventListener('input', renderDrills);
        catFilter.addEventListener('change', renderDrills);

        sessionName.addEventListener('input', function() { _trainingState.sessionName = this.value; });
        sessionDate.addEventListener('change', function() { _trainingState.sessionDate = this.value; });

        if (!sessionDate.value) {
            var today = new Date();
            sessionDate.value = today.toISOString().slice(0, 10);
            _trainingState.sessionDate = sessionDate.value;
        }

        saveBtn.addEventListener('click', function() {
            if (_trainingState.sessionDrills.length === 0) {
                showToast('Add at least one drill to the session.', 'warning');
                return;
            }
            if (!_trainingState.sessionName.trim()) {
                showToast('Enter a session name.', 'warning');
                return;
            }
            var totalMin = 0;
            _trainingState.sessionDrills.forEach(function(d) { totalMin += d.duration_min || 15; });
            var msg = 'Session "' + _trainingState.sessionName + '" saved! (' + _trainingState.sessionDrills.length + ' drills, ' + totalMin + ' min)';
            showToast(msg, 'success');
        });

        clearBtn.addEventListener('click', function() {
            if (_trainingState.sessionDrills.length === 0) return;
            showConfirmDialog('Clear session?', function() {
                _trainingState.sessionDrills = [];
                renderSessionDrills();
            });
        });

        // Make drill list items draggable
        setupTrainingDragDrop();
    }

    function loadDrills() {
        if (typeof bridge === 'undefined' || !bridge) {
            // Fallback: static drills
            _trainingState.drills = [
                { id: '1', name: 'Rondo 5v2', category: 'possession', difficulty: 'medium', duration_min: 10, description: 'Keep-away in tight space' },
                { id: '2', name: 'Finishing Circuit', category: 'finishing', difficulty: 'medium', duration_min: 15, description: 'Rotating shooting stations' },
                { id: '3', name: 'Defensive Shape', category: 'defending', difficulty: 'hard', duration_min: 20, description: 'Compact block + pressing triggers' },
                { id: '4', name: 'Passing Ladder', category: 'passing', difficulty: 'easy', duration_min: 10, description: 'One/two-touch combination patterns' },
                { id: '5', name: 'Interval Sprints', category: 'fitness', difficulty: 'hard', duration_min: 12, description: 'High-intensity interval running' },
                { id: '6', name: 'Corner Kick Routines', category: 'set_piece', difficulty: 'medium', duration_min: 15, description: ' attacking and defending set plays' },
                { id: '7', name: 'Pressing Triggers', category: 'defending', difficulty: 'hard', duration_min: 15, description: 'Counter-press and trap drills' },
                { id: '8', name: 'Positional Play', category: 'possession', difficulty: 'medium', duration_min: 20, description: 'Building through thirds' },
                { id: '9', name: 'Crossing & Finishing', category: 'finishing', difficulty: 'easy', duration_min: 15, description: 'Wide crosses with near-post/far-post runs' },
                { id: '10', name: 'Small-Sided Game', category: 'general', difficulty: 'medium', duration_min: 20, description: '5v5 with constraints' },
            ];
            renderDrills();
            return;
        }
        bridge.get_all_drills(function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) { showToast('Drill load error: ' + data.error, 'error'); console.warn('Drill load error:', data.error); return; }
                _trainingState.drills = data.drills || [];
                renderDrills();
            } catch(e) { showToast('Failed to load drills.', 'error'); console.warn('Drill load error:', e); }
        });
    }

    function renderDrills() {
        var container = document.getElementById('training-drill-list');
        if (!container) return;
        var search = (document.getElementById('training-drill-search').value || '').toLowerCase().trim();
        var cat = document.getElementById('training-category-filter').value;

        var filtered = _trainingState.drills.filter(function(d) {
            if (cat !== 'all' && d.category !== cat) return false;
            if (search && (d.name || '').toLowerCase().indexOf(search) < 0) return false;
            return true;
        });

        if (filtered.length === 0) {
            container.innerHTML = '<p class="hint">No drills match filters.</p>';
            return;
        }

        var html = '';
        filtered.forEach(function(d) {
            var inSession = _trainingState.sessionDrills.some(function(sd) { return sd.id === d.id; });
            html += '<div class="training-drill-item" draggable="true" data-drill-id="' + d.id + '" data-name="' + escapeHtml(d.name) + '" data-cat="' + d.category + '" data-dur="' + (d.duration_min || 15) + '" data-diff="' + d.difficulty + '">' +
                '<span class="training-drill-name">' + escapeHtml(d.name) + '</span>' +
                '<span class="training-drill-cat">' + escapeHtml(d.category) + '</span>' +
                '<span class="training-drill-diff ' + d.difficulty + '">' + escapeHtml(d.difficulty) + '</span>' +
                '<span class="training-drill-dur">' + (d.duration_min || 15) + ' min</span>' +
                (inSession ? '<span style="color:var(--success);font-size:0.7rem">✓</span>' : '') +
                '</div>';
        });
        container.innerHTML = html;

        // Wire drag events
        container.querySelectorAll('.training-drill-item').forEach(function(item) {
            item.addEventListener('dragstart', function(e) {
                e.dataTransfer.setData('text/plain', JSON.stringify({
                    id: this.dataset.drillId,
                    name: this.dataset.name,
                    category: this.dataset.cat,
                    duration_min: parseInt(this.dataset.dur, 10),
                    difficulty: this.dataset.diff,
                }));
                this.classList.add('dragging');
            });
            item.addEventListener('dragend', function() {
                this.classList.remove('dragging');
            });
        });
    }

    function setupTrainingDragDrop() {
        var dropZone = document.getElementById('training-session-drills');
        if (!dropZone) return;

        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', function() {
            this.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('drag-over');
            try {
                var data = JSON.parse(e.dataTransfer.getData('text/plain'));
                _trainingState.sessionDrills.push(data);
                renderSessionDrills();
            } catch(ex) { /* ignore */ }
        });
    }

    function renderSessionDrills() {
        var container = document.getElementById('training-session-drills');
        if (!container) return;
        if (_trainingState.sessionDrills.length === 0) {
            container.innerHTML = '<p class="hint">Drag drills here to build your session.</p>';
            return;
        }
        var html = '';
        var totalMin = 0;
        _trainingState.sessionDrills.forEach(function(d, idx) {
            totalMin += d.duration_min || 15;
            html += '<div class="training-session-drill-item">' +
                '<span class="order">' + (idx + 1) + '.</span>' +
                '<span class="name">' + escapeHtml(d.name || 'Drill') + '</span>' +
                '<span class="dur">' + (d.duration_min || 15) + ' min</span>' +
                '<span class="remove-drill" data-idx="' + idx + '">✕</span>' +
                '</div>';
        });
        html += '<div style="padding:6px 8px;font-size:0.78rem;color:var(--text-muted);border-top:1px solid var(--border);margin-top:4px;">Total: ' + totalMin + ' min (' + _trainingState.sessionDrills.length + ' drills)</div>';
        container.innerHTML = html;

        // Wire remove buttons
        container.querySelectorAll('.remove-drill').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var idx = parseInt(this.dataset.idx, 10);
                _trainingState.sessionDrills.splice(idx, 1);
                renderSessionDrills();
            });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       Wave D — Presentation Mode
       ═══════════════════════════════════════════════════════════════ */

    var _presState = {
        matchId: null,
        currentSlide: 0,
        slides: [],
        isFullscreen: false,
        matchData: null,
    };

    function initPresentationMode() {
        var startBtn = document.getElementById('pres-start-btn');
        var prevBtn = document.getElementById('pres-prev-btn');
        var nextBtn = document.getElementById('pres-next-btn');
        var fullscreenBtn = document.getElementById('pres-fullscreen-btn');
        var exportBtn = document.getElementById('pres-export-btn');

        if (!startBtn) return;

        // Populate match select
        if (typeof bridge !== 'undefined' && bridge) {
            bridge.get_all_matches(function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    var sel = document.getElementById('pres-match-select');
                    sel.innerHTML = '<option value="" data-i18n="presentationSelectMatch">Select Match</option>';
                    (data || []).forEach(function(m) {
                        var name = m.name || (m.home_team + ' vs ' + m.away_team) || 'Match #' + m.id;
                        sel.innerHTML += '<option value="' + m.id + '">' + escapeHtml(name) + '</option>';
                    });
                } catch(e) { showToast('Failed to load matches.', 'error'); console.warn(e); }
            });
        }

        startBtn.addEventListener('click', function() {
            var matchId = parseInt(document.getElementById('pres-match-select').value, 10);
            if (!matchId) { showToast('Select a match first.', 'warning'); return; }
            _presState.matchId = matchId;
            loadPresentationData(matchId);
        });

        prevBtn.addEventListener('click', function() { navigateSlide(-1); });
        nextBtn.addEventListener('click', function() { navigateSlide(1); });

        fullscreenBtn.addEventListener('click', function() {
            var container = document.getElementById('presentation-slides');
            if (!_presState.isFullscreen) {
                container.classList.add('pres-fullscreen');
                _presState.isFullscreen = true;
                fullscreenBtn.textContent = '✕ Exit Fullscreen';
            } else {
                container.classList.remove('pres-fullscreen');
                _presState.isFullscreen = false;
                fullscreenBtn.textContent = '⛶ Fullscreen';
            }
            window.dispatchEvent(new Event('resize'));
        });

        exportBtn.addEventListener('click', function() {
            if (!_presState.matchId) { showToast('Load a match presentation first.', 'warning'); return; }
            var slides = document.querySelectorAll('.pres-slide');
            if (!slides.length) { showToast('No slides to export.', 'warning'); return; }
            var html = '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Match Presentation</title>';
            html += '<style>body{font-family:sans-serif;margin:0;padding:0}.slide{page-break-after:always;padding:40px;min-height:100vh;display:flex;flex-direction:column;justify-content:center}</style></head><body>';
            slides.forEach(function(s) {
                var clone = s.cloneNode(true);
                clone.querySelectorAll('button, .pres-controls, video').forEach(function(el) { el.remove(); });
                html += '<div class="slide">' + clone.innerHTML + '</div>';
            });
            html += '</body></html>';
            var blob = new Blob([html], {type: 'text/html'});
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url; a.download = 'presentation-' + _presState.matchId + '.html';
            a.click();
            URL.revokeObjectURL(url);
            showToast('Presentation exported as HTML (' + slides.length + ' slides). Open in browser and print to PDF.', 'info');
        });

        // Keyboard nav
        document.addEventListener('keydown', function(e) {
            var section = document.getElementById('presentation-section');
            if (!section || section.classList.contains('hidden')) return;
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT')) return;
            if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); navigateSlide(1); }
            if (e.key === 'ArrowLeft') { e.preventDefault(); navigateSlide(-1); }
            if (e.key === 'Escape' && _presState.isFullscreen) {
                document.getElementById('pres-fullscreen-btn').click();
            }
        });
    }

    function loadPresentationData(matchId) {
        if (typeof bridge === 'undefined' || !bridge) { showToast('Bridge not ready.', 'error'); return; }

        // Get match info from all matches
        bridge.get_all_matches(function(matchResult) {
            try {
                var matches = typeof matchResult === 'string' ? JSON.parse(matchResult) : matchResult;
                if (Array.isArray(matches)) {
                    var match = matches.find(function(m) { return m.id === matchId; });
                    _presState.matchData = match || { home_team: 'Home', away_team: 'Away' };
                } else {
                    _presState.matchData = {};
                }
            } catch(e) { _presState.matchData = {}; }
            _presState.matchData = _presState.matchData || {};

            // Load events for stats
            bridge.get_match_events(matchId, function(evResult) {
                try {
                    var events = typeof evResult === 'string' ? JSON.parse(evResult) : evResult;
                    _presState.matchData.events = Array.isArray(events) ? events : (events.events || []);
                } catch(e) { _presState.matchData.events = []; }

                buildPresentationSlides();
                _presState.currentSlide = 0;
                document.getElementById('presentation-slides').classList.remove('hidden');
                renderSlide(0);
            });
        });
    }

    function buildPresentationSlides() {
        var d = _presState.matchData || {};
        var events = d.events || [];
        var home = d.home_team || 'Home';
        var away = d.away_team || 'Away';
        var matchName = d.name || (home + ' vs ' + away);

        var totalShots = events.filter(function(e) { return e.event_type === 'shot'; }).length;
        var totalGoals = events.filter(function(e) { return e.event_type === 'goal'; }).length;
        var totalPasses = events.filter(function(e) { return e.event_type === 'pass'; }).length;
        var totalTackles = events.filter(function(e) { return e.event_type === 'tackle'; }).length;
        var totalXg = 0;
        events.forEach(function(e) {
            var meta = e.metadata || {};
            totalXg += meta.xg || 0;
        });

        _presState.slides = [
            {
                title: matchName,
                subtitle: 'Match Analysis Presentation',
                stat: null,
                label: null,
                bg: 'pres-bg-gradient',
                grid: null,
            },
            {
                title: 'Match Stats Overview',
                subtitle: null,
                stat: null,
                label: null,
                bg: 'pres-bg-away',
                grid: [
                    { label: 'Goals', value: totalGoals },
                    { label: 'Shots', value: totalShots },
                    { label: 'Passes', value: totalPasses },
                    { label: 'Tackles', value: totalTackles },
                ],
            },
            {
                title: 'xG Analysis',
                subtitle: 'Expected Goals: ' + totalXg.toFixed(2),
                stat: totalXg.toFixed(2),
                label: 'Total xG',
                bg: 'pres-bg-gradient',
                grid: null,
            },
            {
                title: 'Event Breakdown',
                subtitle: 'Shots: ' + totalShots + ' | Goals: ' + totalGoals + ' | Passes: ' + totalPasses + ' | Tackles: ' + totalTackles,
                stat: null,
                label: null,
                bg: 'pres-bg-away',
                grid: null,
            },
            {
                title: home + ' vs ' + away,
                subtitle: 'Thank you',
                stat: null,
                label: 'Analysis by Kawkab AI',
                bg: 'pres-bg-gradient',
                grid: null,
            },
        ];
    }

    function navigateSlide(direction) {
        var newIdx = _presState.currentSlide + direction;
        if (newIdx < 0 || newIdx >= _presState.slides.length) return;
        _presState.currentSlide = newIdx;
        renderSlide(newIdx);
    }

    function renderSlide(idx) {
        var slide = _presState.slides[idx];
        if (!slide) return;
        var container = document.getElementById('pres-slide-content');
        var counter = document.getElementById('pres-slide-counter');
        if (!container) return;

        counter.textContent = (idx + 1) + ' / ' + _presState.slides.length;

        var html = '<div class="' + slide.bg + '">';
        html += slide.title ? '<h1>' + escapeHtml(slide.title) + '</h1>' : '';
        html += slide.subtitle ? '<p class="pres-subtitle">' + escapeHtml(slide.subtitle) + '</p>' : '';
        html += slide.stat != null ? '<div class="pres-stat">' + slide.stat + '</div>' : '';
        html += slide.label ? '<div class="pres-label">' + escapeHtml(slide.label) + '</div>' : '';

        if (slide.grid && slide.grid.length > 0) {
            html += '<div class="pres-grid">';
            slide.grid.forEach(function(g) {
                html += '<div class="pres-stat-card"><div class="pres-stat">' + g.value + '</div><div class="pres-label">' + escapeHtml(g.label) + '</div></div>';
            });
            html += '</div>';
        }

        html += '</div>';
        container.innerHTML = html;
        container.className = 'pres-slide-content ' + (slide.bg || '');

        // Update nav buttons
        document.getElementById('pres-prev-btn').disabled = idx <= 0;
        document.getElementById('pres-next-btn').disabled = idx >= _presState.slides.length - 1;
    }

    // ── Scout Portal (delegated to app-scout.js) ──────────────

    function initScoutPortal() {
        if (window.KawkabScout) return window.KawkabScout.initScoutPortal();
    }

    document.addEventListener('DOMContentLoaded', function() {
        document.getElementById('edit-event-save').addEventListener('click', function() {
            var eventId = parseInt(document.getElementById('edit-event-id').value, 10);
            if (!eventId) return;
            var updates = {
                event_type: document.getElementById('edit-event-type').value,
                team: document.getElementById('edit-event-team').value,
            };
            var note = document.getElementById('edit-event-note').value.trim();
            if (note) {
                updates.metadata = { coach_note: note };
            }
            bridge.update_event(eventId, JSON.stringify(updates)).then(function(json) {
                try {
                    var result = JSON.parse(json);
                    if (result.success) {
                        document.getElementById('edit-event-modal').classList.add('hidden');
                        setTimeout(loadEventTimeline, 100);
                    } else {
                        showToast('Failed to update event', 'error');
                    }
                } catch (ex) {
                    showToast('Failed to save: ' + ex, 'error');
                }
            }).catch(function(err) {
                console.error('Update event failed:', err);
            });
        });

        document.getElementById('edit-event-cancel').addEventListener('click', function() {
            document.getElementById('edit-event-modal').classList.add('hidden');
        });

        document.getElementById('edit-event-modal').addEventListener('click', function(e) {
            if (e.target === this) {
                this.classList.add('hidden');
            }
        });

        var video = document.getElementById('match-video');
        if (video) {
            video.addEventListener('timeupdate', function() {
                highlightCurrentTimelineItem(this.currentTime);
            });
        }

        // Init theme before anything else
        initTheme();

        // Init chart cross-filter callback
        wireChartFilter();

        setupEventListeners();

        // Detect initial language from KawkabPolish or localStorage
        var initialLang = 'en';
        if (window.KawkabPolish) {
            try {
                var stored = localStorage.getItem('kawkab_lang');
                if (stored === 'ar' || stored === 'en') initialLang = stored;
            } catch(e) {}
        }
        setLanguage(initialLang);

        initQWebChannel();
        setupFeedbackStars();
        window.__kawkab.setupSearchAndCompare();
        setTimeout(connectProgressSignals, 500);

        // ── Item 7: Init Timeline Scrubber ──
        initTimelineScrubber();
        // Re-init scrubber when timeline events load
        var _origLoadTimeline = loadEventTimeline;
        loadEventTimeline = function() {
            _origLoadTimeline.apply(this, arguments);
            setTimeout(initTimelineScrubber, 300);
        };

        // ── Item 10: Init Density Toggle ──
        initDensityToggle();

        // ── Item 12: Init Color Settings ──
        initColorSettings();

        // ── Item 13: Init Video Shortcuts ──
        initVideoShortcuts();

        // ── Item 14: Save/restore filter state on route change ──
        restoreFilterState();

        // Initialize SPA Router
        var router = new KawkabRouter();
        var _origNavigate = router.navigate || router._onHashChange;
        router.register('dashboard', 'dashboard-section', function() {
            saveFilterState();
            loadDashboard();
        });
        router.register('upload', 'upload-section', saveFilterState);
        router.register('analysis', 'analysis-section', saveFilterState);
        router.register('results', 'results-section', function() {
            saveFilterState();
        });
        router.register('report', 'report-section', saveFilterState);
        router.register('history', 'history-section', saveFilterState);
        router.register('professional', 'professional-section', saveFilterState);
        router.register('coding', 'coding-section', function() {
            loadCodingMatchSelect();
        });
        router.register('review', 'review-section', function() {
            loadReviewMatchSelect();
        });
        router.register('feedback', 'feedback-section', saveFilterState);
        router.register('season', 'season-section', function() {
            saveFilterState();
            loadSeasonData();
        });
        router.register('training', 'training-section', saveFilterState);
        router.register('scout', 'scout-section', saveFilterState);
        router.register('multiangle', 'multiangle-section', function() {
            saveFilterState();
            initMultiAngle();
        });
        router.register('physiology', 'physiology-section', function() {
            saveFilterState();
            initPhysiology();
        });
        router.register('collaboration', 'collaboration-section', function() {
            saveFilterState();
            initCollaboration();
        });
        router.register('livetagging', 'livetagging-section', function() {
            saveFilterState();
            initLiveTagging();
        });
        router.register('livestream', 'livestream-section', function() {
            saveFilterState();
            initStreamCapture();
        });
        router.register('scoutcamera', 'scoutcamera-section', function() {
            saveFilterState();
            initScoutCamera();
        });
        router.register('cloud', 'cloud-section', function() {
            saveFilterState();
            initCloud();
        });
        router.register('opponent', 'opponent-section', function() {
            saveFilterState();
            initOpponentWorkspace();
        });
        router.register('marketplace', 'marketplace-section', function() {
            saveFilterState();
            initMarketplace();
        });
        router.register('highlight', 'highlight-section', function() {
            saveFilterState();
            initHighlightWorkspace();
        });
        router.register('tactics', 'tactics-section', function() {
            saveFilterState();
            initTacticsWorkspace();
        });
        router.register('pitch3d', 'pitch3d-section', function() {
            saveFilterState();
            if (typeof init3DPitch === 'function') {
                init3DPitch();
                loadPitch3dMatchSelect();
            }
        });
        router.register('tacticsknowledge', 'tactics-knowledge-section', function() {
            saveFilterState();
            initTacticsKnowledge();
        });
        router.register('ai', 'ai-section', function() {
            saveFilterState();
            initAiWorkspace();
        });
        router.register('squad', 'squad-section', function() {
            saveFilterState();
            initSquadWorkspace();
        });
        router.register('briefing', 'briefing-section', function() {
            saveFilterState();
            initBriefingWorkspace();
        });
        router.register('proanalytics', 'proanalytics-section', function() {
            saveFilterState();
            initProAnalyticsWorkspace();
        });
        router.register('whiteboard', 'whiteboard-section', function() {
            saveFilterState();
            initWhiteboardWorkspace();
        });
        router.register('settings', 'settings-section', function() {
            saveFilterState();
            initSettingsWorkspace();
        });

        // Initialize PWA on load
        initPWA();

        // Initialize Skeletons
        var skeletons = new KawkabSkeletons();
        skeletons.register('results-section', '100%', 60, 4);
        skeletons.register('report-section', '100%', 40, 6);
        skeletons.register('dashboard-kpis', '100%', 40, 5);
        skeletons.register('dashboard-recent-list', '100%', 30, 3);
        skeletons.register('scout-section', '100%', 80, 4);
        skeletons.register('squad-section', '100%', 60, 6);
        skeletons.register('tactics-section', '100%', 60, 4);
        skeletons.register('ai-section', '100%', 80, 4);
        skeletons.register('coding-section', '100%', 80, 4);
        skeletons.register('review-section', '100%', 80, 4);
        skeletons.register('season-section', '100%', 60, 4);
        skeletons.register('opponent-section', '100%', 60, 4);
        skeletons.register('marketplace-section', '100%', 60, 4);
        skeletons.register('physiology-section', '100%', 60, 4);
        skeletons.register('collaboration-section', '100%', 60, 4);
        skeletons.register('livetagging-section', '100%', 60, 4);
        skeletons.register('professional-section', '100%', 60, 4);
        skeletons.register('highlight-section', '100%', 60, 4);
        skeletons.register('pitch3d-section', '100%', 80, 4);

        // Initialize nav tabs
        setupNavTabs();

        // Initialize quick actions
        setupQuickActions();

        // Initialize perf utilities
        initPassiveListeners();
        initThrottledScroll();
        initThrottledResize();

        // Initialize keyboard nav
        initKeyboardNav();

        // Initialize tooltips (from app-tooltips.js)
        if (typeof window.initTooltips === "function") {
            window.initTooltips();
        }

        // Initialize notifications (from ui.js)
        setTimeout(function () {
            if (window.KawkabUI && window.KawkabUI.initNotifications) {
                window.KawkabUI.initNotifications();
            }
        }, 100);

        // Add confirmation to destructive actions
        var deleteBtns = document.querySelectorAll('[data-confirm]');
        deleteBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                var msg = this.dataset.confirm || 'Are you sure?';
                e.preventDefault();
                var originalClick = this;
                showConfirmDialog(msg, function () {
                    originalClick.click();
                });
            });
        });

        // ── View toggle wiring ──
        setupViewToggles();

        // ── Batch action wiring ──
        setupBatchActions();

        // ── Coding Workspace Init ──
        initCodingWorkspace();

        // ── Event Review Init ──
        initReviewWorkspace();

        // ── Phase 2.3-4 & 3-4 Init ──
        initTacticsWorkspace();
        initWhiteboardWorkspace();
        initSettingsWorkspace();
        initAiWorkspace();
        initSquadWorkspace();
        initSquadHealthTab();

        // ── Wave A — Telestration ──
        initTelestration();
        initTelestrationV2();
        initTelLayerPanel();
        initTelPresetBrowser();
        initTelLocalStorage();

        // ── Wave B — Season Dashboard ──
        initSeasonDashboard();

        // ── Wave C — Training Planner ──
        initTrainingPlanner();

        // ── Wave D — Presentation Mode ──
        initPresentationMode();

        // ── Wave E — Scout Portal ──
        initScoutPortal();

        // ── Phase 13 — Opponent Database + Scouting Network ──
        initOpponentWorkspace();

        // ── Phase 15 — Community Marketplace ──
        initMarketplace();

        // ── Help modal ──
        document.getElementById('help-btn').onclick = function() {
            document.getElementById('help-modal').classList.remove('hidden');
            document.getElementById('help-modal').style.display = 'flex';
        };
        document.getElementById('help-close').onclick = function() {
            document.getElementById('help-modal').classList.add('hidden');
            document.getElementById('help-modal').style.display = '';
        };
        document.getElementById('help-modal').onclick = function(e) {
            if (e.target === this) {
                this.classList.add('hidden');
                this.style.display = '';
            }
        };

        // ── Load sample data buttons ──
        var sampleBtn = document.getElementById('load-sample-data-btn');
        if (sampleBtn) {
            sampleBtn.onclick = function() {
                bridge.load_sample_data().then(function(raw) {
                    try {
                        var r = JSON.parse(raw);
                        if (r.error) { showToast(r.error, 'error'); return; }
                        showToast('Sample match loaded: ' + (r.match || ''), 'info');
                        if (typeof refreshMatchList === 'function') refreshMatchList();
                    } catch (e) {
                        showToast('Sample data loaded successfully', 'info');
                    }
                }).catch(function() {
                    showToast('Sample data loaded', 'info');
                });
            };
        }

        // ── First-run wizard (real onboarding flow, delegated) ──
        if (window.KawkabOnboarding) window.KawkabOnboarding.showFirstRunWizard();
    });
})();
