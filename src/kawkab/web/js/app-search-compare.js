// Kawkab AI - Global Search + Player Comparison
// Split from app.js for modularity (the v0.8.6 sections).
// Follows the app-data-providers.js IIFE + window.__kawkab seam pattern:
// app.js calls window.__kawkab._initSearchCompareBridge(bridge) after
// QWebChannel connects, and window.__kawkab.setupSearchAndCompare() from
// its DOMContentLoaded init. Shared helpers resolve through the globals
// app.js/ui.js/utils.js already expose (window.t, window.formatNumber,
// window.showToast, window.__kawkab.escapeHtml); currentMatchId flows
// through app.js's window.__kawkab.currentMatchId accessor, and match
// navigation goes through window.loadMatch.
(function() {
    'use strict';

    var bridge = null;
    var escapeHtml = window.__kawkab.escapeHtml;
    var t = window.t;
    var formatNumber = window.formatNumber;
    var showToast = window.showToast;

    // ── Global Search state (moved from app.js) ───────────────────────
    var _searchCache = { matches: [], players: [], events: [] };
    var _searchSelectedIdx = -1;

    // ── Player Comparison state (moved from app.js) ───────────────────
    var pcPlayers = [];

    // ── Init hook (called by app.js after QWebChannel connects) ───────
    window.__kawkab._initSearchCompareBridge = function(b) {
        bridge = b;
    };

    // ── Init entry (called from app.js DOMContentLoaded init) ─────────
    window.__kawkab.setupSearchAndCompare = function() {
        setupGlobalSearch();
        setupPlayerComparison();
    };

    // ============================================================
    // Global Search (v0.8.6)
    // ============================================================
    function setupGlobalSearch() {
        var input = document.getElementById('global-search');
        var dropdown = document.getElementById('search-results-dropdown');
        if (!input || !dropdown) return;

        function cacheData() {
            if (!bridge) return;
            bridge.get_all_matches().then(function(json) {
                try { _searchCache.matches = JSON.parse(json) || []; } catch(e) {}
            }).catch(function() {});
            bridge.get_all_player_profiles().then(function(json) {
                try { var d = JSON.parse(json); _searchCache.players = d.profiles || []; } catch(e) {}
            }).catch(function() {});
            if (window.__kawkab.currentMatchId) {
                bridge.get_match_events(window.__kawkab.currentMatchId).then(function(json) {
                    try { _searchCache.events = JSON.parse(json) || []; } catch(e) {}
                }).catch(function() {});
            }
        }

        function filterAndRender(query) {
            if (!query || query.length < 2) {
                dropdown.classList.add('hidden');
                return;
            }
            var q = query.toLowerCase();
            var matchedMatches = _searchCache.matches.filter(function(m) {
                return (m.name || '').toLowerCase().indexOf(q) !== -1;
            }).slice(0, 5);
            var matchedPlayers = _searchCache.players.filter(function(p) {
                return (p.name || '').toLowerCase().indexOf(q) !== -1 || (p.position || '').toLowerCase().indexOf(q) !== -1;
            }).slice(0, 5);
            var matchedEvents = _searchCache.events.filter(function(e) {
                return (e.event_type || '').toLowerCase().indexOf(q) !== -1 || (e.player_name || '').toLowerCase().indexOf(q) !== -1;
            }).slice(0, 5);

            var totalMatches = _searchCache.matches.filter(function(m) { return (m.name || '').toLowerCase().indexOf(q) !== -1; }).length;
            var totalPlayers = _searchCache.players.filter(function(p) { return (p.name || '').toLowerCase().indexOf(q) !== -1 || (p.position || '').toLowerCase().indexOf(q) !== -1; }).length;
            var totalEvents = _searchCache.events.filter(function(e) { return (e.event_type || '').toLowerCase().indexOf(q) !== -1 || (e.player_name || '').toLowerCase().indexOf(q) !== -1; }).length;

            if (matchedMatches.length === 0 && matchedPlayers.length === 0 && matchedEvents.length === 0) {
                dropdown.innerHTML = '<div class="search-dropdown-empty">' + t('noSearchResults') + '</div>';
                dropdown.classList.remove('hidden');
                _searchSelectedIdx = -1;
                return;
            }

            var html = '';
            if (matchedMatches.length > 0) {
                html += '<div class="search-section"><div class="search-section-title">' + t('searchMatches') + '</div>';
                matchedMatches.forEach(function(m, i) {
                    html += '<div class="search-result-item" data-type="match" data-id="' + m.id + '" data-idx="' + i + '">⚽ ' + highlightMatch(escapeHtml(m.name || ''), query) + '</div>';
                });
                if (totalMatches > 5) html += '<div class="search-view-all" data-type="match">' + t('viewAllResults').replace('{n}', totalMatches) + '</div>';
                html += '</div>';
            }
            if (matchedPlayers.length > 0) {
                html += '<div class="search-section"><div class="search-section-title">' + t('searchPlayers') + '</div>';
                matchedPlayers.forEach(function(p, i) {
                    html += '<div class="search-result-item" data-type="player" data-id="' + p.id + '" data-idx="' + i + '">👤 ' + highlightMatch(escapeHtml(p.name || ''), query) + ' <span class="search-result-sub">' + escapeHtml(p.position || '') + '</span></div>';
                });
                if (totalPlayers > 5) html += '<div class="search-view-all" data-type="player">' + t('viewAllResults').replace('{n}', totalPlayers) + '</div>';
                html += '</div>';
            }
            if (matchedEvents.length > 0) {
                html += '<div class="search-section"><div class="search-section-title">' + t('searchEvents') + '</div>';
                matchedEvents.forEach(function(e, i) {
                    var label = (e.event_type || '').replace(/_/g, ' ') + (e.player_name ? ' - ' + e.player_name : '');
                    html += '<div class="search-result-item" data-type="event" data-idx="' + i + '">🔹 ' + highlightMatch(escapeHtml(label), query) + '</div>';
                });
                if (totalEvents > 5) html += '<div class="search-view-all" data-type="event">' + t('viewAllResults').replace('{n}', totalEvents) + '</div>';
                html += '</div>';
            }
            dropdown.innerHTML = html;
            dropdown.classList.remove('hidden');

            _searchSelectedIdx = -1;
            dropdown.querySelectorAll('.search-result-item, .search-view-all').forEach(function(el) {
                el.addEventListener('mousedown', function(e) {
                    e.preventDefault();
                    var type = this.dataset.type;
                    var id = this.dataset.id;
                    if (type === 'match' && id) {
                        dropdown.classList.add('hidden');
                        input.value = '';
                        navigateToMatch(parseInt(id));
                    } else if (type === 'player') {
                        dropdown.classList.add('hidden');
                        input.value = '';
                        window.location.hash = 'professional';
                    } else if (type === 'event') {
                        dropdown.classList.add('hidden');
                        input.value = '';
                    }
                });
            });
        }

        function highlightMatch(text, query) {
            var idx = text.toLowerCase().indexOf(query.toLowerCase());
            if (idx === -1) return text;
            return text.substring(0, idx) + '<strong>' + text.substring(idx, idx + query.length) + '</strong>' + text.substring(idx + query.length);
        }

        function navigateToMatch(matchId) {
            if (matchId) {
                window.location.hash = 'results';
                window.loadMatch(matchId);
            }
        }

        var debounceTimer = null;
        input.addEventListener('input', function() {
            clearTimeout(debounceTimer);
            var val = this.value.trim();
            if (val.length < 2) { dropdown.classList.add('hidden'); return; }
            debounceTimer = setTimeout(function() { filterAndRender(val); }, 300);
        });

        input.addEventListener('focus', function() {
            if (this.value.trim().length >= 2) { filterAndRender(this.value.trim()); }
        });

        input.addEventListener('keydown', function(e) {
            var items = dropdown.querySelectorAll('.search-result-item');
            if (e.key === 'Escape') {
                dropdown.classList.add('hidden');
                this.blur();
                return;
            }
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                _searchSelectedIdx = Math.min(_searchSelectedIdx + 1, items.length - 1);
                updateSelected(items);
                return;
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                _searchSelectedIdx = Math.max(_searchSelectedIdx - 1, -1);
                updateSelected(items);
                return;
            }
            if (e.key === 'Enter' && _searchSelectedIdx >= 0 && items[_searchSelectedIdx]) {
                e.preventDefault();
                items[_searchSelectedIdx].click();
                return;
            }
        });

        function updateSelected(items) {
            items.forEach(function(el, i) {
                el.classList.toggle('search-selected', i === _searchSelectedIdx);
            });
            if (_searchSelectedIdx >= 0 && items[_searchSelectedIdx]) {
                items[_searchSelectedIdx].scrollIntoView({ block: 'nearest' });
            }
        }

        document.addEventListener('click', function(e) {
            if (!input.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.classList.add('hidden');
            }
        });

        // Slash key to focus search
        document.addEventListener('keydown', function(e) {
            if (e.key === '/' && document.activeElement !== input && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                e.preventDefault();
                input.focus();
                input.select();
            }
        });

        // Cache data on bridge connect
        setTimeout(cacheData, 1000);
        // Refresh cache when a match loads (app.js fires this hook right
        // after currentMatchId updates; delay lets its async work settle --
        // same timing the old loadMatch monkey-patch had)
        window.__kawkab._onMatchLoaded = function() { setTimeout(cacheData, 500); };
    }

    // ============================================================
    // Player Comparison (v0.8.6)
    // ============================================================
    function setupPlayerComparison() {
        // Mode toggle
        document.querySelectorAll('[data-pc-mode]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                document.querySelectorAll('[data-pc-mode]').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                var mode = this.dataset.pcMode;
                document.getElementById('pc-match-mode').classList.toggle('hidden', mode !== 'match');
                document.getElementById('pc-player-mode').classList.toggle('hidden', mode !== 'player');
                if (mode === 'player') populatePlayerDropdowns();
            });
        });

        document.getElementById('pc-compare-btn')?.addEventListener('click', handlePlayerCompare);

        // Populate on compare tab activation
        var compareTab = document.querySelector('[data-tab="compare-tab"]');
        if (compareTab) {
            compareTab.addEventListener('click', function() {
                setTimeout(populatePlayerDropdowns, 300);
            });
        }
    }

    async function populatePlayerDropdowns() {
        if (!bridge) return;
        try {
            var data = JSON.parse(await bridge.get_all_player_profiles());
            pcPlayers = data.profiles || [];
            var opts = '<option value="">' + t('pcSelectPlayerA') + '</option>';
            pcPlayers.forEach(function(p) {
                opts += '<option value="' + p.id + '">' + escapeHtml(p.name || '') + ' (#' + (p.jersey || '?') + ')</option>';
            });
            var selA = document.getElementById('pc-player-a');
            var selB = document.getElementById('pc-player-b');
            if (selA) { var v = selA.value; selA.innerHTML = opts; if (v) selA.value = v; }
            if (selB) { var v2 = selB.value; selB.innerHTML = opts.replace('pcSelectPlayerA', 'pcSelectPlayerB'); if (v2) selB.value = v2; }
        } catch (e) {
            console.error('Failed to load players:', e);
        }
    }

    async function handlePlayerCompare() {
        var selA = document.getElementById('pc-player-a');
        var selB = document.getElementById('pc-player-b');
        var playerAId = parseInt(selA.value);
        var playerBId = parseInt(selB.value);
        if (!playerAId || !playerBId) {
            showToast('Select two players to compare', 'warning');
            return;
        }
        if (playerAId === playerBId) {
            showToast('Select two different players', 'warning');
            return;
        }

        try {
            var resultA, resultB;
            if (bridge.compare_players) {
                var json = await bridge.compare_players(playerAId, playerBId);
                var data = JSON.parse(json);
                resultA = data.player_a;
                resultB = data.player_b;
            } else {
                var rawA = await bridge.get_player_stats(playerAId);
                var rawB = await bridge.get_player_stats(playerBId);
                resultA = JSON.parse(rawA);
                resultB = JSON.parse(rawB);
            }

            renderPlayerComparison(resultA, resultB);
        } catch (e) {
            console.error('Player compare failed:', e);
            showToast('Failed to compare players', 'error');
        }
    }

    function renderPlayerComparison(dataA, dataB) {
        var container = document.getElementById('pc-results');
        container.classList.remove('hidden');

        var nameA = dataA.name || t('pcPlayerA');
        var nameB = dataB.name || t('pcPlayerB');
        document.getElementById('pc-radar-a-title').textContent = nameA;
        document.getElementById('pc-radar-b-title').textContent = nameB;

        // Radar charts
        var statKeys = ['passes', 'shots', 'tackles', 'sprints', 'distance', 'xg'];
        var statLabels = [
            t('metric_passes') || 'Passes',
            t('metric_shots') || 'Shots',
            t('alert_tackle') || 'Tackles',
            t('metric_sprints') || 'Sprints',
            t('metric_distance') || 'Distance',
            t('metric_xg') || 'xG',
        ];

        var maxVals = statKeys.map(function(k) {
            return Math.max(dataA[k] || 0, dataB[k] || 0, 0.01);
        });

        function normalize(val, max) { return (val || 0) / max; }

        var chartDataA = { labels: statLabels, values: statKeys.map(function(k, i) { return normalize(dataA[k], maxVals[i]); }), playerName: nameA };
        var chartDataB = { labels: statLabels, values: statKeys.map(function(k, i) { return normalize(dataB[k], maxVals[i]); }), playerName: nameB };

        if (window.KawkabCharts && window.KawkabCharts.renderDualRadar) {
            window.KawkabCharts.renderDualRadar('pc-radar-a-canvas', chartDataA, 'pc-radar-b-canvas', chartDataB, 1);
        } else if (window.KawkabCharts && window.KawkabCharts.renderRadar) {
            window.KawkabCharts.renderRadar(chartDataA);
            // second radar fallback: just use renderRadar on a temp canvas
        }

        // Stat comparison table
        renderPCStatTable(dataA, dataB, statKeys, statLabels, nameA, nameB);

        // Insights
        renderPCInsights(dataA, dataB, statKeys, statLabels, nameA, nameB);
    }

    function renderPCStatTable(dataA, dataB, keys, labels, nameA, nameB) {
        var wrapper = document.getElementById('pc-stat-table-wrapper');
        var html = '<table class="pc-compare-table"><thead><tr>' +
            '<th>' + t('pcStat') + '</th>' +
            '<th>' + escapeHtml(nameA) + '</th>' +
            '<th>' + escapeHtml(nameB) + '</th>' +
            '<th>' + t('pcDelta') + '</th>' +
            '<th>' + t('pcAdvantage') + '</th>' +
            '</tr></thead><tbody>';
        keys.forEach(function(k, i) {
            var vA = dataA[k] || 0;
            var vB = dataB[k] || 0;
            var delta = vA - vB;
            var advantage = delta > 0 ? nameA : (delta < 0 ? nameB : '-');
            var deltaStr = (delta > 0 ? '+' : '') + delta.toFixed(2);
            var deltaClass = delta > 0 ? 'pc-delta-pos' : (delta < 0 ? 'pc-delta-neg' : '');
            html += '<tr>' +
                '<td>' + escapeHtml(labels[i]) + '</td>' +
                '<td>' + formatNumber(vA, 2) + '</td>' +
                '<td>' + formatNumber(vB, 2) + '</td>' +
                '<td class="' + deltaClass + '">' + deltaStr + '</td>' +
                '<td>' + escapeHtml(advantage) + '</td>' +
                '</tr>';
        });
        html += '</tbody></table>';
        wrapper.innerHTML = html;
    }

    function renderPCInsights(dataA, dataB, keys, labels, nameA, nameB) {
        var el = document.getElementById('pc-insights');
        var aBetter = [];
        var bBetter = [];
        keys.forEach(function(k, i) {
            var vA = dataA[k] || 0;
            var vB = dataB[k] || 0;
            if (vA > vB) aBetter.push({ label: labels[i], diff: vA - vB });
            else if (vB > vA) bBetter.push({ label: labels[i], diff: vB - vA });
        });
        aBetter.sort(function(a, b) { return b.diff - a.diff; });
        bBetter.sort(function(a, b) { return b.diff - a.diff; });

        var html = '';
        if (aBetter.length > 0) {
            html += '<p><strong>' + escapeHtml(nameA) + ' ' + t('pcBetterAt') + ':</strong> ' +
                aBetter.slice(0, 3).map(function(s) { return escapeHtml(s.label); }).join(', ') + '</p>';
        }
        if (bBetter.length > 0) {
            html += '<p><strong>' + escapeHtml(nameB) + ' ' + t('pcBetterAt') + ':</strong> ' +
                bBetter.slice(0, 3).map(function(s) { return escapeHtml(s.label); }).join(', ') + '</p>';
        }
        if (aBetter.length === 0 && bBetter.length === 0) {
            html = '<p>' + t('pcNoInsights') + '</p>';
        }
        el.innerHTML = html;
    }
})();
