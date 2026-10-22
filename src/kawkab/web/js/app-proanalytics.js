// Kawkab AI - Pro Analytics (elite module aggregation view)
// Surfaces the orphaned core/ modules: OBV, EPV, pass flow, pressing
// clusters, duels, ball recovery, box entries, switches, crossing, set
// pieces, through balls, off-ball. Every card shows honest
// data_available state — never a fabricated zero.

(function() { 'use strict';

    var escapeHtml = window.__kawkab.escapeHtml;

    var BLOCK_META = {
        epv:             { title: 'EPV — Expected Possession Value', icon: '📊', desc: 'Possession value by control phase' },
        obv:             { title: 'OBV — Off-Ball Value', icon: '🏃', desc: 'Off-ball contribution (needs tracking frames)' },
        off_ball:        { title: 'Off-Ball Metrics', icon: '🧿', desc: 'Space creation & movement (needs tracking frames)' },
        pass_flow:       { title: 'Pass Flow', icon: '➡️', desc: 'Zone-to-zone pass corridors' },
        pressing_clusters: { title: 'Pressing Clusters', icon: '🧱', desc: 'Spatial zones of defensive actions' },
        duels:           { title: 'Duel Analysis', icon: '⚔️', desc: 'Duel outcomes by type & team' },
        ball_recovery:   { title: 'Ball Recovery', icon: '🔁', desc: 'Where possession is won back' },
        box_entries:     { title: 'Box Entries', icon: '📦', desc: 'Penalty-area entries & touches' },
        switch_of_play:  { title: 'Switches of Play', icon: '🔄', desc: 'Lateral play switches' },
        crossing:        { title: 'Crossing Analysis', icon: '✈️', desc: 'Cross quality by zone' },
        set_pieces:      { title: 'Set Pieces', icon: '🚩', desc: 'Set-piece situations & delivery quality' },
        through_balls:   { title: 'Through Balls', icon: '🎯', desc: 'Line-breaking passes' },
        carry_xt:        { title: 'Carry xT', icon: '🏃‍♂️', desc: 'Threat created by ball carrying' },
        xg_chain:        { title: 'xG Chain', icon: '🔗', desc: 'xG credited to every buildup action' },
        game_state:      { title: 'Game State', icon: '⏱️', desc: 'Behavior by scoreline state' },
        flank_analysis:  { title: 'Flank Analysis', icon: '↔️', desc: 'Build-up & attack by side' },
        defensive_xt:    { title: 'Defensive xT', icon: '🛡️', desc: 'Threat prevented by defensive actions' },
        corner_xg:       { title: 'Corner xG', icon: '🚩', desc: 'Corner delivery danger rating' },
        crossing_xg:     { title: 'Crossing xG', icon: '🎯', desc: 'Cross-specific expected goals' },
        expected_pass:   { title: 'Expected Pass (xP)', icon: '📮', desc: 'Pass completion probability model' },
        passing_triangles: { title: 'Passing Triangles', icon: '🔺', desc: 'Rotations among player trios' },
        scoreline:       { title: 'Scoreline Distribution', icon: '🎲', desc: 'Simulated final-score probabilities' },
        velocity:        { title: 'Velocity & Sprints', icon: '⚡', desc: 'Per-player speed profiles (needs calibrated tracking)' },
        influence_map:   { title: 'Influence Map', icon: '🗺️', desc: 'Team spatial dominance (needs calibrated tracking)' },
        lineup_optimizer: { title: 'Lineup Optimizer', icon: '📋', desc: 'Best-X suggestions (needs season history)' },
    };

    var ORDER = ['epv', 'obv', 'off_ball', 'pass_flow', 'pressing_clusters',
                 'duels', 'ball_recovery', 'box_entries', 'switch_of_play',
                 'crossing', 'set_pieces', 'through_balls', 'carry_xt',
                 'xg_chain', 'game_state', 'flank_analysis', 'defensive_xt',
                 'corner_xg', 'crossing_xg', 'expected_pass',
                 'passing_triangles', 'scoreline', 'velocity',
                 'influence_map', 'lineup_optimizer'];

    var _state = { matchId: null, report: null };

    window.initProAnalyticsWorkspace = function() {
        var matchSelect = document.getElementById('proanalytics-match-select');
        var loadBtn = document.getElementById('proanalytics-load-btn');
        if (!matchSelect || !loadBtn) return;

        if (!matchSelect.options || matchSelect.options.length <= 1) {
            loadProAnalyticsMatches(matchSelect);
        }

        loadBtn.addEventListener('click', function() {
            var mid = parseInt(matchSelect.value, 10);
            if (!mid) { showToast('Select a match first.', 'warning'); return; }
            loadProAnalyticsReport(mid);
        });

        var seasonBtn = document.getElementById('proanalytics-season-btn');
        if (seasonBtn) {
            seasonBtn.addEventListener('click', function() {
                loadSeasonProReport(seasonBtn);
            });
        }

        var gamePlanBtn = document.getElementById('proanalytics-gameplan-btn');
        if (gamePlanBtn) {
            gamePlanBtn.addEventListener('click', function() {
                var matchSelect = document.getElementById('proanalytics-match-select');
                var mid = parseInt(matchSelect && matchSelect.value, 10);
                if (!mid) { showToast('Select a match first.', 'warning'); return; }
                loadGamePlan(mid);
            });
        }
    };

    // ── Game Plan (consumes generate_game_plan, dormant since Sprint 12+) ──

    function loadGamePlan(matchId) {
        var status = document.getElementById('proanalytics-status');
        var content = document.getElementById('proanalytics-content');
        if (!content) return;
        if (typeof bridge === 'undefined' || !bridge ||
            typeof bridge.generate_game_plan !== 'function') {
            showToast('Bridge not available', 'error');
            return;
        }
        if (status) status.textContent = 'Building game plan...';
        content.innerHTML = '<p class="hint">Generating opposition game plan from stored match events...</p>';

        bridge.generate_game_plan(matchId, 0, function(result) {
            if (status) status.textContent = '';
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) {
                    content.innerHTML = '<p class="error-message">' + escapeHtml(data.error) + '</p>';
                    return;
                }
                var r = (data && data.result) || {};
                var players = r.key_players_to_neutralize || [];
                content.innerHTML =
                    '<div class="pro-card" style="border:1px solid var(--border);border-radius:var(--radius);padding:12px">' +
                    '<h3>📋 Game Plan' + (r.opponent ? ' — vs ' + escapeHtml(String(r.opponent)) : '') + '</h3>' +
                    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:8px">' +
                    planCell('📐 Formation', r.formation_recommendation) +
                    planCell('🛡 Set Pieces', r.set_piece_plan) +
                    planCell('🔮 Scoreline Prediction', r.scoreline_prediction) +
                    planCell('🎯 Key Players to Neutralize', players.length ? '<ul style="margin:4px 0 0 16px">' +
                        players.map(function(p) { return '<li>' + escapeHtml(String(p)) + '</li>'; }).join('') + '</ul>' : '') +
                    '</div>' +
                    '<p class="hint" style="margin-top:10px">Generated from ' +
                    escapeHtml(String(r.opponent || 'the selected match')) + ' event profile. Review with your staff before matchday.</p>' +
                    '</div>';
            } catch (e) {
                content.innerHTML = '<p class="error-message">Failed to parse game plan.</p>';
                console.warn('game plan parse failed:', e);
            }
        });
    }

    function planCell(title, body) {
        return '<div style="border:1px solid var(--border);border-radius:var(--radius);padding:8px">' +
            '<div style="font-weight:600;margin-bottom:4px">' + title + '</div>' +
            '<div style="font-size:0.85rem">' + (body ? body : '<span class="hint">--</span>') + '</div></div>';
    }

    function loadSeasonProReport(btn) {
        var content = document.getElementById('proanalytics-content');
        if (!content) return;
        if (typeof bridge === 'undefined' || !bridge ||
            typeof bridge.get_season_pro_report !== 'function') {
            showToast('Bridge not available', 'error');
            return;
        }
        btn.disabled = true;
        content.innerHTML = '<p class="hint">Aggregating season analytics across all stored matches...</p>';
        bridge.get_season_pro_report(function(result) {
            btn.disabled = false;
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) {
                    content.innerHTML = '<p class="error-message">' + escapeHtml(data.error) + '</p>';
                    return;
                }
                if (!data.success || !data.data_available) {
                    content.innerHTML = '<p class="hint">' + escapeHtml(data.reason || 'Season analytics needs at least 2 stored matches.') + '</p>';
                    return;
                }
                renderSeasonReport(data);
            } catch (e) {
                content.innerHTML = '<p class="error-message">Failed to parse season report.</p>';
                console.warn('season report parse failed:', e);
            }
        });
    }

    function renderSeasonReport(data) {
        var content = document.getElementById('proanalytics-content');
        var html = '<div class="proanalytics-summary" style="margin-bottom:12px">' +
            '<span class="pro-badge">' + Number(data.n_matches || 0) + ' matches analyzed</span></div>';
        var meta = {
            formation_trends: { title: 'Formation Trends', icon: '📐' },
            discipline: { title: 'Discipline & Suspension Risk', icon: '🟨' },
            fixture_difficulty: { title: 'Fixture Difficulty', icon: '📅' },
        };
        Object.keys(meta).forEach(function(key) {
            var block = (data.blocks || {})[key];
            if (!block) return;
            var ok = block.data_available;
            html += '<div class="pro-card" style="border:1px solid var(--border);border-radius:var(--radius);padding:10px;margin-bottom:10px">' +
                '<strong>' + meta[key].icon + ' ' + meta[key].title + '</strong> ' +
                '<span class="pro-badge ' + (ok ? 'pro-badge-ok' : 'pro-badge-empty') + '" style="float:right">' +
                (ok ? 'available' : 'no data') + '</span>' +
                '<div style="margin-top:8px;font-size:0.82rem">' +
                (ok ? '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>'
                    : '<span class="hint">' + escapeHtml(block.reason || '') + '</span>') +
                '</div></div>';
        });
        content.innerHTML = html;
    }

    function loadProAnalyticsMatches(select) {
        if (typeof bridge === 'undefined' || !bridge) return;
        bridge.get_all_matches(function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                var matches = data.matches || data || [];
                if (!Array.isArray(matches)) return;
                select.innerHTML = '<option value="">-- Select Match --</option>';
                matches.forEach(function(m) {
                    var label = (m.name || (m.home_team + ' vs ' + m.away_team)) +
                                (m.match_date ? ' (' + m.match_date + ')' : '');
                    select.innerHTML += '<option value="' + Number(m.id) + '">' + escapeHtml(label) + '</option>';
                });
            } catch (e) { console.warn('pro-analytics match list failed:', e); }
        });
    }

    function loadProAnalyticsReport(matchId) {
        var status = document.getElementById('proanalytics-status');
        var content = document.getElementById('proanalytics-content');
        if (!content) return;
        _state.matchId = matchId;

        if (typeof bridge === 'undefined' || !bridge ||
            typeof bridge.get_pro_analytics_report !== 'function') {
            if (status) status.textContent = 'Bridge not available';
            return;
        }
        if (status) status.textContent = 'Computing...';
        content.innerHTML = '<p class="hint">Aggregating 12 analytical models — this runs the full elite suite.</p>';

        bridge.get_pro_analytics_report(matchId, function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (status) status.textContent = '';
                if (data.error) {
                    content.innerHTML = '<p class="error-message">' + escapeHtml(data.error) + '</p>';
                    return;
                }
                if (!data.success || !data.data_available) {
                    content.innerHTML = '<p class="hint">' +
                        escapeHtml(data.reason || 'No event data stored for this match — analyze a video or import event data first.') +
                        '</p>';
                    return;
                }
                _state.report = data;
                renderProAnalytics(data);
            } catch (e) {
                if (status) status.textContent = '';
                content.innerHTML = '<p class="error-message">Failed to parse pro-analytics report.</p>';
                console.warn('pro-analytics parse failed:', e);
            }
        });
    }

    function renderProAnalytics(data) {
        var content = document.getElementById('proanalytics-content');
        if (!content) return;

        var html = '<div class="proanalytics-summary" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px">' +
            '<span class="pro-badge">' + Number(data.n_events || 0).toLocaleString() + ' events</span>' +
            '<span class="pro-badge">' + (data.tracking_frames_available ? 'tracking frames ✓' : 'no tracking frames') + '</span>' +
            '</div>';

        html += '<div class="proanalytics-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px">';
        ORDER.forEach(function(key) {
            html += renderBlock(key, (data.blocks || {})[key]);
        });
        html += '</div>';
        content.innerHTML = html;
    }

    function renderBlock(key, block) {
        var meta = BLOCK_META[key] || { title: key, icon: '📈', desc: '' };
        var available = block && block.data_available;

        var header = '<div class="pro-card-header" style="display:flex;justify-content:space-between;align-items:center">' +
            '<strong>' + meta.icon + ' ' + escapeHtml(meta.title) + '</strong>' +
            '<span class="pro-badge ' + (available ? 'pro-badge-ok' : 'pro-badge-empty') + '">' +
            (available ? 'available' : 'no data') + '</span>' +
            '</div>';

        var body;
        if (!available) {
            body = '<p class="hint" style="margin-top:8px;font-size:0.82rem">' +
                escapeHtml((block && block.reason) || meta.desc) + '</p>';
        } else {
            body = '<div class="pro-card-body" style="margin-top:8px;font-size:0.82rem">' +
                renderBlockBody(key, block) + '</div>';
        }

        return '<div class="pro-card collapsible collapsed" style="border:1px solid var(--border);border-radius:var(--radius);padding:10px">' +
            '<div onclick="this.parentElement.classList.toggle(\'collapsed\')" style="cursor:pointer">' +
            header + '</div>' + body + '</div>';
    }

    function fmt(v, digits) {
        if (v === null || v === undefined || isNaN(v)) return '—';
        return Number(v).toFixed(digits === undefined ? 2 : digits);
    }

    function renderBlockBody(key, block) {
        switch (key) {
            case 'epv': {
                return '<div>Home EPV: <strong>' + fmt(block.home_total) + '</strong> · Away: <strong>' + fmt(block.away_total) + '</strong></div>' +
                    '<div style="color:var(--text-muted)">Possessions: ' + Number(block.total_possessions || 0) +
                    ' · per-possession H/A: ' + fmt(block.home_per_possession, 3) + ' / ' + fmt(block.away_per_possession, 3) + '</div>';
            }
            case 'pass_flow': {
                return '<div>' + Number(block.n_links || 0) + ' zone links — top corridors:</div>' +
                    '<ol style="margin:6px 0 0 16px">' +
                    (block.links || []).slice(0, 6).map(function(l) {
                        return '<li>' + escapeHtml(String(l.from_zone || l.origin || '?')) + ' → ' +
                            escapeHtml(String(l.to_zone || l.destination || '?')) +
                            ' (' + Number(l.count || l.n || 0) + ')</li>';
                    }).join('') + '</ol>';
            }
            case 'pressing_clusters': {
                return '<div>' + Number(block.n_clusters || 0) + ' pressing clusters (top 8 shown)</div>' +
                    '<ul style="margin:6px 0 0 16px">' +
                    (block.clusters || []).slice(0, 8).map(function(c) {
                        return '<li>Zone ' + escapeHtml(String(c.zone || c.center_zone || '?')) +
                            ' — ' + Number(c.n_events || c.count || 0) + ' actions</li>';
                    }).join('') + '</ul>';
            }
            case 'duels': {
                var d = JSON.stringify(block).replace(/[{}"]/g, '').slice(0, 400);
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>';
            }
            case 'ball_recovery': {
                var h = (block.home && (block.home.total_recoveries || 0)) || 0;
                var a = (block.away && (block.away.total_recoveries || 0)) || 0;
                var hs = (block.home && block.home.recoveries_leading_to_shot) || 0;
                var as_ = (block.away && block.away.recoveries_leading_to_shot) || 0;
                return '<div>Recoveries — Home: <strong>' + h + '</strong> · Away: <strong>' + a + '</strong></div>' +
                    '<div style="color:var(--text-muted)">Leading to shots: H ' + hs + ' / A ' + as_ + '</div>';
            }
            case 'box_entries': {
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>';
            }
            case 'switch_of_play': {
                var sc = block.switch_count || {};
                return '<div>Switches — Home: <strong>' + Number(sc.home || 0) + '</strong> · Away: <strong>' + Number(sc.away || 0) + '</strong></div>';
            }
            case 'crossing': {
                return '<div>Total crosses: <strong>' + Number(block.total_crosses || 0) + '</strong></div>' +
                    '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block, ['raw'])) + '</pre>';
            }
            case 'set_pieces': {
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>';
            }
            case 'through_balls': {
                return '<div><strong>' + Number(block.n_through_balls || 0) + '</strong> through balls detected</div>' +
                    '<ul style="margin:6px 0 0 16px">' +
                    (block.through_balls || []).slice(0, 6).map(function(tb) {
                        var label = escapeHtml(String(tb.player || tb.passer || 'Player'));
                        if (tb.minute !== undefined && tb.minute !== null) {
                            label += " (" + escapeHtml(String(tb.minute)) + "')";
                        }
                        return '<li>' + label + '</li>';
                    }).join('') + '</ul>';
            }
            case 'obv':
            case 'off_ball': {
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>';
            }
            case 'carry_xt':
            case 'flank_analysis':
            case 'game_state':
            case 'xg_chain': {
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block, ['top_chains', 'chains'])) + '</pre>';
            }
            case 'defensive_xt': {
                return '<div><strong>' + Number(block.n_defensive_actions || 0) + '</strong> defensive actions valued by threat prevented</div>';
            }
            case 'corner_xg': {
                return '<div><strong>' + Number(block.n_corners || 0) + '</strong> corners · avg danger <strong>' + fmt(block.avg_danger_rating, 3) + '</strong></div>';
            }
            case 'crossing_xg': {
                return '<div><strong>' + Number(block.n_crosses || 0) + '</strong> crosses · total cross-xG <strong>' + fmt(block.total_cross_xg, 3) + '</strong></div>';
            }
            case 'expected_pass': {
                return '<div><strong>' + Number(block.n_passes_evaluated || 0) + '</strong> passes · avg xP (completed) <strong>' + fmt(block.avg_completed_ep, 3) + '</strong></div>';
            }
            case 'passing_triangles': {
                return '<div><strong>' + Number(block.n_triangles || 0).toLocaleString() + '</strong> passing triangles detected</div>';
            }
            case 'scoreline': {
                var op = block.outcome_probs || {};
                return '<div>Home win: <strong>' + fmt(op.home_win || op.win_home, 3) + '</strong> · Draw: <strong>' + fmt(op.draw, 3) + '</strong> · Away win: <strong>' + fmt(op.away_win || op.win_away, 3) + '</strong></div>' +
                    '<div style="color:var(--text-muted)">Entropy ' + fmt(block.scoreline_entropy, 2) + ' bits</div>';
            }
            default:
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>';
        }
    }

    function summaryKv(obj, skip) {
        skip = skip || [];
        var parts = [];
        Object.keys(obj || {}).forEach(function(k) {
            if (k === 'data_available' || skip.indexOf(k) >= 0) return;
            var v = obj[k];
            if (typeof v === 'number') parts.push(k + ': ' + (Math.round(v * 1000) / 1000));
            else if (typeof v === 'string' || typeof v === 'boolean') parts.push(k + ': ' + v);
            else if (Array.isArray(v) && v.length) parts.push(k + ': [' + v.length + ' items]');
            else if (v && typeof v === 'object') {
                Object.keys(v).slice(0, 6).forEach(function(k2) {
                    var v2 = v[k2];
                    if (typeof v2 === 'number') parts.push(k + '.' + k2 + ': ' + (Math.round(v2 * 1000) / 1000));
                    else if (typeof v2 === 'string' || typeof v2 === 'boolean') parts.push(k + '.' + k2 + ': ' + v2);
                });
            }
        });
        return parts.slice(0, 14).join('\n');
    }
})();
