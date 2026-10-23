// Kawkab AI - Pro Analytics (elite module aggregation view)
// Surfaces the orphaned core/ modules: OBV, EPV, pass flow, pressing
// clusters, duels, ball recovery, box entries, switches, crossing, set
// pieces, through balls, off-ball. Every card shows honest
// data_available state — never a fabricated zero.

(function() { 'use strict';

    var escapeHtml = window.__kawkab.escapeHtml;

    // i18n lookup that never replaces an authored English label with a
    // bare key: t() returns the key itself when a translation is missing.
    function label(key, fallback) {
        var t = window.t;
        if (typeof t === 'function') {
            var value = t(key);
            if (value && value !== key) return value;
        }
        return fallback;
    }

    var BLOCK_META = {
        elite_benchmark: { title: 'Elite Benchmark', titleKey: 'ebTitle', icon: '📊', desc: 'This match against a defined elite reference population' },
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

    var ORDER = ['elite_benchmark', 'epv', 'obv', 'off_ball', 'pass_flow', 'pressing_clusters',
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
            benchmark_trend: { title: 'Season Benchmark', titleKey: 'sbTitle', icon: '📊' },
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
                (!ok ? '<span class="hint">' + escapeHtml(block.reason || '') + '</span>'
                    : key === 'benchmark_trend' ? renderSeasonBenchmark(block)
                    : '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>') +
                '</div></div>';
        });
        content.innerHTML = html;
    }

    // The season block is the same comparison over N matches, ranked
    // against N-match blocks of the cohort rather than single matches.
    function renderSeasonBenchmark(block) {
        var pop = block.population || {};
        var metrics = block.metrics || {};
        var size = Number((block.block || {}).size || block.n_matches || 0);
        var html = '<div>' + escapeHtml(label('ebReference', 'Reference population')) + ': ' +
            '<strong>' + escapeHtml(pop.label || '') + '</strong> — ' +
            escapeHtml(String(pop.n_team_matches || 0)) + ' ' +
            escapeHtml(label('ebTeamMatches', 'elite team-matches')) + ', ' +
            escapeHtml(String(pop.n_matches || 0)) + ' ' +
            escapeHtml(label('ebMatches', 'matches')) +
            '</div>';
        html += '<div style="margin-top:6px"><strong>' + escapeHtml(String(block.team || '—')) +
            '</strong> — ' + Number(block.n_matches || 0) + ' ' + escapeHtml(label('sbMatches', 'matches')) +
            ' · ' + fmt(block.minutes, 0) + ' ' + escapeHtml(label('sbMinutes', 'minutes')) + '</div>';
        html += '<div class="hint" style="font-size:0.75rem">' +
            escapeHtml(label(
                'sbMatchedNote',
                'percentiles compare your {n}-match block against {n}-match blocks drawn from the cohort'
            ).replace(/\{n\}/g, String(size))) + '</div>';

        METRIC_GROUPS.forEach(function(group) {
            var rows = '';
            METRIC_ORDER.forEach(function(spec) {
                if (spec.group !== group.key) return;
                var metric = metrics[spec.key];
                if (!metric) return;
                var definition = ((pop.definitions || {})[spec.key] || {}).definition || '';
                rows += renderMetricRow({
                    value: metric.value,
                    percentile: metric.percentile,
                    band: metric.band,
                    population: metric.population,
                    comparable: metric.comparable,
                    reason: metric.reason,
                    label: label(spec.labelKey, spec.fallback),
                    digits: spec.digits,
                    definition: definition,
                    interval: metric.interval
                });
            });
            if (rows) {
                html += '<div style="margin-top:8px;font-weight:600;font-size:0.78rem;color:var(--text-muted)">' +
                    escapeHtml(label(group.labelKey, group.fallback)) + '</div>' + rows;
            }
        });

        var trend = block.trend || {};
        html += '<div style="margin-top:8px;font-size:0.8rem">';
        if (trend.available) {
            var delta = Number(trend.delta || 0);
            html += '<strong>' + escapeHtml(label('sbTrend', 'Trend across the block')) + ':</strong> ' +
                escapeHtml(label('ebXg90', 'xG / 90')) + ' ' + fmt(trend.before, 2) + ' → ' +
                fmt(trend.after, 2) + ' (' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + ') ' +
                '<span class="hint">' + escapeHtml(trend.note || '') + '</span>';
        } else {
            html += '<span class="hint">' + escapeHtml(trend.reason || '') + '</span>';
        }
        html += '</div>';
        html += renderLimitations(block.limitations || []);
        return html;
    }

    function renderLimitations(limits) {
        if (!limits.length) return '';
        var html = '<div style="margin-top:10px;color:var(--text-muted);font-size:0.75rem">' +
            '<strong>' + escapeHtml(label('ebLimits', 'What this comparison cannot tell you')) + '</strong>' +
            '<ul style="margin:4px 0 0 16px">';
        limits.forEach(function(limit) {
            var key = (limit && typeof limit === 'object') ? limit.key : null;
            var text = (limit && typeof limit === 'object') ? limit.text : limit;
            html += '<li>' + escapeHtml(key ? label('ebLim_' + key, String(text || '')) : String(limit)) + '</li>';
        });
        return html + '</ul></div>';
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
        var title = meta.titleKey ? label(meta.titleKey, meta.title) : meta.title;

        var header = '<div class="pro-card-header" style="display:flex;justify-content:space-between;align-items:center">' +
            '<strong>' + meta.icon + ' ' + escapeHtml(title) + '</strong>' +
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
            case 'elite_benchmark':
                return renderEliteBenchmark(block);
            default:
                return '<pre style="white-space:pre-wrap;margin:0">' + escapeHtml(summaryKv(block)) + '</pre>';
        }
    }

    // ── Elite benchmark rendering ─────────────────────────────────────
    //
    // The block answers "compared with whom?" — so the population is
    // rendered with the numbers, and every metric that could not be
    // measured says why instead of showing a zero.

    // Output is what a team produced; style is how it played while
    // producing it. The definitions come from the block's own
    // `population.definitions`, so the card can never describe a metric
    // differently from how it was measured.
    var METRIC_GROUPS = [
        { key: 'output', labelKey: 'ebGroupOutput', fallback: 'Output' },
        { key: 'style', labelKey: 'ebGroupStyle', fallback: 'Style' }
    ];

    var METRIC_ORDER = [
        { key: 'shots', group: 'output', labelKey: 'ebShots', fallback: 'Shots', digits: 0 },
        { key: 'shots_per_90', group: 'output', labelKey: 'ebShots90', fallback: 'Shots / 90', digits: 1 },
        { key: 'xg', group: 'output', labelKey: 'ebXg', fallback: 'xG', digits: 2 },
        { key: 'xg_per_90', group: 'output', labelKey: 'ebXg90', fallback: 'xG / 90', digits: 2 },
        { key: 'goals', group: 'output', labelKey: 'ebGoals', fallback: 'Goals', digits: 0 },
        { key: 'ppda', group: 'style', labelKey: 'ebPpda', fallback: 'PPDA', digits: 2 },
        { key: 'field_tilt', group: 'style', labelKey: 'ebFieldTilt', fallback: 'Field tilt', digits: 3 },
        { key: 'progressive_passes_per_90', group: 'style', labelKey: 'ebProgressive', fallback: 'Progressive passes / 90', digits: 1 },
        { key: 'final_third_entries_per_90', group: 'style', labelKey: 'ebEntries', fallback: 'Final-third entries / 90', digits: 1 },
        { key: 'high_turnovers_per_90', group: 'style', labelKey: 'ebHighTurnovers', fallback: 'High turnovers / 90', digits: 1 },
        { key: 'pass_completion_pct', group: 'style', labelKey: 'ebCompletion', fallback: 'Pass completion', digits: 1 }
    ];

    var BAND_LABELS = {
        top_decile: { key: 'ebBand_top_decile', fallback: 'top 10% of the cohort' },
        top_quartile: { key: 'ebBand_top_quartile', fallback: 'top 25% of the cohort' },
        above_median: { key: 'ebBand_above_median', fallback: 'above the cohort median' },
        lower_half: { key: 'ebBand_lower_half', fallback: 'lower half of the cohort' },
        bottom_quartile: { key: 'ebBand_bottom_quartile', fallback: 'bottom 25% of the cohort' },
        bottom_decile: { key: 'ebBand_bottom_decile', fallback: 'bottom 10% of the cohort' }
    };

    // "better than 74% of the cohort" — the percentile said in words. That is
    // what the number means, and unlike an ordinal suffix it survives
    // translation.
    function beatsText(percentile) {
        var pattern = label('ebBeatsShare', 'better than {n}% of the cohort');
        return pattern.replace('{n}', String(Math.round(percentile)));
    }

    function percentileBar(percentile) {
        var filled = Math.max(0, Math.min(10, Math.round((percentile / 100) * 10)));
        var out = '';
        for (var i = 0; i < 10; i++) {
            out += i < filled ? '▮' : '▯';
        }
        return out;
    }

    function renderMetricRow(metric) {
        var name = escapeHtml(metric.label);
        if (metric.definition) {
            name = '<span title="' + escapeHtml(metric.definition) + '">' + name + '</span>';
        }
        if (!metric.comparable || metric.percentile === null || metric.percentile === undefined) {
            var reason = escapeHtml(metric.reason || label('ebNotAvailable', 'not available'));
            return '<div style="display:flex;justify-content:space-between;gap:10px;padding:3px 0">' +
                '<span>' + name + '</span>' +
                '<span style="color:var(--text-muted)">— ' + reason + '</span>' +
                '</div>';
        }
        var pop = metric.population || {};
        var median = (pop.median === undefined || pop.median === null) ? null : pop.median;
        var medianText = median === null ? '' :
            ' · ' + label('ebEliteMedian', 'cohort median') + ' ' + fmt(median, metric.digits);
        var band = BAND_LABELS[metric.band];
        var bandText = band ? label(band.key, band.fallback) : '';
        var intervalText = '';
        if (metric.interval && metric.interval.length === 2) {
            var half = Math.round((metric.interval[1] - metric.interval[0]) / 2);
            if (half > 0) {
                intervalText = ' <span class="hint" title="' +
                    escapeHtml(label('ebIntervalNote', 'sampling band from the cohort size — not a range for the team')) +
                    '">±' + half + '</span>';
            }
        }
        return '<div style="display:flex;justify-content:space-between;gap:10px;padding:3px 0">' +
            '<span>' + name + '</span>' +
            '<span><strong>' + fmt(metric.value, metric.digits) + '</strong> ' +
            percentileBar(metric.percentile) + ' ' +
            escapeHtml(beatsText(metric.percentile)) + intervalText + medianText + '</span>' +
            '</div>' +
            (bandText ? '<div style="color:var(--text-muted);font-size:0.75rem;padding-bottom:2px">' +
                escapeHtml(bandText) + '</div>' : '');
    }

    function renderEliteBenchmark(block) {
        var pop = block.population || {};
        var competitions = (pop.competitions || []).map(function(c) {
            return typeof c === 'string' ? c : c.name;
        }).filter(Boolean);
        var html = '<div>' + escapeHtml(label('ebReference', 'Reference population')) + ': ' +
            '<strong>' + escapeHtml(pop.label || '') + '</strong> — ' +
            escapeHtml(String(pop.n_team_matches || 0)) + ' ' +
            escapeHtml(label('ebTeamMatches', 'elite team-matches')) + ', ' +
            escapeHtml(String(pop.n_matches || 0)) + ' ' +
            escapeHtml(label('ebMatches', 'matches')) + ', ' +
            escapeHtml(String(pop.n_teams || 0)) + ' ' +
            escapeHtml(label('ebTeams', 'teams')) +
            (competitions.length ? ', ' + escapeHtml(competitions.join(', ')) : '') +
            '</div>';

        var card = pop.model_card || {};
        if (card.available && card.n_shots) {
            html += '<div class="hint" style="font-size:0.75rem">' +
                escapeHtml(
                    label(
                        'ebModelCheck',
                        'Model check: Brier {b} on {n} shots (lower is better)'
                    )
                        .replace('{b}', Number(card.brier).toFixed(3))
                        .replace('{n}', Number(card.n_shots).toLocaleString())
                ) + '</div>';
        }

        var teams = block.teams || [];
        teams.forEach(function(team) {
            html += '<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)">' +
                '<strong>' + escapeHtml(String(team.team || '—')) + '</strong>';
            METRIC_GROUPS.forEach(function(group) {
                var rows = '';
                METRIC_ORDER.forEach(function(spec) {
                    if (spec.group !== group.key) return;
                    var metric = (team.metrics || {})[spec.key];
                    if (!metric) return;
                    var definition = ((pop.definitions || {})[spec.key] || {}).definition || '';
                    rows += renderMetricRow({
                        value: metric.value,
                        percentile: metric.percentile,
                        band: metric.band,
                        population: metric.population,
                        comparable: metric.comparable,
                        reason: metric.reason,
                        label: label(spec.labelKey, spec.fallback),
                        digits: spec.digits,
                        definition: definition,
                        interval: metric.interval
                    });
                });
                if (rows) {
                    html += '<div style="margin-top:8px;font-weight:600;font-size:0.78rem;color:var(--text-muted)">' +
                        escapeHtml(label(group.labelKey, group.fallback)) + '</div>' + rows;
                }
            });
            html += '</div>';
        });

        html += renderLimitations(block.limitations || []);
        return html;
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
