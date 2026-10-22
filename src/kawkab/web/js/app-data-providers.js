// Kawkab AI - External Data Provider Functions
// Split from app.js to reduce monolithic size (~1800 lines moved)
(function() {
    'use strict';

    // ── Aliases for shared state from app.js ──────────────────────────
    var bridge = null;                          // set when QWebChannel connects
    var safeParseFloat = function(v, def) { var n = parseFloat(v); return isFinite(n) ? n : (def || 0); };
    var escapeHtml = window.__kawkab.escapeHtml;

    // ── Team ID variables (were in app.js) ────────────────────────────
    var fdHomeTeamId = null;
    var fdAwayTeamId = null;
    var fdHomeSearchTimer = null;
    var fdAwaySearchTimer = null;

    var bzHomeTeamId = null;
    var bzAwayTeamId = null;
    var bzHomeSearchTimer = null;
    var bzAwaySearchTimer = null;

    var afHomeTeamId = null;
    var afAwayTeamId = null;
    var afHomeSearchTimer = null;
    var afAwaySearchTimer = null;

    // ── Bridge initialisation hook (called from app.js) ───────────────
    window.__kawkab._initDataProviderBridge = function(b) {
        bridge = b;
    };

    // ── Football-Data.org (FD) ────────────────────────────────────────

    async function checkFootballDataStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('football-data-status');
        const controls = document.getElementById('football-data-controls');
        try {
            const status = JSON.parse(await bridge.check_football_data_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected (' + status.competitions_count + ' competitions available)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🔴 Offline - ' + (status.error || 'No API key configured');
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🔴 Offline';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    function setupFdTeamSearch(inputId, resultsId, callback) {
        const input = document.getElementById(inputId);
        const results = document.getElementById(resultsId);
        if (!input || !results) return;

        input.addEventListener('input', function() {
            const query = this.value.trim();
            if (query.length < 2) {
                results.classList.add('hidden');
                return;
            }
            clearTimeout(window[inputId + 'Timer']);
            window[inputId + 'Timer'] = setTimeout(async () => {
                try {
                    const data = JSON.parse(await bridge.search_football_team(query));
                    const teams = data.teams || [];
                    if (teams.length === 0) {
                        results.innerHTML = '<div class="fd-result-item" style="color: var(--text-muted)">No teams found</div>';
                        results.classList.remove('hidden');
                        return;
                    }
                    results.innerHTML = teams.map(t => `
                        <div class="fd-result-item" data-team-id="${t.id}" data-team-name="${escapeHtml(t.name)}" data-team-code="${escapeHtml(t.tla || '')}" data-team-crest="${escapeHtml(t.crest || '')}" data-competition-code="${escapeHtml(t.competition_code || '')}">
                            ${t.crest ? '<img src="' + escapeHtml(t.crest) + '" alt="" onerror="this.style.display=\'none\'">' : ''}
                            <span>
                                <span class="fd-result-name">${escapeHtml(t.name)}</span>
                                <span class="fd-result-area">${escapeHtml(t.area_name || '')}${t.competition_name ? ' · ' + escapeHtml(t.competition_name) : ''}</span>
                            </span>
                        </div>
                    `).join('');
                    results.classList.remove('hidden');
                    results.querySelectorAll('.fd-result-item').forEach(el => {
                        el.addEventListener('click', function() {
                            const id = parseInt(this.dataset.teamId);
                            const name = this.dataset.teamName;
                            const compCode = this.dataset.competitionCode;
                            input.value = name;
                            results.classList.add('hidden');
                            callback(id, name, this.dataset.teamCrest, compCode);
                        });
                    });
                } catch (e) {
                    console.error('Team search failed:', e);
                }
            }, 300);
        });

        document.addEventListener('click', function(e) {
            if (!input.contains(e.target) && !results.contains(e.target)) {
                results.classList.add('hidden');
            }
        });
    }

    async function fdImportSquad(matchId, apiTeamId, side, btnId) {
        if (!bridge) return;
        const btn = document.getElementById(btnId);
        if (!btn) return;
        btn.disabled = true;
        btn.textContent = 'Importing...';
        try {
            const result = JSON.parse(await bridge.import_football_team_squad(matchId, apiTeamId, side));
            if (result.success) {
                btn.textContent = '✅ ' + result.created.length + ' imported, ' + result.skipped + ' skipped';
                if (window.__kawkab.loadPlayerProfiles) window.__kawkab.loadPlayerProfiles();
            } else {
                btn.textContent = '❌ ' + (result.error || 'Import failed');
            }
        } catch (e) {
            btn.textContent = '❌ Error';
        }
        setTimeout(() => { btn.disabled = false; }, 3000);
    }

    async function fdVerifyMatch() {
        if (!bridge || !window.__kawkab.currentMatchId) return;
        const apiMatchIdInput = document.getElementById('fd-api-match-id');
        const resultEl = document.getElementById('fd-verify-result');
        const apiMatchId = parseInt(apiMatchIdInput.value);
        if (!apiMatchId) {
            resultEl.textContent = 'Enter an API Match ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Verifying...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.verify_match_with_api(window.__kawkab.currentMatchId, apiMatchId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            if (data.verified) {
                resultEl.innerHTML = '✅ Score verified! API: ' + escapeHtml(data.api_score.home) + '-' + escapeHtml(data.api_score.away) +
                    ' | Status: ' + escapeHtml(data.status) + (data.competition ? ' | ' + escapeHtml(data.competition) : '');
                resultEl.className = 'feedback-result success';
            } else {
                resultEl.innerHTML = '⚠️ Score mismatch — Detected: ' + escapeHtml(data.detected_score.home) + '-' + escapeHtml(data.detected_score.away) +
                    ' | API: ' + (data.api_score ? escapeHtml(data.api_score.home) + '-' + escapeHtml(data.api_score.away) : 'N/A') +
                    ' | Status: ' + escapeHtml(data.status) + (data.reason ? ' (' + escapeHtml(data.reason) + ')' : '');
                resultEl.className = 'feedback-result';
            }
            if (window.__kawkab.loadMatchHistory) window.__kawkab.loadMatchHistory();
        } catch (e) {
            resultEl.textContent = '❌ Verification failed';
            resultEl.className = 'feedback-result error';
        }
    }

    async function fdLoadTeamFixtures(apiTeamId) {
        if (!bridge) return;
        const section = document.getElementById('fd-fixtures-section');
        const list = document.getElementById('fd-fixtures-list');
        if (!apiTeamId) { section.classList.add('hidden'); return; }
        try {
            const data = JSON.parse(await bridge.get_football_team_matches(apiTeamId, '', ''));
            const matches = data.matches || [];
            if (matches.length === 0) {
                section.classList.add('hidden');
                return;
            }
            list.innerHTML = matches.slice(0, 5).map(m => {
                const home = (m.homeTeam || {}).name || '?';
                const away = (m.awayTeam || {}).name || '?';
                const score = (m.score || {}).fullTime || {};
                const sc = (score.home !== null && score.away !== null) ? score.home + '-' + score.away : '-';
                const date = m.utcDate ? new Date(m.utcDate).toLocaleDateString() : '?';
                const status = m.status || 'SCHEDULED';
                return `<div class="roster-item"><span style="font-size:0.85rem">${date}</span><span style="flex:1;text-align:center">${escapeHtml(home)} ${sc} ${escapeHtml(away)}</span><span class="fd-result-area">${status}</span></div>`;
            }).join('');
            section.classList.remove('hidden');
        } catch (e) {
            section.classList.add('hidden');
        }
    }

    async function fdLoadStandings(competitionCode) {
        if (!bridge) return;
        const section = document.getElementById('fd-standings-section');
        const list = document.getElementById('fd-standings-list');
        if (!competitionCode) { section.classList.add('hidden'); return; }
        try {
            const data = JSON.parse(await bridge.get_football_standings(competitionCode));
            if (data.error || !data.standings) { section.classList.add('hidden'); return; }
            const total = data.standings.find(s => s.type === 'TOTAL');
            if (!total || !total.table) { section.classList.add('hidden'); return; }
            list.innerHTML = total.table.slice(0, 5).map(t => `
                <div class="roster-item">
                    <span style="font-weight:700;width:24px">${t.position}</span>
                    <span style="flex:1">${escapeHtml(t.team.name)}</span>
                    <span style="color:var(--text-muted);font-size:0.85rem">P${t.playedGames} · ${t.points}pts</span>
                </div>
            `).join('');
            section.classList.remove('hidden');
        } catch (e) {
            section.classList.add('hidden');
        }
    }

    // --- Bzzoiro (BZ) ---

    async function checkBzzoiroStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('bzzoiro-status');
        const controls = document.getElementById('bzzoiro-controls');
        try {
            const status = JSON.parse(await bridge.check_bzzoiro_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected (' + (status.live_matches || 0) + ' live)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
                bzLoadLeagues();
                bzLoadLive();
            } else {
                statusEl.textContent = '🔴 Offline - ' + (status.error || 'No API key configured');
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🔴 Offline';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    function setupBzTeamSearch(inputId, resultsId, callback) {
        const input = document.getElementById(inputId);
        const results = document.getElementById(resultsId);
        if (!input || !results) return;

        input.addEventListener('input', function() {
            const query = this.value.trim();
            if (query.length < 2) {
                results.classList.add('hidden');
                return;
            }
            clearTimeout(window[inputId + 'Timer']);
            window[inputId + 'Timer'] = setTimeout(async () => {
                try {
                    const data = JSON.parse(await bridge.search_bzzoiro_team(query));
                    const teams = data.teams || [];
                    if (teams.length === 0) {
                        results.innerHTML = '<div class="fd-result-item" style="color: var(--text-muted)">No teams found</div>';
                        results.classList.remove('hidden');
                        return;
                    }
                    results.innerHTML = teams.map(t => `
                        <div class="fd-result-item" data-team-id="${t.id}" data-team-name="${escapeHtml(t.name)}">
                            <span>
                                <span class="fd-result-name">${escapeHtml(t.name)}</span>
                                <span class="fd-result-area">${escapeHtml(t.country || '')}</span>
                            </span>
                        </div>
                    `).join('');
                    results.classList.remove('hidden');
                    results.querySelectorAll('.fd-result-item').forEach(el => {
                        el.addEventListener('click', function() {
                            const id = parseInt(this.dataset.teamId);
                            const name = this.dataset.teamName;
                            input.value = name;
                            results.classList.add('hidden');
                            callback(id, name);
                        });
                    });
                } catch (e) {
                    console.error('Bzzoiro team search failed:', e);
                }
            }, 300);
        });

        document.addEventListener('click', function(e) {
            if (!input.contains(e.target) && !results.contains(e.target)) {
                results.classList.add('hidden');
            }
        });
    }

    async function bzImportSquad(matchId, teamId, side, btnId) {
        if (!bridge) return;
        const btn = document.getElementById(btnId);
        if (!btn) return;
        btn.disabled = true;
        btn.textContent = 'Importing...';
        try {
            const result = JSON.parse(await bridge.import_bzzoiro_team_squad(matchId, teamId, side));
            if (result.success) {
                btn.textContent = '✅ ' + result.created.length + ' imported, ' + result.skipped + ' skipped';
            } else {
                btn.textContent = '❌ ' + (result.error || 'Import failed');
            }
        } catch (e) {
            btn.textContent = '❌ Error';
        }
        setTimeout(() => { btn.disabled = false; }, 3000);
    }

    async function bzVerifyMatch() {
        if (!bridge || !window.__kawkab.currentMatchId) return;
        const eventIdInput = document.getElementById('bz-event-id');
        const resultEl = document.getElementById('bz-verify-result');
        const eventId = parseInt(eventIdInput.value);
        if (!eventId) {
            resultEl.textContent = 'Enter a Bzzoiro Event ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Verifying...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.verify_match_bzzoiro(window.__kawkab.currentMatchId, eventId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            if (data.match_ok) {
                resultEl.innerHTML = '✅ Score verified! ' + escapeHtml(data.match) + ' <strong>' + escapeHtml(data.api_score) + '</strong>';
                resultEl.className = 'feedback-result success';
            } else {
                resultEl.innerHTML = '⚠️ Score mismatch — Detected: ' + escapeHtml(data.detected_score) + ' | API: ' + escapeHtml(data.api_score);
                resultEl.className = 'feedback-result';
            }
        } catch (e) {
            resultEl.textContent = '❌ Verification failed';
            resultEl.className = 'feedback-result error';
        }
    }

    async function bzGetPredictions() {
        if (!bridge) return;
        const eventIdInput = document.getElementById('bz-event-id');
        const resultEl = document.getElementById('bz-predictions-content');
        const section = document.getElementById('bz-predictions-section');
        const eventId = parseInt(eventIdInput.value);
        if (!eventId) { resultEl.textContent = 'Enter an Event ID first'; return; }
        resultEl.textContent = 'Loading predictions...';
        section.classList.remove('hidden');
        try {
            const data = JSON.parse(await bridge.get_bzzoiro_predictions(eventId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            const p = data.predictions || {};
            resultEl.innerHTML = '<pre style="font-size:0.8rem;white-space:pre-wrap">' + escapeHtml(JSON.stringify(p, null, 2)) + '</pre>';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load predictions';
        }
    }

    async function bzLoadStandings() {
        if (!bridge) return;
        const select = document.getElementById('bz-league-select');
        const section = document.getElementById('bz-standings-section');
        const list = document.getElementById('bz-standings-list');
        const leagueId = parseInt(select.value);
        if (!leagueId) { section.classList.add('hidden'); return; }
        section.classList.remove('hidden');
        list.innerHTML = '<div class="roster-item">Loading...</div>';
        try {
            const data = JSON.parse(await bridge.get_bzzoiro_standings(leagueId));
            const standings = data.standings || [];
            if (standings.length === 0) {
                list.innerHTML = '<div class="roster-item">No standings data</div>';
                return;
            }
            list.innerHTML = standings.map(s => `
                <div class="roster-item">
                    <span style="font-weight:700;width:24px">${s.position}</span>
                    <span style="flex:1">${escapeHtml(s.team_name)}</span>
                    <span style="color:var(--text-muted);font-size:0.85rem">P${s.played} · ${s.points}pts</span>
                </div>
            `).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load standings</div>';
        }
    }

    async function bzLoadLeagues() {
        if (!bridge) return;
        const select = document.getElementById('bz-league-select');
        if (!select) return;
        try {
            const data = JSON.parse(await bridge.get_bzzoiro_leagues());
            const leagues = data.leagues || [];
            select.innerHTML = '<option value="">-- Select League for Standings --</option>' +
                leagues.filter(l => l.is_active).map(l =>
                    '<option value="' + l.id + '">' + escapeHtml(l.name) + ' (' + escapeHtml(l.country || '') + ')</option>'
                ).join('');
        } catch (e) {
            console.error('Failed to load leagues:', e);
        }
    }

    async function bzLoadLive() {
        if (!bridge) return;
        const list = document.getElementById('bz-live-list');
        const countEl = document.getElementById('bz-live-count');
        if (!list) return;
        try {
            const data = JSON.parse(await bridge.get_bzzoiro_live());
            const matches = data.matches || [];
            if (countEl) countEl.textContent = '(' + matches.length + ')';
            if (matches.length === 0) {
                list.innerHTML = '<div class="roster-item" style="color:var(--text-muted)">No live matches</div>';
                return;
            }
            list.innerHTML = matches.map(m => {
                const score = (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-');
                const min = m.current_minute ? m.current_minute + "'" : '';
                return `<div class="roster-item"><span style="color:var(--accent);font-size:0.75rem">${escapeHtml(m.league_name || '')}</span><span style="flex:1;text-align:center">${escapeHtml(m.home_team)} <strong>${score}</strong> ${escapeHtml(m.away_team)}</span><span style="color:var(--text-muted);font-size:0.8rem">${min}</span></div>`;
            }).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load live matches</div>';
        }
    }

    // --- API-Football (AF) ---

    async function checkApiFootballStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('apifootball-status');
        const controls = document.getElementById('apifootball-controls');
        try {
            const status = JSON.parse(await bridge.check_apifootball_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected (' + (status.live_matches || 0) + ' live)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
                afLoadLive();
            } else {
                statusEl.textContent = '🔴 Offline - ' + (status.error || 'API key?');
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🔴 Offline';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    function setupAfTeamSearch(inputId, resultsId, callback) {
        const input = document.getElementById(inputId);
        const results = document.getElementById(resultsId);
        if (!input || !results) return;

        input.addEventListener('input', function() {
            const query = this.value.trim();
            if (query.length < 2) {
                results.classList.add('hidden');
                return;
            }
            clearTimeout(window[inputId + 'Timer']);
            window[inputId + 'Timer'] = setTimeout(async () => {
                try {
                    const data = JSON.parse(await bridge.search_apifootball_team(query));
                    const teams = data.teams || [];
                    if (teams.length === 0) {
                        results.innerHTML = '<div class="fd-result-item" style="color: var(--text-muted)">No teams found</div>';
                        results.classList.remove('hidden');
                        return;
                    }
                    results.innerHTML = teams.map(t => `
                        <div class="fd-result-item" data-team-id="${t.id}" data-team-name="${escapeHtml(t.name)}">
                            ${t.logo ? '<img src="' + escapeHtml(t.logo) + '" alt="" onerror="this.style.display=\'none\'">' : ''}
                            <span>
                                <span class="fd-result-name">${escapeHtml(t.name)}</span>
                                <span class="fd-result-area">${escapeHtml(t.country || '')}${t.venue_name ? ' · ' + escapeHtml(t.venue_name) : ''}</span>
                            </span>
                        </div>
                    `).join('');
                    results.classList.remove('hidden');
                    results.querySelectorAll('.fd-result-item').forEach(el => {
                        el.addEventListener('click', function() {
                            const id = parseInt(this.dataset.teamId);
                            const name = this.dataset.teamName;
                            input.value = name;
                            results.classList.add('hidden');
                            callback(id, name);
                        });
                    });
                } catch (e) {
                    console.error('API-Football team search failed:', e);
                }
            }, 300);
        });

        document.addEventListener('click', function(e) {
            if (!input.contains(e.target) && !results.contains(e.target)) {
                results.classList.add('hidden');
            }
        });
    }

    async function afImportSquad(matchId, teamId, side, btnId) {
        if (!bridge) return;
        const btn = document.getElementById(btnId);
        if (!btn) return;
        btn.disabled = true;
        btn.textContent = 'Importing...';
        try {
            const result = JSON.parse(await bridge.import_apifootball_squad(matchId, teamId, side));
            if (result.success) {
                btn.textContent = '✅ ' + result.created.length + ' imported, ' + result.skipped + ' skipped';
            } else {
                btn.textContent = '❌ ' + (result.error || 'Import failed');
            }
        } catch (e) {
            btn.textContent = '❌ Error';
        }
        setTimeout(() => { btn.disabled = false; }, 3000);
    }

    async function afVerifyMatch() {
        if (!bridge || !window.__kawkab.currentMatchId) return;
        const fixtureIdInput = document.getElementById('af-fixture-id');
        const resultEl = document.getElementById('af-verify-result');
        const fixtureId = parseInt(fixtureIdInput.value);
        if (!fixtureId) {
            resultEl.textContent = 'Enter a Fixture ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Verifying...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.verify_match_apifootball(window.__kawkab.currentMatchId, fixtureId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            if (data.match_ok) {
                resultEl.innerHTML = '✅ Score verified! ' + escapeHtml(data.match) + ' <strong>' + escapeHtml(data.api_score) + '</strong>';
                resultEl.className = 'feedback-result success';
            } else {
                resultEl.innerHTML = '⚠️ Score mismatch — Detected: ' + escapeHtml(data.detected_score) + ' | API: ' + escapeHtml(data.api_score);
                resultEl.className = 'feedback-result';
            }
        } catch (e) {
            resultEl.textContent = '❌ Verification failed';
            resultEl.className = 'feedback-result error';
        }
    }

    async function afGetPredictions() {
        if (!bridge) return;
        const fixtureIdInput = document.getElementById('af-fixture-id');
        const resultEl = document.getElementById('af-predictions-content');
        const section = document.getElementById('af-predictions-section');
        const fixtureId = parseInt(fixtureIdInput.value);
        if (!fixtureId) { resultEl.textContent = 'Enter a Fixture ID first'; return; }
        resultEl.textContent = 'Loading predictions...';
        section.classList.remove('hidden');
        try {
            const data = JSON.parse(await bridge.get_apifootball_predictions(fixtureId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            const p = data.predictions || {};
            let html = '';
            if (p.percent) {
                html += '<div style="display:flex;gap:8px;justify-content:center;margin-bottom:8px">';
                html += '<span style="color:#2563eb;font-weight:600">' + escapeHtml(p.percent.home || '') + '</span>';
                html += '<span style="color:#64748b">' + escapeHtml(p.percent.draw || '') + '</span>';
                html += '<span style="color:#dc2626;font-weight:600">' + escapeHtml(p.percent.away || '') + '</span>';
                html += '</div>';
            }
            if (p.advice) html += '<div style="font-size:0.85rem;background:#f8fafc;padding:6px;border-radius:4px">Advice: ' + escapeHtml(p.advice) + '</div>';
            if (p.winner) html += '<div style="margin-top:4px;font-size:0.85rem">Predicted winner: ' + escapeHtml(p.winner.name || p.winner) + '</div>';
            resultEl.innerHTML = html || '<pre style="font-size:0.8rem">' + escapeHtml(JSON.stringify(p, null, 2)) + '</pre>';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load predictions';
        }
    }

    async function afLoadStandings() {
        if (!bridge) return;
        const section = document.getElementById('af-standings-section');
        const list = document.getElementById('af-standings-list');
        section.classList.remove('hidden');
        list.innerHTML = '<div class="roster-item">Loading standings...</div>';
        try {
            const data = JSON.parse(await bridge.get_apifootball_standings(200, 2024));
            const standings = data.standings || [];
            if (standings.length === 0) {
                list.innerHTML = '<div class="roster-item">No standings data</div>';
                return;
            }
            list.innerHTML = standings.map(s => `
                <div class="roster-item">
                    <span style="font-weight:700;width:24px">${s.rank}</span>
                    <span style="flex:1">${escapeHtml(s.team_name)}</span>
                    <span style="color:var(--text-muted);font-size:0.85rem">P${s.played} · ${s.points}pts</span>
                </div>
            `).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load standings</div>';
        }
    }

    async function afLoadLive() {
        if (!bridge) return;
        const list = document.getElementById('af-live-list');
        if (!list) return;
        try {
            const data = JSON.parse(await bridge.get_apifootball_live());
            const matches = data.matches || [];
            if (matches.length === 0) {
                list.innerHTML = '<div class="roster-item" style="color:var(--text-muted)">No live matches</div>';
                return;
            }
            list.innerHTML = matches.map(m => {
                const score = (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-');
                const min = m.elapsed ? m.elapsed + "'" : '';
                return `<div class="roster-item"><span style="color:var(--accent);font-size:0.75rem">${escapeHtml(m.league_name || '')}</span><span style="flex:1;text-align:center">${escapeHtml(m.home_team)} <strong>${score}</strong> ${escapeHtml(m.away_team)}</span><span style="color:var(--text-muted);font-size:0.8rem">${min}</span></div>`;
            }).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load live matches</div>';
        }
    }

    // --- EasySoccer (ES) ---

    async function checkEasySoccerStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('easysoccer-status');
        const controls = document.getElementById('easysoccer-controls');
        try {
            const status = JSON.parse(await bridge.check_easy_soccer_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
                esLoadLive();
            } else {
                statusEl.textContent = '🟡 Not available (pip install EasySoccerData)';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function esGetEvent() {
        if (!bridge) return;
        const input = document.getElementById('es-event-id');
        const resultEl = document.getElementById('es-event-result');
        const eventId = parseInt(input.value);
        if (!eventId) { resultEl.textContent = 'Enter a Sofascore Event ID'; return; }
        resultEl.textContent = 'Loading...';
        try {
            const data = JSON.parse(await bridge.get_easy_soccer_event(eventId));
            if (data.error) { resultEl.textContent = '❌ ' + data.error; return; }
            const e = data.event;
            resultEl.innerHTML = '<div style="font-size:0.85rem"><strong>' + escapeHtml(e.home_team) + '</strong> vs <strong>' + escapeHtml(e.away_team) + '</strong><br>Score: ' +
                (e.home_score !== null ? e.home_score : '-') + '-' + (e.away_score !== null ? e.away_score : '-') +
                ' | Status: ' + escapeHtml(e.status) + '</div>';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load event';
        }
    }

    async function esGetIncidents() {
        if (!bridge) return;
        const input = document.getElementById('es-event-id');
        const resultEl = document.getElementById('es-event-result');
        const eventId = parseInt(input.value);
        if (!eventId) { resultEl.textContent = 'Enter a Sofascore Event ID'; return; }
        resultEl.textContent = 'Loading incidents...';
        try {
            const data = JSON.parse(await bridge.get_easy_soccer_incidents(eventId));
            if (data.error) { resultEl.textContent = '❌ ' + data.error; return; }
            const incidents = data.incidents || [];
            if (incidents.length === 0) { resultEl.textContent = 'No incidents found'; return; }
            resultEl.innerHTML = incidents.slice(0, 15).map(i =>
                '<div style="font-size:0.8rem;padding:2px 0">' + (i.minute ? escapeHtml(String(i.minute)) + "' " : '') +
                escapeHtml(i.type || '') + ' - ' + escapeHtml(i.player || '') + ' (' + escapeHtml(i.team || '') + ')</div>'
            ).join('');
        } catch (e) {
            resultEl.textContent = '❌ Failed to load incidents';
        }
    }

    async function esLoadLive() {
        if (!bridge) return;
        const list = document.getElementById('es-live-list');
        if (!list) return;
        try {
            const data = JSON.parse(await bridge.get_easy_soccer_live());
            const matches = data.matches || [];
            if (matches.length === 0) {
                list.innerHTML = '<div class="roster-item" style="color:var(--text-muted)">No live matches</div>';
                return;
            }
            list.innerHTML = matches.map(m => {
                const score = (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-');
                return `<div class="roster-item"><span style="flex:1">${escapeHtml(m.home_team)} <strong>${score}</strong> ${escapeHtml(m.away_team)}</span></div>`;
            }).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load live matches</div>';
        }
    }

    // --- TheSportsDB (TSDB) ---

    async function checkTheSportsDBStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('thesportsdb-status');
        const controls = document.getElementById('thesportsdb-controls');
        try {
            const status = JSON.parse(await bridge.check_thesportsdb_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected (public free tier)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Not available';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    function setupTsdbTeamSearch(inputId, resultsId, callback) {
        const input = document.getElementById(inputId);
        const results = document.getElementById(resultsId);
        if (!input || !results) return;

        input.addEventListener('input', function() {
            const query = this.value.trim();
            if (query.length < 2) {
                results.classList.add('hidden');
                return;
            }
            clearTimeout(window[inputId + 'Timer']);
            window[inputId + 'Timer'] = setTimeout(async () => {
                try {
                    const data = JSON.parse(await bridge.search_thesportsdb_team(query));
                    const teams = data.teams || [];
                    if (teams.length === 0) {
                        results.innerHTML = '<div class="fd-result-item" style="color: var(--text-muted)">No teams found</div>';
                        results.classList.remove('hidden');
                        return;
                    }
                    results.innerHTML = teams.map(t => `
                        <div class="fd-result-item" data-team-id="${t.id}" data-team-name="${escapeHtml(t.name)}" data-league="${escapeHtml(t.league || '')}" data-league-id="${t.league_id || ''}" data-badge="${escapeHtml(t.badge || '')}">
                            <span>
                                <span class="fd-result-name">${escapeHtml(t.name)}</span>
                                <span class="fd-result-area">${escapeHtml(t.league || '')}${t.location ? ' · ' + escapeHtml(t.location) : ''}</span>
                            </span>
                        </div>
                    `).join('');
                    results.classList.remove('hidden');
                    results.querySelectorAll('.fd-result-item').forEach(el => {
                        el.addEventListener('click', function() {
                            const id = this.dataset.teamId;
                            const name = this.dataset.teamName;
                            input.value = name;
                            results.classList.add('hidden');
                            callback(id, name, this.dataset.leagueId, this.dataset.badge);
                        });
                    });
                } catch (e) {
                    console.error('TheSportsDB team search failed:', e);
                }
            }, 300);
        });

        document.addEventListener('click', function(e) {
            if (!input.contains(e.target) && !results.contains(e.target)) {
                results.classList.add('hidden');
            }
        });
    }

    async function tsdbGetTeamInfo() {
        if (!bridge) return;
        const infoEl = document.getElementById('tsdb-team-info');
        const searchInput = document.getElementById('tsdb-team-search');
        const resultsDiv = document.getElementById('tsdb-team-results');
        const teamId = window._tsdbSelectedTeamId || searchInput?.value;
        if (!teamId) {
            infoEl.textContent = 'Search and click a team first';
            return;
        }
        try {
            const data = JSON.parse(await bridge.get_thesportsdb_team_info(teamId));
            if (data.error || !data.team) {
                infoEl.textContent = '❌ ' + (data.error || 'Team not found');
                return;
            }
            const t = data.team;
            infoEl.innerHTML = '<div style="font-size:0.85rem">' +
                (t.badge ? '<img src="' + escapeHtml(t.badge) + '" style="height:32px;vertical-align:middle;margin-right:8px">' : '') +
                '<strong>' + escapeHtml(t.name) + '</strong>' +
                (t.alternate_name ? ' (' + escapeHtml(t.alternate_name) + ')' : '') +
                '<br>League: ' + escapeHtml(t.league) + ' (ID: ' + escapeHtml(String(t.league_id)) + ')' +
                (t.location ? '<br>Location: ' + escapeHtml(t.location) : '') +
                (t.stadium ? '<br>Stadium: ' + escapeHtml(t.stadium) + (t.capacity ? ' (' + escapeHtml(String(t.capacity)) + ')' : '') : '') +
                (t.formed_year ? '<br>Founded: ' + escapeHtml(String(t.formed_year)) : '') +
                (t.api_football_id && t.api_football_id !== '0' ? '<br>API-Football ID: ' + escapeHtml(String(t.api_football_id)) : '') +
                (t.description ? '<br><br><em>' + escapeHtml(t.description.substring(0, 300)) + '</em>' : '') +
                '</div>';
            infoEl.className = 'feedback-result';
        } catch (e) {
            infoEl.textContent = '❌ Failed to load team info';
        }
    }

    async function tsdbLoadStandings() {
        if (!bridge) return;
        const leagueIdInput = document.getElementById('tsdb-league-id');
        const section = document.getElementById('tsdb-standings-section');
        const list = document.getElementById('tsdb-standings-list');
        const leagueId = leagueIdInput.value.trim();
        if (!leagueId) { section.classList.add('hidden'); return; }
        try {
            const data = JSON.parse(await bridge.get_thesportsdb_standings(leagueId));
            if (data.error || !data.standings) { section.classList.add('hidden'); return; }
            list.innerHTML = data.standings.map(s => {
                const formDots = s.form ? s.form.split('').map(f => {
                    const cls = f === 'W' ? 'form-w' : f === 'D' ? 'form-d' : 'form-l';
                    return '<span class="' + cls + '">' + f + '</span>';
                }).join('') : '';
                return '<div class="roster-item">' +
                    '<span style="font-weight:700;width:28px">' + s.rank + '</span>' +
                    '<span style="flex:1">' + (s.badge ? '<img src="' + escapeHtml(s.badge) + '" style="height:18px;vertical-align:middle;margin-right:6px">' : '') + escapeHtml(s.team) + '</span>' +
                    '<span style="color:var(--text-muted);font-size:0.85rem;margin:0 6px">P' + s.played + '</span>' +
                    '<span style="color:var(--text-muted);font-size:0.85rem">' + s.points + 'pts</span>' +
                    (formDots ? '<span style="margin-left:8px">' + formDots + '</span>' : '') +
                    (s.description ? '<span style="color:var(--text-muted);font-size:0.7rem;margin-left:6px">' + escapeHtml(s.description) + '</span>' : '') +
                    '</div>';
            }).join('');
            section.classList.remove('hidden');
        } catch (e) {
            section.classList.add('hidden');
        }
    }

    async function tsdbLoadEvents(type) {
        if (!bridge) return;
        const teamIdInput = document.getElementById('tsdb-team-id-events');
        const section = document.getElementById('tsdb-events-section');
        const list = document.getElementById('tsdb-events-list');
        const teamId = teamIdInput.value.trim();
        if (!teamId) { section.classList.add('hidden'); return; }
        try {
            const method = type === 'last' ? 'get_thesportsdb_team_events_last' : 'get_thesportsdb_team_events_next';
            const data = JSON.parse(await bridge[method](teamId));
            if (data.error || !data.events) { section.classList.add('hidden'); return; }
            list.innerHTML = data.events.slice(0, 10).map(e => {
                const score = (e.home_score !== null ? e.home_score : '-') + '-' + (e.away_score !== null ? e.away_score : '-');
                return '<div class="roster-item">' +
                    '<span style="color:var(--text-muted);font-size:0.8rem;width:80px">' + escapeHtml(e.date || '') + '</span>' +
                    '<span style="flex:1;text-align:center">' + escapeHtml(e.home) + ' <strong>' + score + '</strong> ' + escapeHtml(e.away) + '</span>' +
                    (e.round ? '<span style="color:var(--text-muted);font-size:0.8rem">R' + escapeHtml(String(e.round)) + '</span>' : '') +
                    '</div>';
            }).join('');
            section.classList.remove('hidden');
        } catch (e) {
            section.classList.add('hidden');
        }
    }

    // --- StatsBomb (SB) ---

    async function checkStatsBombStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('statsbomb-status');
        const controls = document.getElementById('statsbomb-controls');
        try {
            const status = JSON.parse(await bridge.check_statsbomb_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected (' + (status.competitions || 0) + ' competitions)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Not available (network error)';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function sbGetEvents() {
        if (!bridge) return;
        const matchIdInput = document.getElementById('sb-match-id');
        const resultEl = document.getElementById('sb-events-result');
        const matchId = parseInt(matchIdInput.value);
        if (!matchId) {
            resultEl.textContent = 'Enter a StatsBomb Match ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Loading events...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.get_statsbomb_events(matchId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            const s = data.summary;
            const shotsHtml = (data.shots || []).map(sh =>
                '<div style="font-size:0.78rem;padding:2px 0">' + sh.minute + "' " +
                escapeHtml(sh.team) + ' · ' + escapeHtml(sh.player) +
                (sh.xg !== null ? ' (xG: ' + sh.xg.toFixed(3) + ')' : '') +
                ' · ' + escapeHtml(sh.outcome || '') + '</div>'
            ).join('');
            resultEl.innerHTML =
                '<div style="font-size:0.85rem">' +
                '<strong>' + s.total_events + '</strong> events · ' +
                '<strong>' + s.shots + '</strong> shots · ' +
                '<strong>' + s.passes + '</strong> passes · ' +
                'xG: <strong>' + s.total_xg + '</strong><br>' +
                'Teams: ' + escapeHtml(s.teams.join(' vs ')) +
                '</div>' + shotsHtml;
            resultEl.className = 'feedback-result';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load events';
            resultEl.className = 'feedback-result error';
        }
    }

    async function sbImportToDb() {
        if (!bridge) return;
        const matchIdInput = document.getElementById('sb-match-id');
        const resultEl = document.getElementById('sb-import-result');
        const matchId = matchIdInput.value.trim();
        if (!matchId) {
            resultEl.textContent = 'Enter a StatsBomb Match ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Importing events to local database...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.import_statsbomb_match(matchId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            resultEl.textContent = '✅ Imported ' + data.imported + ' events from match ' + data.match_id;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Import failed';
            resultEl.className = 'feedback-result error';
        }
    }

    async function sbGetLineups() {
        if (!bridge) return;
        const matchIdInput = document.getElementById('sb-match-id');
        const resultEl = document.getElementById('sb-lineups-result');
        const matchId = parseInt(matchIdInput.value);
        if (!matchId) {
            resultEl.textContent = 'Enter a StatsBomb Match ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Loading lineups...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.get_statsbomb_lineups(matchId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            resultEl.innerHTML = (data.lineups || []).map(team =>
                '<div style="font-size:0.85rem;margin-top:6px"><strong>' + escapeHtml(team.team) + '</strong> (' + team.players.length + ' players)' +
                '<div style="font-size:0.75rem;color:var(--text-muted)">' +
                team.players.slice(0, 11).map(p => '#' + p.jersey_number + ' ' + escapeHtml(p.name)).join(' · ') +
                '</div></div>'
            ).join('');
            resultEl.className = 'feedback-result';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load lineups';
            resultEl.className = 'feedback-result error';
        }
    }

    async function sbSearchTeam() {
        if (!bridge) return;
        const searchInput = document.getElementById('sb-team-search');
        const resultsDiv = document.getElementById('sb-team-results');
        const query = searchInput.value.trim();
        if (query.length < 2) {
            resultsDiv.classList.add('hidden');
            return;
        }
        try {
            const data = JSON.parse(await bridge.search_statsbomb_team(query));
            const matches = data.matches || [];
            if (matches.length === 0) {
                resultsDiv.innerHTML = '<div class="fd-result-item" style="color: var(--text-muted)">No matches found</div>';
                resultsDiv.classList.remove('hidden');
                return;
            }
            resultsDiv.innerHTML = matches.slice(0, 10).map(m =>
                '<div class="fd-result-item" data-match-id="' + m.match_id + '">' +
                '<span style="font-size:0.8rem">' +
                escapeHtml(m.home) + ' ' + (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-') + ' ' + escapeHtml(m.away) +
                '<br><span style="color:var(--text-muted);font-size:0.7rem">' + escapeHtml(m.competition) + ' · ' + escapeHtml(m.season) + ' · ' + escapeHtml(m.date) + '</span>' +
                '</span></div>'
            ).join('');
            resultsDiv.classList.remove('hidden');
            resultsDiv.querySelectorAll('.fd-result-item').forEach(el => {
                el.addEventListener('click', function() {
                    document.getElementById('sb-match-id').value = this.dataset.matchId;
                    resultsDiv.classList.add('hidden');
                });
            });
        } catch (e) {
            console.error('StatsBomb team search failed:', e);
        }
    }

    // --- OpenFootball (OFB) ---

    async function checkOpenFootballStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('openfootball-status');
        const controls = document.getElementById('openfootball-controls');
        try {
            const status = JSON.parse(await bridge.check_openfootball_status());
            if (status.available) {
                statusEl.textContent = '🟢 Connected (CC0 public domain data)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
                ofbPopulateCompetitions();
            } else {
                statusEl.textContent = '🟡 Not available (network error)';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function ofbPopulateCompetitions() {
        if (!bridge) return;
        try {
            const data = JSON.parse(await bridge.get_openfootball_competitions());
            const compSelect = document.getElementById('ofb-competition');
            const seasonSelect = document.getElementById('ofb-season');
            compSelect.innerHTML = '<option value="">-- Competition --</option>' +
                (data.competitions || []).map(c =>
                    '<option value="' + c.id + '">' + escapeHtml(c.name) + '</option>'
                ).join('');
            compSelect.onchange = function() {
                const comp = (data.competitions || []).find(c => c.id === this.value);
                seasonSelect.innerHTML = '<option value="">-- Season --</option>' +
                    (comp ? comp.seasons : []).map(s =>
                        '<option value="' + s + '">' + s + '</option>'
                    ).join('');
            };
            const wcSelect = document.getElementById('ofb-wc-year');
            wcSelect.innerHTML = '<option value="">-- World Cup Year --</option>' +
                [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]
                .map(y => '<option value="' + y + '">' + y + '</option>').join('');
        } catch (e) {
            console.error('OpenFootball populate failed:', e);
        }
    }

    async function ofbLoadMatches() {
        if (!bridge) return;
        const comp = document.getElementById('ofb-competition').value;
        const season = document.getElementById('ofb-season').value;
        const list = document.getElementById('ofb-matches-list');
        if (!comp || !season) { list.innerHTML = ''; return; }
        list.innerHTML = '<div class="roster-item">Loading...</div>';
        try {
            const data = JSON.parse(await bridge.get_openfootball_matches(comp, season));
            if (data.error) { list.innerHTML = '<div class="roster-item">❌ ' + escapeHtml(data.error) + '</div>'; return; }
            const matches = data.matches || [];
            if (matches.length === 0) { list.innerHTML = '<div class="roster-item">No matches</div>'; return; }
            list.innerHTML = matches.map(m => {
                const score = (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-');
                return '<div class="roster-item">' +
                    '<span style="color:var(--text-muted);font-size:0.75rem;width:60px">' + escapeHtml(m.date) + '</span>' +
                    '<span style="flex:1">' + escapeHtml(m.home) + ' <strong>' + score + '</strong> ' + escapeHtml(m.away) + '</span>' +
                    '<span style="color:var(--text-muted);font-size:0.7rem">' + escapeHtml(m.round) + '</span>' +
                    '</div>';
            }).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load</div>';
        }
    }

    async function ofbSearchTeam() {
        if (!bridge) return;
        const searchInput = document.getElementById('ofb-team-search');
        const resultsDiv = document.getElementById('ofb-team-results');
        const query = searchInput.value.trim();
        if (query.length < 2) { resultsDiv.classList.add('hidden'); return; }
        clearTimeout(window.ofbTimer);
        window.ofbTimer = setTimeout(async () => {
            try {
                const data = JSON.parse(await bridge.search_openfootball_team(query));
                const matches = data.matches || [];
                if (matches.length === 0) {
                    resultsDiv.innerHTML = '<div class="fd-result-item" style="color: var(--text-muted)">No matches found</div>';
                    resultsDiv.classList.remove('hidden');
                    return;
                }
                resultsDiv.innerHTML = matches.slice(0, 15).map(m =>
                    '<div class="fd-result-item">' +
                    '<span style="font-size:0.8rem">' +
                    escapeHtml(m.home) + ' ' + (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-') + ' ' + escapeHtml(m.away) +
                    '<br><span style="color:var(--text-muted);font-size:0.7rem">' + escapeHtml(m.competition) + ' · ' + escapeHtml(m.season) + ' · ' + escapeHtml(m.date) + '</span>' +
                    '</span></div>'
                ).join('');
                resultsDiv.classList.remove('hidden');
            } catch (e) {
                console.error('OpenFootball team search failed:', e);
            }
        }, 400);
    }

    async function ofbLoadWorldCup() {
        if (!bridge) return;
        const yearInput = document.getElementById('ofb-wc-year');
        const list = document.getElementById('ofb-wc-list');
        const year = parseInt(yearInput.value);
        if (!year) { list.innerHTML = ''; return; }
        list.innerHTML = '<div class="roster-item">Loading WC ' + year + '...</div>';
        try {
            const data = JSON.parse(await bridge.get_openfootball_worldcup(year));
            if (data.error) { list.innerHTML = '<div class="roster-item">❌ ' + escapeHtml(data.error) + '</div>'; return; }
            const matches = data.matches || [];
            if (matches.length === 0) { list.innerHTML = '<div class="roster-item">No WC matches for ' + year + '</div>'; return; }
            list.innerHTML = matches.map(m => {
                const score = (m.home_score !== null ? m.home_score : '-') + '-' + (m.away_score !== null ? m.away_score : '-');
                return '<div class="roster-item">' +
                    '<span style="color:var(--text-muted);font-size:0.75rem;width:80px">' + escapeHtml(m.date || '') + '</span>' +
                    '<span style="flex:1">' + escapeHtml(m.home) + ' <strong>' + score + '</strong> ' + escapeHtml(m.away) + '</span>' +
                    '<span style="color:var(--text-muted);font-size:0.7rem">' + escapeHtml(m.round) + '</span>' +
                    '</div>';
            }).join('');
        } catch (e) {
            list.innerHTML = '<div class="roster-item">Failed to load WC</div>';
        }
    }

    // --- Pose Analysis ---

    async function checkPoseStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('pose-status');
        const controls = document.getElementById('pose-controls');
        try {
            const status = JSON.parse(await bridge.check_pose_status());
            if (status.available) {
                statusEl.textContent = '🟢 YOLO26-pose model loaded';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Pose model not available (ultralytics not installed)';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function poseGetSummary() {
        if (!bridge) return;
        const trackIdInput = document.getElementById('pose-track-id');
        const resultEl = document.getElementById('pose-result');
        const trackId = parseInt(trackIdInput.value);
        if (isNaN(trackId)) {
            resultEl.textContent = 'Enter a player track ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Loading...';
        try {
            const data = JSON.parse(await bridge.get_activity_summary(trackId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            const entries = Object.entries(data.summary || {});
            if (entries.length === 0) {
                resultEl.textContent = 'No activity data for track ' + trackId;
                resultEl.className = 'feedback-result';
                return;
            }
            const html = entries.map(([act, dur]) =>
                '<span style="display:inline-block;margin:2px 6px;padding:3px 8px;background:#e2e8f0;border-radius:4px;font-size:0.85rem">' +
                escapeHtml(act) + ': ' + dur.toFixed(1) + 's</span>'
            ).join('');
            resultEl.innerHTML = '<div style="font-size:0.85rem">Activity for track #' + trackId + ': ' + html + '</div>';
            resultEl.className = 'feedback-result';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load activity';
            resultEl.className = 'feedback-result error';
        }
    }

    async function poseGetSegments() {
        if (!bridge) return;
        const trackIdInput = document.getElementById('pose-track-id');
        const resultEl = document.getElementById('pose-result');
        const trackId = parseInt(trackIdInput.value);
        if (isNaN(trackId)) {
            resultEl.textContent = 'Enter a player track ID';
            resultEl.className = 'feedback-result error';
            return;
        }
        resultEl.textContent = 'Loading...';
        try {
            const data = JSON.parse(await bridge.get_activity_segments(trackId));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            const segs = data.segments || [];
            if (segs.length === 0) {
                resultEl.textContent = 'No segments for track ' + trackId;
                resultEl.className = 'feedback-result';
                return;
            }
            resultEl.innerHTML = segs.slice(0, 20).map(s =>
                '<div style="font-size:0.78rem;padding:2px 0">' +
                s.start_time.toFixed(1) + 's → ' + s.end_time.toFixed(1) + 's: ' +
                '<strong>' + s.activity + '</strong> (' + s.duration_s.toFixed(1) + 's)' +
                '</div>'
            ).join('');
            resultEl.className = 'feedback-result';
        } catch (e) {
            resultEl.textContent = '❌ Failed to load segments';
            resultEl.className = 'feedback-result error';
        }
    }

    // --- MuJoCo Ball Simulation ---

    async function checkMuJoCoStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('mujoco-status');
        const controls = document.getElementById('mujoco-controls');
        try {
            const status = JSON.parse(await bridge.check_mujoco_status());
            if (status.available) {
                let info = '🟢 Trajectory simulation available';
                if (status.uses_mujoco) info += ' (using MuJoCo)';
                else info += ' (analytical fallback)';
                statusEl.textContent = info;
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
                mujocoLoadPresets();
            } else {
                statusEl.textContent = '🟡 Not available';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function mujocoLoadPresets() {
        if (!bridge) return;
        try {
            const data = JSON.parse(await bridge.get_setpiece_presets());
            const select = document.getElementById('mujoco-preset');
            select.innerHTML = '<option value="">-- Select Preset --</option>' +
                (data.presets || []).map((p, idx) =>
                    '<option value="' + idx + '">' + escapeHtml(p.name) + '</option>'
                ).join('');
        } catch (e) {
            console.error('MuJoCo preset load failed:', e);
        }
    }

    function mujocoApplyPreset() {
        if (!bridge) return;
        bridge.get_setpiece_presets().then(json => {
            try {
                const data = JSON.parse(json);
                const idx = parseInt(document.getElementById('mujoco-preset').value);
                if (isNaN(idx) || !data.presets || !data.presets[idx]) return;
                const p = data.presets[idx];
                document.getElementById('mujoco-speed').value = p.initial_speed;
                document.getElementById('mujoco-angle').value = p.launch_angle_deg;
                document.getElementById('mujoco-spin').value = p.spin_rps;
                document.getElementById('mujoco-direction').value = p.direction_deg;
            } catch (e) { showToast('Failed to apply MuJoCo preset.', 'error'); console.warn('mujocoApplyPreset:', e); }
        });
    }

    async function mujocoSimulate() {
        if (!bridge) return;
        const speed = safeParseFloat(document.getElementById('mujoco-speed').value, 25);
        const angle = safeParseFloat(document.getElementById('mujoco-angle').value, 30);
        const spin = safeParseFloat(document.getElementById('mujoco-spin').value, 0);
        const direction = safeParseFloat(document.getElementById('mujoco-direction').value, 0);
        const resultEl = document.getElementById('mujoco-result');
        resultEl.textContent = 'Simulating...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.simulate_trajectory(speed, angle, spin, direction, 2.5));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            resultEl.innerHTML =
                '<div style="font-size:0.85rem">' +
                '<strong>' + data.method + '</strong> simulation<br>' +
                'Landing: (' + data.landing_x.toFixed(1) + ', ' + data.landing_y.toFixed(1) + ') m<br>' +
                'Max height: ' + data.max_height.toFixed(1) + ' m<br>' +
                'Duration: ' + data.duration_s.toFixed(2) + ' s<br>' +
                'Final speed: ' + data.final_speed_mps.toFixed(1) + ' m/s' +
                '</div>';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Simulation failed';
            resultEl.className = 'feedback-result error';
        }
    }

    // --- Weather ---

    async function checkWeatherStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('weather-status');
        const controls = document.getElementById('weather-controls');
        try {
            const status = JSON.parse(await bridge.check_weather_status());
            if (status.available) {
                statusEl.textContent = '🟢 Available' +
                    (status.has_video_classifier ? ' (video classifier ready)' : ' (Open-Meteo + manual)');
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Not available';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function wxAnalyzeManual() {
        if (!bridge) return;
        const resultEl = document.getElementById('wx-result');
        const temp = safeParseFloat(document.getElementById('wx-temp').value, 20);
        const precip = safeParseFloat(document.getElementById('wx-precip').value, 0);
        const wind = safeParseFloat(document.getElementById('wx-wind').value, 5);
        const humidity = safeParseFloat(document.getElementById('wx-humidity').value, 50);
        const conditions = document.getElementById('wx-conditions').value;
        resultEl.textContent = 'Analyzing...';
        try {
            const data = JSON.parse(await bridge.set_manual_weather(temp, precip, wind, humidity, conditions));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            const c = data.conditions, i = data.impact;
            resultEl.innerHTML = '<div style="font-size:0.85rem">' +
                '<strong>Conditions:</strong> ' + c.conditions + ' (' + c.pitch_state + ' pitch)<br>' +
                '<strong>Impact:</strong> ' + i.goals_delta + ' goals, ' + i.passing_delta_pct + '% passing, ' +
                i.sprint_delta_pct + '% sprint<br>' +
                i.notes.map(n => '• ' + escapeHtml(n)).join('<br>') + '</div>';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Failed to analyze';
        }
    }

    async function wxFetchFromAPI() {
        if (!bridge) return;
        const resultEl = document.getElementById('wx-result');
        const lat = parseFloat(document.getElementById('wx-lat').value);
        const lon = parseFloat(document.getElementById('wx-lon').value);
        const date = document.getElementById('wx-date').value;
        if (!date) {
            resultEl.textContent = 'Enter a date';
            return;
        }
        resultEl.textContent = 'Fetching from Open-Meteo...';
        try {
            const data = JSON.parse(await bridge.fetch_match_weather(lat, lon, date, false));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            document.getElementById('wx-temp').value = data.temperature_c.toFixed(1);
            document.getElementById('wx-precip').value = data.precipitation_mm.toFixed(1);
            document.getElementById('wx-wind').value = data.wind_speed_kmh.toFixed(1);
            document.getElementById('wx-humidity').value = data.humidity_pct.toFixed(0);
            document.getElementById('wx-conditions').value = data.conditions;
            resultEl.textContent = '✅ Fetched: ' + data.temperature_c.toFixed(1) + '°C, ' +
                data.conditions + ', pitch=' + data.pitch_state + ' (source: ' + data.source + ')';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Fetch failed';
        }
    }

    async function wxAnalyzeVideo() {
        if (!bridge) return;
        const resultEl = document.getElementById('wx-video-result');
        const videoPath = document.getElementById('wx-video-path').value.trim();
        if (!videoPath) {
            resultEl.textContent = 'Enter a video file path';
            return;
        }
        resultEl.textContent = 'Analyzing video weather (this may take 10-30 seconds)...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.classify_video_weather(videoPath));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            const rd = data.raindrop_detection;
            const wc = data.weather_classification;
            let html = '<div style="font-size:0.85rem"><strong>Combined analysis:</strong> ' + escapeHtml(data.method) + '<br>';
            if (rd) {
                html += '<strong>Raindrops:</strong> ' + rd.raindrop_count + ' detected across ' + rd.frame_count + ' frames ' +
                    '(density: ' + rd.raindrop_density.toFixed(1) + '/Mpx, conf: ' + rd.avg_confidence.toFixed(2) + ', method: ' + escapeHtml(rd.method) + ')<br>';
            }
            if (wc) {
                html += '<strong>Multi-class:</strong> ' + escapeHtml(wc.predicted_class) + ' (conf: ' + (wc.confidence * 100).toFixed(1) + '%, method: ' + escapeHtml(wc.method) + ')<br>';
                html += '<small>Probabilities: ';
                const probs = Object.entries(wc.class_probabilities).sort((a, b) => b[1] - a[1]);
                html += probs.map(([c, p]) => escapeHtml(c) + ':' + (p * 100).toFixed(0) + '%').join(', ') + '</small><br>';
            }
            html += '<strong>Final: </strong>' + (data.is_rainy ? '🌧️ Rainy' : '☀️ Not rainy');
            if (data.conditions) {
                html += ' (' + escapeHtml(data.conditions.conditions) + ', ' + escapeHtml(data.conditions.pitch_state) + ' pitch)';
            }
            html += '</div>';
            resultEl.innerHTML = html;
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Video analysis failed';
        }
    }

    // --- Psychology ---

    async function checkPsychologyStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('psy-status');
        const controls = document.getElementById('psy-controls');
        try {
            const status = JSON.parse(await bridge.check_psychology_status());
            if (status.available) {
                statusEl.textContent = '🟢 Available (momentum, score-state, late-game)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Not available';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function psyAnalyze() {
        if (!bridge) return;
        const summaryEl = document.getElementById('psy-summary');
        const momentumEl = document.getElementById('psy-momentum');
        const eventsEl = document.getElementById('psy-events');
        const matchId = parseInt(document.getElementById('psy-match-id').value) || 0;
        const home = document.getElementById('psy-home').value || 'Home';
        const away = document.getElementById('psy-away').value || 'Away';
        summaryEl.textContent = 'Analyzing...';
        const events = JSON.stringify([]);
        try {
            const data = JSON.parse(await bridge.analyze_match_psychology(matchId, home, away, events));
            if (data.error) {
                summaryEl.textContent = '❌ ' + data.error;
                return;
            }
            summaryEl.innerHTML = '<div style="font-size:0.85rem">' +
                '<strong>Score state transitions:</strong> ' + data.score_state_transitions.length + '<br>' +
                '<strong>Post-goal lulls:</strong> ' + data.post_goal_lull_count + '<br>' +
                '<strong>Comebacks:</strong> ' + data.comeback_count + '<br>' +
                '<strong>Capitulations:</strong> ' + data.capitulation_count + '<br>' +
                '<strong>Avg late-game passing drop:</strong> ' + (data.avg_late_game_passing_drop * 100).toFixed(1) + '%<br>' +
                data.notes.map(n => '• ' + escapeHtml(n)).join('<br>') + '</div>';
            summaryEl.className = 'feedback-result success';
            if (data.momentum_timeline && data.momentum_timeline.length > 0) {
                const last10 = data.momentum_timeline.slice(-10);
                momentumEl.innerHTML = '<div style="font-size:0.78rem">Momentum (last 10 windows): ' +
                    last10.map(m => 'M' + m.minute.toFixed(0) + ':' + m.home.toFixed(2)).join(' | ') + '</div>';
            } else {
                momentumEl.textContent = 'No momentum data (no events)';
            }
            if (data.psychology_events && data.psychology_events.length > 0) {
                eventsEl.innerHTML = data.psychology_events.slice(0, 10).map(e =>
                    '<div style="font-size:0.78rem;padding:2px 0">' +
                    escapeHtml(String(e.minute)) + "' " + escapeHtml(e.team) + ': <strong>' + escapeHtml(e.type) + '</strong> — ' +
                    escapeHtml(e.description) + '</div>'
                ).join('');
            } else {
                eventsEl.textContent = 'No psychology events (provide events JSON for full analysis)';
            }
        } catch (e) {
            summaryEl.textContent = '❌ Analysis failed';
        }
    }

    // --- Rules / IFAB ---

    async function checkRulesStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('rules-status');
        const controls = document.getElementById('rules-controls');
        try {
            const status = JSON.parse(await bridge.check_rules_status());
            if (status.available) {
                statusEl.textContent = '🟢 IFAB Laws loaded (' + status.laws_count + ' laws)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
                rulesLoadLaws();
            } else {
                statusEl.textContent = '🟡 Not available';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function rulesLoadLaws() {
        if (!bridge) return;
        try {
            const data = JSON.parse(await bridge.get_all_laws());
            const select = document.getElementById('rules-law-select');
            select.innerHTML = '<option value="">-- Select Law --</option>' +
                (data.laws || []).map(l =>
                    '<option value="' + l.number + '">Law ' + l.number + ': ' + escapeHtml(l.name) + '</option>'
                ).join('');
        } catch (e) {
            console.error('Failed to load laws:', e);
        }
    }

    async function rulesShowLaw() {
        if (!bridge) return;
        const num = parseInt(document.getElementById('rules-law-select').value);
        const contentEl = document.getElementById('rules-law-content');
        if (!num) {
            contentEl.textContent = 'Select a law';
            return;
        }
        try {
            const law = JSON.parse(await bridge.get_law_summary(num));
            if (law.error) {
                contentEl.textContent = '❌ ' + law.error;
                return;
            }
            contentEl.innerHTML = '<div style="font-size:0.85rem">' +
                '<strong>Law ' + law.number + ': ' + escapeHtml(law.name) + '</strong><br>' +
                escapeHtml(law.summary || '') +
                (law.key_constraints ? '<br><br><em>Key constraints:</em><ul style="margin:4px 0">' +
                    law.key_constraints.map(c => '<li style="font-size:0.78rem">' + escapeHtml(c) + '</li>').join('') + '</ul>' : '') +
                '</div>';
            contentEl.className = 'feedback-result';
        } catch (e) {
            contentEl.textContent = '❌ Failed to load law';
        }
    }

    async function rulesClassifyEvent() {
        if (!bridge) return;
        const resultEl = document.getElementById('cls-result');
        const event = document.getElementById('cls-event').value;
        const x = parseFloat(document.getElementById('cls-x').value);
        const y = parseFloat(document.getElementById('cls-y').value);
        const side = document.getElementById('cls-side').value;
        resultEl.textContent = 'Classifying...';
        try {
            const data = JSON.parse(await bridge.classify_event_rule(event, x, y, side));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            resultEl.innerHTML = '<div style="font-size:0.85rem">' +
                '<strong>Law ' + data.law + ': ' + escapeHtml(data.law_name) + '</strong><br>' +
                'Restart: ' + (data.restart || 'none') + '<br>' +
                escapeHtml(data.description) + '<br>' +
                'Card likely: ' + (data.card_likely || 'none') + '</div>';
            resultEl.className = 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Classification failed';
        }
    }

    async function rulesCheckOffside() {
        if (!bridge) return;
        const resultEl = document.getElementById('off-result');
        const ax = parseFloat(document.getElementById('off-attacker').value);
        const dx = parseFloat(document.getElementById('off-defender').value);
        const bx = parseFloat(document.getElementById('off-ball').value);
        resultEl.textContent = 'Checking...';
        try {
            const data = JSON.parse(await bridge.check_offside(ax, dx, bx, 'right'));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                return;
            }
            resultEl.innerHTML = '<div style="font-size:0.85rem">' +
                (data.is_offside ? '🚩 <strong>OFFSIDE</strong>' : '✅ <strong>Onside</strong>') + '<br>' +
                escapeHtml(data.explanation) + '</div>';
            resultEl.className = data.is_offside ? 'feedback-result error' : 'feedback-result success';
        } catch (e) {
            resultEl.textContent = '❌ Check failed';
        }
    }

    // --- Cards ---

    async function checkCardsStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('cards-status');
        const controls = document.getElementById('cards-controls');
        try {
            const status = JSON.parse(await bridge.check_cards_status());
            if (status.available) {
                statusEl.textContent = '🟢 Available (visual + audio + tactical + external)';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Not available';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function cardsInferTactical() {
        if (!bridge) return;
        const listEl = document.getElementById('cards-list');
        const events = [
            { type: 'foul', team: 'home', minute: 23, second: 12, x: 92, severity: 0.7, player_track_id: 5, player_name: 'Defender A' },
            { type: 'foul', team: 'away', minute: 45, second: 30, x: 12, severity: 0.8, player_track_id: 12, player_name: 'Midfielder B' },
            { type: 'foul', team: 'home', minute: 78, second: 5, x: 88, severity: 0.65, player_track_id: 7, player_name: 'Defender C' },
        ];
        listEl.innerHTML = '<div class="roster-item">Inferring...</div>';
        try {
            const data = JSON.parse(await bridge.infer_cards_tactically(JSON.stringify(events)));
            if (data.error) {
                listEl.innerHTML = '<div class="roster-item">❌ ' + escapeHtml(data.error) + '</div>';
                return;
            }
            const cards = data.cards || [];
            if (cards.length === 0) {
                listEl.innerHTML = '<div class="roster-item">No cards inferred</div>';
                return;
            }
            listEl.innerHTML = cards.map(c =>
                '<div class="roster-item" style="font-size:0.85rem">' +
                '<strong style="color:' + (c.card_type === 'red' ? '#dc2626' : '#eab308') + '">[' + c.card_type.toUpperCase() + ']</strong> ' +
                c.minute + "' " + escapeHtml(c.team) + ' — ' + escapeHtml(c.player_name) +
                ' <span style="color:var(--text-muted);font-size:0.75rem">(' + c.source + ', ' + (c.confidence * 100).toFixed(0) + '%)</span>' +
                '<div style="font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(c.description) + '</div>' +
                '</div>'
            ).join('');
        } catch (e) {
            listEl.innerHTML = '<div class="roster-item">❌ Failed</div>';
        }
    }

    async function cardsFetchExternal() {
        if (!bridge) return;
        const listEl = document.getElementById('cards-list');
        const matchIdInput = document.getElementById('cards-match-id');
        const matchId = parseInt(matchIdInput.value) || 0;
        if (!matchId) {
            listEl.innerHTML = '<div class="roster-item">Enter a Match ID first</div>';
            return;
        }
        listEl.innerHTML = '<div class="roster-item">Fetching from external...</div>';
        try {
            const data = JSON.parse(await bridge.fetch_external_cards(matchId));
            if (data.error) {
                listEl.innerHTML = '<div class="roster-item">❌ ' + escapeHtml(data.error) + '</div>';
                return;
            }
            const cards = data.cards || [];
            if (cards.length === 0) {
                listEl.innerHTML = '<div class="roster-item">No external cards for match ' + matchId + ' (not in StatsBomb or service unavailable)</div>';
                return;
            }
            listEl.innerHTML = cards.map(c =>
                '<div class="roster-item" style="font-size:0.85rem">' +
                '<strong style="color:' + (c.card_type === 'red' ? '#dc2626' : '#eab308') + '">[' + c.card_type.toUpperCase() + ']</strong> ' +
                c.minute + "' " + escapeHtml(c.team) + ' — ' + escapeHtml(c.player_name) +
                ' <span style="color:var(--text-muted);font-size:0.75rem">(verified)</span></div>'
            ).join('');
        } catch (e) {
            listEl.innerHTML = '<div class="roster-item">❌ Failed</div>';
        }
    }

    // --- FluidX3D / CFD ---

    async function checkFluidX3DStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('fluidx3d-status');
        const controls = document.getElementById('fluidx3d-controls');
        try {
            const status = JSON.parse(await bridge.check_fluidx3d_status());
            if (status.available) {
                statusEl.textContent = '🟢 FluidX3D binary configured';
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.innerHTML = '🟡 Binary not configured<br><small style="color:var(--text-muted)">' +
                    escapeHtml(status.license_notice || '') + '</small>';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function cfdRun() {
        if (!bridge) return;
        const wind = parseFloat(document.getElementById('cfd-wind').value);
        const spin = parseFloat(document.getElementById('cfd-spin').value);
        const radius = parseFloat(document.getElementById('cfd-radius').value);
        const resultEl = document.getElementById('cfd-result');
        resultEl.textContent = 'Running CFD simulation...';
        resultEl.className = 'feedback-result';
        try {
            const data = JSON.parse(await bridge.simulate_ball_cfd(wind, spin, radius));
            resultEl.innerHTML = '<div style="font-size:0.85rem">' +
                (data.success ? '✅ ' : '❌ ') + escapeHtml(data.notes || '') +
                (data.error ? '<br><small style="color:var(--text-muted)">' + escapeHtml(data.error) + '</small>' : '') +
                '</div>';
            resultEl.className = data.success ? 'feedback-result success' : 'feedback-result error';
        } catch (e) {
            resultEl.textContent = '❌ CFD failed';
            resultEl.className = 'feedback-result error';
        }
    }

    // --- Roboflow Sports (RF) ---

    async function checkRoboflowSportsStatus() {
        if (!bridge) return;
        const statusEl = document.getElementById('roboflow-sports-status');
        const controls = document.getElementById('roboflow-sports-controls');
        try {
            const status = JSON.parse(await bridge.check_roboflow_sports_status());
            if (status.available) {
                let info = '🟢 Available';
                if (status.has_team_classifier) info += ' (team classifier ✅)';
                if (status.has_view_transformer) info += ' (view transformer ✅)';
                statusEl.textContent = info;
                statusEl.className = 'feedback-result success';
                controls.classList.remove('hidden');
            } else {
                statusEl.textContent = '🟡 Not installed (pip install git+https://github.com/roboflow/sports.git)';
                statusEl.className = 'feedback-result';
                controls.classList.add('hidden');
            }
        } catch (e) {
            statusEl.textContent = '🟡 Not available';
            statusEl.className = 'feedback-result';
            controls.classList.add('hidden');
        }
    }

    async function rfDrawPitch() {
        if (!bridge) return;
        const scaleInput = document.getElementById('rf-pitch-scale');
        const resultEl = document.getElementById('rf-pitch-result');
        const imgEl = document.getElementById('rf-pitch-image');
        const scale = parseFloat(scaleInput.value) || 0.5;
        resultEl.textContent = 'Drawing pitch...';
        resultEl.className = 'feedback-result';
        imgEl.innerHTML = '';
        try {
            const data = JSON.parse(await bridge.rf_draw_pitch(scale));
            if (data.error) {
                resultEl.textContent = '❌ ' + data.error;
                resultEl.className = 'feedback-result error';
                return;
            }
            resultEl.textContent = '✅ Pitch drawn (' + data.shape[0] + 'x' + data.shape[1] + ')';
            resultEl.className = 'feedback-result success';
            imgEl.innerHTML = '<img src="data:image/png;base64,' + escapeHtml(data.image_b64) + '" style="max-width:100%;border:1px solid #ccc;border-radius:6px">';
        } catch (e) {
            resultEl.textContent = '❌ Failed to draw pitch';
            resultEl.className = 'feedback-result error';
        }
    }

    // ── Exported boot helpers ──────────────────────────────────────────

    /** Called from app.js initQWebChannel callback after bridge connects */
    window.__kawkab.checkAllDataProviderStatuses = function() {
        if (!bridge) { bridge = window.__kawkab.bridge; if (!bridge) return; }
        checkFootballDataStatus();
        checkBzzoiroStatus();
        checkEasySoccerStatus();
        checkApiFootballStatus();
        checkTheSportsDBStatus();
        checkStatsBombStatus();
        checkOpenFootballStatus();
        checkRoboflowSportsStatus();
        checkPoseStatus();
        checkMuJoCoStatus();
        checkFluidX3DStatus();
        checkWeatherStatus();
        checkPsychologyStatus();
        checkRulesStatus();
        checkCardsStatus();
    };

    // ── Data provider event listener registration ─────────────────────

    window.__kawkab.setupDataProviderListeners = function() {
        // Football-Data.org
        var fdHomeSearch = document.getElementById('fd-home-search');
        var fdAwaySearch = document.getElementById('fd-away-search');
        if (fdHomeSearch) setupFdTeamSearch('fd-home-search', 'fd-home-results', function(id, name, crest, compCode) {
            fdHomeTeamId = id;
            document.getElementById('fd-import-home-btn').disabled = false;
            document.getElementById('fd-import-home-btn').textContent = 'Import Home Squad';
            fdLoadTeamFixtures(id);
            if (compCode) { fdLoadStandings(compCode); }
        });
        if (fdAwaySearch) setupFdTeamSearch('fd-away-search', 'fd-away-results', function(id, name, crest, compCode) {
            fdAwayTeamId = id;
            document.getElementById('fd-import-away-btn').disabled = false;
            document.getElementById('fd-import-away-btn').textContent = 'Import Away Squad';
            fdLoadTeamFixtures(id);
            if (compCode) { fdLoadStandings(compCode); }
        });
        document.getElementById('fd-import-home-btn')?.addEventListener('click', function() {
            if (window.__kawkab.currentMatchId && fdHomeTeamId) fdImportSquad(window.__kawkab.currentMatchId, fdHomeTeamId, 'home', 'fd-import-home-btn');
        });
        document.getElementById('fd-import-away-btn')?.addEventListener('click', function() {
            if (window.__kawkab.currentMatchId && fdAwayTeamId) fdImportSquad(window.__kawkab.currentMatchId, fdAwayTeamId, 'away', 'fd-import-away-btn');
        });
        document.getElementById('fd-verify-btn')?.addEventListener('click', fdVerifyMatch);

        // Bzzoiro
        var bzHomeSearch = document.getElementById('bz-home-search');
        var bzAwaySearch = document.getElementById('bz-away-search');
        if (bzHomeSearch) setupBzTeamSearch('bz-home-search', 'bz-home-results', function(id, name) {
            bzHomeTeamId = id;
            document.getElementById('bz-import-home-btn').disabled = false;
            document.getElementById('bz-import-home-btn').textContent = 'Import Home Squad';
        });
        if (bzAwaySearch) setupBzTeamSearch('bz-away-search', 'bz-away-results', function(id, name) {
            bzAwayTeamId = id;
            document.getElementById('bz-import-away-btn').disabled = false;
            document.getElementById('bz-import-away-btn').textContent = 'Import Away Squad';
        });
        document.getElementById('bz-import-home-btn')?.addEventListener('click', function() {
            if (window.__kawkab.currentMatchId && bzHomeTeamId) bzImportSquad(window.__kawkab.currentMatchId, bzHomeTeamId, 'home', 'bz-import-home-btn');
        });
        document.getElementById('bz-import-away-btn')?.addEventListener('click', function() {
            if (window.__kawkab.currentMatchId && bzAwayTeamId) bzImportSquad(window.__kawkab.currentMatchId, bzAwayTeamId, 'away', 'bz-import-away-btn');
        });
        document.getElementById('bz-verify-btn')?.addEventListener('click', bzVerifyMatch);
        document.getElementById('bz-predict-btn')?.addEventListener('click', bzGetPredictions);
        document.getElementById('bz-standings-btn')?.addEventListener('click', bzLoadStandings);

        // EasySoccerData
        document.getElementById('es-get-event-btn')?.addEventListener('click', esGetEvent);
        document.getElementById('es-get-incidents-btn')?.addEventListener('click', esGetIncidents);

        // API-Football
        var afHomeSearch = document.getElementById('af-home-search');
        var afAwaySearch = document.getElementById('af-away-search');
        if (afHomeSearch) setupAfTeamSearch('af-home-search', 'af-home-results', function(id, name) {
            afHomeTeamId = id;
            document.getElementById('af-import-home-btn').disabled = false;
            document.getElementById('af-import-home-btn').textContent = 'Import Home Squad';
        });
        if (afAwaySearch) setupAfTeamSearch('af-away-search', 'af-away-results', function(id, name) {
            afAwayTeamId = id;
            document.getElementById('af-import-away-btn').disabled = false;
            document.getElementById('af-import-away-btn').textContent = 'Import Away Squad';
        });
        document.getElementById('af-import-home-btn')?.addEventListener('click', function() {
            if (window.__kawkab.currentMatchId && afHomeTeamId) afImportSquad(window.__kawkab.currentMatchId, afHomeTeamId, 'home', 'af-import-home-btn');
        });
        document.getElementById('af-import-away-btn')?.addEventListener('click', function() {
            if (window.__kawkab.currentMatchId && afAwayTeamId) afImportSquad(window.__kawkab.currentMatchId, afAwayTeamId, 'away', 'af-import-away-btn');
        });
        document.getElementById('af-verify-btn')?.addEventListener('click', afVerifyMatch);
        document.getElementById('af-predict-btn')?.addEventListener('click', afGetPredictions);

        // TheSportsDB
        var tsdbSearch = document.getElementById('tsdb-team-search');
        if (tsdbSearch) setupTsdbTeamSearch('tsdb-team-search', 'tsdb-team-results', function(id, name, leagueId, badge) {
            window._tsdbSelectedTeamId = id;
            document.getElementById('tsdb-get-info-btn').disabled = false;
        });
        document.getElementById('tsdb-get-info-btn')?.addEventListener('click', tsdbGetTeamInfo);
        document.getElementById('tsdb-standings-btn')?.addEventListener('click', tsdbLoadStandings);
        document.getElementById('tsdb-last-events-btn')?.addEventListener('click', function() { tsdbLoadEvents('last'); });
        document.getElementById('tsdb-next-events-btn')?.addEventListener('click', function() { tsdbLoadEvents('next'); });

        // StatsBomb
        var sbTeamSearch = document.getElementById('sb-team-search');
        if (sbTeamSearch) {
            sbTeamSearch.addEventListener('input', function() { sbSearchTeam(); });
        }
        document.getElementById('sb-events-btn')?.addEventListener('click', sbGetEvents);
        document.getElementById('sb-lineups-btn')?.addEventListener('click', sbGetLineups);
        document.getElementById('sb-import-btn')?.addEventListener('click', sbImportToDb);

        // OpenFootball
        document.getElementById('ofb-load-btn')?.addEventListener('click', ofbLoadMatches);
        var ofbTeamSearch = document.getElementById('ofb-team-search');
        if (ofbTeamSearch) {
            ofbTeamSearch.addEventListener('input', function() { ofbSearchTeam(); });
        }
        document.getElementById('ofb-wc-btn')?.addEventListener('click', ofbLoadWorldCup);

        // Roboflow Sports
        document.getElementById('rf-draw-pitch-btn')?.addEventListener('click', rfDrawPitch);

        // Pose Analysis
        document.getElementById('pose-summary-btn')?.addEventListener('click', poseGetSummary);
        document.getElementById('pose-segments-btn')?.addEventListener('click', poseGetSegments);

        // MuJoCo Trajectory
        document.getElementById('mujoco-load-preset-btn')?.addEventListener('click', mujocoApplyPreset);
        document.getElementById('mujoco-simulate-btn')?.addEventListener('click', mujocoSimulate);

        // FluidX3D
        document.getElementById('cfd-run-btn')?.addEventListener('click', cfdRun);

        // Weather
        document.getElementById('wx-analyze-btn')?.addEventListener('click', wxAnalyzeManual);
        document.getElementById('wx-fetch-btn')?.addEventListener('click', wxFetchFromAPI);
        document.getElementById('wx-analyze-video-btn')?.addEventListener('click', wxAnalyzeVideo);

        // Psychology
        document.getElementById('psy-analyze-btn')?.addEventListener('click', psyAnalyze);

        // Football Rules
        document.getElementById('rules-show-btn')?.addEventListener('click', rulesShowLaw);
        document.getElementById('cls-classify-btn')?.addEventListener('click', rulesClassifyEvent);
        document.getElementById('off-check-btn')?.addEventListener('click', rulesCheckOffside);

        // Card Detection
        document.getElementById('cards-infer-btn')?.addEventListener('click', cardsInferTactical);
        document.getElementById('cards-external-btn')?.addEventListener('click', cardsFetchExternal);
    };
})();
