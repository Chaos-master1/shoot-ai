    // ── Phase 13 — Opponent Database + Scouting Network + Transfermarkt ──

    // Not wrapped in its own IIFE (split out of app.js without one) -- the
    // bundler is a plain concatenation, so this and every escapeHtml(...)
    // call below resolved to nothing at runtime (ReferenceError) until this
    // import was added. Matches app-data-providers.js's own fix for the
    // same gap.
    var escapeHtml = window.__kawkab.escapeHtml;

    function initOpponentWorkspace() {
        // Tab switching
        function switchOppTab(tabId) {
            document.querySelectorAll('#opponent-tabs .tab').forEach(function(t) { t.classList.remove('active'); });
            document.querySelectorAll('#opponent-section .tab-content').forEach(function(c) { c.classList.add('hidden'); });
            var tab = document.querySelector('#opponent-tabs .tab[data-tab="' + tabId + '"]');
            if (tab) tab.classList.add('active');
            var content = document.getElementById(tabId);
            if (content) content.classList.remove('hidden');
        }

        document.querySelectorAll('#opponent-tabs .tab').forEach(function(tab) {
            tab.addEventListener('click', function() { switchOppTab(this.dataset.tab); });
        });

        // === Opponent Profiles ===
        function loadOpponents() {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.opponent_list(function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (!data.success) return;
                    var list = document.getElementById('opp-list');
                    list.innerHTML = '';
                    (data.profiles || []).forEach(function(p) {
                        var card = document.createElement('div');
                        card.className = 'pro-card opp-card';
                        card.style.cursor = 'pointer';
                        card.innerHTML = '<div style="font-weight:700">' + escapeHtml(p.team_name) + '</div>' +
                            '<div style="font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(p.league || 'N/A') +
                            ' | ' + escapeHtml(p.formation) + ' | Pressing: ' + escapeHtml(p.pressing_style) +
                            ' | ' + p.matches + ' matches</div>';
                        card.addEventListener('click', function() { loadOpponentDetail(p.id); });
                        list.appendChild(card);
                    });
                    if (data.profiles.length === 0) {
                        list.innerHTML = '<p class="hint">No opponents yet. Create your first opponent profile.</p>';
                    }
                } catch(e) {}
            });
        }

        function loadOpponentDetail(profileId) {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.opponent_get(profileId, function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (!data.success) return;
                    var p = data.profile;
                    document.getElementById('opp-detail').classList.remove('hidden');
                    document.getElementById('opp-detail-name').textContent = p.team_name;
                    document.getElementById('opp-detail-tactical').innerHTML =
                        '<p><strong>Formations:</strong> ' + escapeHtml((p.formation_tendencies || []).join(', ') || 'N/A') + '</p>' +
                        '<p><strong>Pressing:</strong> ' + escapeHtml(p.pressing_style || 'N/A') + '</p>' +
                        '<p><strong>Attack:</strong> ' + escapeHtml((p.attacking_patterns || []).join(', ') || 'N/A') + '</p>' +
                        '<p><strong>Defensive gaps:</strong> ' + escapeHtml((p.defensive_vulnerabilities || []).join(', ') || 'N/A') + '</p>' +
                        '<p><strong>Set pieces:</strong> ' + escapeHtml((p.set_piece_routines || []).join(', ') || 'N/A') + '</p>';
                    var mu = document.getElementById('opp-matchups');
                    mu.innerHTML = '';
                    (data.matchups || []).forEach(function(m) {
                        mu.innerHTML += '<div style="font-size:0.75rem;padding:4px 0;border-bottom:1px solid var(--border)">' +
                            escapeHtml(m.date) + ' — ' + escapeHtml(m.score) + ' (xG: ' + m.our_xg + '-' + m.their_xg + ')' +
                            '</div>';
                    });
                    if (!data.matchups || data.matchups.length === 0) {
                        mu.innerHTML = '<p style="font-size:0.8rem;color:var(--text-muted)">No matchups recorded yet.</p>';
                    }
                    document.getElementById('opp-generate-report-btn').onclick = function() {
                        bridge.opponent_scouting_report(profileId, function(r2) {
                            try { var d2 = JSON.parse(r2); document.getElementById('opp-scouting-report').textContent = d2.report || 'Error'; } catch(e) {}
                        });
                    };
                    document.getElementById('opp-delete-btn').onclick = function() {
                        if (!confirm('Delete opponent profile?')) return;
                        bridge.opponent_delete(profileId, function() {
                            document.getElementById('opp-detail').classList.add('hidden');
                            loadOpponents();
                            showToast('Deleted', 'info');
                        });
                    };
                } catch(e) {}
            });
        }

        document.getElementById('opp-refresh-btn').onclick = loadOpponents;
        document.getElementById('opp-create-btn').onclick = function() {
            var name = prompt('Opponent team name:');
            if (!name) return;
            var league = prompt('League (optional):') || '';
            var country = prompt('Country (optional):') || '';
            bridge.opponent_create(name, league, country, function() {
                loadOpponents();
                showToast('Opponent created: ' + name, 'success');
            });
        };

        // === Scouting Network ===
        function searchScoutNetwork() {
            var query = document.getElementById('scout-net-search').value;
            var position = document.getElementById('scout-net-position').value;
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.scout_network_search(query, position, '0', '99', '', '0', function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (!data.success) return;
                    var container = document.getElementById('scout-net-results');
                    container.innerHTML = '';
                    (data.players || []).forEach(function(p) {
                        var card = document.createElement('div');
                        card.className = 'pro-card';
                        card.style.margin = '4px 0';
                        card.innerHTML = '<div style="display:flex;justify-content:space-between">' +
                            '<div><strong>' + escapeHtml(p.name) + '</strong> — ' + escapeHtml(p.position || 'N/A') +
                            ' | ' + escapeHtml(p.club || 'N/A') + ' | Age: ' + p.age + '</div>' +
                            '<div><span class="badge" style="background:var(--primary)">' + p.rating + '/10</span></div></div>' +
                            '<div style="font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(p.league || '') +
                            (p.estimated_value ? ' | €' + (p.estimated_value/1e6).toFixed(1) + 'M' : '') +
                            (p.strengths && p.strengths.length ? ' | Strengths: ' + escapeHtml(p.strengths.join(', ')) : '') + '</div>';
                        container.appendChild(card);
                    });
                    if (!data.players || data.players.length === 0) {
                        container.innerHTML = '<p class="hint">No players found. Add a player or broaden your search.</p>';
                    }
                } catch(e) {}
            });
        }

        document.getElementById('scout-net-search-btn').onclick = searchScoutNetwork;
        document.getElementById('scout-net-add-btn').onclick = function() {
            var name = prompt('Player name:');
            if (!name) return;
            var position = prompt('Position:') || '';
            var club = prompt('Club:') || '';
            var league = prompt('League:') || '';
            var rating = prompt('Rating (0-10):') || '5';
            bridge.scout_network_add(name, position, club, league, rating, '[]', '[]', '', '', '[]', function() {
                showToast('Player added: ' + name, 'success');
                searchScoutNetwork();
            });
        };

        if (typeof bridge !== 'undefined' && bridge) {
            bridge.scout_network_stats(function(result) {
                try {
                    var data = JSON.parse(result);
                    if (data.success && data.stats) {
                        document.getElementById('scout-net-stats').textContent =
                            data.stats.total + ' players, avg rating: ' + data.stats.avg_rating;
                    }
                } catch(e) {}
            });
        }

        // === Transfermarkt ===
        document.getElementById('tm-search-btn').onclick = function() {
            var name = document.getElementById('tm-search-input').value.trim();
            if (!name) { showToast('Enter a player name.', 'warning'); return; }
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.transfermarkt_search(name, function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (!data.success) return;
                    var container = document.getElementById('tm-results');
                    container.innerHTML = '';
                    (data.results || []).forEach(function(p) {
                        var card = document.createElement('div');
                        card.className = 'pro-card';
                        card.style.cursor = 'pointer';
                        card.style.margin = '4px 0';
                        card.innerHTML = '<div style="display:flex;justify-content:space-between">' +
                            '<div><strong>' + escapeHtml(p.name) + '</strong> — ' + escapeHtml(p.position || 'N/A') +
                            ' | ' + escapeHtml(p.club || 'N/A') + '</div>' +
                            '<div>€' + (p.market_value/1e6).toFixed(1) + 'M</div></div>' +
                            '<div style="font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(p.league || '') + ' | Age: ' + p.age + ' | ' + escapeHtml(p.nationality || '') + '</div>';
                        card.addEventListener('click', function() {
                            bridge.transfermarkt_get(String(p.id), function(r2) {
                                try {
                                    var d2 = JSON.parse(r2);
                                    if (!d2.success) return;
                                    var det = d2.details;
                                    document.getElementById('tm-detail').classList.remove('hidden');
                                    document.getElementById('tm-player-name').textContent = det.name;
                                    var html = '<p><strong>Position:</strong> ' + escapeHtml(det.position) + '</p>' +
                                        '<p><strong>Club:</strong> ' + escapeHtml(det.club) + '</p>' +
                                        '<p><strong>League:</strong> ' + escapeHtml(det.league) + '</p>' +
                                        '<p><strong>Market Value:</strong> €' + (det.market_value/1e6).toFixed(1) + 'M</p>' +
                                        '<p><strong>Age:</strong> ' + det.age + '</p>' +
                                        '<p><strong>Height:</strong> ' + (det.height_cm || 'N/A') + ' cm</p>' +
                                        '<p><strong>Foot:</strong> ' + escapeHtml(det.foot || 'N/A') + '</p>' +
                                        '<p><strong>Contract:</strong> ' + escapeHtml(det.contract_until || 'N/A') + '</p>';
                                    if (det.stats) {
                                        html += '<h4>Stats</h4><p>Apps: ' + det.stats.appearances + ' | Goals: ' + det.stats.goals + ' | Assists: ' + det.stats.assists + '</p>';
                                    }
                                    document.getElementById('tm-player-details').innerHTML = html;
                                } catch(e) {}
                            });
                        });
                        container.appendChild(card);
                    });
                    if (!data.results || data.results.length === 0) {
                        container.innerHTML = '<p class="hint">No results found.</p>';
                    }
                } catch(e) {}
            });
        };

        loadOpponents();
    }

