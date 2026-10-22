// Kawkab AI - Scout Portal (Enhanced: FIFA-style cards, transfer value, strengths/weaknesses)
// Extracted from app.js for modularity

(function() { 'use strict';

    var _scoutState = {
        searchResults: [],
        shortlist: [],
        compareA: null,
        compareB: null,
        detailView: false,
        selectedPlayer: null,
    };

    function initScoutPortal() {
        var tabs = document.querySelectorAll('.scout-tabs .tab-btn');
        tabs.forEach(function(tab) {
            tab.addEventListener('click', function() {
                tabs.forEach(function(t) { t.classList.remove('active'); });
                this.classList.add('active');
                document.querySelectorAll('.scout-tab-content').forEach(function(c) { c.classList.remove('active'); });
                var target = document.getElementById('scout-' + this.dataset.stab);
                if (target) target.classList.add('active');
            });
        });

        var searchBtn = document.getElementById('scout-search-btn');
        var searchInput = document.getElementById('scout-search-input');
        var compareBtn = document.getElementById('scout-compare-btn');
        var reportBtn = document.getElementById('scout-report-btn');
        var refreshBtn = document.getElementById('scout-shortlist-refresh');
        var detailToggle = document.getElementById('scout-detail-toggle');
        var clearSearch = document.getElementById('scout-clear-search');

        if (!searchBtn) return;

        searchBtn.addEventListener('click', function() {
            var query = searchInput.value.trim();
            var pos = document.getElementById('scout-position-input').value.trim();
            if (!query) { showToast('Enter a player name.', 'warning'); return; }
            scoutSearch(query, pos);
        });

        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') searchBtn.click();
        });

        if (clearSearch) {
            clearSearch.addEventListener('click', function() {
                searchInput.value = '';
                _scoutState.searchResults = [];
                document.getElementById('scout-search-results').innerHTML = '<p class="hint">Enter a player name to search.</p>';
            });
        }

        if (detailToggle) {
            detailToggle.addEventListener('click', function() {
                _scoutState.detailView = !_scoutState.detailView;
                this.textContent = _scoutState.detailView ? '📋 List View' : '🃏 Card View';
                renderScoutResults();
            });
        }

        compareBtn.addEventListener('click', function() {
            var a = document.getElementById('scout-compare-a').value;
            var b = document.getElementById('scout-compare-b').value;
            if (!a || !b) { showToast('Select two players to compare.', 'warning'); return; }
            if (a === b) { showToast('Select two different players.', 'warning'); return; }
            scoutCompare(a, b);
        });

        reportBtn.addEventListener('click', function() {
            generateScoutReport();
        });

        refreshBtn.addEventListener('click', loadShortlist);

        loadShortlist();
        populateCompareSelects();
    }

    function scoutSearch(query, position) {
        if (typeof bridge === 'undefined' || !bridge) {
            useMockData(query, position);
            return;
        }
        bridge.scout_search_players(query, position, function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                _scoutState.searchResults = data.results || data.players || (Array.isArray(data) ? data : []);
                renderScoutResults();
            } catch(e) { showToast('Scout search failed.', 'error'); console.warn('Scout search error:', e); }
        });

        // Also search external APIs for richer data
        bridge.search_external_player(query, position, function(result) {
            try {
                var external = typeof result === 'string' ? JSON.parse(result) : result;
                if (external && external.players) {
                    var existingIds = {};
                    _scoutState.searchResults.forEach(function(p) { existingIds[p.track_id || p.id] = true; });
                    external.players.forEach(function(p) {
                        var pid = p.track_id || p.id || p.player_id;
                        if (!existingIds[pid]) {
                            _scoutState.searchResults.push(p);
                            existingIds[pid] = true;
                        }
                    });
                    renderScoutResults();
                }
            } catch(e) { /* silent - external search is best-effort */ }
        });
    }

    function useMockData(query, position) {
        var mockPlayers = [
            { track_id: 1, name: 'Erling Haaland', position: 'FW', team: 'Manchester City', age: 22, matches: 28, goals: 32, assists: 5, xg: 28.5, passes: 412, tackles: 12, nation: 'NO', rating: 91, pac: 89, sho: 94, pas: 66, dri: 80, def: 35, phy: 88, strengths: ['Finishing', 'Positioning', 'Strength'], weaknesses: ['Passing under pressure'], transfer_value: 180 },
            { track_id: 2, name: 'Kevin De Bruyne', position: 'MF', team: 'Manchester City', age: 30, matches: 25, goals: 8, assists: 16, xg: 7.2, passes: 1250, tackles: 34, nation: 'BE', rating: 91, pac: 72, sho: 82, pas: 93, dri: 84, def: 62, phy: 78, strengths: ['Passing', 'Vision', 'Long shots'], weaknesses: ['Pace'], transfer_value: 80 },
            { track_id: 3, name: 'Virgil van Dijk', position: 'DF', team: 'Liverpool', age: 31, matches: 30, goals: 3, assists: 2, xg: 2.8, passes: 1800, tackles: 45, nation: 'NL', rating: 89, pac: 71, sho: 60, pas: 72, dri: 66, def: 92, phy: 86, strengths: ['Aerial duels', 'Positioning', 'Leadership'], weaknesses: ['Pace decline'], transfer_value: 45 },
            { track_id: 4, name: 'Kylian Mbappé', position: 'FW', team: 'PSG', age: 24, matches: 26, goals: 28, assists: 8, xg: 24.1, passes: 380, tackles: 8, nation: 'FR', rating: 91, pac: 97, sho: 89, pas: 72, dri: 92, def: 33, phy: 82, strengths: ['Pace', 'Dribbling', 'Finishing'], weaknesses: ['Defensive contribution'], transfer_value: 200 },
            { track_id: 5, name: 'Jude Bellingham', position: 'MF', team: 'Real Madrid', age: 20, matches: 27, goals: 15, assists: 7, xg: 12.8, passes: 890, tackles: 42, nation: 'EN', rating: 89, pac: 78, sho: 82, pas: 80, dri: 86, def: 72, phy: 84, strengths: ['Box-to-box', 'Dribbling', 'Work rate'], weaknesses: ['Experience'], transfer_value: 150 },
        ];
        var q = query.toLowerCase();
        _scoutState.searchResults = mockPlayers.filter(function(p) {
            var nameMatch = (p.name || '').toLowerCase().indexOf(q) >= 0;
            var posMatch = !position || (p.position || '').toLowerCase().indexOf(position.toLowerCase()) >= 0;
            return nameMatch && posMatch;
        });
        renderScoutResults();
    }

    function renderScoutResults() {
        var container = document.getElementById('scout-search-results');
        if (!container) return;
        var results = _scoutState.searchResults;
        var detailToggle = document.getElementById('scout-detail-toggle');

        if (!results || results.length === 0) {
            container.innerHTML = '<p class="hint">No players found. Try a different search.</p>';
            return;
        }

        var html = '';
        if (_scoutState.detailView) {
            html = renderDetailCards(results);
        } else {
            html = renderCompactList(results);
        }
        container.innerHTML = html;

        // Wire up shortlist buttons
        container.querySelectorAll('[data-action="shortlist"]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var trackId = parseInt(this.dataset.trackId, 10);
                var name = this.dataset.name;
                var pos = this.dataset.pos;
                toggleShortlist(trackId, name, pos, this);
            });
        });

        container.querySelectorAll('[data-action="compare"]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var trackId = this.dataset.trackId;
                var name = this.dataset.name;
                addToCompare(trackId, name);
                document.querySelector('.scout-tabs [data-stab="compare"]').click();
            });
        });

        container.querySelectorAll('[data-action="detail"]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var trackId = parseInt(this.dataset.trackId, 10);
                var player = results.find(function(p) { return (p.track_id || p.id) == trackId; });
                if (player) showPlayerDetail(player);
            });
        });

        container.querySelectorAll('.scout-card-view-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                _scoutState.detailView = !_scoutState.detailView;
                if (detailToggle) detailToggle.textContent = _scoutState.detailView ? '📋 List View' : '🃏 Card View';
                renderScoutResults();
            });
        });
    }

    function renderCompactList(results) {
        var html = '';
        results.forEach(function(p) {
            var initial = (p.name || '?').charAt(0).toUpperCase();
            var tid = p.track_id || p.id || p.player_id;
            var onShortlist = _scoutState.shortlist.some(function(s) { return s.track_id === tid; });
            var val = p.transfer_value || p.estimated_value || '';
            var valStr = val ? '€' + (val >= 100 ? (val/100).toFixed(1) + 'M' : val + 'M') : '';

            html += '<div class="scout-player-card">' +
                '<div class="scout-player-avatar">' + (p.nation ? getFlagEmoji(p.nation) : initial) + '</div>' +
                '<div class="scout-player-info">' +
                '<div class="scout-player-name">' + escapeHtml(p.name || 'Unknown') +
                (p.rating ? ' <span class="scout-rating-badge">' + p.rating + '</span>' : '') +
                '</div>' +
                '<div class="scout-player-meta">' +
                '<span>' + escapeHtml(p.position || '--') + '</span>' +
                '<span>' + escapeHtml(p.team || '--') + '</span>' +
                '<span>Age: ' + (p.age || '--') + '</span>' +
                (valStr ? '<span class="scout-value">' + valStr + '</span>' : '') +
                '</div></div>' +
                '<div class="scout-player-stats">' +
                '<div class="stat"><div class="stat-val">' + (p.goals != null ? p.goals : '--') + '</div><div class="stat-label">G</div></div>' +
                '<div class="stat"><div class="stat-val">' + (p.assists != null ? p.assists : '--') + '</div><div class="stat-label">A</div></div>' +
                '<div class="stat"><div class="stat-val">' + (p.matches || '--') + '</div><div class="stat-label">M</div></div>' +
                '</div>' +
                '<div class="scout-icons">' +
                '<button class="scout-icon scout-detail-btn" data-action="detail" data-track-id="' + tid + '" title="View details">👤</button>' +
                '<button class="scout-icon ' + (onShortlist ? 'added' : '') + '" data-action="shortlist" data-track-id="' + tid + '" data-name="' + escapeHtml(p.name) + '" data-pos="' + escapeHtml(p.position || '') + '">' + (onShortlist ? '★' : '☆') + '</button>' +
                '<button class="scout-icon" data-action="compare" data-track-id="' + tid + '" data-name="' + escapeHtml(p.name) + '">⇄</button>' +
                '</div></div>';
        });
        return html;
    }

    function renderDetailCards(results) {
        var html = '<div class="scout-card-grid">';
        results.forEach(function(p) {
            var tid = p.track_id || p.id || p.player_id;
            var onShortlist = _scoutState.shortlist.some(function(s) { return s.track_id === tid; });
            var val = p.transfer_value || p.estimated_value || '';
            var valStr = val ? '€' + (val >= 100 ? (val/100).toFixed(1) + 'M' : val + 'M') : '€--';

            html += '<div class="fut-card" data-player-id="' + tid + '">';
            html += '<div class="fut-card-bg" style="background:linear-gradient(135deg,' + getTeamColor(p.team) + '30,' + getTeamColor(p.team, true) + '80)">';
            html += '<div class="fut-card-nation">' + (p.nation ? getFlagEmoji(p.nation) : '🌍') + '</div>';
            html += '<div class="fut-card-rating">' + (p.rating || 75) + '</div>';
            html += '<div class="fut-card-pos">' + escapeHtml(p.position || '--') + '</div>';
            html += '<div class="fut-card-photo">' + (p.photo_url ? '<img src="' + p.photo_url + '" alt="">' : '<div class="fut-card-silhouette">' + (p.name ? p.name.charAt(0).toUpperCase() : '?') + '</div>') + '</div>';
            html += '<div class="fut-card-name">' + escapeHtml(p.name || 'Unknown') + '</div>';
            html += '<div class="fut-card-team">' + escapeHtml(p.team || '--') + '</div>';
            html += '<div class="fut-card-stats">';
            var statItems = [
                { key: 'pac', label: 'PAC', val: p.pac || p.sprint_speed },
                { key: 'sho', label: 'SHO', val: p.sho || p.shooting },
                { key: 'pas', label: 'PAS', val: p.pas || p.pass_accuracy },
                { key: 'dri', label: 'DRI', val: p.dri || p.dribbling },
                { key: 'def', label: 'DEF', val: p.def || p.defending },
                { key: 'phy', label: 'PHY', val: p.phy || p.physical },
            ];
            statItems.forEach(function(s) {
                var v = s.val != null ? Math.min(99, Math.round(s.val)) : 50;
                var color = v >= 80 ? '#22c55e' : (v >= 60 ? '#eab308' : '#ef4444');
                html += '<div class="fut-stat-row"><span class="fut-stat-label">' + s.label + '</span><div class="fut-stat-bar"><div class="fut-stat-fill" style="width:' + (v/99*100) + '%;background:' + color + '"></div></div><span class="fut-stat-val">' + v + '</span></div>';
            });
            html += '</div>';
            html += '<div class="fut-card-footer">';
            html += '<span class="fut-card-value">' + valStr + '</span>';
            html += '<div class="fut-card-actions">';
            html += '<button class="scout-icon ' + (onShortlist ? 'added' : '') + '" data-action="shortlist" data-track-id="' + tid + '" data-name="' + escapeHtml(p.name) + '" data-pos="' + escapeHtml(p.position || '') + '">' + (onShortlist ? '★' : '☆') + '</button>';
            html += '<button class="scout-icon" data-action="compare" data-track-id="' + tid + '" data-name="' + escapeHtml(p.name) + '">⇄</button>';
            html += '</div></div></div></div>';
        });
        html += '</div>';
        return html;
    }

    function showPlayerDetail(player) {
        _scoutState.selectedPlayer = player;
        var tid = player.track_id || player.id || player.player_id;
        var val = player.transfer_value || player.estimated_value || '';
        var valStr = val ? '€' + (val >= 100 ? (val/100).toFixed(1) + 'M' : val + 'M') : '--';

        // Build modal
        var html = '<div class="modal-scout-detail">';
        html += '<div class="modal-scout-header">';
        html += '<h3>' + escapeHtml(player.name || 'Player Detail') + ' <span class="scout-rating-badge">' + (player.rating || '--') + '</span></h3>';
        html += '<button id="scout-detail-close" class="modal-close">&times;</button>';
        html += '</div>';
        html += '<div class="modal-scout-body">';
        html += '<div class="scout-detail-grid">';

        // Info column
        html += '<div class="scout-detail-info">';
        html += '<p><strong>Position:</strong> ' + escapeHtml(player.position || '--') + '</p>';
        html += '<p><strong>Team:</strong> ' + escapeHtml(player.team || '--') + '</p>';
        html += '<p><strong>Age:</strong> ' + (player.age || '--') + '</p>';
        html += '<p><strong>Nation:</strong> ' + (player.nation ? getFlagEmoji(player.nation) + ' ' + player.nation : '--') + '</p>';
        html += '<p><strong>Est. Value:</strong> ' + valStr + '</p>';
        html += '</div>';

        // Stats column
        html += '<div class="scout-detail-stats">';
        var statItems = [
            { key: 'matches', label: 'Matches', val: player.matches },
            { key: 'goals', label: 'Goals', val: player.goals },
            { key: 'assists', label: 'Assists', val: player.assists },
            { key: 'xg', label: 'xG', val: player.xg },
            { key: 'passes', label: 'Passes', val: player.passes },
            { key: 'tackles', label: 'Tackles', val: player.tackles },
        ];
        html += '<table class="data-table"><thead><tr><th>Stat</th><th>Value</th></tr></thead><tbody>';
        statItems.forEach(function(s) {
            html += '<tr><td>' + s.label + '</td><td>' + (s.val != null ? s.val : '--') + '</td></tr>';
        });
        html += '</tbody></table></div>';

        // Strengths & Weaknesses
        html += '<div class="scout-detail-sw">';
        if (player.strengths && player.strengths.length > 0) {
            html += '<div class="scout-strengths"><h4>✓ Strengths</h4><ul>';
            player.strengths.forEach(function(s) { html += '<li>' + escapeHtml(s) + '</li>'; });
            html += '</ul></div>';
        }
        if (player.weaknesses && player.weaknesses.length > 0) {
            html += '<div class="scout-weaknesses"><h4>✗ Weaknesses</h4><ul>';
            player.weaknesses.forEach(function(s) { html += '<li>' + escapeHtml(s) + '</li>'; });
            html += '</ul></div>';
        }
        html += '</div></div></div></div>';

        var modal = document.createElement('div');
        modal.className = 'modal visible';
        modal.style.display = 'flex';
        modal.innerHTML = html;
        document.body.appendChild(modal);

        var closeBtn = modal.querySelector('#scout-detail-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', function() { document.body.removeChild(modal); });
        }
        modal.addEventListener('click', function(e) {
            if (e.target === modal) document.body.removeChild(modal);
        });
    }

    function toggleShortlist(trackId, name, position, btn) {
        var idx = _scoutState.shortlist.findIndex(function(s) { return String(s.track_id) === String(trackId); });
        if (idx >= 0) {
            removeFromShortlist(_scoutState.shortlist[idx].entry_id);
            if (btn) { btn.textContent = '☆'; btn.classList.remove('added'); }
        } else {
            if (typeof bridge === 'undefined' || !bridge) {
                _scoutState.shortlist.push({ track_id: trackId, name: name, position: position });
                renderShortlist();
                return;
            }
            bridge.add_shortlist_entry(JSON.stringify({
                player_id: String(trackId),
                player_name: name,
                position: position || '',
                status: 'shortlisted',
            }), function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (data.error) throw new Error(data.error);
                    if (btn) { btn.textContent = '★'; btn.classList.add('added'); }
                    showToast('Added to shortlist!', 'success');
                    loadShortlist();
                } catch (e) { showToast('Could not save to shortlist.', 'error'); }
            });
            return;
        }
        renderShortlist();
    }

    function renderShortlist() {
        var container = document.getElementById('scout-shortlist-content');
        if (!container) return;
        var players = _scoutState.shortlist;

        if (!players || players.length === 0) {
            container.innerHTML = '<p class="hint">No shortlisted players yet.</p>';
            return;
        }

        var html = '<div class="scout-card-grid scout-shortlist-grid">';
        players.forEach(function(p) {
            var initial = (p.name || '?').charAt(0).toUpperCase();
            html += '<div class="scout-shortlist-card">' +
                '<div class="scout-player-avatar">' + initial + '</div>' +
                '<div class="scout-player-info">' +
                '<div class="scout-player-name">' + escapeHtml(p.name || 'Unknown') + '</div>' +
                '<div class="scout-player-meta"><span>' + escapeHtml(p.position || '--') + '</span>' +
                (p.status ? ' <span class="scout-status-badge">' + escapeHtml(p.status) + '</span>' : '') +
                (p.estimated_value ? ' <span>£' + Number(p.estimated_value).toLocaleString() + '</span>' : '') +
                '</div>' +
                '</div>' +
                '<div class="scout-icons">' +
                '<button class="scout-icon" data-action="advance-shortlist" data-entry-id="' + p.entry_id + '" title="Advance status">▶</button>' +
                '<button class="scout-icon" data-action="remove-shortlist" data-entry-id="' + p.entry_id + '">✕</button>' +
                '</div></div>';
        });
        html += '</div>';
        container.innerHTML = html;

        container.querySelectorAll('[data-action="remove-shortlist"]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                removeFromShortlist(parseInt(this.dataset.entryId, 10));
            });
        });
        container.querySelectorAll('[data-action="advance-shortlist"]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                advanceShortlistStatus(parseInt(this.dataset.entryId, 10));
            });
        });
    }

    var STATUS_FLOW = ['scouted', 'shortlisted', 'contacted', 'trial', 'signed'];

    function advanceShortlistStatus(entryId) {
        if (typeof bridge === 'undefined' || !bridge) return;
        var entry = _scoutState.shortlist.find(function(s) { return s.entry_id === entryId; });
        if (!entry) return;
        var next = STATUS_FLOW[Math.min(STATUS_FLOW.indexOf(entry.status || 'scouted') + 1, STATUS_FLOW.length - 1)];
        bridge.update_shortlist_entry(entryId, JSON.stringify({ status: next }), function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) throw new Error(data.error);
                entry.status = next;
                renderShortlist();
                showToast('Status: ' + next, 'success');
            } catch (e) { showToast('Update failed.', 'error'); }
        });
    }

    function removeFromShortlist(entryId) {
        if (typeof bridge === 'undefined' || !bridge) return;
        bridge.delete_shortlist_entry(entryId, function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) throw new Error(data.error);
                var idx = _scoutState.shortlist.findIndex(function(s) { return s.entry_id === entryId; });
                if (idx >= 0) {
                    _scoutState.shortlist.splice(idx, 1);
                    renderShortlist();
                    renderScoutResults();
                }
                showToast('Removed from shortlist.', 'info');
            } catch (e) { showToast('Remove failed.', 'error'); }
        });
    }

    function loadShortlist() {
        if (typeof bridge === 'undefined' || !bridge) {
            renderShortlist();
            return;
        }
        // Server-backed shortlist (player_shortlist table, migration 016):
        // previously this UI kept a local array that died on restart, and the
        // get_shortlist result was fetched and then ignored by the renderer.
        bridge.get_shortlist(function(result) {
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                if (data.error) throw new Error(data.error);
                _scoutState.shortlist = (data.players || []).map(function(p) {
                    return {
                        entry_id: p.id,
                        track_id: p.player_id,
                        name: p.player_name,
                        position: p.position,
                        team: p.team,
                        status: p.status,
                        priority: p.priority,
                        scout_rating: p.scout_rating,
                        estimated_value: p.estimated_value,
                    };
                });
                renderShortlist();
            } catch(e) { showToast('Failed to load shortlist.', 'error'); console.warn('Shortlist load error:', e); }
        });
    }

    function generateScoutReport() {
        if (_scoutState.shortlist.length === 0) {
            showToast('Add players to your shortlist first.', 'warning');
            return;
        }

        var report = '# Scout Report\n## Shortlisted Players\n\n';
        _scoutState.shortlist.forEach(function(p) {
            report += '- **' + p.name + '** (' + (p.position || 'N/A') + ')\n';
        });
        report += '\n*Generated by Kawkab AI Scout Portal*\n';

        var blob = new Blob([report], { type: 'text/markdown' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'scout-report-' + Date.now() + '.md';
        document.body.appendChild(a);
        a.click();
        setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
        showToast('Scout report downloaded!', 'success');
    }

    function populateCompareSelects() {}

    function addToCompare(trackId, name) {
        var selA = document.getElementById('scout-compare-a');
        var selB = document.getElementById('scout-compare-b');
        var opt = document.createElement('option');
        opt.value = trackId;
        opt.textContent = name;

        var exists = false;
        [selA, selB].forEach(function(sel) {
            for (var i = 0; i < sel.options.length; i++) {
                if (sel.options[i].value === trackId) exists = true;
            }
        });

        if (!exists) {
            selA.appendChild(opt.cloneNode(true));
            selB.appendChild(opt.cloneNode(true));
        }
        if (!selA.value) selA.value = trackId;
        else if (!selB.value) selB.value = trackId;
    }

    function scoutCompare(trackIdA, trackIdB) {
        var a = _scoutState.searchResults.find(function(p) { return (p.track_id || p.id) == trackIdA; });
        var b = _scoutState.searchResults.find(function(p) { return (p.track_id || p.id) == trackIdB; });

        var container = document.getElementById('scout-compare-results');
        if (!container) return;
        container.classList.remove('hidden');

        if (!a || !b) {
            container.innerHTML = '<p class="hint">Player data not found. Search for players first.</p>';
            return;
        }

        var html = '<div class="pro-card"><h4>Comparison: ' + escapeHtml(a.name) + ' vs ' + escapeHtml(b.name) + '</h4>';
        html += '<table class="data-table"><thead><tr><th>Stat</th><th>' + escapeHtml(a.name) + '</th><th>' + escapeHtml(b.name) + '</th></tr></thead><tbody>';

        var stats = [
            { key: 'rating', label: 'Rating' },
            { key: 'goals', label: 'Goals' },
            { key: 'assists', label: 'Assists' },
            { key: 'matches', label: 'Matches' },
            { key: 'xg', label: 'xG' },
            { key: 'passes', label: 'Passes' },
            { key: 'tackles', label: 'Tackles' },
            { key: 'transfer_value', label: 'Value (€M)' },
        ];

        stats.forEach(function(s) {
            var va = a[s.key] != null ? a[s.key] : '--';
            var vb = b[s.key] != null ? b[s.key] : '--';
            html += '<tr><td>' + s.label + '</td><td class="' + (va !== '--' && vb !== '--' && typeof va === 'number' && typeof vb === 'number' ? (va >= vb ? 'cmp-better' : 'cmp-worse') : '') + '">' + va + '</td>' +
                '<td class="' + (va !== '--' && vb !== '--' && typeof va === 'number' && typeof vb === 'number' ? (vb >= va ? 'cmp-better' : 'cmp-worse') : '') + '">' + vb + '</td></tr>';
        });

        html += '</tbody></table></div>';
        container.innerHTML = html;
    }

    function getTeamColor(teamName, darker) {
        var colors = {
            'Manchester City': darker ? '#1a237e' : '#6ab2f0',
            'Liverpool': darker ? '#7f1d1d' : '#dc2626',
            'PSG': darker ? '#1a237e' : '#004170',
            'Real Madrid': darker ? '#1a237e' : '#fbbf24',
            'Arsenal': darker ? '#7f1d1d' : '#ef4444',
            'Chelsea': darker ? '#1a3a5c' : '#2563eb',
            'FC Barcelona': darker ? '#1a3a5c' : '#a50044',
            'Bayern Munich': darker ? '#7f1d1d' : '#dc2626',
            'Manchester United': darker ? '#7f1d1d' : '#e11d48',
        };
        return colors[teamName] || (darker ? '#334155' : '#64748b');
    }

    function getFlagEmoji(countryCode) {
        if (!countryCode || countryCode.length !== 2) return '🌍';
        var codePoints = countryCode.toUpperCase().split('').map(function(c) {
            return 0x1F1E6 + c.charCodeAt(0) - 0x41;
        });
        return String.fromCodePoint(codePoints[0], codePoints[1]);
    }

    function escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    window.initScoutPortal = initScoutPortal;
    window.scoutSearch = scoutSearch;
    window.renderScoutResults = renderScoutResults;
    window.renderShortlist = renderShortlist;
    window.loadShortlist = loadShortlist;
    window.generateScoutReport = generateScoutReport;
    window.scoutCompare = scoutCompare;
    window.toggleShortlist = toggleShortlist;
    window.advanceShortlistStatus = advanceShortlistStatus;
    window.removeFromShortlist = removeFromShortlist;
    window.addToCompare = addToCompare;
    window.populateCompareSelects = populateCompareSelects;
    window.showPlayerDetail = showPlayerDetail;

    window.KawkabScout = {
        initScoutPortal: initScoutPortal,
        scoutSearch: scoutSearch,
        renderScoutResults: renderScoutResults,
        renderShortlist: renderShortlist,
        generateScoutReport: generateScoutReport,
        scoutCompare: scoutCompare,
        showPlayerDetail: showPlayerDetail,
    };

})();
