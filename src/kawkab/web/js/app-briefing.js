// Kawkab AI - Pre-Match Manager Briefing
// Football Manager-style match preview with tactical analysis

(function() { 'use strict';

    var _briefingState = {
        matchId: null,
        briefing: null,
        format: 'html',
    };

    window.initBriefingWorkspace = function() {
        loadBriefingMatchSelect();

        var generateBtn = document.getElementById('briefing-generate-btn');
        var exportBtn = document.getElementById('briefing-export-btn');
        var matchSelect = document.getElementById('briefing-match-select');

        if (!generateBtn || !matchSelect) return;

        generateBtn.addEventListener('click', function() {
            var mid = parseInt(matchSelect.value, 10);
            if (!mid) { showToast('Select a match first.', 'warning'); return; }
            generateBriefing(mid);
        });

        matchSelect.addEventListener('change', function() {
            _briefingState.matchId = parseInt(this.value, 10) || null;
        });

        if (exportBtn) {
            exportBtn.addEventListener('click', function() {
                if (!_briefingState.briefing) { showToast('Generate a briefing first.', 'warning'); return; }
                exportBriefing();
            });
        }
    };

    function loadBriefingMatchSelect() {
        if (typeof bridge === 'undefined' || !bridge) return;
        bridge.get_all_matches(function(result) {
            var data = typeof result === 'string' ? JSON.parse(result) : result;
            if (data && data.error) data = [];
            var select = document.getElementById('briefing-match-select');
            if (!select) return;
            select.innerHTML = '<option value="">-- Select Match --</option>';
            var matches = Array.isArray(data) ? data : (data.matches || data.results || []);
            matches.forEach(function(m) {
                var label = m.name || m.home_team + ' vs ' + m.away_team || 'Match #' + m.id;
                select.innerHTML += '<option value="' + m.id + '">' + escapeHtml(label) + '</option>';
            });
        });
    }

    window.generateBriefing = function(matchId) {
        var status = document.getElementById('briefing-status');
        var content = document.getElementById('briefing-content');
        if (!status || !content) return;

        status.textContent = 'Generating briefing...';
        content.innerHTML = '<div class="skeleton" style="height:400px"></div>';

        bridge.generate_briefing(String(matchId), function(result) {
            var data = typeof result === 'string' ? JSON.parse(result) : result;
            if (data && data.error) {
                status.textContent = 'Error';
                content.innerHTML = '<p class="error-message">Failed to generate briefing: ' + escapeHtml(data.error) + '</p>';
                showToast('Briefing generation failed.', 'error');
                return;
            }

            if (data && data.success && data.briefing) {
                _briefingState.briefing = data.briefing;
                _briefingState.matchId = matchId;

                if (data.html) {
                    content.innerHTML = data.html;
                } else {
                    content.innerHTML = '<pre style="white-space:pre-wrap;font-size:0.82rem">' + escapeHtml(data.markdown || JSON.stringify(data.briefing, null, 2)) + '</pre>';
                }

                status.textContent = 'Briefing ready';

                var matchInfo = data.briefing.match_info || {};
                var ourTeam = data.briefing.our_team || {};
                var opponent = data.briefing.opponent || {};
                var prediction = data.briefing.prediction || {};

                showToast('Briefing generated for ' + matchInfo.home_team + ' vs ' + matchInfo.away_team, 'success');
            } else {
                status.textContent = 'Error';
                content.innerHTML = '<p class="hint">No briefing data returned.</p>';
                showToast('Unexpected briefing response.', 'error');
            }
        });
    };

    window.exportBriefing = function() {
        if (!_briefingState.briefing) {
            showToast('Generate a briefing first.', 'warning');
            return;
        }
        var md = _briefingState.briefing.markdown;
        if (!md) {
            showToast('No markdown version available.', 'warning');
            return;
        }
        var matchInfo = _briefingState.briefing.match_info || {};
        var filename = 'briefing_' + (matchInfo.home_team || 'home') + '_vs_' + (matchInfo.away_team || 'away') + '.md';
        var blob = new Blob([md], { type: 'text/markdown' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Briefing exported as ' + filename, 'success');
    };

    function escapeHtml(text) {
        if (typeof text !== 'string') return '';
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

})();
