// Kawkab AI - Coding Workspace (Video Tagging Engine)
// Extracted from app.js for modularity

(function() { 'use strict';

    var _codingState = {
        matchId: null,
        tags: [],
        templates: [],
        video: null,
        currentTime: 0,
        activeTagId: null,
        shortcuts: {},
        selectedPlayer: 0,
        selectedTeam: '',
        selectedPeriod: 1,
        leadMs: 2000,
        lagMs: 3000,
        notes: '',
        filterText: '',
        isLoading: false,
        keyboardEnabled: false,
    };

    function initCodingWorkspace() {
        var loadBtn = document.getElementById('coding-load-btn');
        var matchSelect = document.getElementById('coding-match-select');
        var clearBtn = document.getElementById('coding-clear-btn');
        var exportAllBtn = document.getElementById('coding-export-all-btn');
        var filterInput = document.getElementById('coding-filter-input');
        var exportCsv = document.getElementById('coding-export-csv');
        var exportJson = document.getElementById('coding-export-json');
        var timelineCanvas = document.getElementById('coding-timeline-canvas');
        var notesInput = document.getElementById('coding-notes-input');

        if (!loadBtn) return;

        window.loadCodingMatchSelect = function() {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.get_all_matches(function(result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error || !Array.isArray(data)) {
                        data = typeof data === 'object' && data.matches ? data.matches : [];
                    }
                    var sel = document.getElementById('coding-match-select');
                    sel.innerHTML = '<option value="">-- Select Match --</option>';
                    (data || []).forEach(function(m) {
                        var name = m.name || m.home_team + ' vs ' + m.away_team || 'Match #' + m.id;
                        sel.innerHTML += '<option value="' + m.id + '">' + escapeHtml(name) + '</option>';
                    });
                } catch(e) {
                    showToast('Failed to load matches for coding.', 'error');
                    console.warn('loadCodingMatchSelect:', e);
                }
            });
        };

        loadBtn.addEventListener('click', function() {
            var matchId = parseInt(matchSelect.value, 10);
            if (!matchId) {
                showToast('Please select a match first.', 'warning');
                return;
            }
            loadCodingWorkspace(matchId);
        });

        clearBtn.addEventListener('click', function() {
            if (_codingState.tags.length === 0) return;
            showConfirmDialog('Delete all ' + _codingState.tags.length + ' tags for this match?', function() {
                var ids = _codingState.tags.map(function(t) { return t.id; });
                var deleted = 0;
                ids.forEach(function(id) {
                    bridge.delete_coding_tag(id, function(result) {
                        var data = JSON.parse(result);
                        if (data.success) deleted++;
                        if (deleted === ids.length) {
                            _codingState.tags = [];
                            renderCodingTagList();
                            renderCodingTimeline();
                            updateCodingStats();
                            showToast('All tags deleted.', 'success');
                        }
                    });
                });
            });
        });

        exportAllBtn.addEventListener('click', function() {
            if (_codingState.tags.length === 0) {
                showToast('No tags to export.', 'warning');
                return;
            }
            showToast('Extracting clips...', 'info');
            var tagIds = _codingState.tags.map(function(t) { return t.id; });
            bridge.extract_tag_clips_batch(_codingState.matchId, JSON.stringify(tagIds), function(result) {
                try {
                    var data = JSON.parse(result);
                    if (data.success) {
                        var done = data.results.filter(function(r) { return r.success; }).length;
                        var failed = data.results.filter(function(r) { return r.error; }).length;
                        showToast('Exported ' + done + ' clips' + (failed ? ', ' + failed + ' failed' : ''), done > 0 ? 'success' : 'error');
                    } else {
                        showToast('Export failed: ' + (data.error || 'Unknown error'), 'error');
                    }
                } catch(e) {
                    showToast('Export failed: ' + e.message, 'error');
                }
            });
        });

        filterInput.addEventListener('input', function() {
            _codingState.filterText = this.value.toLowerCase();
            renderCodingTagList();
        });

        exportCsv.addEventListener('click', function() {
            exportCodingTags('csv');
        });
        exportJson.addEventListener('click', function() {
            exportCodingTags('json');
        });

        document.getElementById('coding-player-select').addEventListener('change', function() {
            _codingState.selectedPlayer = parseInt(this.value, 10) || 0;
        });

        document.getElementById('coding-team-select').addEventListener('change', function() {
            _codingState.selectedTeam = this.value;
        });

        document.getElementById('coding-period-select').addEventListener('change', function() {
            _codingState.selectedPeriod = parseInt(this.value, 10) || 1;
        });

        document.getElementById('coding-lead-ms').addEventListener('change', function() {
            _codingState.leadMs = parseInt(this.value, 10) || 2000;
        });
        document.getElementById('coding-lag-ms').addEventListener('change', function() {
            _codingState.lagMs = parseInt(this.value, 10) || 3000;
        });

        notesInput.addEventListener('change', function() {
            _codingState.notes = this.value;
        });

        timelineCanvas.addEventListener('click', function(e) {
            var rect = this.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var pct = x / rect.width;
            var video = document.getElementById('coding-video');
            if (video && video.duration) {
                video.currentTime = pct * video.duration;
            }
        });

        document.addEventListener('keydown', function(e) {
            if (!_codingState.keyboardEnabled) return;
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT')) return;
            if (e.ctrlKey || e.metaKey || e.altKey) return;

            var key = e.key.toLowerCase();
            if (key === ' ' || key === 'k') {
                e.preventDefault();
                var vid = document.getElementById('coding-video');
                if (vid && vid.src) {
                    if (vid.paused) vid.play(); else vid.pause();
                }
                return;
            }
            if (key === 'arrowleft') {
                e.preventDefault();
                var vid = document.getElementById('coding-video');
                if (vid) vid.currentTime = Math.max(0, vid.currentTime - 3);
                return;
            }
            if (key === 'arrowright') {
                e.preventDefault();
                var vid = document.getElementById('coding-video');
                if (vid) vid.currentTime = Math.min(vid.duration || 0, vid.currentTime + 3);
                return;
            }

            var shortcutMap = _codingState.shortcuts;
            if (shortcutMap[key]) {
                e.preventDefault();
                triggerTag(shortcutMap[key]);
            }
        });

        var video = document.getElementById('coding-video');
        if (video) {
            video.addEventListener('timeupdate', function() {
                _codingState.currentTime = this.currentTime;
                updateCodingTimelineCursor();
                highlightTagAtTime(this.currentTime);
            });
        }

        loadCodingTemplates();
        loadCodingMatchSelect();
    }

    function loadCodingTemplates() {
        if (typeof bridge === 'undefined' || !bridge || !bridge.get_coding_templates) return;
        bridge.get_coding_templates(function(result) {
            try {
                var data = JSON.parse(result);
                if (data.success) {
                    _codingState.templates = data.templates;
                    renderCodingMatrix(data.templates);
                    var shortcuts = {};
                    (data.templates.categories || []).forEach(function(cat) {
                        (cat.buttons || []).forEach(function(btn) {
                            if (btn.shortcut) shortcuts[btn.shortcut.toLowerCase()] = btn;
                        });
                    });
                    _codingState.shortcuts = shortcuts;
                }
            } catch(e) {
                showToast('Failed to load coding templates.', 'error');
                console.warn('loadCodingTemplates:', e);
            }
        });
    }

    // WCAG relative luminance -> pick black or white text so a coach's
    // custom (or the shipped default) event-color palette stays readable.
    // Several shipped defaults (e.g. #a3e635, #a5f3fc, #fca5a5) are light
    // enough that the fixed white button text this replaces dropped to
    // ~1.2-1.9:1 contrast, far under the 3:1 floor for bold UI text.
    function contrastTextColor(hex) {
        var m = /^#?([0-9a-f]{6}|[0-9a-f]{3})$/i.exec(hex || '');
        if (!m) return '#fff';
        var h = m[1];
        if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
        var r = parseInt(h.substr(0, 2), 16) / 255;
        var g = parseInt(h.substr(2, 2), 16) / 255;
        var b = parseInt(h.substr(4, 2), 16) / 255;
        function lin(c) { return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
        var luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
        var contrastWithWhite = 1.05 / (luminance + 0.05);
        var contrastWithBlack = (luminance + 0.05) / 0.05;
        return contrastWithWhite >= contrastWithBlack ? '#fff' : '#000';
    }

    function renderCodingMatrix(templates) {
        var container = document.getElementById('coding-matrix');
        if (!container) return;
        container.innerHTML = '';
        (templates.categories || []).forEach(function(cat) {
            var catEl = document.createElement('div');
            catEl.className = 'coding-matrix-category';
            catEl.innerHTML = '<div class="coding-category-label" style="color:' + escapeHtml(cat.color || '#fff') + '">' + escapeHtml(cat.label) + '</div>';
            var grid = document.createElement('div');
            grid.className = 'coding-button-grid';
            (cat.buttons || []).forEach(function(btn) {
                var btnEl = document.createElement('button');
                btnEl.className = 'coding-matrix-btn';
                var btnColor = btn.color || '#555';
                btnEl.style.background = btnColor;
                btnEl.style.color = contrastTextColor(btnColor);
                btnEl.dataset.eventType = btn.id;
                btnEl.dataset.shortcut = btn.shortcut || '';
                btnEl.innerHTML = escapeHtml(btn.label) + (btn.shortcut ? '<span class="shortcut-hint">' + escapeHtml(btn.shortcut) + '</span>' : '');
                btnEl.addEventListener('click', function() {
                    triggerTag(btn);
                });
                grid.appendChild(btnEl);
            });
            catEl.appendChild(grid);
            container.appendChild(catEl);
        });
    }

    function loadCodingWorkspace(matchId) {
        _codingState.matchId = matchId;
        _codingState.isLoading = true;

        var status = document.getElementById('coding-match-status');
        status.textContent = 'Loading...';

        bridge.get_video_path(matchId, function(result) {
            try {
                var data = JSON.parse(result);
                if (data && data.video_path) {
                    var video = document.getElementById('coding-video');
                    video.src = data.video_path;
                    video.load();
                    _codingState.keyboardEnabled = true;
                    status.textContent = 'Ready';
                } else {
                    status.textContent = 'No video found';
                    showToast('No video found for this match.', 'warning');
                }
            } catch(e) {
                status.textContent = 'Error loading video';
                showToast('Error loading coding video.', 'error');
                console.warn('loadCodingWorkspace video:', e);
            }
        });

        if (bridge.get_coding_players) {
            bridge.get_coding_players(matchId, function(result) {
                try {
                    var data = JSON.parse(result);
                    if (data.success && data.players) {
                        var sel = document.getElementById('coding-player-select');
                        sel.innerHTML = '<option value="0">-- None --</option>';
                        data.players.forEach(function(p) {
                            sel.innerHTML += '<option value="' + p.track_id + '">' +
                                escapeHtml(p.name || 'Player ' + p.track_id) +
                                ' (#' + p.jersey + ')</option>';
                        });
                    }
                } catch(e) {}
            });
        }

        bridge.get_coding_tags(matchId, function(result) {
            try {
                var data = JSON.parse(result);
                if (data.success) {
                    _codingState.tags = data.tags || [];
                    renderCodingTagList();
                    renderCodingTimeline();
                    updateCodingStats();
                }
            } catch(e) {
                showToast('Failed to load existing tags.', 'error');
                console.warn('loadCodingWorkspace tags:', e);
            }
            _codingState.isLoading = false;

            document.getElementById('coding-workspace').classList.remove('hidden');
            document.getElementById('coding-empty-state').classList.add('hidden');

            showToast('Loaded ' + _codingState.tags.length + ' existing tags.', 'info');
        });
    }

    function triggerTag(btn) {
        if (!_codingState.matchId) {
            showToast('Select a match first.', 'warning');
            return;
        }

        var video = document.getElementById('coding-video');
        if (!video || !video.src) {
            showToast('No video loaded.', 'warning');
            return;
        }

        var videoTime = video.currentTime;
        var tag = {
            event_type: btn.id,
            sub_type: '',
            video_time: videoTime,
            player_track_id: _codingState.selectedPlayer,
            player_name: getSelectedPlayerName(),
            team: _codingState.selectedTeam,
            period: _codingState.selectedPeriod,
            notes: _codingState.notes || '',
            lead_ms: _codingState.leadMs,
            lag_ms: _codingState.lagMs,
        };

        var container = document.querySelector('.coding-video-container');
        container.classList.remove('coding-flash');
        void container.offsetWidth;
        container.classList.add('coding-flash');

        bridge.save_coding_tag(_codingState.matchId, JSON.stringify(tag), function(result) {
            try {
                var data = JSON.parse(result);
                if (data.success) {
                    tag.id = data.tag_id;
                    _codingState.tags.push(tag);
                    renderCodingTagList();
                    renderCodingTimeline();
                    updateCodingStats();
                    updateCodingLastTag(btn.label || btn.id, videoTime);
                } else {
                    showToast('Failed to save tag: ' + (data.error || 'Unknown'), 'error');
                }
            } catch(e) {
                showToast('Error saving tag.', 'error');
                console.warn('triggerTag save:', e);
            }
        });
    }

    function getSelectedPlayerName() {
        var sel = document.getElementById('coding-player-select');
        if (!sel) return '';
        var opt = sel.options[sel.selectedIndex];
        return opt ? opt.text.split(' (#')[0] : '';
    }

    function renderCodingTagList() {
        var list = document.getElementById('coding-tag-list');
        if (!list) return;
        var filter = _codingState.filterText;
        var tags = _codingState.tags;

        if (filter) {
            tags = tags.filter(function(t) {
                return (t.event_type || '').toLowerCase().indexOf(filter) !== -1 ||
                       (t.player_name || '').toLowerCase().indexOf(filter) !== -1 ||
                       (t.notes || '').toLowerCase().indexOf(filter) !== -1;
            });
        }

        if (tags.length === 0) {
            list.innerHTML = '<div class="coding-tag-list-empty">' +
                (filter ? 'No tags match your filter.' : 'No tags yet. Click a matrix button to tag the current video time!') +
                '</div>';
            document.getElementById('coding-tag-count-badge').textContent = '0';
            return;
        }

        var html = '';
        tags.forEach(function(tag, idx) {
            var timeStr = formatCodingTime(tag.video_time || 0);
            var typeLabel = tag.event_type || 'unknown';
            var playerLabel = tag.player_name || '';
            var activeClass = tag.id === _codingState.activeTagId ? ' active' : '';
            var color = getTagColor(tag.event_type);

            html += '<div class="coding-tag-item' + activeClass + '" data-tag-id="' + tag.id + '" data-video-time="' + (tag.video_time || 0) + '">';
            html += '  <span class="tag-color-dot" style="background:' + color + '"></span>';
            html += '  <div class="tag-info">';
            html += '    <div class="tag-type">' + escapeHtml(typeLabel) + '</div>';
            html += '    <div class="tag-sub">' + escapeHtml(timeStr) + (playerLabel ? ' · ' + escapeHtml(playerLabel) : '') + '</div>';
            html += '  </div>';
            html += '  <div class="tag-time">' + timeStr + '</div>';
            html += '  <div class="tag-actions">';
            html += '    <button class="tag-action-btn tag-seek-btn" title="Seek to time">⏩</button>';
            html += '    <button class="tag-action-btn tag-clip-btn" title="Extract clip">✂️</button>';
            html += '    <button class="tag-action-btn tag-delete-btn" title="Delete tag">✕</button>';
            html += '  </div>';
            html += '</div>';
        });
        list.innerHTML = html;

        document.getElementById('coding-tag-count-badge').textContent = tags.length;

        list.querySelectorAll('.coding-tag-item').forEach(function(item) {
            item.addEventListener('click', function(e) {
                if (e.target.closest('.tag-actions')) return;
                var time = parseFloat(this.dataset.videoTime);
                var video = document.getElementById('coding-video');
                if (video) video.currentTime = time;
                _codingState.activeTagId = parseInt(this.dataset.tagId, 10);
                renderCodingTagList();
            });

            item.querySelector('.tag-seek-btn').addEventListener('click', function(e) {
                e.stopPropagation();
                var time = parseFloat(item.dataset.videoTime);
                var video = document.getElementById('coding-video');
                if (video) video.currentTime = time;
            });

            item.querySelector('.tag-clip-btn').addEventListener('click', function(e) {
                e.stopPropagation();
                var tagId = parseInt(item.dataset.tagId, 10);
                if (!tagId) return;
                showToast('Extracting clip...', 'info');
                bridge.extract_tag_clip(_codingState.matchId, tagId, function(result) {
                    try {
                        var data = JSON.parse(result);
                        if (data.success) {
                            showToast('Clip saved: ' + data.clip_path, 'success');
                        } else {
                            showToast('Failed: ' + (data.error || 'Unknown'), 'error');
                        }
                    } catch(e) {
                        showToast('Clip extraction failed.', 'error');
                    }
                });
            });

            item.querySelector('.tag-delete-btn').addEventListener('click', function(e) {
                e.stopPropagation();
                var tagId = parseInt(item.dataset.tagId, 10);
                if (!tagId) return;
                showConfirmDialog('Delete this tag?', function() {
                    bridge.delete_coding_tag(tagId, function(result) {
                        try {
                            var data = JSON.parse(result);
                            if (data.success) {
                                _codingState.tags = _codingState.tags.filter(function(t) { return t.id !== tagId; });
                                renderCodingTagList();
                                renderCodingTimeline();
                                updateCodingStats();
                            } else {
                                showToast('Failed to delete tag.', 'error');
                            }
                        } catch(e) {}
                    });
                });
            });
        });
    }

    function renderCodingTimeline() {
        var canvas = document.getElementById('coding-timeline-canvas');
        if (!canvas) return;
        var video = document.getElementById('coding-video');
        var duration = video && video.duration ? video.duration : 90 * 60;

        var rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width || 600;
        canvas.height = 40;

        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        var markerInterval = Math.max(60, Math.floor(duration / 10));
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.font = '9px monospace';
        ctx.fillStyle = '#64748b';
        for (var t = 0; t <= duration; t += markerInterval) {
            var x = (t / duration) * canvas.width;
            ctx.beginPath();
            ctx.moveTo(x, 24);
            ctx.lineTo(x, 40);
            ctx.stroke();
            ctx.fillText(formatCodingTime(t), x + 2, 35);
        }

        (_codingState.tags || []).forEach(function(tag) {
            var time = tag.video_time || 0;
            var x = (time / duration) * canvas.width;
            var color = getTagColor(tag.event_type);

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, 12, 5, 0, Math.PI * 2);
            ctx.fill();

            if (tag.id === _codingState.activeTagId) {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, 12, 8, 0, Math.PI * 2);
                ctx.stroke();
            }
        });

        if (video && video.currentTime != null) {
            var cursorX = (video.currentTime / duration) * canvas.width;
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cursorX, 0);
            ctx.lineTo(cursorX, 40);
            ctx.stroke();
        }

        var labels = document.getElementById('coding-timeline-labels');
        if (labels) {
            labels.innerHTML = '0:00';
            var half = Math.floor(duration / 2);
            labels.innerHTML += '<span>' + formatCodingTime(half) + '</span>';
            labels.innerHTML += '<span>' + formatCodingTime(duration) + '</span>';
        }
    }

    function updateCodingTimelineCursor() {
        renderCodingTimeline();
    }

    function highlightTagAtTime(time) {
        var tags = _codingState.tags;
        var closest = null;
        var closestDist = Infinity;
        tags.forEach(function(tag) {
            var dist = Math.abs((tag.video_time || 0) - time);
            if (dist < closestDist) {
                closestDist = dist;
                closest = tag;
            }
        });
        var activeId = closest && closestDist < 5 ? closest.id : null;
        if (activeId !== _codingState.activeTagId) {
            _codingState.activeTagId = activeId;
            renderCodingTagList();
        }
    }

    function updateCodingStats() {
        var count = _codingState.tags.length;
        document.getElementById('coding-tag-count').textContent = count;
    }

    function updateCodingLastTag(type, time) {
        var el = document.getElementById('coding-last-tag');
        if (el) {
            el.innerHTML = 'Last: <strong>' + escapeHtml(type) + '</strong> at ' + formatCodingTime(time);
        }
    }

    function formatCodingTime(seconds) {
        if (seconds == null || !isFinite(seconds)) return '0:00';
        var m = Math.floor(seconds / 60);
        var s = Math.floor(seconds % 60);
        return m + ':' + (s < 10 ? '0' : '') + s;
    }

    function getTagColor(eventType) {
        if (!eventType) return '#64748b';
        var colorMap = {
            'pass': '#22c55e',
            'through_ball': '#4ade80',
            'shot': '#ef4444',
            'goal': '#dc2626',
            'dribble': '#3b82f6',
            'cross': '#60a5fa',
            'carry': '#818cf8',
            'key_pass': '#a3e635',
            'tackle': '#f97316',
            'interception': '#fb923c',
            'press': '#a855f7',
            'clearance': '#c084fc',
            'block': '#e879f9',
            'foul': '#f43f5e',
            'error_positional': '#92400e',
            'error_technical': '#b45309',
            'error_decision': '#d97706',
            'error_physical': '#f59e0b',
            'missed_tackle': '#ef4444',
            'bad_pass': '#fca5a5',
            'corner': '#06b6d4',
            'free_kick': '#22d3ee',
            'throw_in': '#67e8f9',
            'goal_kick': '#a5f3fc',
            'penalty': '#2dd4bf',
        };
        return colorMap[eventType] || '#64748b';
    }

    function escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    function exportCodingTags(format) {
        if (_codingState.tags.length === 0) {
            showToast('No tags to export.', 'warning');
            return;
        }

        if (format === 'csv') {
            var headers = 'id,event_type,video_time,player_name,team,period,notes,lead_ms,lag_ms';
            var rows = _codingState.tags.map(function(t) {
                return (t.id || '') + ',' +
                       (t.event_type || '') + ',' +
                       (t.video_time || 0) + ',' +
                       '"' + (t.player_name || '') + '",' +
                       (t.team || '') + ',' +
                       (t.period || 1) + ',' +
                       '"' + (t.notes || '').replace(/"/g, '""') + '",' +
                       (t.lead_ms || 2000) + ',' +
                       (t.lag_ms || 3000);
            });
            var csv = headers + '\n' + rows.join('\n');
            downloadFile(csv, 'coding_tags_' + _codingState.matchId + '.csv', 'text/csv');
        } else {
            var json = JSON.stringify(_codingState.tags, null, 2);
            downloadFile(json, 'coding_tags_' + _codingState.matchId + '.json', 'application/json');
        }

        showToast('Exported ' + _codingState.tags.length + ' tags.', 'success');
    }

    function downloadFile(content, filename, mimeType) {
        var blob = new Blob([content], { type: mimeType });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Expose globally for backward compat
    window.initCodingWorkspace = initCodingWorkspace;
    window.loadCodingWorkspace = loadCodingWorkspace;
    window.triggerTag = triggerTag;
    window.renderCodingTagList = renderCodingTagList;
    window.renderCodingTimeline = renderCodingTimeline;
    window.exportCodingTags = exportCodingTags;
    window.formatCodingTime = formatCodingTime;
    window.getTagColor = getTagColor;
    window.loadCodingTemplates = loadCodingTemplates;
    window.renderCodingMatrix = renderCodingMatrix;
    window.updateCodingTimelineCursor = updateCodingTimelineCursor;
    window.highlightTagAtTime = highlightTagAtTime;
    window.updateCodingStats = updateCodingStats;
    window.updateCodingLastTag = updateCodingLastTag;
    window.getSelectedPlayerName = getSelectedPlayerName;
    window.downloadFile = downloadFile;

    // Namespace
    window.KawkabCoding = {
        initCodingWorkspace: initCodingWorkspace,
        loadCodingWorkspace: loadCodingWorkspace,
        triggerTag: triggerTag,
        renderCodingTagList: renderCodingTagList,
        renderCodingTimeline: renderCodingTimeline,
        exportCodingTags: exportCodingTags,
    };

})();
