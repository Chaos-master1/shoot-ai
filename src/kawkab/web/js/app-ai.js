    function _catchBridge(method) {
        return function(err) { console.error('bridge.' + method + ' failed:', err); };
    }

    // Not wrapped in its own IIFE (split out of app.js without one) -- the
    // bundler is a plain concatenation, so this and every escapeHtml(...)
    // call below resolved to nothing at runtime (ReferenceError) until this
    // import was added. Matches app-data-providers.js's own fix for the
    // same gap.
    var escapeHtml = window.__kawkab.escapeHtml;

    // ── AI Coach Assistant v2 (Phase 12) ─────────────────────────

    function initAiWorkspace() {
        var askBtn = document.getElementById('ai-ask-btn');
        var input = document.getElementById('ai-question-input');
        var matchSelect = document.getElementById('ai-match-select');
        var convSelect = document.getElementById('ai-conv-select');
        var voiceBtn = document.getElementById('ai-voice-btn');
        var langSelect = document.getElementById('ai-lang-select');
        var autoReportBtn = document.getElementById('ai-auto-report-btn');
        var exportChatBtn = document.getElementById('ai-export-chat-btn');
        var newConvBtn = document.getElementById('ai-new-conv-btn');
        var delConvBtn = document.getElementById('ai-del-conv-btn');
        if (!askBtn) return;

        var currentConvId = '';
        var convCache = {};

        function getMatchId() { return parseInt(matchSelect.value, 10); }
        function getLang() { return (langSelect ? langSelect.value : 'en') || 'en'; }

        window.loadAiMatchSelect = function() {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.get_all_matches(function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (data.error) data = [];
                    matchSelect.innerHTML = '<option value="">-- Select Match --</option>';
                    (data || []).forEach(function(m) {
                        var name = m.name || (m.home_team + ' vs ' + m.away_team) || 'Match #' + m.id;
                        matchSelect.innerHTML += '<option value="' + m.id + '">' + escapeHtml(name) + '</option>';
                    });
                } catch(e) { showToast('Failed to load matches.', 'error'); console.warn(e); }
            });
        };

        function loadConversations() {
            var mid = getMatchId();
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.ai_v2_list_convs(String(mid || ''), function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (!data.success) return;
                    convSelect.innerHTML = '<option value="">-- New Chat --</option>';
                    (data.conversations || []).forEach(function(c) {
                        var opt = document.createElement('option');
                        opt.value = c.id;
                        opt.textContent = c.title + ' (' + c.message_count + ' msgs)';
                        convSelect.appendChild(opt);
                    });
                    convCache = {};
                    (data.conversations || []).forEach(function(c) { convCache[c.id] = c; });
                } catch(e) {}
            });
        }

        function ensureConv(callback) {
            if (currentConvId) { callback(currentConvId); return; }
            var title = 'Chat ' + new Date().toLocaleTimeString();
            bridge.ai_v2_create_conv(String(getMatchId() || ''), title, function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (data.success) {
                        currentConvId = data.conv_id;
                        loadConversations();
                        callback(data.conv_id);
                    } else {
                        callback('');
                    }
                } catch(e) { callback(''); }
            });
        }

        function addMessage(role, content) {
            var messages = document.getElementById('ai-chat-messages');
            var clean = escapeHtml(content);
            messages.innerHTML += '<div class="ai-message ' + role + '">' + clean + '</div>';
            messages.scrollTop = messages.scrollHeight;
        }

        function showTyping() {
            var messages = document.getElementById('ai-chat-messages');
            messages.innerHTML += '<div class="ai-message assistant ai-typing" id="ai-typing-msg">Thinking</div>';
            messages.scrollTop = messages.scrollHeight;
        }

        function removeTyping() {
            var t = document.getElementById('ai-typing-msg');
            if (t) t.remove();
        }

        function askQuestion(questionText) {
            var mid = getMatchId();
            if (!mid) { showToast('Select a match first.', 'warning'); return; }
            var question = questionText || input.value.trim();
            if (!question) { showToast('Enter a question.', 'warning'); return; }

            addMessage('user', question);
            showTyping();
            if (input) input.value = '';

            ensureConv(function(convId) {
                if (!convId) { removeTyping(); addMessage('assistant', 'Failed to create conversation.'); return; }
                currentConvId = convId;

                bridge.ai_v2_ask(convId, question, '', getLang(), function(result) {
                    removeTyping();
                    try {
                        var data = typeof result === 'string' ? JSON.parse(result) : result;
                        if (data.success) {
                            addMessage('assistant', data.answer);
                        } else {
                            // fallback to old method
                            bridge.ask_llm(mid, question, function(r2) {
                                try {
                                    var d2 = typeof r2 === 'string' ? JSON.parse(r2) : r2;
                                    if (d2.success) {
                                        addMessage('assistant', d2.answer);
                                    } else {
                                        addMessage('assistant', 'Error: ' + escapeHtml(d2.error || 'Unknown'));
                                    }
                                } catch(e) {
                                    addMessage('assistant', 'Failed to get answer.');
                                }
                            });
                        }
                        loadConversations();
                    } catch(e) { removeTyping(); addMessage('assistant', 'Parse error.'); }
                });
            });
        }

        askBtn.addEventListener('click', function() { askQuestion(); });
        input.addEventListener('keydown', function(e) { if (e.key === 'Enter') askQuestion(); });

        // Voice input
        if (voiceBtn && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            var recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.interimResults = false;
            voiceBtn.addEventListener('click', function() {
                voiceBtn.textContent = '🎤 Listening...';
                recognition.start();
            });
            recognition.onresult = function(e) {
                var transcript = e.results[0][0].transcript;
                input.value = transcript;
                voiceBtn.textContent = '🎤';
                askQuestion(transcript);
            };
            recognition.onerror = function() { voiceBtn.textContent = '🎤'; };
            recognition.onend = function() { voiceBtn.textContent = '🎤'; };
        } else if (voiceBtn) {
            voiceBtn.title = 'Voice not supported in this browser';
            voiceBtn.style.opacity = '0.4';
        }

        // Suggestion chips
        document.querySelectorAll('#ai-suggestion-chips .chip').forEach(function(chip) {
            chip.addEventListener('click', function() {
                var topic = this.getAttribute('data-topic');
                if (!topic) return;
                askQuestion('Analyze ' + topic.replace('_', ' ') + ' for this match in detail.');
            });
        });

        // Auto report
        if (autoReportBtn) {
            autoReportBtn.addEventListener('click', function() {
                var mid = getMatchId();
                if (!mid) { showToast('Select a match first.', 'warning'); return; }
                showTyping();
                bridge.ai_v2_auto_report(String(mid), getLang(), function(result) {
                    removeTyping();
                    try {
                        var data = typeof result === 'string' ? JSON.parse(result) : result;
                        if (data.success) {
                            addMessage('assistant', data.report);
                        } else {
                            addMessage('assistant', 'Auto report failed: ' + escapeHtml(data.error || 'Unknown'));
                        }
                    } catch(e) { removeTyping(); addMessage('assistant', 'Report generation error.'); }
                });
            });
        }

        // Export chat
        if (exportChatBtn) {
            exportChatBtn.addEventListener('click', function() {
                var msgs = document.querySelectorAll('#ai-chat-messages .ai-message');
                var text = '';
                msgs.forEach(function(m) {
                    var role = m.classList.contains('user') ? 'Coach' : m.classList.contains('assistant') ? 'Analyst' : 'System';
                    text += '### ' + role + '\n' + m.textContent + '\n\n';
                });
                var blob = new Blob([text], { type: 'text/markdown' });
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'ai-chat-' + new Date().toISOString().slice(0, 10) + '.md';
                a.click();
                showToast('Chat exported as Markdown', 'success');
            });
        }

        // Conversation management
        if (convSelect) {
            convSelect.addEventListener('change', function() {
                currentConvId = this.value;
                if (!currentConvId) {
                    document.getElementById('ai-chat-messages').innerHTML =
                        '<div class="ai-message system">New conversation. Ask a question to start.</div>';
                    return;
                }
                showToast('Switched conversation', 'info');
            });
            newConvBtn.addEventListener('click', function() {
                currentConvId = '';
                convSelect.value = '';
                document.getElementById('ai-chat-messages').innerHTML =
                    '<div class="ai-message system">New conversation. Ask a question to start.</div>';
            });
            delConvBtn.addEventListener('click', function() {
                var cid = currentConvId || convSelect.value;
                if (!cid) { showToast('No conversation selected.', 'warning'); return; }
                if (!confirm('Delete this conversation?')) return;
                bridge.ai_v2_delete_conv(cid, function() {
                    currentConvId = '';
                    convSelect.value = '';
                    loadConversations();
                    document.getElementById('ai-chat-messages').innerHTML =
                        '<div class="ai-message system">Conversation deleted.</div>';
                    showToast('Deleted', 'info');
                });
            });
        }

        loadAiMatchSelect();
        loadConversations();
    }

    // ── Squad + Player Ratings (delegated to app-squad.js) ───

    function initSquadWorkspace() {
        if (window.KawkabSquad) return window.KawkabSquad.initSquadWorkspace();
    }

    // ── Event Review Workspace (Phase 2) ─────────────────────────

    var _reviewState = {
        matchId: null,
        events: [],
        queue: [],
        currentIndex: -1,
        video: null,
        autoAdvance: true,
        isLoading: false,
    };

    function initReviewWorkspace() {
        var loadBtn = document.getElementById('review-load-btn');
        var matchSelect = document.getElementById('review-match-select');
        var autoAdvBtn = document.getElementById('review-auto-advance-btn');
        var typeFilter = document.getElementById('review-type-filter');
        var confirmBtn = document.getElementById('review-confirm-btn');
        var editBtn = document.getElementById('review-edit-btn');
        var rejectBtn = document.getElementById('review-reject-btn');
        var seekBtn = document.getElementById('review-seek-btn');
        var prevBtn = document.getElementById('review-prev-btn');
        var nextBtn = document.getElementById('review-next-btn');
        var saveEditBtn = document.getElementById('review-save-edit-btn');

        if (!loadBtn) return;

        window.loadReviewMatchSelect = function() {
            if (typeof bridge === 'undefined' || !bridge) return;
            bridge.get_all_matches(function(result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    if (data.error) data = [];
                    var sel = document.getElementById('review-match-select');
                    sel.innerHTML = '<option value="">-- Select Match --</option>';
                    (data || []).forEach(function(m) {
                        var name = m.name || (m.home_team + ' vs ' + m.away_team) || 'Match #' + m.id;
                        sel.innerHTML += '<option value="' + m.id + '">' + escapeHtml(name) + '</option>';
                    });
                } catch(e) { showToast('Failed to load matches.', 'error'); console.warn('loadReviewMatchSelect:', e); }
            });
        };

        loadBtn.addEventListener('click', function() {
            var matchId = parseInt(matchSelect.value, 10);
            if (!matchId) { showToast('Select a match first.', 'warning'); return; }
            loadReviewWorkspace(matchId);
        });

        autoAdvBtn.addEventListener('click', function() {
            _reviewState.autoAdvance = !_reviewState.autoAdvance;
            this.classList.toggle('btn-primary');
            this.classList.toggle('btn-secondary');
            this.innerHTML = _reviewState.autoAdvance ? '⏩ Auto-Advance: ON' : '⏩ Auto-Advance: OFF';
        });

        typeFilter.addEventListener('change', function() {
            renderReviewQueue();
        });

        confirmBtn.addEventListener('click', function() {
            if (_reviewState.currentIndex < 0) return;
            var item = _reviewState.queue[_reviewState.currentIndex];
            submitReviewAction(item.id, 'confirm', '', function() {
                removeFromQueue(_reviewState.currentIndex);
            });
        });

        editBtn.addEventListener('click', function() {
            var fields = document.getElementById('review-edit-fields');
            fields.classList.toggle('hidden');
            if (!fields.classList.contains('hidden')) {
                populateEditFields();
            }
        });

        saveEditBtn.addEventListener('click', function() {
            var item = _reviewState.queue[_reviewState.currentIndex];
            if (!item) return;
            var corrections = {
                event_type: document.getElementById('review-edit-type').value,
                team: document.getElementById('review-edit-team').value,
                completed: document.getElementById('review-edit-completed').checked,
            };
            submitReviewAction(item.id, 'edit', JSON.stringify(corrections), function() {
                document.getElementById('review-edit-fields').classList.add('hidden');
                removeFromQueue(_reviewState.currentIndex);
            });
        });

        rejectBtn.addEventListener('click', function() {
            if (_reviewState.currentIndex < 0) return;
            var item = _reviewState.queue[_reviewState.currentIndex];
            showConfirmDialog('Reject this auto-detected event? It will be deleted.', function() {
                submitReviewAction(item.id, 'reject', '', function() {
                    removeFromQueue(_reviewState.currentIndex);
                });
            });
        });

        seekBtn.addEventListener('click', function() {
            if (_reviewState.currentIndex < 0) return;
            var item = _reviewState.queue[_reviewState.currentIndex];
            var video = document.getElementById('review-video');
            if (video && item.video_time != null) {
                video.currentTime = item.video_time;
            }
        });

        prevBtn.addEventListener('click', function() {
            navigateReview(-1);
        });
        nextBtn.addEventListener('click', function() {
            navigateReview(1);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT')) return;
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            var section = document.getElementById('review-section');
            if (!section || section.classList.contains('hidden')) return;

            switch (e.key) {
                case 'Enter':
                    e.preventDefault();
                    confirmBtn.click();
                    break;
                case 'Delete':
                case 'd':
                    e.preventDefault();
                    rejectBtn.click();
                    break;
                case 'e':
                    e.preventDefault();
                    editBtn.click();
                    break;
                case 'ArrowLeft':
                    prevBtn.click();
                    break;
                case 'ArrowRight':
                    nextBtn.click();
                    break;
                case ' ':
                case 'k':
                    e.preventDefault();
                    var vid = document.getElementById('review-video');
                    if (vid && vid.src) { if (vid.paused) vid.play(); else vid.pause(); }
                    break;
            }
        });

        // Video timeupdate to highlight current event
        var video = document.getElementById('review-video');
        if (video) {
            video.addEventListener('timeupdate', function() {
                highlightReviewEventAtTime(this.currentTime);
            });
        }

        loadReviewMatchSelect();
    }

    function loadReviewWorkspace(matchId) {
        _reviewState.matchId = matchId;
        _reviewState.isLoading = true;

        var status = document.getElementById('review-match-status');
        status.textContent = 'Loading...';

        // Load video
        bridge.get_video_path(matchId, function(result) {
            try {
                var data = JSON.parse(result);
                if (data && data.video_path) {
                    var video = document.getElementById('review-video');
                    video.src = data.video_path;
                    video.load();
                    _reviewState.video = video;
                    status.textContent = 'Ready';
                } else {
                    status.textContent = 'No video';
                }
            } catch(e) {
                status.textContent = 'Error';
                showToast('Failed to load review workspace.', 'error');
                console.warn(e);
            }
        });

        // Load events + detection summary
        loadReviewEvents(matchId);
    }

    function loadReviewEvents(matchId) {
        bridge.get_match_events(matchId, function(result) {
            try {
                var data = JSON.parse(result);
                _reviewState.events = Array.isArray(data) ? data : (data.events || []);
            } catch(e) { _reviewState.events = []; }

            // Separate unreviewed (low confidence, not user_corrected)
            _reviewState.queue = _reviewState.events.filter(function(ev) {
                return !ev.user_corrected && (ev.confidence == null || ev.confidence < 0.7);
            });
            _reviewState.queue.sort(function(a, b) {
                return (a.confidence || 0) - (b.confidence || 0);
            });

            _reviewState.currentIndex = _reviewState.queue.length > 0 ? 0 : -1;

            // Show workspace
            document.getElementById('review-workspace').classList.remove('hidden');
            document.getElementById('review-empty-state').classList.add('hidden');

            renderReviewSummary();
            renderReviewQueue();
            renderReviewEventDetail();
            updateReviewNavButtons();

            _reviewState.isLoading = false;
            status.textContent = _reviewState.queue.length + ' events need review';
        });
    }

    function renderReviewSummary() {
        var container = document.getElementById('review-summary-content');
        if (!container) return;

        var total = _reviewState.events.length;
        var unreviewed = _reviewState.queue.length;
        var corrected = total - unreviewed;
        var pct = total > 0 ? Math.round(corrected / total * 100) : 0;

        // Count by type
        var byType = {};
        _reviewState.events.forEach(function(ev) {
            var t = ev.event_type || 'unknown';
            if (!byType[t]) byType[t] = { total: 0, unreviewed: 0 };
            byType[t].total++;
            if (!ev.user_corrected && (ev.confidence == null || ev.confidence < 0.7)) {
                byType[t].unreviewed++;
            }
        });

        var typeNames = Object.keys(byType).sort();
        var breakdownHtml = '';
        typeNames.forEach(function(t) {
            var stats = byType[t];
            var barW = stats.total > 0 ? Math.round((stats.total - stats.unreviewed) / stats.total * 100) : 0;
            breakdownHtml += '<div class="review-breakdown-row">' +
                '<span>' + escapeHtml(t) + '</span>' +
                '<span>' + stats.unreviewed + '/' + stats.total + '</span>' +
                '</div>';
        });

        container.innerHTML =
            '<div class="review-summary-stats">' +
                '<div class="review-summary-stat"><div class="stat-value">' + total + '</div><div class="stat-label">Total</div></div>' +
                '<div class="review-summary-stat"><div class="stat-value ' + (unreviewed > 0 ? 'danger' : 'success') + '">' + unreviewed + '</div><div class="stat-label">To Review</div></div>' +
                '<div class="review-summary-stat"><div class="stat-value success">' + corrected + '</div><div class="stat-label">Reviewed</div></div>' +
                '<div class="review-summary-stat"><div class="stat-value">' + pct + '%</div><div class="stat-label">Progress</div></div>' +
            '</div>' +
            '<div class="review-summary-breakdown">' + breakdownHtml + '</div>';
    }

    function renderReviewQueue() {
        var list = document.getElementById('review-queue-list');
        var count = document.getElementById('review-queue-count');
        if (!list) return;

        var filter = document.getElementById('review-type-filter').value;
        var queue = _reviewState.queue;
        if (filter) {
            queue = queue.filter(function(ev) { return ev.event_type === filter; });
        }

        count.textContent = queue.length;

        if (queue.length === 0) {
            list.innerHTML = '<div class="review-queue-empty">All events reviewed! 🎉</div>';
            return;
        }

        // Populate type filter options if not done
        var typeFilter = document.getElementById('review-type-filter');
        if (typeFilter.options.length <= 1) {
            var types = {};
            _reviewState.queue.forEach(function(ev) {
                var t = ev.event_type || 'unknown';
                types[t] = true;
            });
            Object.keys(types).sort().forEach(function(t) {
                typeFilter.innerHTML += '<option value="' + escapeHtml(t) + '">' + escapeHtml(t) + '</option>';
            });
        }

        var html = '';
        queue.forEach(function(ev, idx) {
            var globalIdx = _reviewState.queue.indexOf(ev);
            var isActive = globalIdx === _reviewState.currentIndex;
            var conf = ev.confidence || 0;
            var confClass = conf < 0.3 ? 'q-conf-low' : (conf < 0.5 ? 'q-conf-mid' : 'q-conf-high');
            var timeStr = formatCodingTime(ev.timestamp || 0);
            var typeLabel = ev.event_type || 'unknown';
            var teamLabel = ev.team || '';

            html += '<div class="review-queue-item' + (isActive ? ' active' : '') + '" data-event-id="' + ev.id + '" data-idx="' + globalIdx + '">' +
                '<div class="q-type">' + escapeHtml(typeLabel) + '</div>' +
                '<div class="q-time">' + timeStr + (teamLabel ? ' · ' + escapeHtml(teamLabel) : '') + '</div>' +
                '<div class="q-conf ' + confClass + '">' + conf.toFixed(2) + '</div>' +
                '</div>';
        });
        list.innerHTML = html;

        // Click to select
        list.querySelectorAll('.review-queue-item').forEach(function(item) {
            item.addEventListener('click', function() {
                var idx = parseInt(this.dataset.idx, 10);
                if (!isNaN(idx)) {
                    _reviewState.currentIndex = idx;
                    renderReviewQueue();
                    renderReviewEventDetail();
                    updateReviewNavButtons();
                    seekToReviewEvent();
                }
            });
        });
    }

    function renderReviewEventDetail() {
        var detailContainer = document.getElementById('review-event-detail');
        var infoContainer = document.getElementById('review-event-details');
        var badge = document.getElementById('review-confidence-badge');
        var confirmBtn = document.getElementById('review-confirm-btn');
        var editBtn = document.getElementById('review-edit-btn');
        var rejectBtn = document.getElementById('review-reject-btn');
        var seekBtn = document.getElementById('review-seek-btn');

        var idx = _reviewState.currentIndex;
        if (idx < 0 || idx >= _reviewState.queue.length) {
            detailContainer.innerHTML = '<div class="review-detail-empty">All events reviewed! 🎉</div>';
            infoContainer.innerHTML = '<p class="review-detail-placeholder">No event selected.</p>';
            badge.textContent = '--';
            badge.className = 'review-confidence-badge';
            confirmBtn.disabled = true;
            editBtn.disabled = true;
            rejectBtn.disabled = true;
            seekBtn.disabled = true;
            return;
        }

        confirmBtn.disabled = false;
        editBtn.disabled = false;
        rejectBtn.disabled = false;
        seekBtn.disabled = false;

        var ev = _reviewState.queue[idx];
        var conf = ev.confidence || 0;
        var confClass = conf < 0.3 ? 'low' : (conf < 0.5 ? 'mid' : 'high');
        badge.textContent = 'Conf: ' + conf.toFixed(3);
        badge.className = 'review-confidence-badge ' + confClass;

        // Detail in center panel
        var meta = ev._meta || {};
        var metaHtml = '';
        if (meta && typeof meta === 'object') {
            Object.keys(meta).slice(0, 8).forEach(function(k) {
                var v = typeof meta[k] === 'object' ? JSON.stringify(meta[k]) : meta[k];
                metaHtml += '<div class="detail-row"><span class="detail-label">' + escapeHtml(k) + '</span><span class="detail-value">' + escapeHtml(String(v)) + '</span></div>';
            });
        }

        detailContainer.innerHTML = '<div class="review-detail-content">' +
            '<div class="detail-row"><span class="detail-label">Type</span><span class="detail-value">' + escapeHtml(ev.event_type || 'unknown') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Time</span><span class="detail-value">' + formatCodingTime(ev.timestamp || 0) + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Team</span><span class="detail-value">' + escapeHtml(ev.team || '--') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Completed</span><span class="detail-value">' + (ev.completed ? 'Yes' : 'No') + '</span></div>' +
            (ev.from_track_id ? '<div class="detail-row"><span class="detail-label">Player</span><span class="detail-value">#' + ev.from_track_id + '</span></div>' : '') +
            (conf < 0.35 ? '<div class="detail-row" style="color:var(--warning);font-size:0.75rem">⚠ Low confidence — likely needs review</div>' : '') +
            '</div>';

        // Info in right panel
        infoContainer.innerHTML =
            '<div class="detail-row"><span class="detail-label">Event ID</span><span class="detail-value">#' + ev.id + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Type</span><span class="detail-value">' + escapeHtml(ev.event_type || 'unknown') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Timestamp</span><span class="detail-value">' + (ev.timestamp ? ev.timestamp.toFixed(2) + 's' : '--') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Team</span><span class="detail-value">' + escapeHtml(ev.team || '--') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Player</span><span class="detail-value">' + (ev.from_track_id ? '#' + ev.from_track_id : '--') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Confidence</span><span class="detail-value">' + (conf * 100).toFixed(1) + '%</span></div>' +
            '<div class="detail-row"><span class="detail-label">Completed</span><span class="detail-value">' + (ev.completed ? 'Yes' : 'No') + '</span></div>' +
            (metaHtml ? '<hr style="margin:6px 0;border-color:var(--border)">' + metaHtml : '');
    }

    function updateReviewNavButtons() {
        var prevBtn = document.getElementById('review-prev-btn');
        var nextBtn = document.getElementById('review-next-btn');
        var pos = document.getElementById('review-position');
        var idx = _reviewState.currentIndex;
        var total = _reviewState.queue.length;

        prevBtn.disabled = idx <= 0;
        nextBtn.disabled = idx >= total - 1 || total === 0;
        pos.textContent = total > 0 ? (idx + 1) + ' / ' + total : '0 / 0';
    }

    function navigateReview(direction) {
        var newIdx = _reviewState.currentIndex + direction;
        if (newIdx < 0 || newIdx >= _reviewState.queue.length) return;
        _reviewState.currentIndex = newIdx;
        renderReviewQueue();
        renderReviewEventDetail();
        updateReviewNavButtons();
        seekToReviewEvent();
    }

    function seekToReviewEvent() {
        var idx = _reviewState.currentIndex;
        if (idx < 0) return;
        var ev = _reviewState.queue[idx];
        var video = document.getElementById('review-video');
        if (video && ev.timestamp != null) {
            time = Math.max(0, (ev.timestamp || 0) - 2);
            video.currentTime = time;
        }
    }

    function highlightReviewEventAtTime(time) {
        // Highlight nearest unreviewed event within 3 seconds
        var closest = -1;
        var closestDist = 3;
        _reviewState.queue.forEach(function(ev, idx) {
            var dist = Math.abs((ev.timestamp || 0) - time);
            if (dist < closestDist) {
                closestDist = dist;
                closest = idx;
            }
        });
        if (closest >= 0 && closest !== _reviewState.currentIndex) {
            _reviewState.currentIndex = closest;
            renderReviewQueue();
            renderReviewEventDetail();
            updateReviewNavButtons();
        }
    }

    function submitReviewAction(eventId, action, correctionsJson, callback) {
        var mid = _reviewState.matchId;
        bridge.submit_event_correction(mid, eventId, action, correctionsJson, function(result) {
            try {
                var data = JSON.parse(result);
                if (data.success) {
                    showToast('Event ' + data.action + '!', 'success');
                    if (callback) callback();
                } else {
                    showToast('Failed: ' + (data.error || 'Unknown'), 'error');
                }
            } catch(e) {
                showToast('Error submitting correction.', 'error');
            }
        });
    }

    function removeFromQueue(idx) {
        _reviewState.queue.splice(idx, 1);
        if (_reviewState.queue.length === 0) {
            _reviewState.currentIndex = -1;
        } else if (idx >= _reviewState.queue.length) {
            _reviewState.currentIndex = _reviewState.queue.length - 1;
        }
        renderReviewSummary();
        renderReviewQueue();
        renderReviewEventDetail();
        updateReviewNavButtons();

        // Auto-advance to next
        if (_reviewState.autoAdvance && _reviewState.currentIndex >= 0) {
            seekToReviewEvent();
        }

        // Recalculate status
        var status = document.getElementById('review-match-status');
        if (status) {
            status.textContent = _reviewState.queue.length + ' events need review';
        }
    }

    function populateEditFields() {
        var idx = _reviewState.currentIndex;
        if (idx < 0) return;
        var ev = _reviewState.queue[idx];

        // Populate type dropdown
        var typeSel = document.getElementById('review-edit-type');
        var allTypes = ['pass', 'shot', 'goal', 'tackle', 'interception', 'dribble', 'corner',
            'free_kick', 'throw_in', 'clearance', 'cross', 'block', 'carry', 'duel',
            'foul', 'offside', 'hand_ball', 'yellow_card', 'red_card', 'save', 'ball_out'];
        typeSel.innerHTML = '';
        allTypes.forEach(function(t) {
            typeSel.innerHTML += '<option value="' + t + '"' + (t === ev.event_type ? ' selected' : '') + '>' + t + '</option>';
        });

        document.getElementById('review-edit-team').value = ev.team || 'home';
        document.getElementById('review-edit-completed').checked = !!ev.completed;
    }

    // ── End Event Review Workspace ──────────────────────────────

    document.addEventListener('change', function(e) {
        if (e.target.id === 'timeline-filter-type') {
            _timelinePageState.page = 1;
            renderTimeline(window._timelineEvents || []);
        }
        // Per-page select
        if (e.target.classList.contains('per-page-select')) {
            _timelinePageState.perPage = parseInt(e.target.value, 10);
            _timelinePageState.page = 1;
            renderTimelineTable(window._timelineEvents || []);
        }
    });

    document.addEventListener('input', function(e) {
        // Column filter inputs in table header
        if (e.target.classList.contains('col-filter-input')) {
            var key = e.target.dataset.filterKey;
            _timelineFilters[key] = e.target.value;
            _timelinePageState.page = 1;
            renderTimelineTable(window._timelineEvents || []);
        }
        // Timeline search input
        if (e.target.id === 'timeline-search') {
            _timelineSearchText = e.target.value;
            _timelinePageState.page = 1;
            renderTimeline(window._timelineEvents || []);
        }
        // Roster search input
        if (e.target.id === 'roster-search') {
            _rosterSearchText = e.target.value;
            _rosterPageState.page = 1;
            if (_currentRosterView === 'roster-table') {
                renderRosterTable();
            } else {
                loadPlayerProfiles();
            }
        }
    });

    // ── Roster table rendering ──────────────────────────────

    function renderRosterTable() {
        var wrapper = document.getElementById('player-roster-table-wrapper');
        var roster = document.getElementById('player-roster');
        if (!wrapper) return;
        if (!roster) return;
        roster.classList.add('hidden');
        wrapper.classList.remove('hidden');

        var data = window._rosterData || [];
        var text = (_rosterSearchText || '').toLowerCase().trim();
        if (text) {
            data = data.filter(function(p) {
                return (p.name || '').toLowerCase().indexOf(text) >= 0 ||
                       (p.position || '').toLowerCase().indexOf(text) >= 0;
            });
        }

        var key = _rosterSortState.key;
        var dir = _rosterSortState.dir === 'asc' ? 1 : -1;
        data.sort(function(a, b) {
            var va = a[key] != null ? a[key] : '';
            var vb = b[key] != null ? b[key] : '';
            if (typeof va === 'string') return va.localeCompare(vb) * dir;
            return (va - vb) * dir;
        });

        var perPage = _rosterPageState.perPage || 25;
        var total = data.length;
        var totalPages = Math.max(1, Math.ceil(total / perPage));
        var page = Math.min(_rosterPageState.page, totalPages);
        var start = (page - 1) * perPage;
        var pageData = data.slice(start, start + perPage);

        var sortKey = _rosterSortState.key;
        var sortDir = _rosterSortState.dir;

        var html = '<table class="data-table" id="roster-data-table"><thead><tr>';
        var rosterCols = [
            { key: 'name', label: 'Name', filterable: true },
            { key: 'position', label: 'Position', filterable: true },
            { key: 'minutes', label: 'Minutes', filterable: false },
            { key: 'xg', label: 'xG', filterable: false },
            { key: 'xa', label: 'xA', filterable: false },
            { key: 'pass_pct', label: 'Pass%', filterable: false },
            { key: 'rating', label: 'Rating', filterable: false },
        ];
        rosterCols.forEach(function(col) {
            var isSorted = sortKey === col.key;
            var sortClass = isSorted ? (sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc') : '';
            html += '<th class="sortable ' + sortClass + '" data-roster-sort="' + col.key + '">' +
                '<span class="th-label">' + col.label + '</span><span class="sort-indicator"></span></th>';
        });
        html += '</tr></thead><tbody>';

        if (pageData.length === 0) {
            html += '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:1.5rem">No players match filters</td></tr>';
        } else {
            pageData.forEach(function(p) {
                html += '<tr>';
                html += '<td>' + escapeHtml(p.name || '') + '</td>';
                html += '<td>' + escapeHtml(p.position || '') + '</td>';
                html += '<td class="numeric">' + (p.minutes != null ? p.minutes : '--') + '</td>';
                html += '<td class="numeric">' + (p.xg != null ? p.xg.toFixed(3) : '--') + '</td>';
                html += '<td class="numeric">' + (p.xa != null ? p.xa.toFixed(3) : '--') + '</td>';
                html += '<td class="numeric">' + (p.pass_pct != null ? p.pass_pct.toFixed(1) + '%' : '--') + '</td>';
                html += '<td class="numeric">' + (p.rating != null ? p.rating.toFixed(1) : '--') + '</td>';
                html += '</tr>';
            });
        }
        html += '</tbody></table>';

        // Pagination
        html += '<div class="data-table-pagination">';
        html += '<span class="pagination-info">Showing ' + (total > 0 ? (start + 1) + '-' + Math.min(start + perPage, total) : 0) + ' of ' + total + ' players</span>';
        html += '<div class="pagination-controls">';
        html += '<button class="pagination-btn" data-roster-page="prev" ' + (page <= 1 ? 'disabled' : '') + '>&#9664;</button>';
        var maxB = 5, sP = Math.max(1, page - Math.floor(maxB / 2)), eP = Math.min(totalPages, sP + maxB - 1);
        if (eP - sP < maxB - 1) sP = Math.max(1, eP - maxB + 1);
        if (sP > 1) html += '<button class="pagination-btn" data-roster-page="1">1</button>' + (sP > 2 ? '<span style="color:var(--text-muted);padding:0 2px">...</span>' : '');
        for (var pi = sP; pi <= eP; pi++) {
            html += '<button class="pagination-btn ' + (pi === page ? 'active' : '') + '" data-roster-page="' + pi + '">' + pi + '</button>';
        }
        if (eP < totalPages) html += (eP < totalPages - 1 ? '<span style="color:var(--text-muted);padding:0 2px">...</span>' : '') + '<button class="pagination-btn" data-roster-page="' + totalPages + '">' + totalPages + '</button>';
        html += '<button class="pagination-btn" data-roster-page="next" ' + (page >= totalPages ? 'disabled' : '') + '>&#9654;</button>';
        html += '<select class="per-page-select">';
        [25, 50, 100].forEach(function(pp) {
            html += '<option value="' + pp + '" ' + (pp === perPage ? 'selected' : '') + '>' + pp + ' / page</option>';
        });
        html += '</select></div></div>';

        wrapper.innerHTML = html;
    }

    // ── View toggle wiring ──

    function setupViewToggles() {
        // Timeline view toggle
        document.querySelectorAll('.view-toggle-btn[data-view="timeline"], .view-toggle-btn[data-view="table"]').forEach(function(btn) {
            if (btn.dataset.view === 'timeline' || btn.dataset.view === 'table') {
                btn.addEventListener('click', function() {
                    var view = this.dataset.view;
                    var parent = this.closest('.view-toggle');
                    parent.querySelectorAll('.view-toggle-btn').forEach(function(b) { b.classList.remove('active'); });
                    this.classList.add('active');
                    _currentTimelineView = view;
                    _timelinePageState.page = 1;
                    renderTimeline(window._timelineEvents || []);
                });
            }
        });

        // Roster view toggle
        document.querySelectorAll('.view-toggle-btn[data-view="roster-cards"], .view-toggle-btn[data-view="roster-table"]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var view = this.dataset.view;
                var parent = this.closest('.view-toggle');
                parent.querySelectorAll('.view-toggle-btn').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                _currentRosterView = view;
                if (view === 'roster-table') {
                    renderRosterTable();
                } else {
                    var wrapper = document.getElementById('player-roster-table-wrapper');
                    var roster = document.getElementById('player-roster');
                    if (wrapper) wrapper.classList.add('hidden');
                    if (roster) {
                        roster.classList.remove('hidden');
                        loadPlayerProfiles();
                    }
                }
            });
        });

        // Roster table sort
        document.addEventListener('click', function(e) {
            var th = e.target.closest('th[data-roster-sort]');
            if (th) {
                var key = th.dataset.rosterSort;
                if (_rosterSortState.key === key) {
                    _rosterSortState.dir = _rosterSortState.dir === 'asc' ? 'desc' : 'asc';
                } else {
                    _rosterSortState.key = key;
                    _rosterSortState.dir = 'asc';
                }
                renderRosterTable();
                return;
            }
            // Roster pagination
            if (e.target.closest('[data-roster-page]')) {
                var btn = e.target.closest('[data-roster-page]');
                var page = btn.dataset.rosterPage;
                if (page === 'prev') {
                    _rosterPageState.page = Math.max(1, _rosterPageState.page - 1);
                } else if (page === 'next') {
                    var total = (window._rosterData || []).length;
                    var maxP = Math.ceil(total / _rosterPageState.perPage);
                    _rosterPageState.page = Math.min(maxP, _rosterPageState.page + 1);
                } else {
                    _rosterPageState.page = parseInt(page, 10);
                }
                renderRosterTable();
                return;
            }
        });

        document.addEventListener('change', function(e) {
            if (e.target.closest('.data-table-pagination .per-page-select') && document.getElementById('roster-data-table')) {
                _rosterPageState.perPage = parseInt(e.target.value, 10);
                _rosterPageState.page = 1;
                renderRosterTable();
            }
        });
    }

    // ── Batch action wiring ──

    function setupBatchActions() {
        document.getElementById('batch-delete-btn')?.addEventListener('click', function() {
            var ids = Array.from(_selectedEventIds);
            if (ids.length === 0) return;
            showConfirmDialog('Delete ' + ids.length + ' selected event(s)?', function() {
                var promises = ids.map(function(eid) {
                    return bridge.delete_event(eid).then(function(json) {
                        try {
                            var result = JSON.parse(json);
                            return result.success;
                        } catch (ex) { return false; }
                    }).catch(function() { return false; });
                });
                Promise.all(promises).then(function() {
                    _selectedEventIds.clear();
                    _updateBatchActionBar();
                    setTimeout(loadEventTimeline, 100);
                    showToast('Deleted ' + ids.length + ' event(s)', 'info');
                });
            });
        });

        document.getElementById('batch-export-csv-btn')?.addEventListener('click', function() {
            batchExport('csv');
        });

        document.getElementById('batch-export-json-btn')?.addEventListener('click', function() {
            batchExport('json');
        });
    }

    function batchExport(format) {
        var ids = Array.from(_selectedEventIds);
        if (ids.length === 0) return;
        var events = (window._timelineEvents || []).filter(function(e) {
            return ids.indexOf(e.id) >= 0;
        });
        if (events.length === 0) return;

        if (format === 'csv') {
            var headers = ['id', 'event_type', 'team', 'player_name', 'timestamp', 'xg', 'xa', 'xt'];
            var rows = events.map(function(e) {
                return [e.id, e.event_type, e.team, e.player_name || '', e.timestamp || 0, e.xg != null ? e.xg : '', e.xa != null ? e.xa : '', e.xt != null ? e.xt : ''];
            });
            var csv = headers.join(',') + '\n' + rows.map(function(r) {
                return r.map(function(c) {
                    var s = String(c != null ? c : '');
                    return s.indexOf(',') >= 0 || s.indexOf('"') >= 0 ? '"' + s.replace(/"/g, '""') + '"' : s;
                }).join(',');
            }).join('\n');
            var blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'selected-events.csv';
            document.body.appendChild(a);
            a.click();
            setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
        } else {
            var json = JSON.stringify(events, null, 2);
            var blob = new Blob([json], { type: 'application/json' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'selected-events.json';
            document.body.appendChild(a);
            a.click();
            setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
        }
    }

    var _aiMatchVideo = document.getElementById('match-video');
    if (_aiMatchVideo) {
        _aiMatchVideo.addEventListener('timeupdate', function() {
            highlightCurrentTimelineItem(this.currentTime);
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       Wave A — Telestration Engine
       ═══════════════════════════════════════════════════════════════ */

    var _telestrateState = {
        active: false,
        tool: 'arrow',
        color: '#ff0000',
        width: 3,
        strokes: [],
        redoStack: [],
        isDrawing: false,
        startX: 0, startY: 0,
        currentText: '',
        textInput: null,
    };

    // ── Sprint 1: Multi-Angle Video, Trimming, Highlight Reel ──
    var _maInitialized = false;
    var _maSources = [];
    var _maTrimIn = null;
    var _maTrimOut = null;
    var _maVideoSync = {};

    function initMultiAngle() {
        if (_maInitialized) return;
        _maInitialized = true;

        var addBtn = document.getElementById('ma-add-source-btn');
        var loadBtn = document.getElementById('ma-load-btn');
        var clearBtn = document.getElementById('ma-clear-btn');
        var syncMasterBtn = document.getElementById('ma-sync-master-btn');
        var playAllBtn = document.getElementById('ma-play-all-btn');
        var pauseAllBtn = document.getElementById('ma-pause-all-btn');
        var setInBtn = document.getElementById('ma-set-in-btn');
        var setOutBtn = document.getElementById('ma-set-out-btn');
        var trimExportBtn = document.getElementById('ma-trim-export-btn');
        var reelGenBtn = document.getElementById('ma-reel-generate-btn');
        var fileInput = document.getElementById('ma-source-file-input');
        var sourceList = document.getElementById('ma-source-list');

        addBtn.addEventListener('click', function () { fileInput.click(); });
        fileInput.addEventListener('change', function (e) {
            Array.from(e.target.files).forEach(function (f) {
                var path = f.name;
                var url = URL.createObjectURL(f);
                _maSources.push({ label: f.name, path: path, url: url });
                renderSourceList();
            });
            fileInput.value = '';
        });

        clearBtn.addEventListener('click', function () {
            _maSources = [];
            _maTrimIn = null;
            _maTrimOut = null;
            renderSourceList();
            document.getElementById('ma-workspace').classList.add('hidden');
            document.getElementById('ma-status').textContent = '';
        });

        loadBtn.addEventListener('click', function () {
            var paths = _maSources.map(function (s) { return { path: s.path, label: s.label }; });
            if (paths.length === 0) { showToast('Add at least one video source', 'warning'); return; }
            bridge.sync_load(JSON.stringify(paths), function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    loadSyncWorkspace(data);
                } catch (e) { showToast('Failed to parse sync response', 'error'); }
            });
        });

        syncMasterBtn.addEventListener('click', syncAllToMaster);
        playAllBtn.addEventListener('click', function () { playAllVideos(true); });
        pauseAllBtn.addEventListener('click', function () { playAllVideos(false); });

        setInBtn.addEventListener('click', function () {
            var master = document.getElementById('ma-video-0');
            if (master) { _maTrimIn = master.currentTime; updateTrimDisplay(); }
        });
        setOutBtn.addEventListener('click', function () {
            var master = document.getElementById('ma-video-0');
            if (master) { _maTrimOut = master.currentTime; updateTrimDisplay(); }
        });
        trimExportBtn.addEventListener('click', function () {
            if (_maTrimIn === null || _maTrimOut === null) { showToast('Set in and out points first', 'warning'); return; }
            if (_maTrimIn >= _maTrimOut) { showToast('In point must be before out point', 'warning'); return; }
            var masterPath = _maSources.length > 0 ? _maSources[0].path : '';
            if (!masterPath) { showToast('No source loaded', 'warning'); return; }
            var outName = document.getElementById('ma-trim-output').value || ('trim_' + Math.round(_maTrimIn) + '_' + Math.round(_maTrimOut) + '.mp4');
            bridge.trim_video(masterPath, _maTrimIn, _maTrimOut, outName, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    showToast('Trim exported: ' + (data.output || ''), 'success');
                } catch (e) { showToast('Trim export failed', 'error'); }
            });
        });
        reelGenBtn.addEventListener('click', function () {
            if (_maSources.length === 0) { showToast('Load a video first', 'warning'); return; }
            var clip = [{
                video_path: _maSources[0].path,
                start_s: _maTrimIn || 0,
                end_s: _maTrimOut || 60,
                label: 'clip_1'
            }];
            var outName = document.getElementById('ma-reel-output').value || 'highlight_reel.mp4';
            bridge.reel_compose(JSON.stringify(clip), outName, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    document.getElementById('ma-reel-status').textContent = data.output_path ? 'Reel saved: ' + data.output_path : data.clip_count + ' clips, ' + data.total_duration_s + 's';
                    showToast('Reel generated: ' + data.clip_count + ' clips', 'success');
                } catch (e) { showToast('Reel generation failed', 'error'); }
            });
        });

        document.querySelectorAll('.ma-offset-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var target = parseInt(this.dataset.target);
                var delta = parseFloat(this.dataset.delta);
                if (isNaN(target) || isNaN(delta)) return;
                bridge.sync_set_offset(target, 0, function (cur) {
                    bridge.sync_set_offset(target, delta, function (r) {
                        try {
                            var d = JSON.parse(r);
                            if (d.ok) { document.getElementById('ma-offset-' + target).textContent = (d.offset_s || 0).toFixed(1) + 's'; }
                        } catch (e) {}
                    });
                });
            });
        });
    }

    function renderSourceList() {
        var list = document.getElementById('ma-source-list');
        if (!list) return;
        if (_maSources.length === 0) {
            list.innerHTML = '<p class="hint" data-i18n="maSourceHint">Add video sources and click "Load Videos".</p>';
            return;
        }
        var html = '';
        _maSources.forEach(function (s, i) {
            html += '<div class="ma-source-item">';
            html += '<span class="ma-source-label" title="' + escapeHtml(s.label) + '">' + escapeHtml(s.label) + '</span>';
            html += '<span class="ma-source-remove" data-index="' + i + '">✕</span>';
            html += '</div>';
        });
        list.innerHTML = html;
        list.querySelectorAll('.ma-source-remove').forEach(function (el) {
            el.addEventListener('click', function () {
                var idx = parseInt(this.dataset.index);
                if (!isNaN(idx) && idx >= 0 && idx < _maSources.length) {
                    if (_maSources[idx].url) URL.revokeObjectURL(_maSources[idx].url);
                    _maSources.splice(idx, 1);
                    renderSourceList();
                }
            });
        });
    }

    function loadSyncWorkspace(data) {
        var ws = document.getElementById('ma-workspace');
        ws.classList.remove('hidden');
        document.getElementById('ma-status').textContent = data.sources.length + ' sources loaded';
        var sources = data.sources || [];
        for (var i = 0; i < 3; i++) {
            var video = document.getElementById('ma-video-' + i);
            var nameEl = document.getElementById('ma-name-' + i);
            if (i < sources.length) {
                var src = sources[i];
                if (_maSources[i] && _maSources[i].url) {
                    video.src = _maSources[i].url;
                }
                nameEl.textContent = src.label || 'Angle ' + (i + 1);
                var cell = video.closest('.ma-video-cell');
                if (cell) cell.style.display = '';
            } else {
                var cell = video.closest('.ma-video-cell');
                if (cell) cell.style.display = 'none';
            }
        }
        // Wire master video timeupdate to sync slaves
        var masterVideo = document.getElementById('ma-video-0');
        if (masterVideo) {
            masterVideo.removeEventListener('timeupdate', _maSyncHandler);
            _maSyncHandler = function () {
                var t = masterVideo.currentTime;
                bridge.sync_positions(t, function (result) {
                    try {
                        var pos = JSON.parse(result);
                        if (pos.error) return;
                        (pos.positions || []).forEach(function (p) {
                            if (p.index === 0) return;
                            var slave = document.getElementById('ma-video-' + p.index);
                            if (slave && Math.abs(slave.currentTime - p.time_s) > 0.3) {
                                slave.currentTime = p.time_s;
                            }
                            var offEl = document.getElementById('ma-offset-' + p.index);
                            if (offEl) offEl.textContent = (p.time_s - t).toFixed(1) + 's';
                        });
                    } catch (e) {}
                });
            };
            masterVideo.addEventListener('timeupdate', _maSyncHandler);
        }
    }
    var _maSyncHandler = null;

    // ── Sprint 3: Team Collaboration ──
    var _collabInitialized = false;

    function initCollaboration() {
        if (_collabInitialized) return;
        _collabInitialized = true;

        // User tabs
        document.querySelectorAll('[data-ctab]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                document.querySelectorAll('[data-ctab]').forEach(function (b) { b.classList.remove('active'); });
                this.classList.add('active');
                document.querySelectorAll('.collab-tab-content').forEach(function (tc) { tc.classList.add('hidden'); });
                var tab = document.getElementById('collab-' + this.dataset.ctab);
                if (tab) tab.classList.remove('hidden');
            });
        });

        // Users
        document.getElementById('collab-add-user-btn').addEventListener('click', function () {
            var uname = document.getElementById('collab-username').value.trim();
            var dname = document.getElementById('collab-display-name').value.trim();
            var role = document.getElementById('collab-role').value;
            if (!uname) { showToast('Username required', 'warning'); return; }
            bridge.create_collab_user(uname, dname || uname, role, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    showToast('User added: ' + data.user.username, 'success');
                    loadCollabUsers();
                } catch (e) { showToast('Failed', 'error'); }
            });
        });

        // Comments
        document.getElementById('collab-add-comment-btn').addEventListener('click', function () {
            var mid = parseInt(document.getElementById('collab-comment-match').value);
            var eid = parseInt(document.getElementById('collab-comment-event-select').value) || 0;
            var text = document.getElementById('collab-comment-text').value.trim();
            if (!mid || !text) { showToast('Match ID and comment text required', 'warning'); return; }
            bridge.add_comment(mid, eid, 0, text, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    showToast('Comment added', 'success');
                    document.getElementById('collab-comment-text').value = '';
                    loadCollabComments();
                } catch (e) { showToast('Failed', 'error'); }
            });
        });
        document.getElementById('collab-load-comments-btn').addEventListener('click', loadCollabComments);
        document.getElementById('collab-load-events-btn').addEventListener('click', function () {
            var mid = parseInt(document.getElementById('collab-comment-match').value);
            if (!mid) { showToast('Match ID required', 'warning'); return; }
            bridge.get_match_events(mid, function (result) {
                try {
                    var data = typeof result === 'string' ? JSON.parse(result) : result;
                    var events = data.events || data || [];
                    var select = document.getElementById('collab-comment-event-select');
                    select.innerHTML = '<option value="0">All Events</option>';
                    events.forEach(function (ev) {
                        var opt = document.createElement('option');
                        opt.value = ev.id || ev.event_id || 0;
                        opt.textContent = '#' + opt.value + ' ' + (ev.type || ev.event_type || '') + ' (' + (ev.timestamp || ev.time || '?') + ')';
                        select.appendChild(opt);
                    });
                    showToast('Loaded ' + events.length + ' events', 'success');
                } catch (e) { showToast('Failed to load events', 'error'); }
            });
        });

        function loadCollabComments() {
            var mid = parseInt(document.getElementById('collab-comment-match').value);
            var eid = parseInt(document.getElementById('collab-comment-event-select').value) || 0;
            if (!mid) { showToast('Match ID required', 'warning'); return; }
            if (eid > 0) {
                bridge.get_event_comments(mid, eid, function (result) {
                    try {
                        var data = JSON.parse(result);
                        renderCollabComments(data.comments || []);
                    } catch (e) { showToast('Failed', 'error'); }
                });
            } else {
                bridge.get_comments(mid, 0, function (result) {
                    try {
                        var data = JSON.parse(result);
                        renderCollabComments(data.comments || []);
                    } catch (e) { showToast('Failed', 'error'); }
                });
            }
        }

        // Projects
        document.getElementById('collab-export-btn').addEventListener('click', function () {
            var mid = parseInt(document.getElementById('collab-export-match').value);
            if (!mid) { showToast('Match ID required', 'warning'); return; }
            bridge.export_project(mid, function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.error) { showToast(data.error, 'error'); return; }
                    var blob = new Blob([JSON.stringify(data.project, null, 2)], { type: 'application/json' });
                    var url = URL.createObjectURL(blob);
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = 'match_' + mid + '.kawkab';
                    a.click();
                    URL.revokeObjectURL(url);
                    showToast('Project exported', 'success');
                } catch (e) { showToast('Export failed', 'error'); }
            });
        });

        var importFileInput = document.getElementById('collab-import-file');
        document.getElementById('collab-import-btn').addEventListener('click', function () { importFileInput.click(); });
        importFileInput.addEventListener('change', function (e) {
            if (e.target.files.length === 0) return;
            var reader = new FileReader();
            reader.onload = function (ev) {
                bridge.import_project(ev.target.result, function (result) {
                    try {
                        var data = JSON.parse(result);
                        if (data.error) { showToast(data.error, 'error'); return; }
                        showToast('Project imported: ' + (data.comments_imported || 0) + ' comments', 'success');
                        document.getElementById('collab-project-result').textContent = 'Match ID: ' + (data.match.id || '?') + ' imported successfully';
                    } catch (err) { showToast('Import failed', 'error'); }
                });
            };
            reader.readAsText(e.target.files[0]);
            importFileInput.value = '';
        });

        // Activity
        document.getElementById('collab-refresh-activity-btn').addEventListener('click', loadCollabActivity);

        // Sidebar activity
        initSidebarActivity();

        // Mention polling
        initMentionPolling();

        // Load initial data
        loadCollabUsers();
        loadCollabActivity();
    }

    function loadCollabUsers() {
        bridge.get_collab_users(function (result) {
            try {
                var data = JSON.parse(result);
                var el = document.getElementById('collab-user-list');
                if (!el) return;
                if (!data.users || data.users.length === 0) {
                    el.innerHTML = '<p class="hint">No team members.</p>';
                    return;
                }
                var html = '';
                data.users.forEach(function (u) {
                    html += '<div class="collab-user-item">' +
                        '<span class="collab-user-name">' + escapeHtml(u.display_name || u.username) + '</span>' +
                        '<span class="collab-user-role">' + escapeHtml(u.role) + '</span>' +
                        '<span class="collab-user-uname">@' + escapeHtml(u.username) + '</span>' +
                        '</div>';
                });
                el.innerHTML = html;
            } catch (e) {}
        });
    }

    function renderCollabComments(comments) {
        var el = document.getElementById('collab-comment-list');
        if (!el) return;
        if (!comments || comments.length === 0) {
            el.innerHTML = '<p class="hint">No comments found.</p>';
            return;
        }
        var html = '';
        comments.forEach(function (c) {
            var seekLink = '';
            if (c.event_id && c.event_id > 0) {
                seekLink = ' <a href="#" class="collab-seek-link" data-event-id="' + c.event_id + '" data-match-id="' + c.match_id + '" title="Seek to event">(seek)</a>';
            }
            html += '<div class="collab-comment-item">' +
                '<strong>' + escapeHtml(c.username) + '</strong> ' +
                '<span class="collab-comment-text">' + escapeHtml(c.text) + '</span>' +
                '<span class="collab-comment-meta">Match ' + c.match_id + (c.event_id ? ' | Event #' + c.event_id : '') + ' | ' + (c.created_at || '').slice(0, 19).replace('T', ' ') + '</span>' +
                seekLink +
                '</div>';
        });
        el.innerHTML = html;
        el.querySelectorAll('.collab-seek-link').forEach(function (link) {
            link.addEventListener('click', function (e) {
                e.preventDefault();
                var eventId = parseInt(this.dataset.eventId);
                var matchId = parseInt(this.dataset.matchId);
                if (typeof video !== 'undefined' && video && typeof video.currentTime !== 'undefined') {
                    bridge.get_event_timestamp(matchId, eventId, function (result) {
                        try {
                            var data = JSON.parse(result);
                            var ts = data.timestamp || data.time || 0;
                            video.currentTime = ts;
                            showToast('Seeked to event #' + eventId + ' at ' + ts.toFixed(1) + 's', 'info');
                        } catch (e) { showToast('Could not seek', 'error'); }
                    });
                } else {
                    showToast('Video player not available', 'warning');
                }
            });
        });
    }

    function loadCollabActivity() {
        bridge.get_activity_feed(50, function (result) {
            try {
                var data = JSON.parse(result);
                var el = document.getElementById('collab-activity-list');
                if (!el) return;
                if (!data.activities || data.activities.length === 0) {
                    el.innerHTML = '<p class="hint">No activity yet.</p>';
                    return;
                }
                var html = '';
                data.activities.forEach(function (a) {
                    html += '<div class="collab-activity-item">' +
                        '<span class="collab-activity-user">' + escapeHtml(a.username) + '</span> ' +
                        '<span class="collab-activity-action">' + escapeHtml(a.action) + '</span> ' +
                        '<span class="collab-activity-desc">' + escapeHtml(a.description) + '</span>' +
                        '<span class="collab-activity-time">' + (a.created_at || '').slice(0, 19).replace('T', ' ') + '</span>' +
                        '</div>';
                });
                el.innerHTML = html;
            } catch (e) {}
        });
    }

    // Phase 6 Sprint 5: Sidebar Activity + Mention Polling
    var _sidebarActivityInterval = null;
    var _mentionPollInterval = null;

    function initSidebarActivity() {
        var sidebarBtn = document.querySelector('[data-route="collaboration"]');
        var sidebarPanel = document.getElementById('sidebar-activity');
        if (!sidebarBtn || !sidebarPanel) return;

        document.getElementById('sidebar-activity-close').addEventListener('click', function () {
            sidebarPanel.classList.add('hidden');
        });

        loadSidebarActivity();
        if (_sidebarActivityInterval) clearInterval(_sidebarActivityInterval);
        _sidebarActivityInterval = setInterval(loadSidebarActivity, 30000);
    }

    function loadSidebarActivity() {
        bridge.get_activity_feed(10, function (result) {
            try {
                var data = JSON.parse(result);
                var el = document.getElementById('sidebar-activity-list');
                if (!el) return;
                if (!data.activities || data.activities.length === 0) {
                    el.innerHTML = '<p class="hint">No recent activity.</p>';
                    return;
                }
                var iconMap = { analyze: '📊', comment: '💬', export: '📤', import: '📥', tag: '🏷️', review: '🔍', user_created: '👤' };
                var html = '';
                data.activities.forEach(function (a) {
                    var icon = iconMap[a.action] || '📋';
                    var timeAgo = '';
                    if (a.created_at) {
                        var then = new Date(a.created_at);
                        var now = new Date();
                        var diffSec = Math.floor((now - then) / 1000);
                        if (diffSec < 60) timeAgo = diffSec + 's ago';
                        else if (diffSec < 3600) timeAgo = Math.floor(diffSec / 60) + 'm ago';
                        else if (diffSec < 86400) timeAgo = Math.floor(diffSec / 3600) + 'h ago';
                        else timeAgo = Math.floor(diffSec / 86400) + 'd ago';
                    }
                    html += '<div class="sidebar-activity-item">' +
                        '<span class="sidebar-activity-icon">' + icon + '</span>' +
                        '<div class="sidebar-activity-body">' +
                        '<span class="activity-user">' + escapeHtml(a.username) + '</span> ' +
                        '<span class="activity-desc">' + escapeHtml(a.description) + '</span>' +
                        '<span class="activity-time">' + timeAgo + '</span>' +
                        '</div></div>';
                });
                el.innerHTML = html;
            } catch (e) {}
        });
    }

    function initMentionPolling() {
        if (_mentionPollInterval) clearInterval(_mentionPollInterval);
        pollMentions();
        _mentionPollInterval = setInterval(pollMentions, 30000);
    }

    function pollMentions() {
        bridge.get_collab_users(function (result) {
            try {
                var data = JSON.parse(result);
                var users = data.users || [];
                if (users.length === 0) return;
                var me = users[0].username;
                bridge.get_mentions(me, function (res) {
                    try {
                        var mData = JSON.parse(res);
                        var dot = document.getElementById('collab-mention-dot');
                        if (!dot) return;
                        var unread = mData.unread || 0;
                        if (unread > 0) {
                            dot.classList.remove('hidden');
                            dot.title = unread + ' unread mention' + (unread > 1 ? 's' : '');
                        } else {
                            dot.classList.add('hidden');
                        }
                    } catch (e) {}
                });
            } catch (e) {}
        });
    }

    // ── Sprint 4: Live Tagging ──
    var _liveInitialized = false;
    var _liveSessionActive = false;
    var _liveHotkeys = {};

    function initLiveTagging() {
        if (_liveInitialized) return;
        _liveInitialized = true;

        var startBtn = document.getElementById('live-start-btn');
        var stopBtn = document.getElementById('live-stop-btn');
        var clearBtn = document.getElementById('live-clear-btn');
        var exportBtn = document.getElementById('live-export-btn');
        var homeInput = document.getElementById('live-home-team');
        var awayInput = document.getElementById('live-away-team');
        var status = document.getElementById('live-status');

        startBtn.onclick = function() {
            if (_liveSessionActive) return;
            var home = homeInput.value.trim() || 'Home';
            var away = awayInput.value.trim() || 'Away';
            bridge.live_start_session(home, away).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { showToast(r.error, 'error'); return; }
                _liveSessionActive = true;
                startBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
                document.getElementById('live-dashboard-toggle').classList.remove('hidden');
                status.textContent = r.message || 'Session active';
                loadLiveHotkeys();
                updateLiveStats();
                initLiveDashboard();
            }).catch(_catchBridge('live_start_session'));
        };

        stopBtn.onclick = function() {
            if (!_liveSessionActive) return;
            bridge.live_stop_session().then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { showToast(r.error, 'error'); return; }
                _liveSessionActive = false;
                startBtn.classList.remove('hidden');
                stopBtn.classList.add('hidden');
                document.getElementById('live-dashboard-toggle').classList.add('hidden');
                document.getElementById('live-kpi-dashboard').classList.add('hidden');
                document.getElementById('live-dashboard-panels').classList.add('hidden');
                status.textContent = r.message || 'Session stopped';
                loadLiveTags();
                if (_liveXgChartInstance) { _liveXgChartInstance.destroy(); _liveXgChartInstance = null; }
            });
        };

        clearBtn.onclick = function() {
            bridge.live_clear_tags().then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { showToast(r.error, 'error'); return; }
                updateLiveStats();
                loadLiveTags();
                showToast('Tags cleared', 'info');
            }).catch(_catchBridge('live_stop_session'));
        };

        exportBtn.onclick = function() {
            bridge.live_export().then(function(raw) {
                var d = JSON.parse(raw);
                if (d.error) { showToast(d.error, 'error'); return; }
                var blob = new Blob([JSON.stringify(d, null, 2)], {type:'application/json'});
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'live-tags.json';
                a.click();
                URL.revokeObjectURL(a.href);
                showToast('Exported ' + d.total + ' tags', 'info');
            }).catch(_catchBridge('live_clear_tags'));
        };

        // Period buttons
        document.querySelectorAll('.live-period-btn').forEach(function(btn) {
            btn.onclick = function() {
                if (!_liveSessionActive) return;
                var period = parseInt(this.getAttribute('data-period'), 10);
                bridge.live_set_period(period);
                document.querySelectorAll('.live-period-btn').forEach(function(b) { b.classList.remove('btn-primary'); b.classList.add('btn-sm'); });
                this.classList.add('btn-primary');
                this.classList.remove('btn-sm');
            };
        });

        // Keyboard listener
        document.addEventListener('keydown', function liveKeyHandler(e) {
            if (!_liveSessionActive) return;
            var section = document.getElementById('livetagging-section');
            if (!section || section.classList.contains('hidden')) return;
            var key = e.key.toLowerCase();
            if (key === ' ' || key === 'enter' || key === 'tab') return;
            if (document.activeElement && ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
            var eventType = _liveHotkeys[key];
            if (!eventType) return;
            e.preventDefault();
            bridge.live_tag_event(eventType, '', 0, '', null, null).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) return;
                // Flash button
                var btn = document.querySelector('.live-hotkey-btn[data-type="' + eventType + '"]');
                if (btn) { btn.classList.add('active'); setTimeout(function(){ btn.classList.remove('active'); }, 200); }
                updateLiveStats();
                loadLiveTags();
            }).catch(_catchBridge('live_tag_event'));
        });

        updateLiveStats();
    }

    function loadLiveHotkeys() {
        bridge.live_get_hotkeys().then(function(raw) {
            var r = JSON.parse(raw);
            if (r.error) return;
            _liveHotkeys = r.hotkeys || {};
            var grid = document.getElementById('live-hotkeys-grid');
            if (!grid) return;
            grid.innerHTML = '';
            Object.keys(_liveHotkeys).forEach(function(key) {
                var label = _liveHotkeys[key].replace(/_/g, ' ').replace(/\b\w/g, function(c){ return c.toUpperCase(); });
                var btn = document.createElement('div');
                btn.className = 'live-hotkey-btn';
                btn.setAttribute('data-type', _liveHotkeys[key]);
                btn.innerHTML = '<span class="hk-key">' + key + '</span><span class="hk-label">' + label + '</span>';
                btn.onclick = function() {
                    if (!_liveSessionActive) { showToast('Start a session first', 'warning'); return; }
                    bridge.live_tag_event(_liveHotkeys[key], '', 0, '', null, null).then(function(raw2) {
                        var r2 = JSON.parse(raw2);
                        if (r2.error) { showToast(r2.error, 'error'); return; }
                        btn.classList.add('active');
                        setTimeout(function(){ btn.classList.remove('active'); }, 200);
                        updateLiveStats();
                loadLiveTags();
                    }).catch(_catchBridge('live_tag_event'));
                };
                grid.appendChild(btn);
            });
        }).catch(_catchBridge('live_get_hotkeys'));
    }

    function updateLiveStats() {
        var container = document.getElementById('live-stats-content');
        if (!container) return;
        bridge.live_get_stats().then(function(raw) {
            var r = JSON.parse(raw);
            if (r.error || !r.stats) return;
            var s = r.stats;
            container.innerHTML = '';
            var items = [
                { label: 'Tags', value: s.tags_count },
                { label: 'Home Goals', value: s.home_goals },
                { label: 'Away Goals', value: s.away_goals },
                { label: 'Home Shots', value: s.home_shots },
                { label: 'Away Shots', value: s.away_shots },
                { label: 'Possession (Home)', value: s.home_possession_pct + '%' },
                { label: 'Elapsed', value: formatLiveTime(s.elapsed_s) },
            ];
            items.forEach(function(it) {
                var d = document.createElement('div');
                d.style.cssText = 'display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)';
                d.innerHTML = '<span>' + it.label + '</span><strong>' + it.value + '</strong>';
                container.appendChild(d);
            });
        }).catch(_catchBridge('live_get_stats'));
    }

    function loadLiveTags() {
        var container = document.getElementById('live-tag-list');
        var counter = document.getElementById('live-tag-count');
        if (!container) return;
        bridge.live_get_tags().then(function(raw) {
            var r = JSON.parse(raw);
            if (r.error || !r.tags) return;
            if (counter) counter.textContent = r.total;
            container.innerHTML = '';
            if (!r.tags.length) {
                container.innerHTML = '<p class="hint">No tags yet. Use hotkeys or click buttons.</p>';
                return;
            }
            r.tags.slice().reverse().forEach(function(tag) {
                var div = document.createElement('div');
                div.className = 'live-tag-entry';
                div.innerHTML = '<span class="live-tag-type">' + escapeHtml(tag.type.replace(/_/g, ' ')) + '</span>'
                    + '<span class="live-tag-team">' + escapeHtml(tag.team || '') + '</span>'
                    + '<span class="live-tag-time">' + formatLiveTime(tag.t) + '</span>'
                    + '<span class="live-tag-notes" style="flex:1;font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(tag.notes || '') + '</span>';
                container.appendChild(div);
            });
        });
    }

    function formatLiveTime(seconds) {
        if (!seconds && seconds !== 0) return '0:00';
        var m = Math.floor(seconds / 60);
        var s = Math.floor(seconds % 60);
        return m + ':' + (s < 10 ? '0' : '') + s;
    }

    // ── Phase 6 Sprint 2: Live Tagging Dashboard ──

    function initLiveDashboard() {
        var toggleBtn = document.getElementById('live-dashboard-toggle');
        var kpiSection = document.getElementById('live-kpi-dashboard');
        var panels = document.getElementById('live-dashboard-panels');
        if (!toggleBtn || !kpiSection || !panels) return;
        var visible = false;
        toggleBtn.onclick = function() {
            visible = !visible;
            kpiSection.classList.toggle('hidden', !visible);
            panels.classList.toggle('hidden', !visible);
            toggleBtn.textContent = visible ? '\u{1F4CA} Dashboard' : '\u{1F4CA} Dashboard';
            if (visible) {
                updateLiveDashboard();
            }
        };
    }

    function updateLiveDashboard() {
        bridge.get_live_kpis('live').then(function(raw) {
            var r = JSON.parse(raw);
            if (r.error) return;
            document.getElementById('live-possession').textContent = r.possession_pct + '%';
            document.getElementById('live-shots').textContent = r.shots;
            document.getElementById('live-xg').textContent = r.xg.toFixed(2);
            document.getElementById('live-xg-diff').textContent = (r.xg_diff >= 0 ? '+' : '') + r.xg_diff.toFixed(2);
            document.getElementById('live-goals').textContent = r.goals;
        }).catch(_catchBridge('get_live_kpis'));
        bridge.get_live_pitch_map('live').then(function(raw) {
            var r = JSON.parse(raw);
            if (r.error) return;
            renderLivePitchMap(r);
        }).catch(_catchBridge('get_live_pitch_map'));
        bridge.get_live_xg_chart('live').then(function(raw) {
            var r = JSON.parse(raw);
            if (r.error) return;
            renderLiveXgChart(r);
        }).catch(_catchBridge('get_live_xg_chart'));
    }

    function renderLivePitchMap(data) {
        var container = document.getElementById('live-pitch-events');
        if (!container) return;
        container.innerHTML = '';
        var svg = document.getElementById('live-pitch-svg');
        if (!svg) return;
        var vb = svg.getAttribute('viewBox').split(' ').map(Number);
        var pw = vb[2] || 1050;
        var ph = vb[3] || 680;

        function addEvent(ev, team) {
            if (ev.x == null || ev.y == null) return;
            var ex = (ev.x / 100) * pw * 0.85 + pw * 0.075;
            var ey = (ev.y / 100) * ph * 0.85 + ph * 0.075;
            var el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            var color = team === 'home' ? '#2563eb' : '#dc2626';
            var r = 6;
            var cls = 'pitch-event-marker';
            if (ev.type === 'goal') {
                el.setAttribute('r', '10');
                el.setAttribute('fill', '#fbbf24');
                el.setAttribute('stroke', color);
                el.setAttribute('stroke-width', '3');
                el.innerHTML = '<title>Goal at ' + formatLiveTime(ev.t) + '</title>';
            } else if (ev.type === 'shot') {
                el.setAttribute('r', '7');
                el.setAttribute('fill', 'none');
                el.setAttribute('stroke', color);
                el.setAttribute('stroke-width', '2.5');
                el.setAttribute('stroke-dasharray', '3,3');
                el.innerHTML = '<title>Shot at ' + formatLiveTime(ev.t) + '</title>';
            } else if (ev.type === 'pass') {
                r = 4;
                el.setAttribute('fill', color);
                el.setAttribute('opacity', '0.6');
                el.innerHTML = '<title>Pass at ' + formatLiveTime(ev.t) + '</title>';
            } else {
                el.setAttribute('r', '5');
                el.setAttribute('fill', color);
                el.setAttribute('opacity', '0.5');
                el.innerHTML = '<title>' + escapeHtml(ev.type || 'event') + ' at ' + formatLiveTime(ev.t) + '</title>';
            }
            el.setAttribute('cx', ex);
            el.setAttribute('cy', ey);
            el.setAttribute('class', cls);
            container.appendChild(el);
        }
        (data.home_events || []).forEach(function(ev) { addEvent(ev, 'home'); });
        (data.away_events || []).forEach(function(ev) { addEvent(ev, 'away'); });
    }

    var _liveXgChartInstance = null;

    function renderLiveXgChart(data) {
        var canvas = document.getElementById('live-xg-chart-canvas');
        if (!canvas) return;
        if (_liveXgChartInstance) {
            _liveXgChartInstance.destroy();
            _liveXgChartInstance = null;
        }
        var ctx = canvas.getContext('2d');
        if (typeof Chart === 'undefined') return;
        var labels = (data.timeline || []).map(function(p) { return p.minute + "'"; });
        var homeData = (data.timeline || []).map(function(p) { return p.home_xg; });
        var awayData = (data.timeline || []).map(function(p) { return p.away_xg; });
        if (labels.length === 0) {
            labels = ['0\''];
            homeData = [0];
            awayData = [0];
        }
        _liveXgChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Home xG',
                        data: homeData,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37,99,235,0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 3,
                    },
                    {
                        label: 'Away xG',
                        data: awayData,
                        borderColor: '#dc2626',
                        backgroundColor: 'rgba(220,38,38,0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { labels: { color: '#f1f5f9' } },
                },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.15)' } },
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.15)' } },
                },
            },
        });
    }

    // ── Sprint 5: Scout Camera (Mobile/Tablet) ──
    var _scoutCamInitialized = false;
    var _scoutCamStream = null;
    var _scoutCaptures = [];

    function initScoutCamera() {
        if (_scoutCamInitialized) return;
        _scoutCamInitialized = true;

        var video = document.getElementById('scout-camera-video');
        var startBtn = document.getElementById('scout-cam-start-btn');
        var stopBtn = document.getElementById('scout-cam-stop-btn');
        var captureBtn = document.getElementById('scout-cam-capture-btn');
        var clearBtn = document.getElementById('scout-cam-clear-btn');
        var capturesDiv = document.getElementById('scout-camera-captures');

        startBtn.onclick = function() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showToast('Camera not available on this device', 'error');
                return;
            }
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } })
                .then(function(stream) {
                    _scoutCamStream = stream;
                    video.srcObject = stream;
                    video.play();
                    startBtn.classList.add('hidden');
                    stopBtn.classList.remove('hidden');
                    captureBtn.classList.remove('hidden');
                    showToast('Camera started', 'info');
                })
                .catch(function(err) {
                    showToast('Camera error: ' + err.message, 'error');
                });
        };

        stopBtn.onclick = function() {
            if (_scoutCamStream) {
                _scoutCamStream.getTracks().forEach(function(t) { t.stop(); });
                _scoutCamStream = null;
            }
            video.srcObject = null;
            startBtn.classList.remove('hidden');
            stopBtn.classList.add('hidden');
            captureBtn.classList.add('hidden');
        };

        captureBtn.onclick = function() {
            if (!_scoutCamStream) return;
            var canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            var dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            _scoutCaptures.push(dataUrl);
            renderScoutCaptures();
            showToast('Snapshot captured', 'info');
        };

        clearBtn.onclick = function() {
            _scoutCaptures = [];
            renderScoutCaptures();
        };

        function renderScoutCaptures() {
            if (_scoutCaptures.length === 0) {
                capturesDiv.innerHTML = '<p class="hint">Captured snapshots will appear here.</p>';
                return;
            }
            capturesDiv.innerHTML = '';
            _scoutCaptures.forEach(function(url, i) {
                var item = document.createElement('div');
                item.className = 'scout-capture-item';
                item.innerHTML = '<img src="' + url + '" alt="Capture ' + (i+1) + '"><div class="capture-label">#' + (i+1) + '</div>';
                item.onclick = function() {
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = 'scout-capture-' + (i+1) + '.jpg';
                    a.click();
                };
                capturesDiv.appendChild(item);
            });
        }
    }

    // ── PWA install prompt handler ──
    var _deferredPrompt = null;
    var _pwaInitialized = false;

    function initPWA() {
        if (_pwaInitialized) return;
        _pwaInitialized = true;

        window.addEventListener('beforeinstallprompt', function(e) {
            e.preventDefault();
            _deferredPrompt = e;
            var banner = document.getElementById('pwa-install-banner');
            if (banner) banner.classList.add('visible');
        });

        window.addEventListener('appinstalled', function() {
            _deferredPrompt = null;
            var banner = document.getElementById('pwa-install-banner');
            if (banner) banner.classList.remove('visible');
            showToast('App installed!', 'info');
        });

        // Offline/online detection
        function updateOnlineStatus() {
            var indicator = document.getElementById('offline-indicator');
            if (!indicator) return;
            if (navigator.onLine) {
                indicator.classList.remove('visible');
            } else {
                indicator.classList.add('visible');
            }
        }
        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
        updateOnlineStatus();
    }

    // ── Install button handler ──
    document.addEventListener('click', function(e) {
        if (e.target && e.target.matches('#pwa-install-btn')) {
            if (_deferredPrompt) {
                _deferredPrompt.prompt();
                _deferredPrompt.userChoice.then(function() {
                    _deferredPrompt = null;
                    var banner = document.getElementById('pwa-install-banner');
                    if (banner) banner.classList.remove('visible');
                });
            }
        }
        if (e.target && e.target.matches('#pwa-install-close, #pwa-install-close *')) {
            var banner = document.getElementById('pwa-install-banner');
            if (banner) banner.classList.remove('visible');
        }
    });

    // ── Phase 9: Live Stream Capture ──
    var _streamInitialized = false;
    var _currentStreamId = null;

    function initStreamCapture() {
        if (_streamInitialized) return;
        _streamInitialized = true;

        var statusEl = document.getElementById('stream-status');
        var markerList = document.getElementById('stream-marker-list');
        var recordingsDiv = document.getElementById('stream-recordings');

        document.getElementById('stream-detect-btn').onclick = function() {
            var url = document.getElementById('stream-url-input').value.trim();
            if (!url) { showToast('Enter a URL first', 'warning'); return; }
            bridge.stream_detect_source(url).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { statusEl.textContent = 'Error: ' + r.error; return; }
                statusEl.textContent = 'Detected: ' + r.source_type;
                showToast('Source: ' + r.source_type, 'info');
            }).catch(_catchBridge('stream_detect_source'));
        };

        document.getElementById('stream-start-btn').onclick = function() {
            var url = document.getElementById('stream-url-input').value.trim();
            if (!url) { showToast('Enter a stream URL', 'warning'); return; }
            bridge.stream_start_capture(url, '', '').then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { statusEl.textContent = 'Error: ' + r.error; return; }
                _currentStreamId = r.stream_id;
                statusEl.textContent = 'Capturing: ' + r.stream_id + ' (' + r.source_type + ') -> ' + r.output;
                document.getElementById('stream-start-btn').classList.add('hidden');
                document.getElementById('stream-stop-btn').classList.remove('hidden');
                showToast('Stream capture started', 'info');
            }).catch(_catchBridge('stream_start_capture'));
        };

        document.getElementById('stream-stop-btn').onclick = function() {
            if (!_currentStreamId) return;
            bridge.stream_stop_capture(_currentStreamId).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { showToast(r.error, 'error'); return; }
                statusEl.textContent = 'Capture stopped. Chapters: ' + (r.chapters || 0);
                document.getElementById('stream-start-btn').classList.remove('hidden');
                document.getElementById('stream-stop-btn').classList.add('hidden');
                _currentStreamId = null;
                showToast('Capture stopped', 'info');
                loadStreamRecordings();
            }).catch(_catchBridge('stream_stop_capture'));
        };

        document.getElementById('stream-add-marker-btn').onclick = function() {
            if (!_currentStreamId) { showToast('Start a capture first', 'warning'); return; }
            var label = document.getElementById('stream-marker-label').value.trim() || '';
            bridge.stream_add_marker(_currentStreamId, label).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { showToast(r.error, 'error'); return; }
                loadStreamMarkers();
                document.getElementById('stream-marker-label').value = '';
            }).catch(_catchBridge('stream_add_marker'));
        };

        document.getElementById('stream-refresh-recordings-btn').onclick = function() {
            loadStreamRecordings();
        };

        function loadStreamMarkers() {
            if (!_currentStreamId) return;
            bridge.stream_get_status(_currentStreamId).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) return;
                // We re-fetch by just showing chapter count
                if (markerList) markerList.innerHTML = 'Markers: ' + r.chapters;
            }).catch(_catchBridge('stream_get_status'));
        }

        function loadStreamRecordings() {
            if (!recordingsDiv) return;
            bridge.stream_list_recordings().then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error || !r.recordings || r.recordings.length === 0) {
                    recordingsDiv.innerHTML = '<p class="hint">No recordings yet.</p>';
                    return;
                }
                recordingsDiv.innerHTML = r.recordings.slice(0, 20).map(function(f) {
                    var size = (f.size / 1024 / 1024).toFixed(1) + ' MB';
                    return '<div class="stream-rec-entry" style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--border);font-size:0.8rem">'
                        + '<span>' + f.name + '</span><span>' + size + '</span></div>';
                }).join('');
            }).catch(_catchBridge('stream_list_recordings'));
        }

        loadStreamRecordings();
    }

    // ── Phase 8: Cloud Sync ──
    var _cloudInitialized = false;

    function initCloud() {
        if (_cloudInitialized) return;
        _cloudInitialized = true;

        var statusEl = document.getElementById('cloud-connection');
        var authResult = document.getElementById('cloud-auth-result');
        var syncResult = document.getElementById('cloud-sync-result');
        var inviteResult = document.getElementById('cloud-invite-result');

        function updateCloudUI() {
            bridge.cloud_is_logged_in().then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) return;
                if (r.logged_in) {
                    document.getElementById('cloud-logged-out').classList.add('hidden');
                    document.getElementById('cloud-logged-in').classList.remove('hidden');
                    if (r.user) {
                        document.getElementById('cloud-user-display').textContent = r.user.display_name || r.user.username;
                        document.getElementById('cloud-user-email').textContent = r.user.email;
                    }
                    if (statusEl) statusEl.textContent = 'Online';
                    if (statusEl) statusEl.style.background = 'var(--success)';
                } else {
                    document.getElementById('cloud-logged-out').classList.remove('hidden');
                    document.getElementById('cloud-logged-in').classList.add('hidden');
                    if (statusEl) statusEl.textContent = 'Offline';
                    if (statusEl) statusEl.style.background = 'var(--text-muted)';
                }
            }).catch(_catchBridge('cloud_is_logged_in'));
            loadCloudTeams();
        }

        // Cloud server control
        document.getElementById('cloud-start-server-btn').onclick = function() {
            bridge.cloud_start_server(8741).then(function(raw) {
                var r = JSON.parse(raw);
                showToast(r.message || 'Server started', r.error ? 'error' : 'info');
                checkCloudHealth();
            }).catch(_catchBridge('cloud_start_server'));
        };

        document.getElementById('cloud-refresh-btn').onclick = function() {
            checkCloudHealth();
            updateCloudUI();
        };

        // Auth
        document.getElementById('cloud-login-btn').onclick = function() {
            var email = document.getElementById('cloud-email').value.trim();
            var pw = document.getElementById('cloud-password').value;
            if (!email || !pw) { showToast('Enter email and password', 'warning'); return; }
            bridge.cloud_login(email, pw).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { authResult.textContent = 'Error: ' + r.error; return; }
                authResult.textContent = 'Logged in as ' + (r.user?.display_name || r.user?.username);
                updateCloudUI();
            }).catch(_catchBridge('cloud_login'));
        };

        document.getElementById('cloud-register-btn').onclick = function() {
            var email = document.getElementById('cloud-email').value.trim();
            var pw = document.getElementById('cloud-password').value;
            var username = email.split('@')[0];
            if (!email || !pw || pw.length < 8) { showToast('Email + password (min 8 chars)', 'warning'); return; }
            bridge.cloud_register(username, email, pw, username).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { authResult.textContent = 'Error: ' + r.error; return; }
                authResult.textContent = 'Registered! Logged in as ' + (r.user?.display_name || r.user?.username);
                updateCloudUI();
            }).catch(_catchBridge('cloud_register'));
        };

        document.getElementById('cloud-logout-btn').onclick = function() {
            bridge.cloud_logout().then(function() {
                updateCloudUI();
            }).catch(_catchBridge('cloud_logout'));
        };

        // OAuth — sign in with Google, GitHub, Apple
        document.querySelectorAll('.oauth-btn').forEach(function(btn) {
            btn.onclick = function() {
                var provider = btn.getAttribute('data-provider');
                bridge.cloud_oauth_authorize_url(provider).then(function(raw) {
                    var r = JSON.parse(raw);
                    if (r.error) { authResult.textContent = 'Error: ' + r.error; return; }
                    if (!r.authorize_url) { authResult.textContent = 'Provider not configured (set env vars)'; return; }
                    authResult.innerHTML = 'Opened browser for ' + provider + ' login. <br><small>After authorizing, paste the full callback URL below:</small>';
                    var codeInput = document.createElement('div');
                    codeInput.id = 'oauth-code-input';
                    codeInput.style.marginTop = '6px';
                    codeInput.innerHTML = '<div class="form-row"><input type="text" id="oauth-code-field" class="input" placeholder="Paste callback URL or code" style="flex:1"><button id="oauth-submit-btn" class="btn btn-sm btn-primary">Verify</button></div>';
                    var existing = document.getElementById('oauth-code-input');
                    if (existing) existing.remove();
                    btn.parentNode.appendChild(codeInput);
                    document.getElementById('oauth-submit-btn').onclick = function() {
                        var codeOrUrl = document.getElementById('oauth-code-field').value.trim();
                        if (!codeOrUrl) { showToast('Enter the authorization code', 'warning'); return; }
                        var code = codeOrUrl;
                        var state = r.state || '';
                        var urlMatch = codeOrUrl.match(/[?&]code=([^&]+)/);
                        if (urlMatch) code = decodeURIComponent(urlMatch[1]);
                        var stateMatch = codeOrUrl.match(/[?&]state=([^&]+)/);
                        if (stateMatch) state = decodeURIComponent(stateMatch[1]);
                        bridge.cloud_oauth_exchange(provider, code, state).then(function(raw2) {
                            var r2 = JSON.parse(raw2);
                            if (r2.error) { authResult.textContent = 'Auth error: ' + r2.error; return; }
                            authResult.textContent = 'Logged in with ' + provider + ' as ' + (r2.user?.display_name || r2.user?.username);
                            updateCloudUI();
                            var ci = document.getElementById('oauth-code-input');
                            if (ci) ci.remove();
                        }).catch(_catchBridge('cloud_oauth_exchange'));
                    };
                }).catch(_catchBridge('cloud_oauth_authorize_url'));
            };
        });

        // Teams
        document.getElementById('cloud-create-team-btn').onclick = function() {
            var name = document.getElementById('cloud-team-name').value.trim();
            if (!name) { showToast('Enter a team name', 'warning'); return; }
            bridge.cloud_create_team(name, '').then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { showToast(r.error, 'error'); return; }
                showToast('Team created!', 'info');
                loadCloudTeams();
            }).catch(_catchBridge('cloud_create_team'));
        };

        // Invite
        document.getElementById('cloud-invite-btn').onclick = function() {
            var teamId = parseInt(document.getElementById('cloud-invite-team').value, 10);
            var email = document.getElementById('cloud-invite-email').value.trim();
            if (!teamId || !email) { showToast('Select team and enter email', 'warning'); return; }
            bridge.cloud_invite_member(teamId, email).then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { inviteResult.textContent = 'Error: ' + r.error; return; }
                inviteResult.textContent = 'Invite sent! Token: ' + (r.invite_token || '');
            }).catch(_catchBridge('cloud_invite_member'));
        };

        // Sync
        document.getElementById('cloud-sync-push-btn').onclick = function() {
            bridge.cloud_sync_push('desktop-001', '[]').then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { syncResult.textContent = 'Error: ' + r.error; return; }
                syncResult.textContent = 'Pushed: ' + (r.operations?.length || 0) + ' ops, ' + (r.conflicts?.length || 0) + ' conflicts';
            }).catch(_catchBridge('cloud_sync_push'));
        };

        document.getElementById('cloud-sync-pull-btn').onclick = function() {
            bridge.cloud_sync_pull('desktop-001').then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error) { syncResult.textContent = 'Error: ' + r.error; return; }
                syncResult.textContent = 'Pulled: ' + (r.operations?.length || 0) + ' items';
            }).catch(_catchBridge('cloud_sync_pull'));
        };

        function loadCloudTeams() {
            var list = document.getElementById('cloud-team-list');
            var select = document.getElementById('cloud-invite-team');
            if (!list || !select) return;
            bridge.cloud_list_teams().then(function(raw) {
                var r = JSON.parse(raw);
                if (r.error || !Array.isArray(r)) {
                    list.innerHTML = '<p class="hint">Log in to manage teams.</p>';
                    return;
                }
                if (r.length === 0) {
                    list.innerHTML = '<p class="hint">No teams yet. Create one above.</p>';
                    select.innerHTML = '<option value="">Select team</option>';
                    return;
                }
                list.innerHTML = r.map(function(t) {
                    return '<div class="collab-user-item" style="padding:6px 8px"><span class="collab-user-name">' + escapeHtml(t.name) + '</span><span class="collab-user-role">' + escapeHtml(t.role || 'member') + '</span></div>';
                }).join('');
                select.innerHTML = '<option value="">Select team</option>' + r.map(function(t) { return '<option value="' + escapeHtml(t.id) + '">' + escapeHtml(t.name) + '</option>'; }).join('');
            }).catch(_catchBridge('cloud_list_teams'));
        }

        function checkCloudHealth() {
            bridge.cloud_check_health().then(function(raw) {
                var r = JSON.parse(raw);
                if (statusEl) {
                    if (r.status === 'ok') {
                        statusEl.textContent = 'Server Online';
                        statusEl.style.background = 'var(--success)';
                    } else {
                        statusEl.textContent = 'Server Offline';
                        statusEl.style.background = 'var(--danger)';
                    }
                }
            }).catch(_catchBridge('cloud_check_health'));
        }

        checkCloudHealth();
    }

