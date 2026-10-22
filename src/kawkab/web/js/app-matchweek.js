// Kawkab AI - Match Week: the diagnosis -> plan -> re-check loop.
//
// One owner for the loop panel, plan generation and plan rendering. The
// state itself is composed server-side (services/match_week_service.py);
// this module never derives a delta, never diagnoses, and never decides
// whether a measurement is trustworthy -- it renders what the service
// reports, including its refusals ("pending", "not observable").
//
// Text is written with textContent throughout: plan/diagnosis strings
// come from the knowledge base and the coach's own data, so no HTML
// escaping is needed here at all.

(function () {
    'use strict';

    var _seq = 0;
    var _initialized = false;

    function getBridge() {
        if (window.KawkabUtils && typeof window.KawkabUtils.getBridge === 'function') {
            return window.KawkabUtils.getBridge();
        }
        if (window.__kawkab && window.__kawkab.bridge) return window.__kawkab.bridge;
        return null;
    }

    function currentMatchId() {
        var id = window.__kawkab ? window.__kawkab.currentMatchId : null;
        var n = parseInt(id, 10);
        return isNaN(n) || n <= 0 ? 0 : n;
    }

    function label(key, fallback) {
        var t = window.t;
        if (typeof t === 'function') {
            var value = t(key);
            if (value && value !== key) return value;
        }
        return fallback;
    }

    function toast(message, kind) {
        if (typeof window.showToast === 'function') window.showToast(message, kind);
    }

    function el(tag, className, text) {
        var node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null && text !== '') node.textContent = String(text);
        return node;
    }

    function clear(node) {
        if (!node) return;
        while (node.firstChild) node.removeChild(node.firstChild);
    }

    function parse(raw) {
        var data = typeof raw === 'string' ? JSON.parse(raw) : raw;
        if (!data) throw new Error('empty response');
        if (data.error) throw new Error(data.error);
        return data;
    }

    function pct(value) {
        var n = parseFloat(value);
        return isNaN(n) ? '' : Math.round(n * 100) + '%';
    }

    function num(value) {
        var n = parseFloat(value);
        return isNaN(n) ? '' : String(Math.round(n * 100) / 100);
    }

    // ── Rendering ────────────────────────────────────────────────────

    function renderStage(stage) {
        var badge = document.getElementById('mw-stage');
        if (badge) {
            badge.textContent = label('mwStage_' + stage, stage.replace(/_/g, ' '));
            badge.setAttribute('data-stage', stage || '');
        }
    }

    function renderNote(note) {
        var node = document.getElementById('mw-note');
        if (node) node.textContent = note || '';
    }

    function renderDiagnoses(diagnoses) {
        var host = document.getElementById('mw-diagnoses');
        if (!host) return;
        clear(host);
        if (!diagnoses || diagnoses.length === 0) {
            host.appendChild(el('p', 'hint', label(
                'mwNoDiagnoses',
                'No diagnosis is stored yet. Generating a plan runs the reasoning engine and saves its diagnoses.'
            )));
            return;
        }
        diagnoses.forEach(function (d) {
            var card = el('div', 'mw-diagnosis');
            card.appendChild(el('strong', '', d.rule_name || d.rule_id || ''));
            var conf = pct(d.confidence);
            if (conf) card.appendChild(el('span', 'mw-confidence', conf));
            host.appendChild(card);
        });
    }

    function renderGenerateButton(state) {
        var btn = document.getElementById('mw-generate');
        if (!btn) return;
        var needsPlan = state.stage === 'no_plan';
        btn.classList.toggle('hidden', !needsPlan);
        btn.disabled = false;
        btn.textContent = label('mwGenerate', 'Generate this week\'s plan');
    }

    function renderPlan(plan) {
        var host = document.getElementById('mw-plan');
        var preview = document.getElementById('training-plan-preview');
        var weeksHost = document.getElementById('training-plan-weeks');
        if (!host) return;
        clear(host);
        if (!plan) {
            host.appendChild(el('p', 'hint', label('mwNoPlan', 'No plan yet.')));
            if (preview) preview.classList.add('hidden');
            return;
        }

        var summary = el('div', 'mw-plan-summary');
        summary.appendChild(el('strong', '', plan.title || label('mwPlan', 'Training plan')));
        var meta = [];
        if (plan.duration_weeks) meta.push(plan.duration_weeks + ' ' + label('mwWeeks', 'weeks'));
        if (plan.total_drills) meta.push(plan.total_drills + ' ' + label('mwDrills', 'drills'));
        if (plan.status) meta.push(plan.status);
        if (meta.length) summary.appendChild(el('span', 'mw-meta', meta.join(' · ')));
        host.appendChild(summary);

        if (plan.priority_addressed && plan.priority_addressed.length) {
            host.appendChild(el('div', 'mw-priorities', plan.priority_addressed.join(' · ')));
        }
        if (plan.expected_overall_improvement) {
            host.appendChild(el('p', 'hint', plan.expected_overall_improvement));
        }
        if (plan.unresolved_drills && plan.unresolved_drills.length) {
            host.appendChild(el('p', 'mw-warning', label(
                'mwUnresolvedDrills',
                'Referenced drills missing from the knowledge base: '
            ) + plan.unresolved_drills.join(', ')));
        }

        if (!weeksHost) return;
        clear(weeksHost);
        (plan.weeks || []).forEach(function (week) {
            var card = el('div', 'pro-card');
            var header = el('div', 'pro-card-header');
            header.appendChild(el('strong', '', 'Week ' + week.week_number + ': ' + (week.theme || '')));
            header.appendChild(el('span', 'mw-meta', week.primary_focus || ''));
            card.appendChild(header);

            var body = el('div', 'pro-card-body');
            if (week.expected_improvements && week.expected_improvements.length) {
                body.appendChild(el('div', 'mw-line',
                    label('mwExpected', 'Expected') + ': ' + week.expected_improvements.join(', ')));
            }
            if (week.re_test_focus) {
                body.appendChild(el('div', 'mw-line',
                    label('mwRetest', 'Re-test') + ': ' + week.re_test_focus));
            }
            (week.sessions || []).forEach(function (session) {
                var row = el('div', 'mw-session');
                row.appendChild(el('strong', '', session.day || ''));
                if (session.intensity) {
                    row.appendChild(el('span', 'risk-badge risk-' + (
                        session.intensity === 'low' ? 'low'
                            : (session.intensity === 'medium' || session.intensity === 'moderate') ? 'moderate'
                                : 'high'
                    ), session.intensity));
                }
                row.appendChild(el('div', 'mw-line',
                    (session.focus || '') + ' · ' + (session.total_duration_min || 0) + ' min'));
                if (session.drills && session.drills.length) {
                    row.appendChild(el('div', 'mw-line', session.drills.join(', ')));
                }
                body.appendChild(row);
            });
            card.appendChild(body);
            weeksHost.appendChild(card);
        });
        if (preview) preview.classList.toggle('hidden', !(plan.weeks && plan.weeks.length));
    }

    function ruleRow(rule) {
        var row = el('div', 'mw-rule');
        var head = el('div', 'mw-rule-head');
        head.appendChild(el('strong', '', rule.rule_name || rule.rule_id || ''));
        if (rule.metric) head.appendChild(el('span', 'mw-meta', rule.metric));
        row.appendChild(head);

        if (rule.state === 'measured') {
            var values = el('div', 'mw-delta');
            values.appendChild(el('span', '', num(rule.mean_before)));
            values.appendChild(el('span', 'mw-arrow', '→'));
            values.appendChild(el('span', '', num(rule.mean_after)));
            var delta = num(rule.delta);
            values.appendChild(el('span', 'mw-delta-value',
                (parseFloat(rule.delta) > 0 ? '+' : '') + delta));
            if (rule.improved === true) values.appendChild(el('span', 'mw-improved', label('mwImproved', 'improved')));
            else if (rule.improved === false) values.appendChild(el('span', 'mw-worse', label('mwWorse', 'worse')));
            row.appendChild(values);
        }

        if (rule.note) row.appendChild(el('p', 'hint', rule.note));
        if (rule.state && rule.state !== 'measured') {
            row.appendChild(el('div', 'mw-state', label(
                'mwState_' + rule.state,
                String(rule.state).replace(/_/g, ' ')
            )));
        }
        if (rule.reasons && rule.reasons.length) {
            row.appendChild(el('p', 'hint', rule.reasons.join('; ')));
        }
        return row;
    }

    function renderRecheck(recheck) {
        var host = document.getElementById('mw-recheck');
        if (!host) return;
        clear(host);
        if (!recheck) {
            host.appendChild(el('p', 'hint', label('mwNoRecheck', 'No measurement yet.')));
            return;
        }
        var rules = recheck.rules || [];
        if (rules.length === 0) {
            host.appendChild(el('p', 'hint', label('mwNoRules', 'The plan carries no measurable diagnosis.')));
            return;
        }
        if (!recheck.after_window_complete) {
            host.appendChild(el('p', 'hint', label(
                'mwAwaiting',
                'Waiting for the next two matches. A delta from one match is noise, not evidence.'
            )));
        }
        rules.forEach(function (rule) {
            host.appendChild(ruleRow(rule));
        });
    }

    function renderProvenance(provenance) {
        var host = document.getElementById('mw-provenance');
        if (!host) return;
        clear(host);
        if (!provenance) return;

        var chips = el('div', 'mw-chips');
        var calibrated = provenance.calibrated;
        chips.appendChild(el('span', 'mw-chip' + (calibrated === true ? ' mw-chip-ok' : calibrated === false ? ' mw-chip-warn' : ''),
            calibrated === true
                ? label('mwCalibrated', 'Calibrated — spatial stats in meters')
                : calibrated === false
                    ? label('mwUncalibrated', 'Not calibrated — spatial stats are pixel-based estimates')
                    : label('mwCalibrationUnknown', 'Calibration state unknown')));

        if (provenance.based_on_match_ids && provenance.based_on_match_ids.length) {
            chips.appendChild(el('span', 'mw-chip',
                label('mwBasedOn', 'Plan based on matches') + ' ' + provenance.based_on_match_ids.join(', ')));
        }
        if (provenance.after_window_complete !== undefined) {
            chips.appendChild(el('span', 'mw-chip',
                label('mwWindow', 'Post-plan window') + ' ' + (provenance.after_window || []).length + '/2'));
        }
        host.appendChild(chips);
    }

    function renderEmpty(message) {
        renderStage('no_analysis');
        renderNote(message);
        renderDiagnoses([]);
        renderPlan(null);
        renderRecheck(null);
        renderProvenance(null);
    }

    // ── Data ─────────────────────────────────────────────────────────

    function refresh() {
        var host = document.getElementById('matchweek-panel');
        if (!host) return Promise.resolve(null);

        var matchId = currentMatchId();
        if (!matchId) {
            renderEmpty(label('mwSelectMatch', 'Select or analyse a match to see its week.'));
            return Promise.resolve(null);
        }

        var bridge = getBridge();
        if (!bridge || typeof bridge.get_match_week_state !== 'function') {
            renderEmpty(label('mwBridgeUnavailable', 'The backend bridge is not available.'));
            return Promise.resolve(null);
        }

        var seq = ++_seq;
        host.classList.add('mw-loading');
        return new Promise(function (resolve) {
            bridge.get_match_week_state(String(matchId), function (raw) {
                if (seq !== _seq) return resolve(null);
                host.classList.remove('mw-loading');
                var state;
                try {
                    state = parse(raw);
                } catch (e) {
                    renderStage('error');
                    renderNote(String(e.message || e));
                    resolve(null);
                    return;
                }
                renderStage(state.stage);
                renderNote(state.note);
                renderDiagnoses(state.diagnoses || []);
                renderPlan(state.plan);
                renderRecheck(state.recheck);
                renderProvenance(state.provenance);
                renderGenerateButton(state);
                resolve(state);
            });
        });
    }

    function generatePlan() {
        var bridge = getBridge();
        var matchId = currentMatchId();
        if (!bridge || !matchId) return;
        if (typeof bridge.generate_training_plan !== 'function') return;

        var btn = document.getElementById('mw-generate');
        if (btn) {
            btn.disabled = true;
            btn.textContent = label('mwGenerating', 'Generating…');
        }
        bridge.generate_training_plan(String(matchId), function (raw) {
            try {
                var data = parse(raw);
                toast(label('mwPlanGenerated', 'Training plan generated'), 'success');
                if (data.unresolved_drills && data.unresolved_drills.length) {
                    toast(label('mwUnresolvedDrills', 'Some referenced drills are missing from the knowledge base.'), 'warning');
                }
            } catch (e) {
                toast(String(e.message || e), 'error');
            }
            if (btn) btn.disabled = false;
            refresh();
        });
    }

    function init() {
        if (_initialized) return;
        _initialized = true;

        var refreshBtn = document.getElementById('mw-refresh');
        if (refreshBtn) refreshBtn.addEventListener('click', function () { refresh(); });

        var genBtn = document.getElementById('mw-generate');
        if (genBtn) genBtn.addEventListener('click', generatePlan);

        document.addEventListener('kawkab:match-loaded', function () { refresh(); });
        window.addEventListener('hashchange', function () {
            if (window.location.hash === '#training-section') refresh();
        });
        // The sidebar toggles sections without necessarily changing the hash
        // when you are already on the section.
        document.addEventListener('click', function (event) {
            var target = event.target;
            if (!target || typeof target.closest !== 'function') return;
            if (target.closest('[data-section="training-section"]')) {
                setTimeout(refresh, 0);
            }
        });

        refresh();
    }

    window.KawkabMatchWeek = {
        init: init,
        refresh: refresh,
        generatePlan: generatePlan
    };

    // Self-initialise. Depending on app.js's DOMContentLoaded handler to
    // reach its last line is fragile -- any earlier module throwing (e.g.
    // a 3D-pitch call outside the Qt shell) would leave this panel dead.
    // init() is idempotent, so an explicit call is harmless too.
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
