// Kawkab AI - Tactical Whiteboard
// Coach drawing board on the frontend for the TacticalWhiteboard backend
// (src/kawkab/analysis/tactical_whiteboard.py). Before this file existed the
// backend exposed 19 bridge slots with zero JS callers -- a fully tested
// feature that no user could reach.
//
// Coordinate system: the backend templates store normalized 0-100 positions
// (x = along the pitch, y = across), and generate_svg scales them onto the
// requested pixel size. This UI mirrors that: the SVG canvas uses
// viewBox="0 0 100 100" so board coordinates ARE svg user units -- no
// client-side scaling math, and what you draw is exactly what exports.

(function() {
    'use strict';

    var _state = {
        boardId: null,
        name: '',
        tool: 'arrow',
        color: '#ffffff',
        moveMode: false,
        drawing: false,
        draftPoints: [],
        annotations: [],     // annotations added this session (not yet saved)
        playerIndex: null,   // player being dragged
        team: null,
        dirty: false,
    };

    var _templates = [];

    // Resolve the QWebChannel bridge at call time via the shared helper
    // in utils.js (KawkabUtils.getBridge).
    function resolveBridge() {
        return (window.KawkabUtils && KawkabUtils.getBridge) ? KawkabUtils.getBridge() : null;
    }

    function escapeHtml(s) {
        if (typeof window.escapeHtml === 'function') return window.escapeHtml(s);
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function status(msg) {
        var el = document.getElementById('wb-status');
        if (el) el.textContent = msg;
    }

    function _parse(result) {
        var data = typeof result === 'string' ? JSON.parse(result) : result;
        if (data && data.error) throw new Error(data.error);
        return data;
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------

    function initWhiteboardWorkspace() {
        var bridge = resolveBridge();
        if (!bridge) return;
        if (document.getElementById('wb-new-btn') === null) return;
        if (document.getElementById('wb-new-btn').dataset.wbInit === '1') return;
        document.getElementById('wb-new-btn').dataset.wbInit = '1';

        wireToolbar();
        refreshBoardList();
        loadTemplates();
        checkBackendStatus();
    }

    function checkBackendStatus() {
        var bridge = resolveBridge();
        if (!bridge || !bridge.check_whiteboard_status) return;
        bridge.check_whiteboard_status(function(result) {
            try {
                var data = _parse(result);
                if (data && data.available === false) {
                    status('Whiteboard backend unavailable in this build.');
                    return;
                }
                // Channel is healthy: surface a real hint instead of a toast.
                bridge.whiteboard_list(function(listResult) {
                    var boards;
                    try { boards = _parse(listResult) || []; } catch (e) { boards = []; }
                    if (boards.length) {
                        status(boards.length + ' saved board' + (boards.length === 1 ? '' : 's') +
                            ' -- pick one below or start a new one.');
                    }
                });
            } catch (e) { /* non-fatal */ }
        });
    }

    function wireToolbar() {
        document.getElementById('wb-new-btn').addEventListener('click', onNewBoard);
        document.getElementById('wb-create-btn').addEventListener('click', onCreateBoard);
        document.getElementById('wb-delete-btn').addEventListener('click', onDeleteBoard);
        document.getElementById('wb-board-select').addEventListener('change', onBoardSelected);
        document.getElementById('wb-tool-select').addEventListener('change', function() {
            _state.tool = this.value;
        });
        document.getElementById('wb-color-select').addEventListener('change', function() {
            _state.color = this.value;
        });
        document.getElementById('wb-undo-btn').addEventListener('click', onUndo);
        document.getElementById('wb-clear-btn').addEventListener('click', onClear);
        document.getElementById('wb-ball-btn').addEventListener('click', onPlaceBall);
        document.getElementById('wb-apply-formations-btn').addEventListener('click', onApplyFormations);
        document.getElementById('wb-template-both-btn').addEventListener('click', function() {
            var name = document.getElementById('wb-home-formation').value;
            if (!name) { status('Pick a Home formation first.'); return; }
            applyTemplatePair(name);
        });
        document.getElementById('wb-move-btn').addEventListener('click', onToggleMove);
        document.getElementById('wb-save-btn').addEventListener('click', onSave);
        document.getElementById('wb-export-btn').addEventListener('click', onExportSvg);

        wireCanvasEvents();
    }

    function loadTemplates() {
        var bridge = resolveBridge();
        if (!bridge) return;
        bridge.whiteboard_list_templates(function(result) {
            try {
                _templates = _parse(result) || [];
            } catch (e) {
                _templates = [];
                console.warn('whiteboard_list_templates failed:', e);
                return;
            }
            var options = '<option value="">-- none --</option>' + _templates.map(function(t) {
                return '<option value="' + escapeHtml(t.name) + '">' + escapeHtml(t.label) + '</option>';
            }).join('');
            ['wb-formation-select', 'wb-home-formation', 'wb-away-formation'].forEach(function(id) {
                var sel = document.getElementById(id);
                if (sel) sel.innerHTML = options;
            });
        });
    }

    /** Apply a formation template as both teams in one click.
     * Consumes whiteboard_get_template (positions for a named template) --
     * the last slot without a UI caller.
     */
    function applyTemplatePair(templateName) {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        bridge.whiteboard_get_template(templateName, function(result) {
            var positions;
            try { positions = _parse(result).positions; }
            catch (e) { status('Template failed: ' + e.message); return; }
            var toPlayers = function(team, mirror) {
                return positions.map(function(p, i) {
                    return {
                        x: mirror ? 100 - p.x : p.x,
                        y: p.y,
                        number: p.num,
                        position: p.pos,
                        team: team,
                    };
                });
            };
            bridge.whiteboard_update(_state.boardId, JSON.stringify({
                players_home: toPlayers('home', false),
                players_away: toPlayers('away', true),
                formation_home: templateName,
                formation_away: templateName,
            }), function(updResult) {
                try {
                    var st = _parse(updResult);
                    renderState(st);
                    document.getElementById('wb-home-formation').value = templateName;
                    document.getElementById('wb-away-formation').value = templateName;
                    status('Template ' + templateName + ' applied to both teams.');
                } catch (e) { status('Template apply failed: ' + e.message); }
            });
        });
    }

    // ------------------------------------------------------------------
    // Board lifecycle
    // ------------------------------------------------------------------

    function refreshBoardList(selectId) {
        var bridge = resolveBridge();
        if (!bridge) return;
        bridge.whiteboard_list(function(result) {
            var boards;
            try { boards = _parse(result) || []; } catch (e) { boards = []; }
            var sel = document.getElementById('wb-board-select');
            if (!sel) return;
            sel.innerHTML = '<option value="">-- Saved Boards --</option>' + boards.map(function(s) {
                return '<option value="' + escapeHtml(s.id) + '">' +
                    escapeHtml(s.name) + ' (' + (s.annotation_count || 0) + ')</option>';
            }).join('');
            if (selectId) sel.value = selectId;
        });
    }

    function onNewBoard() {
        _state.boardId = null;
        _state.name = '';
        _state.annotations = [];
        _state.dirty = false;
        document.getElementById('wb-unnamed').classList.remove('hidden');
        document.getElementById('wb-board').classList.add('hidden');
        document.getElementById('wb-name-input').value = '';
        document.getElementById('wb-name-input').focus();
        status('');
    }

    function onCreateBoard() {
        var bridge = resolveBridge();
        if (!bridge) return;
        var name = document.getElementById('wb-name-input').value.trim();
        var formation = document.getElementById('wb-formation-select').value;
        bridge.whiteboard_create(name, formation, function(result) {
            try {
                var st = _parse(result);
                _state.boardId = st.id;
                _state.name = st.name;
                document.getElementById('wb-unnamed').classList.add('hidden');
                document.getElementById('wb-board').classList.remove('hidden');
                renderState(st);
                refreshBoardList(st.id);
                status('Board created.');
            } catch (e) {
                status('Create failed: ' + e.message);
            }
        });
    }

    function onBoardSelected() {
        var bridge = resolveBridge();
        var id = this.value;
        if (!bridge || !id) return;
        bridge.whiteboard_get(id, function(result) {
            try {
                var st = _parse(result);
                _state.boardId = st.id;
                _state.name = st.name;
                _state.annotations = [];
                _state.dirty = false;
                document.getElementById('wb-unnamed').classList.add('hidden');
                document.getElementById('wb-board').classList.remove('hidden');
                renderState(st);
                status('Loaded "' + st.name + '".');
            } catch (e) {
                status('Load failed: ' + e.message);
            }
        });
    }

    function onDeleteBoard() {
        var bridge = resolveBridge();
        var id = _state.boardId || document.getElementById('wb-board-select').value;
        if (!bridge || !id) { status('Select a board to delete.'); return; }
        bridge.whiteboard_delete(id, function(result) {
            try {
                var data = _parse(result);
                if (data && data.ok) {
                    if (_state.boardId === id) hideBoard();
                    refreshBoardList('');
                    status('Board deleted.');
                } else {
                    status('Delete failed.');
                }
            } catch (e) { status('Delete failed: ' + e.message); }
        });
    }

    function hideBoard() {
        _state.boardId = null;
        document.getElementById('wb-board').classList.add('hidden');
        document.getElementById('wb-unnamed').classList.add('hidden');
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    var SVGNS = 'http://www.w3.org/2000/svg';

    function layer(name) { return document.getElementById('wb-layer-' + name); }

    function clearDynamicLayers() {
        ['annotations', 'players', 'draft'].forEach(function(n) {
            var el = layer(n);
            if (el) el.innerHTML = '';
        });
    }

    function renderState(st) {
        clearDynamicLayers();
        st.players_home = st.players_home || [];
        st.players_away = st.players_away || [];
        (st.players_home).forEach(function(p, i) { renderPlayer(p, i, 'home'); });
        (st.players_away).forEach(function(p, i) { renderPlayer(p, i, 'away'); });
        (st.annotations || []).forEach(renderAnnotation);
        if (st.ball_position) renderBall(st.ball_position);
        document.getElementById('wb-home-formation').value = st.formation_home || '';
        document.getElementById('wb-away-formation').value = st.formation_away || '';
    }

    function renderPlayer(p, index, team) {
        var g = document.createElementNS(SVGNS, 'g');
        g.setAttribute('class', 'wb-player');
        g.dataset.wbTeam = team;
        g.dataset.wbIndex = String(index);

        var c = document.createElementNS(SVGNS, 'circle');
        c.setAttribute('cx', String(p.x));
        c.setAttribute('cy', String(p.y));
        c.setAttribute('r', '2.6');
        c.setAttribute('fill', team === 'home' ? '#e74c3c' : '#3498db');
        c.setAttribute('stroke', 'white');
        c.setAttribute('stroke-width', '0.2');
        c.setAttribute('vector-effect', 'non-scaling-stroke');
        g.appendChild(c);

        var t = document.createElementNS(SVGNS, 'text');
        t.setAttribute('x', String(p.x));
        t.setAttribute('y', String(p.y));
        t.setAttribute('fill', 'white');
        t.setAttribute('font-size', '2.4');
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('dominant-baseline', 'central');
        t.setAttribute('style', 'pointer-events:none;user-select:none');
        t.textContent = String(p.number != null ? p.number : '');
        g.appendChild(t);

        g.addEventListener('pointerdown', function(ev) {
            if (!_state.moveMode) return;
            ev.stopPropagation();
            _state.playerIndex = index;
            _state.team = team;
        });

        layer('players').appendChild(g);
    }

    function renderBall(pos) {
        var b = document.createElementNS(SVGNS, 'circle');
        b.setAttribute('cx', String(pos.x));
        b.setAttribute('cy', String(pos.y));
        b.setAttribute('r', '1.6');
        b.setAttribute('fill', 'white');
        b.setAttribute('stroke', 'black');
        b.setAttribute('stroke-width', '0.15');
        b.dataset.wbBall = '1';
        layer('players').appendChild(b);
    }

    function renderAnnotation(a) {
        var g = layer('annotations');
        var el;
        var pts = a.points || [];
        var stroke = a.color || '#ffffff';
        var w = a.width || 3;
        var opacity = a.opacity != null ? a.opacity : 1;
        var dash = a.dashed ? '1.5,1' : null;

        function svgEl(tag) {
            var e = document.createElementNS(SVGNS, tag);
            e.setAttribute('stroke', stroke);
            e.setAttribute('stroke-width', String(w));
            e.setAttribute('opacity', String(opacity));
            e.setAttribute('fill', 'none');
            e.setAttribute('vector-effect', 'non-scaling-stroke');
            if (dash) e.setAttribute('stroke-dasharray', dash);
            return e;
        }

        if (a.type === 'circle' && pts.length >= 1) {
            el = svgEl('circle');
            el.setAttribute('cx', String(pts[0].x));
            el.setAttribute('cy', String(pts[0].y));
            el.setAttribute('r', String(pts[0].radius != null ? pts[0].radius : 8));
        } else if ((a.type === 'rectangle' || a.type === 'zone') && pts.length >= 2) {
            el = svgEl(a.type === 'zone' ? 'polygon' : 'rect');
            if (a.type === 'rectangle') {
                el.setAttribute('x', String(Math.min(pts[0].x, pts[1].x)));
                el.setAttribute('y', String(Math.min(pts[0].y, pts[1].y)));
                el.setAttribute('width', String(Math.abs(pts[1].x - pts[0].x)));
                el.setAttribute('height', String(Math.abs(pts[1].y - pts[0].y)));
            } else {
                el.setAttribute('points', pts.map(function(p) { return p.x + ',' + p.y; }).join(' '));
                el.setAttribute('fill', stroke);
                el.setAttribute('fill-opacity', String(opacity * 0.3));
            }
        } else if (a.type === 'freehand' && pts.length >= 2) {
            el = svgEl('polyline');
            el.setAttribute('points', pts.map(function(p) { return p.x + ',' + p.y; }).join(' '));
        } else if ((a.type === 'line' || a.type === 'arrow') && pts.length >= 2) {
            el = svgEl('line');
            el.setAttribute('x1', String(pts[0].x));
            el.setAttribute('y1', String(pts[0].y));
            el.setAttribute('x2', String(pts[1].x));
            el.setAttribute('y2', String(pts[1].y));
            g.appendChild(el);
            if (a.type === 'arrow') {
                // Arrowhead in board units; counter-rotate not needed because
                // viewBox is square (no aspect distortion).
                var dx = pts[1].x - pts[0].x, dy = pts[1].y - pts[0].y;
                var len = Math.sqrt(dx * dx + dy * dy) || 1;
                var ux = dx / len, uy = dy / len;
                var head = 2.4, spread = 0.45;
                function pt(ang) {
                    var ca = Math.cos(ang), sa = Math.sin(ang);
                    return (pts[1].x - head * (ux * ca - uy * sa)) + ',' +
                           (pts[1].y - head * (ux * sa + uy * ca));
                }
                var tri = document.createElementNS(SVGNS, 'polygon');
                tri.setAttribute('points', pts[1].x + ',' + pts[1].y + ' ' + pt(spread) + ' ' + pt(-spread));
                tri.setAttribute('fill', stroke);
                tri.setAttribute('opacity', String(opacity));
                g.appendChild(tri);
            }
            if (a.label) {
                var tl = document.createElementNS(SVGNS, 'text');
                tl.setAttribute('x', String(pts[1].x));
                tl.setAttribute('y', String(pts[1].y - 2));
                tl.setAttribute('fill', stroke);
                tl.setAttribute('font-size', '3');
                tl.setAttribute('text-anchor', 'middle');
                tl.textContent = a.label;
                g.appendChild(tl);
            }
            return;
        } else if (a.type === 'text' && pts.length >= 1) {
            el = document.createElementNS(SVGNS, 'text');
            el.setAttribute('x', String(pts[0].x));
            el.setAttribute('y', String(pts[0].y));
            el.setAttribute('fill', stroke);
            el.setAttribute('font-size', '3.5');
            el.textContent = a.label || '';
        } else {
            return;
        }
        g.appendChild(el);
    }

    // ------------------------------------------------------------------
    // Drawing interaction
    // ------------------------------------------------------------------

    function boardCoords(ev) {
        var svg = document.getElementById('wb-canvas');
        var rect = svg.getBoundingClientRect();
        var x = ((ev.clientX - rect.left) / rect.width) * 100;
        var y = ((ev.clientY - rect.top) / rect.height) * 100;
        return {
            x: Math.max(0, Math.min(100, Math.round(x * 10) / 10)),
            y: Math.max(0, Math.min(100, Math.round(y * 10) / 10)),
        };
    }

    function wireCanvasEvents() {
        var svg = document.getElementById('wb-canvas');
        if (!svg) return;

        svg.addEventListener('pointerdown', function(ev) {
            if (_state.moveMode || !_state.boardId) return;
            ev.preventDefault();
            var pt = boardCoords(ev);
            if (_state.tool === 'text') {
                var label = window.prompt ? window.prompt('Text label:') : '';
                if (label) addAnnotation({ type: 'text', points: [pt], label: label, color: _state.color });
                return;
            }
            if (_state.tool === 'circle') {
                addAnnotation({ type: 'circle', points: [{ x: pt.x, y: pt.y, radius: 8 }], color: _state.color });
                return;
            }
            _state.drawing = true;
            _state.draftPoints = [pt];
            try { svg.setPointerCapture(ev.pointerId); } catch (e) { /* jsdom */ }
        });

        svg.addEventListener('pointermove', function(ev) {
            if (_state.moveMode && _state.playerIndex !== null) {
                moveDraggedPlayer(ev);
                return;
            }
            if (!_state.drawing) return;
            var pt = boardCoords(ev);
            _state.draftPoints.push(pt);
            renderDraft();
        });

        function finish(ev) {
            if (_state.moveMode && _state.playerIndex !== null) {
                commitDraggedPlayer(ev);
                return;
            }
            if (!_state.drawing) return;
            _state.drawing = false;
            layer('draft').innerHTML = '';
            var pts = _state.draftPoints.slice();
            _state.draftPoints = [];
            if (pts.length < 2) return;
            if (_state.tool === 'run' || _state.tool === 'pass') {
                // Run/Pass use the backend's canonical generator slots
                // (whiteboard_generate_player_run / _generate_pass) so the
                // stored shape matches the backend's own semantics.
                addGeneratedAnnotation(_state.tool, pts[0], pts[pts.length - 1]);
                return;
            }
            var a = { type: _state.tool, points: pts, color: _state.color };
            if (_state.tool === 'rectangle' || _state.tool === 'zone') {
                a.points = [pts[0], pts[pts.length - 1]];
            }
            addAnnotation(a);
        }

        svg.addEventListener('pointerup', finish);
        svg.addEventListener('pointerleave', function(ev) {
            if (_state.drawing) finish(ev);
        });
    }

    function renderDraft() {
        var g = layer('draft');
        if (!g) return;
        g.innerHTML = '';
        var pts = _state.draftPoints;
        if (!pts.length) return;
        if (_state.tool === 'freehand' && pts.length >= 2) {
            var pl = document.createElementNS(SVGNS, 'polyline');
            pl.setAttribute('points', pts.map(function(p) { return p.x + ',' + p.y; }).join(' '));
            pl.setAttribute('stroke', _state.color);
            pl.setAttribute('fill', 'none');
            pl.setAttribute('vector-effect', 'non-scaling-stroke');
            g.appendChild(pl);
        } else if (pts.length >= 2) {
            var ln = document.createElementNS(SVGNS, 'line');
            var a = pts[0], b = pts[pts.length - 1];
            ln.setAttribute('x1', String(a.x)); ln.setAttribute('y1', String(a.y));
            ln.setAttribute('x2', String(b.x)); ln.setAttribute('y2', String(b.y));
            ln.setAttribute('stroke', _state.color);
            ln.setAttribute('vector-effect', 'non-scaling-stroke');
            g.appendChild(ln);
        }
    }

    function moveDraggedPlayer(ev) {
        var g = layer('players');
        if (!g) return;
        var sel = g.querySelector('[data-wb-team="' + _state.team + '"][data-wb-index="' + _state.playerIndex + '"]');
        if (!sel) return;
        var pt = boardCoords(ev);
        var circle = sel.querySelector('circle');
        var text = sel.querySelector('text');
        if (circle) { circle.setAttribute('cx', String(pt.x)); circle.setAttribute('cy', String(pt.y)); }
        if (text) { text.setAttribute('x', String(pt.x)); text.setAttribute('y', String(pt.y)); }
    }

    function commitDraggedPlayer(ev) {
        var bridge = resolveBridge();
        var idx = _state.playerIndex, team = _state.team;
        _state.playerIndex = null;
        _state.team = null;
        if (!bridge || idx === null) return;
        var pt = boardCoords(ev);
        bridge.whiteboard_move_player(_state.boardId, idx, pt.x, pt.y, team, function(result) {
            try { _parse(result); _state.dirty = true; } catch (e) { status('Move failed: ' + e.message); }
        });
    }

    // ------------------------------------------------------------------
    // Toolbar actions
    // ------------------------------------------------------------------

    function addGeneratedAnnotation(kind, start, end) {
        function onGenerated(genResult) {
            var gen;
            try { gen = _parse(genResult); } catch (e) { status('Generator failed: ' + e.message); return; }
            addAnnotation(gen);
        }
        if (kind === 'pass') {
            bridge.whiteboard_generate_pass(start.x, start.y, end.x, end.y, '#2ecc71', onGenerated);
        } else {
            bridge.whiteboard_generate_player_run(start.x, start.y, end.x, end.y, '#f39c12', '', onGenerated);
        }
    }

    function addAnnotation(annotation) {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        annotation.width = 1.2;
        bridge.whiteboard_add_annotation(_state.boardId, JSON.stringify(annotation), function(result) {
            try {
                var created = _parse(result);
                _state.annotations.push(created);
                renderAnnotation(created);
                _state.dirty = true;
            } catch (e) { status('Draw failed: ' + e.message); }
        });
    }

    function onUndo() {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        // Operate on the live board (backend view of annotations), so Undo
        // works for annotations created in an earlier session too.
        bridge.whiteboard_get(_state.boardId, function(getResult) {
            var st;
            try { st = _parse(getResult); } catch (e) { status('Undo failed: ' + e.message); return; }
            var annos = st.annotations || [];
            if (!annos.length) { status('Nothing to undo.'); return; }
            var last = annos[annos.length - 1];
            bridge.whiteboard_remove_annotation(_state.boardId, last.id, function(result) {
                try {
                    _parse(result);
                    _state.annotations = _state.annotations.filter(function(a) { return a.id !== last.id; });
                    reloadAnnotations();
                } catch (e) { status('Undo failed: ' + e.message); }
            });
        });
    }

    function reloadAnnotations() {
        var bridge = resolveBridge();
        if (!bridge) return;
        bridge.whiteboard_get(_state.boardId, function(result) {
            try {
                var st = _parse(result);
                clearDynamicLayers();
                (st.players_home || []).forEach(function(p, i) { renderPlayer(p, i, 'home'); });
                (st.players_away || []).forEach(function(p, i) { renderPlayer(p, i, 'away'); });
                if (st.ball_position) renderBall(st.ball_position);
                (st.annotations || []).forEach(renderAnnotation);
            } catch (e) { status('Refresh failed: ' + e.message); }
        });
    }

    function onClear() {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        bridge.whiteboard_clear_annotations(_state.boardId, function(result) {
            try {
                _parse(result);
                _state.annotations = [];
                _state.dirty = true;
                reloadAnnotations();
                status('Drawings cleared.');
            } catch (e) { status('Clear failed: ' + e.message); }
        });
    }

    function onPlaceBall() {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        var label = window.prompt ? window.prompt('Ball position as "x,y" (0-100, e.g. 50,50 for kickoff):', '50,50') : null;
        if (!label) return;
        var parts = String(label).split(',').map(function(s) { return parseFloat(s, 10); });
        if (parts.length !== 2 || isNaN(parts[0]) || isNaN(parts[1])) {
            status('Ball position must be "x,y" numbers 0-100.');
            return;
        }
        var pos = { x: Math.max(0, Math.min(100, parts[0])), y: Math.max(0, Math.min(100, parts[1])) };
        bridge.whiteboard_update(_state.boardId, JSON.stringify({ ball_position: pos }), function(result) {
            try {
                var st = _parse(result);
                renderState(st);
                _state.dirty = true;
                status('Ball placed.');
            } catch (e) { status('Ball placement failed: ' + e.message); }
        });
    }

    function onApplyFormations() {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        var home = document.getElementById('wb-home-formation').value;
        var away = document.getElementById('wb-away-formation').value;
        var pending = 0;
        function done() {
            pending -= 1;
            if (pending === 0) {
                _state.dirty = true;
                reloadAnnotations();
                status('Formations applied.');
            }
        }
        ['home', 'away'].forEach(function(team) {
            var name = team === 'home' ? home : away;
            if (!name) return;
            pending += 1;
            bridge.whiteboard_set_formation(_state.boardId, name, team, function(result) {
                try { _parse(result); } catch (e) { status(team + ' formation failed: ' + e.message); }
                done();
            });
        });
        if (pending === 0) status('Pick at least one formation to apply.');
    }

    function onToggleMove() {
        _state.moveMode = !_state.moveMode;
        this.setAttribute('aria-pressed', _state.moveMode ? 'true' : 'false');
        this.classList.toggle('active', _state.moveMode);
        var svg = document.getElementById('wb-canvas');
        if (svg) svg.classList.toggle('wb-move-mode', _state.moveMode);
        status(_state.moveMode ? 'Move mode: drag players, click again to exit.' : 'Draw mode.');
    }

    function onSave() {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        // Send metadata only: annotations already persist server-side on each
        // add_annotation call, and update_state overwrites every key it
        // receives -- sending annotations: [] here would WIPE the board.
        bridge.whiteboard_update(_state.boardId, JSON.stringify({
            metadata: { saved_from_ui: true },
        }), function(result) {
            try {
                _parse(result);
                _state.annotations = [];
                _state.dirty = false;
                refreshBoardList(_state.boardId);
                status('Board saved.');
            } catch (e) { status('Save failed: ' + e.message); }
        });
    }

    function onExportSvg() {
        var bridge = resolveBridge();
        if (!bridge || !_state.boardId) return;
        bridge.whiteboard_generate_svg(_state.boardId, 1050, 680, function(result) {
            var svgText = null;
            try {
                var data = typeof result === 'string' ? JSON.parse(result) : result;
                svgText = data && data.svg;
            } catch (e) {
                svgText = typeof result === 'string' && result.indexOf('<svg') !== -1 ? result : null;
            }
            if (!svgText) {
                status('Export failed: backend returned no SVG.');
                return;
            }
            var blob = new Blob([String(svgText)], { type: 'image/svg+xml' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = (_state.name || 'whiteboard').replace(/[^\w\-]+/g, '_') + '.svg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
            status('SVG exported.');
        });
    }

    // Export for app.js delegation (same pattern as KawkabTactics).
    window.KawkabWhiteboard = {
        initWhiteboardWorkspace: initWhiteboardWorkspace,
    };
    window.initWhiteboardWorkspace = initWhiteboardWorkspace;
})();
