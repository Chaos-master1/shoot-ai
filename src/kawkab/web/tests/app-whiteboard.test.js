/* Tests for app-whiteboard.js — tactical whiteboard UI (CommonJS).
 *
 * Loads the real IIFE against a mock QWebChannel bridge and exercises the
 * full user flows: board creation, formation application, drawing tools,
 * run/pass generator slots, undo, ball placement, save, export, delete.
 * The backend contract these flows rely on is pinned by
 * tests/unit/test_bridge_contract.py (every bridge.X() call must have a
 * real @Slot); this suite pins the *frontend behavior* on top of that
 * contract: which slots get called with which arguments for each user
 * action, and what the user sees in response.
 */

const fs = require('fs');
const path = require('path');

const WB_JS_PATH = path.resolve(__dirname, '../js/app-whiteboard.js');
const wbCode = fs.readFileSync(WB_JS_PATH, 'utf-8');

// The shipped bundle always loads utils.js before the module IIFEs
// (frontend-manifest.json order); app-whiteboard.js resolves the bridge
// through KawkabUtils.getBridge, so the harness must load it too.
const UTILS_JS_PATH = path.resolve(__dirname, '../js/utils.js');
const utilsCode = fs.readFileSync(UTILS_JS_PATH, 'utf-8');

// A minimal but faithful WhiteboardState shape (mirrors
// kawkab/analysis/tactical_whiteboard.py to_dict()).
function makeState(overrides = {}) {
    return Object.assign({
        id: 'abc123',
        name: 'Test Board',
        pitch_orientation: 'horizontal',
        annotations: [],
        players_home: [],
        players_away: [],
        ball_position: null,
        formation_home: '',
        formation_away: '',
        timestamp: '2026-09-16T00:00:00Z',
        metadata: {},
    }, overrides);
}

// Mock bridge: records calls, replies from a scripted handler map.
function makeMockBridge() {
    const calls = [];
    const handlers = {};
    const bridge = new Proxy({}, {
        get(_target, prop) {
            if (prop === '__calls') return calls;
            if (prop === '__on') return handlers;
            return function (...args) {
                const cb = typeof args[args.length - 1] === 'function' ? args[args.length - 1] : null;
                calls.push({ method: prop, args: args.slice(0, cb ? -1 : undefined) });
                const handler = handlers[prop];
                if (cb) {
                    if (handler) cb(handler(args));
                    // No handler => simulate a silent backend (never replies).
                }
            };
        },
    });
    return bridge;
}

function loadIIFE() {
    // eslint-disable-next-line no-eval
    (function () { eval(utilsCode); }).call(window);
    // eslint-disable-next-line no-eval
    (function () { eval(wbCode); }).call(window);
}

const FORMATION_433 = [
    { x: 5, y: 50, pos: 'GK', num: 1 },
    { x: 15, y: 15, pos: 'LB', num: 3 },
    { x: 15, y: 38, pos: 'CB', num: 4 },
];

function setSectionDOM() {
    document.body.innerHTML = `
    <section id="whiteboard-section">
      <select id="wb-board-select"><option value="">-- Saved Boards --</option></select>
      <button id="wb-new-btn"></button>
      <button id="wb-delete-btn"></button>
      <span id="wb-status"></span>
      <div id="wb-unnamed" class="hidden">
        <input id="wb-name-input" value="Derby plan" />
        <select id="wb-formation-select"><option value="4-3-3">4-3-3</option></select>
        <button id="wb-create-btn"></button>
      </div>
      <div id="wb-board" class="hidden">
        <select id="wb-tool-select">
          <option value="arrow">arrow</option>
          <option value="run">run</option>
          <option value="pass">pass</option>
          <option value="line">line</option>
          <option value="freehand">freehand</option>
          <option value="circle">circle</option>
          <option value="rectangle">rectangle</option>
          <option value="zone">zone</option>
          <option value="text">text</option>
        </select>
        <select id="wb-color-select"><option value="#ffffff">white</option></select>
        <button id="wb-undo-btn"></button>
        <button id="wb-clear-btn"></button>
        <button id="wb-ball-btn"></button>
        <select id="wb-home-formation"><option value="">-- none --</option><option value="4-3-3">4-3-3</option></select>
        <select id="wb-away-formation"><option value="">-- none --</option><option value="4-3-3">4-3-3</option></select>
        <button id="wb-apply-formations-btn"></button>
        <button id="wb-template-both-btn"></button>
        <button id="wb-move-btn"></button>
        <button id="wb-save-btn"></button>
        <button id="wb-export-btn"></button>
        <svg id="wb-canvas" viewBox="0 0 100 100">
          <g id="wb-layer-zones"></g>
          <g id="wb-layer-annotations"></g>
          <g id="wb-layer-players"></g>
          <g id="wb-layer-draft"></g>
        </svg>
      </div>
    </section>`;
}

let bridge;

beforeEach(() => {
    jest.clearAllMocks();
    setSectionDOM();
    bridge = makeMockBridge();
    window.__kawkab = { bridge };
    window.bridge = bridge;
    delete window.KawkabWhiteboard;
    delete window.KawkabUtils;
    delete window.initWhiteboardWorkspace;
    // jsdom returns zero rects; give every element a 100x100 box. Element
    // (not HTMLElement) covers SVGElement too.
    const spy = jest.spyOn(window.Element.prototype, 'getBoundingClientRect');
    spy.mockReturnValue({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0 });
    // jsdom lacks the blob URL helpers the export path uses.
    URL.createObjectURL = jest.fn(() => 'blob:mock');
    URL.revokeObjectURL = jest.fn();
});

afterEach(() => {
    jest.restoreAllMocks();
});

function init() {
    loadIIFE();
    window.KawkabWhiteboard.initWhiteboardWorkspace();
}

function lastCall(method) {
    const hits = bridge.__calls.filter((c) => c.method === method);
    return hits.length ? hits[hits.length - 1] : null;
}

describe('whiteboard init', () => {
    test('init wires toolbar and loads templates + board list + status', () => {
        bridge.__on.whiteboard_list_templates = () => [{ name: '4-3-3', label: '4-3-3', count: 11 }];
        bridge.__on.whiteboard_list = () => [];
        bridge.__on.check_whiteboard_status = () => ({ available: true });
        bridge.__on.whiteboard_get_template = () => ({ positions: FORMATION_433 });

        init();

        expect(lastCall('whiteboard_list_templates')).not.toBeNull();
        expect(lastCall('whiteboard_list')).not.toBeNull();
        expect(lastCall('check_whiteboard_status')).not.toBeNull();
        expect(window.KawkabWhiteboard).toBeDefined();
    });

    test('init is idempotent (guard flag) so router + boot init do not double-wire', () => {
        bridge.__on.whiteboard_list_templates = () => [];
        bridge.__on.whiteboard_list = () => [];
        init();
        init();
        const templateCalls = bridge.__calls.filter((c) => c.method === 'whiteboard_list_templates');
        expect(templateCalls.length).toBe(1);
    });

    test('init without a bridge is a safe no-op', () => {
        window.__kawkab = {};
        window.bridge = undefined;
        expect(() => init()).not.toThrow();
    });
});

describe('board lifecycle', () => {
    beforeEach(() => {
        bridge.__on.whiteboard_list_templates = () => [{ name: '4-3-3', label: '4-3-3', count: 11 }];
        bridge.__on.whiteboard_list = () => [];
        bridge.__on.check_whiteboard_status = () => ({ available: true });
        init();
    });

    test('New Board reveals the naming panel', () => {
        document.getElementById('wb-new-btn').click();
        expect(document.getElementById('wb-unnamed').classList.contains('hidden')).toBe(false);
        expect(document.getElementById('wb-board').classList.contains('hidden')).toBe(true);
    });

    test('Create calls whiteboard_create with name + formation and renders the board', () => {
        bridge.__on.whiteboard_create = (args) => makeState({
            name: args[0],
            players_home: FORMATION_433.map((p) => ({ x: p.x, y: p.y, number: p.num, position: p.pos, team: 'home' })),
            formation_home: args[1],
        });

        document.getElementById('wb-new-btn').click();
        // New Board clears the input; the coach types the name now.
        document.getElementById('wb-name-input').value = 'Derby plan';
        document.getElementById('wb-formation-select').value = '4-3-3';
        document.getElementById('wb-create-btn').click();

        const call = lastCall('whiteboard_create');
        expect(call).not.toBeNull();
        expect(call.args[0]).toBe('Derby plan');
        expect(call.args[1]).toBe('4-3-3');

        expect(document.getElementById('wb-board').classList.contains('hidden')).toBe(false);
        // Players rendered from the created state
        const players = document.querySelectorAll('#wb-layer-players .wb-player');
        expect(players.length).toBe(FORMATION_433.length);
        // Board list refreshed and reselected
        expect(lastCall('whiteboard_list')).not.toBeNull();
    });

    test('selecting a saved board loads it via whiteboard_get', () => {
        bridge.__on.whiteboard_get = () => makeState({
            annotations: [{ id: 'a1', type: 'arrow', points: [{ x: 1, y: 1 }, { x: 9, y: 9 }], color: '#ffffff', width: 1.2, opacity: 1, dashed: false, label: '', player_number: '', layer: 0, created_at: '' }],
        });
        const sel = document.getElementById('wb-board-select');
        const opt = document.createElement('option');
        opt.value = 'abc123';
        sel.appendChild(opt);
        sel.value = 'abc123';
        sel.dispatchEvent(new Event('change'));

        expect(lastCall('whiteboard_get').args[0]).toBe('abc123');
        expect(document.getElementById('wb-board').classList.contains('hidden')).toBe(false);
        expect(document.querySelectorAll('#wb-layer-annotations line').length).toBe(1);
    });

    test('Delete removes the current board and hides the editor', () => {
        bridge.__on.whiteboard_create = () => makeState();
        bridge.__on.whiteboard_delete = () => ({ ok: true });
        document.getElementById('wb-new-btn').click();
        document.getElementById('wb-create-btn').click();
        document.getElementById('wb-delete-btn').click();

        expect(lastCall('whiteboard_delete').args[0]).toBe('abc123');
        expect(document.getElementById('wb-board').classList.contains('hidden')).toBe(true);
    });
});

describe('formations and templates', () => {
    beforeEach(() => {
        // Return the 4-3-3 template so the formation selects get real
        // options (loadTemplates overwrites their innerHTML during init).
        bridge.__on.whiteboard_list_templates = () => [{ name: '4-3-3', label: '4-3-3', count: 11 }];
        bridge.__on.whiteboard_list = () => [];
        bridge.__on.check_whiteboard_status = () => ({ available: true });
        bridge.__on.whiteboard_create = () => makeState();
        init();
        document.getElementById('wb-new-btn').click();
        document.getElementById('wb-create-btn').click();
    });

    test('Apply Formations calls whiteboard_set_formation per selected team', () => {
        bridge.__on.whiteboard_set_formation = () => ({ players: [] });
        document.getElementById('wb-home-formation').value = '4-3-3';
        document.getElementById('wb-away-formation').value = '4-3-3';
        document.getElementById('wb-apply-formations-btn').click();

        const formationCalls = bridge.__calls.filter((c) => c.method === 'whiteboard_set_formation');
        expect(formationCalls.length).toBe(2);
        expect(formationCalls.map((c) => c.args[2]).sort()).toEqual(['away', 'home']);
    });

    test('Template Both Teams mirrors positions for the away side', () => {
        bridge.__on.whiteboard_get_template = () => ({ positions: FORMATION_433 });
        bridge.__on.whiteboard_update = (args) => {
            const data = JSON.parse(args[1]);
            return makeState({
                players_home: data.players_home,
                players_away: data.players_away,
                formation_home: data.formation_home,
                formation_away: data.formation_away,
            });
        };
        document.getElementById('wb-home-formation').value = '4-3-3';
        document.getElementById('wb-template-both-btn').click();

        expect(lastCall('whiteboard_get_template').args[0]).toBe('4-3-3');
        const upd = lastCall('whiteboard_update');
        const data = JSON.parse(upd.args[1]);
        expect(data.formation_home).toBe('4-3-3');
        expect(data.formation_away).toBe('4-3-3');
        expect(data.players_home[0].x).toBe(5);
        expect(data.players_away[0].x).toBe(95); // mirrored
        expect(data.players_away[0].team).toBe('away');
    });
});

describe('drawing', () => {
    beforeEach(() => {
        bridge.__on.whiteboard_list_templates = () => [];
        bridge.__on.whiteboard_list = () => [];
        bridge.__on.check_whiteboard_status = () => ({ available: true });
        bridge.__on.whiteboard_create = () => makeState();
        init();
        document.getElementById('wb-new-btn').click();
        document.getElementById('wb-create-btn').click();
    });

    function pointer(type, x, y) {
        // jsdom has no PointerEvent; the module only reads clientX/clientY
        // (setPointerCapture is try/catch-guarded), so MouseEvent works.
        return new MouseEvent(type, { clientX: x, clientY: y, bubbles: true });
    }

    function drag(svg, from, to) {
        svg.dispatchEvent(pointer('pointerdown', from[0], from[1]));
        svg.dispatchEvent(pointer('pointermove', to[0], to[1]));
        svg.dispatchEvent(pointer('pointerup', to[0], to[1]));
    }

    test('drag with the arrow tool adds an annotation via the backend', () => {
        bridge.__on.whiteboard_add_annotation = (args) => {
            const a = JSON.parse(args[1]);
            return Object.assign({ id: 'n1', created_at: '' }, a);
        };
        const svg = document.getElementById('wb-canvas');
        drag(svg, [10, 10], [40, 40]);

        const call = lastCall('whiteboard_add_annotation');
        expect(call).not.toBeNull();
        expect(call.args[0]).toBe('abc123');
        const ann = JSON.parse(call.args[1]);
        expect(ann.type).toBe('arrow');
        expect(ann.points[0]).toEqual({ x: 10, y: 10 });
        expect(ann.points[1]).toEqual({ x: 40, y: 40 });
        // Rendered with an arrowhead polygon
        expect(document.querySelectorAll('#wb-layer-annotations polygon').length).toBe(1);
    });

    test('rectangle collapses the drag path to two corners', () => {
        bridge.__on.whiteboard_add_annotation = (args) => Object.assign({ id: 'n2', created_at: '' }, JSON.parse(args[1]));
        document.getElementById('wb-tool-select').value = 'rectangle';
        const svg = document.getElementById('wb-canvas');
        drag(svg, [10, 10], [50, 60]);

        const ann = JSON.parse(lastCall('whiteboard_add_annotation').args[1]);
        expect(ann.points).toEqual([{ x: 10, y: 10 }, { x: 50, y: 60 }]);
    });

    test('run tool uses the canonical generator slot then persists the shape', () => {
        bridge.__on.whiteboard_generate_player_run = (args) => ({
            id: 'g1', type: 'arrow', points: [{ x: args[0], y: args[1] }, { x: args[2], y: args[3] }],
            color: args[4], width: 3, opacity: 1, dashed: false, label: args[5], player_number: '', layer: 0, created_at: '',
        });
        bridge.__on.whiteboard_add_annotation = (args) => Object.assign({ created_at: '' }, JSON.parse(args[1]));

        document.getElementById('wb-tool-select').value = 'run';
        document.getElementById('wb-tool-select').dispatchEvent(new Event('change'));
        const svg = document.getElementById('wb-canvas');
        drag(svg, [20, 30], [60, 30]);

        const gen = lastCall('whiteboard_generate_player_run');
        expect(gen).not.toBeNull();
        expect(gen.args[4]).toBe('#f39c12');
        const added = JSON.parse(lastCall('whiteboard_add_annotation').args[1]);
        expect(added.type).toBe('arrow');
    });

    test('pass tool uses the pass generator slot', () => {
        bridge.__on.whiteboard_generate_pass = (args) => ({
            id: 'g2', type: 'arrow', points: [{ x: args[0], y: args[1] }, { x: args[2], y: args[3] }],
            color: args[4], width: 2, opacity: 1, dashed: true, label: '', player_number: '', layer: 0, created_at: '',
        });
        bridge.__on.whiteboard_add_annotation = (args) => Object.assign({ created_at: '' }, JSON.parse(args[1]));

        document.getElementById('wb-tool-select').value = 'pass';
        document.getElementById('wb-tool-select').dispatchEvent(new Event('change'));
        const svg = document.getElementById('wb-canvas');
        drag(svg, [50, 50], [70, 20]);

        const gen = lastCall('whiteboard_generate_pass');
        expect(gen).not.toBeNull();
        expect(gen.args[4]).toBe('#2ecc71');
        expect(lastCall('whiteboard_add_annotation')).not.toBeNull();
    });

    test('text tool prompts and stores a label', () => {
        const promptSpy = jest.spyOn(window, 'prompt').mockReturnValue('Press high');
        bridge.__on.whiteboard_add_annotation = (args) => Object.assign({ id: 'n3', created_at: '' }, JSON.parse(args[1]));
        document.getElementById('wb-tool-select').value = 'text';
        document.getElementById('wb-tool-select').dispatchEvent(new Event('change'));
        const svg = document.getElementById('wb-canvas');
        svg.dispatchEvent(pointer('pointerdown', 30, 30));

        const ann = JSON.parse(lastCall('whiteboard_add_annotation').args[1]);
        expect(ann.type).toBe('text');
        expect(ann.label).toBe('Press high');
        promptSpy.mockRestore();
    });

    test('draw before creating a board is a safe no-op', () => {
        document.getElementById('wb-new-btn').click(); // board hidden, boardId null
        const svg = document.getElementById('wb-canvas');
        drag(svg, [10, 10], [40, 40]);
        expect(lastCall('whiteboard_add_annotation')).toBeNull();
    });
});

describe('undo / clear / ball / save / export', () => {
    beforeEach(() => {
        bridge.__on.whiteboard_list_templates = () => [];
        bridge.__on.whiteboard_list = () => [];
        bridge.__on.check_whiteboard_status = () => ({ available: true });
        bridge.__on.whiteboard_create = () => makeState();
        init();
        document.getElementById('wb-new-btn').click();
        document.getElementById('wb-create-btn').click();
    });

    test('undo removes the newest backend annotation and re-renders', () => {
        bridge.__on.whiteboard_get = () => makeState({
            annotations: [
                { id: 'old', type: 'circle', points: [{ x: 5, y: 5, radius: 8 }], color: '#fff', width: 1.2, opacity: 1, dashed: false, label: '', player_number: '', layer: 0, created_at: '' },
                { id: 'new', type: 'circle', points: [{ x: 9, y: 9, radius: 8 }], color: '#fff', width: 1.2, opacity: 1, dashed: false, label: '', player_number: '', layer: 0, created_at: '' },
            ],
        });
        bridge.__on.whiteboard_remove_annotation = () => ({ ok: true });
        document.getElementById('wb-undo-btn').click();

        expect(lastCall('whiteboard_remove_annotation').args[1]).toBe('new');
        expect(lastCall('whiteboard_get')).not.toBeNull();
    });

    test('clear wipes annotations through the dedicated slot', () => {
        bridge.__on.whiteboard_clear_annotations = () => ({ ok: true });
        bridge.__on.whiteboard_get = () => makeState();
        document.getElementById('wb-clear-btn').click();
        expect(lastCall('whiteboard_clear_annotations').args[0]).toBe('abc123');
    });

    test('place ball persists ball_position via whiteboard_update', () => {
        const promptSpy = jest.spyOn(window, 'prompt').mockReturnValue('50,50');
        bridge.__on.whiteboard_update = (args) => makeState(JSON.parse(args[1]));
        document.getElementById('wb-ball-btn').click();

        const data = JSON.parse(lastCall('whiteboard_update').args[1]);
        expect(data.ball_position).toEqual({ x: 50, y: 50 });
        expect(document.querySelector('#wb-layer-players [data-wb-ball]')).not.toBeNull();
        promptSpy.mockRestore();
    });

    test('save sends metadata only (never annotations: [], which would wipe the board)', () => {
        bridge.__on.whiteboard_update = () => makeState();
        document.getElementById('wb-save-btn').click();
        const data = JSON.parse(lastCall('whiteboard_update').args[1]);
        expect('annotations' in data).toBe(false);
        expect(data.metadata.saved_from_ui).toBe(true);
    });

    test('export fetches SVG from the backend and triggers a download', () => {
        bridge.__on.whiteboard_generate_svg = () => ({ svg: '<svg xmlns="http://www.w3.org/2000/svg"></svg>' });
        const clickSpy = jest.fn();
        jest.spyOn(window.HTMLAnchorElement.prototype, 'click').mockImplementation(clickSpy);

        document.getElementById('wb-export-btn').click();

        expect(lastCall('whiteboard_generate_svg')).not.toBeNull();
        expect(clickSpy).toHaveBeenCalled();
    });
});
