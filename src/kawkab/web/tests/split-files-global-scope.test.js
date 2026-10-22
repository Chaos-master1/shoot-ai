/* Regression test for a real bug: app-opponent.js, app-marketplace.js, and
 * app-ai.js are NOT wrapped in their own IIFE (split out of app.js without
 * one), unlike every other web/js/*.js file. scripts/bundle-js.mjs
 * concatenates raw source with no per-file wrapper, so their top-level
 * `function`/`var` declarations become real globals in the shipped
 * bundle -- and so did every escapeHtml(...) call inside them, which
 * resolved to nothing (ReferenceError) until each file gained the same
 * `var escapeHtml = window.__kawkab.escapeHtml;` import line
 * app-data-providers.js already had.
 *
 * require() alone would NOT catch this: Node's CommonJS wrapper gives
 * each required file its own function scope, accidentally providing the
 * IIFE isolation these files lack in the real concatenated bundle. This
 * uses vm.runInThisContext to replicate the bundler's actual behavior --
 * raw source, executed against the same global object, in load order --
 * so top-level declarations land on the real `window`/`global` the same
 * way they do once bundled.
 *
 * vm.runInThisContext runs against Node's own global context, not
 * jest-environment-jsdom's patched one (confirmed: it throws "window is
 * not defined" even for utils.js's own first line). Indirect eval
 * (`(0, eval)(...)`) always runs in whatever the *caller's* global scope
 * is, which in this Jest environment is jsdom's -- so top-level
 * declarations correctly land on the same `window` the assertions below
 * read from.
 */
const fs = require('fs');
const path = require('path');
const indirectEval = eval;

function loadConcatenated(...files) {
    const raw = files
        .map((f) => fs.readFileSync(path.join(__dirname, '../js', f), 'utf-8'))
        .join('\n');
    indirectEval(raw);
}

beforeEach(() => {
    window.__kawkab = undefined;
    window.escapeHtml = undefined;
});

describe('files split out of app.js without their own IIFE', () => {
    it('app-opponent.js resolves escapeHtml via window.__kawkab when utils.js loads first (real manifest order)', () => {
        expect(() => loadConcatenated('utils.js', 'app-opponent.js')).not.toThrow();
        expect(typeof escapeHtml).toBe('function');
        expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
        expect(typeof initOpponentWorkspace).toBe('function');
    });

    it('app-marketplace.js resolves escapeHtml via window.__kawkab when utils.js loads first', () => {
        expect(() => loadConcatenated('utils.js', 'app-marketplace.js')).not.toThrow();
        expect(typeof escapeHtml).toBe('function');
        expect(typeof initMarketplace).toBe('function');
    });

    it('app-ai.js resolves escapeHtml via window.__kawkab when utils.js loads first', () => {
        expect(() => loadConcatenated('utils.js', 'app-ai.js')).not.toThrow();
        expect(typeof escapeHtml).toBe('function');
        expect(typeof initAiWorkspace).toBe('function');
    });

    it('would throw if utils.js did not load first -- proves the fix is load-order dependent, matching frontend-manifest.json (utils.js at position 6, well before these three)', () => {
        expect(() => loadConcatenated('app-opponent.js')).toThrow();
    });
});
