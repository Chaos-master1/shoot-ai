/* Tests for ui.js — toast, skeleton, collapsible, modal (IIFE attaching to window) */

/** ui.js is a plain IIFE (like every other web/js/*.js file), not an ES
 * module -- it has never had `export function` since this test was
 * written. require()-ing it runs the IIFE, which attaches its public
 * functions directly onto `window` (see the "Public API" section at the
 * bottom of ui.js); jest.resetModules() forces a fresh run each test so
 * state (e.g. the notification list closure) doesn't leak between tests,
 * matching the working pattern in js/__tests__/router.test.js. */
function loadUi() {
    jest.resetModules();
    require('../js/ui.js');
    return {
        showToast: window.showToast,
        showSkeleton: window.showSkeleton,
        hideSkeleton: window.hideSkeleton,
        toggleCollapsible: window.toggleCollapsible,
        updateWorkflowStep: window.updateWorkflowStep,
        openModal: window.openModal,
        closeModal: window.closeModal,
        loadMissingKeys: window.loadMissingKeys,
    };
}

var ui;

beforeEach(function() {
    document.body.innerHTML = '';
    jest.useFakeTimers();
    ui = loadUi();
});

afterEach(function() {
    jest.useRealTimers();
});

// ── showToast ────────────────────────────────────────────────────────────────

describe('showToast', function() {
    it('creates toast container if missing and appends toast', function() {
        ui.showToast('Hello', 'info');
        var container = document.getElementById('toast-container');
        expect(container).toBeTruthy();
        expect(container.childElementCount).toBe(1);
        expect(container.firstChild.textContent).toBe('Hello');
        expect(container.getAttribute('aria-live')).toBe('polite');
        expect(container.getAttribute('role')).toBe('alert');
    });

    it('sets correct background for error type', function() {
        ui.showToast('Error!', 'error');
        var toast = document.querySelector('#toast-container div');
        expect(toast.style.background).toBe('rgb(220, 38, 38)');
    });

    it('sets correct background for success type', function() {
        ui.showToast('Success!', 'success');
        var toast = document.querySelector('#toast-container div');
        expect(toast.style.background).toBe('rgb(22, 163, 74)');
    });

    it('sets default background for info type', function() {
        ui.showToast('Info', 'info');
        var toast = document.querySelector('#toast-container div');
        expect(toast.style.background).toBe('rgb(59, 130, 246)');
    });

    it('defaults to info type when type is omitted', function() {
        ui.showToast('Default');
        var toast = document.querySelector('#toast-container div');
        expect(toast.style.background).toBe('rgb(59, 130, 246)');
    });

    it('removes toast after 5 seconds', function() {
        ui.showToast('Timed', 'info');
        expect(document.querySelector('#toast-container').childElementCount).toBe(1);
        jest.advanceTimersByTime(5000);
        expect(document.querySelector('#toast-container').childElementCount).toBe(0);
    });

    it('removes toast on click', function() {
        ui.showToast('Clickable', 'info');
        var toast = document.querySelector('#toast-container div');
        toast.click();
        expect(document.querySelector('#toast-container div')).toBeNull();
    });
});

// ── showSkeleton / hideSkeleton ─────────────────────────────────────────────

describe('showSkeleton / hideSkeleton', function() {
    beforeEach(function() {
        var container = document.createElement('div');
        container.id = 'test-container';
        document.body.appendChild(container);
    });

    it('showSkeleton adds 3 skeleton divs to container', function() {
        ui.showSkeleton('test-container');
        var skeletons = document.querySelectorAll('.skeleton');
        expect(skeletons.length).toBe(3);
    });

    it('showSkeleton does nothing for missing container', function() {
        expect(function() { ui.showSkeleton('non-existent'); }).not.toThrow();
    });

    it('hideSkeleton removes skeleton elements', function() {
        var container = document.getElementById('test-container');
        container.innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div>real</div>';
        ui.hideSkeleton('test-container');
        expect(container.querySelectorAll('.skeleton').length).toBe(0);
        expect(container.textContent).toBe('real');
    });

    it('hideSkeleton does nothing for missing container', function() {
        expect(function() { ui.hideSkeleton('non-existent'); }).not.toThrow();
    });
});

// ── toggleCollapsible ───────────────────────────────────────────────────────

describe('toggleCollapsible', function() {
    it('toggles collapsed class on parent pro-card', function() {
        var card = document.createElement('div');
        card.className = 'pro-card';
        var header = document.createElement('div');
        header.className = 'pro-card-header';
        var body = document.createElement('div');
        body.className = 'pro-card-body';
        card.appendChild(header);
        card.appendChild(body);
        document.body.appendChild(card);

        ui.toggleCollapsible(header);
        expect(card.classList.contains('collapsed')).toBe(true);
        expect(body.style.display).toBe('none');

        ui.toggleCollapsible(header);
        expect(card.classList.contains('collapsed')).toBe(false);
        expect(body.style.display).toBe('');
    });

    it('does nothing if header has no pro-card parent', function() {
        var orphan = document.createElement('div');
        expect(function() { ui.toggleCollapsible(orphan); }).not.toThrow();
    });
});

// ── updateWorkflowStep ──────────────────────────────────────────────────────

describe('updateWorkflowStep', function() {
    function createWorkflow() {
        var container = document.createElement('div');
        container.className = 'workflow-steps';
        container.innerHTML =
            '<div class="workflow-step"><span class="workflow-circle">1</span></div>' +
            '<div class="workflow-step"><span class="workflow-circle">2</span></div>' +
            '<div class="workflow-step"><span class="workflow-circle">3</span></div>' +
            '<div class="workflow-step"><span class="workflow-circle">4</span></div>';
        document.body.appendChild(container);
        return container;
    }

    it('sets step 1 as active, others as neither', function() {
        var container = createWorkflow();
        ui.updateWorkflowStep(1);
        var steps = container.querySelectorAll('.workflow-step');
        expect(steps[0].classList.contains('active')).toBe(true);
        expect(steps[0].classList.contains('completed')).toBe(false);
        for (var i = 1; i < 4; i++) {
            expect(steps[i].classList.contains('active')).toBe(false);
            expect(steps[i].classList.contains('completed')).toBe(false);
        }
    });

    it('sets earlier steps as completed, current as active', function() {
        var container = createWorkflow();
        ui.updateWorkflowStep(3);
        var steps = container.querySelectorAll('.workflow-step');
        expect(steps[0].classList.contains('completed')).toBe(true);
        expect(steps[1].classList.contains('completed')).toBe(true);
        expect(steps[2].classList.contains('active')).toBe(true);
        expect(steps[3].classList.contains('active')).toBe(false);
        expect(steps[3].classList.contains('completed')).toBe(false);
    });

    it('updates aria attributes on container', function() {
        var container = createWorkflow();
        ui.updateWorkflowStep(2);
        expect(container.getAttribute('aria-label')).toBe('Workflow progress: step 2 of 4');
        expect(container.getAttribute('aria-valuenow')).toBe('2');
    });

    it('does nothing if no workflow-steps container exists', function() {
        expect(function() { ui.updateWorkflowStep(1); }).not.toThrow();
    });
});

// ── openModal / closeModal ──────────────────────────────────────────────────

describe('openModal / closeModal', function() {
    it('openModal removes hidden class', function() {
        var modal = document.createElement('div');
        modal.id = 'test-modal';
        modal.className = 'hidden';
        document.body.appendChild(modal);
        ui.openModal('test-modal');
        expect(modal.classList.contains('hidden')).toBe(false);
    });

    it('closeModal adds hidden class', function() {
        var modal = document.createElement('div');
        modal.id = 'test-modal';
        document.body.appendChild(modal);
        ui.closeModal('test-modal');
        expect(modal.classList.contains('hidden')).toBe(true);
    });

    it('openModal does nothing for missing modal', function() {
        expect(function() { ui.openModal('non-existent'); }).not.toThrow();
    });

    it('closeModal does nothing for missing modal', function() {
        expect(function() { ui.closeModal('non-existent'); }).not.toThrow();
    });
});

// ── Modal focus trap (WCAG 2.1 §2.4.3 / §2.1.2) ─────────────────────────────

describe('modal focus trap', function() {
    var modal;

    function buildModal() {
        modal = document.createElement('div');
        modal.id = 'trap-modal';
        modal.className = 'hidden';
        modal.innerHTML =
            '<button id="trap-first">First</button>' +
            '<input id="trap-mid" type="text" />' +
            '<button id="trap-last">Last</button>';
        document.body.appendChild(modal);
        return modal;
    }

    function pressTab(modal, shiftKey) {
        var ev = new window.KeyboardEvent('keydown', {
            key: 'Tab', shiftKey: !!shiftKey, bubbles: true, cancelable: true
        });
        modal.dispatchEvent(ev);
        return ev;
    }

    afterEach(function() {
        if (modal && modal.parentNode) modal.parentNode.removeChild(modal);
        modal = null;
    });

    it('openModal moves focus to the first focusable element', function() {
        buildModal();
        ui.openModal('trap-modal');
        expect(document.activeElement.id).toBe('trap-first');
        ui.closeModal('trap-modal');
    });

    it('Tab on the last element wraps to the first', function() {
        buildModal();
        ui.openModal('trap-modal');
        document.getElementById('trap-last').focus();
        var ev = pressTab(modal, false);
        expect(ev.defaultPrevented).toBe(true);
        expect(document.activeElement.id).toBe('trap-first');
        ui.closeModal('trap-modal');
    });

    it('Shift+Tab on the first element wraps to the last', function() {
        buildModal();
        ui.openModal('trap-modal');
        var ev = pressTab(modal, true);
        expect(ev.defaultPrevented).toBe(true);
        expect(document.activeElement.id).toBe('trap-last');
        ui.closeModal('trap-modal');
    });

    it('Escape closes the modal', function() {
        buildModal();
        ui.openModal('trap-modal');
        modal.dispatchEvent(new window.KeyboardEvent('keydown', {
            key: 'Escape', bubbles: true, cancelable: true
        }));
        expect(modal.classList.contains('hidden')).toBe(true);
    });

    it('closeModal restores focus to the element that opened it', function() {
        buildModal();
        var opener = document.createElement('button');
        opener.id = 'trap-opener';
        document.body.appendChild(opener);
        opener.focus();
        ui.openModal('trap-modal');
        expect(document.activeElement.id).toBe('trap-first');
        ui.closeModal('trap-modal');
        expect(document.activeElement.id).toBe('trap-opener');
        opener.parentNode.removeChild(opener);
    });

    it('closeModal does not crash when the opener element is gone', function() {
        buildModal();
        var opener = document.createElement('button');
        document.body.appendChild(opener);
        opener.focus();
        ui.openModal('trap-modal');
        opener.parentNode.removeChild(opener);
        expect(function() { ui.closeModal('trap-modal'); }).not.toThrow();
    });

    it('openModal focuses the modal itself when it has no focusable children', function() {
        var bare = document.createElement('div');
        bare.id = 'trap-bare';
        bare.className = 'hidden';
        document.body.appendChild(bare);
        ui.openModal('trap-bare');
        expect(document.activeElement).toBe(bare);
        ui.closeModal('trap-bare');
        bare.parentNode.removeChild(bare);
    });
});
