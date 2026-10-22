/* Kawkab AI — UI utilities: toast, skeleton, collapsible, modal, notifications */

(function () {
    "use strict";

    // ============================================================
    // Toast
    // ============================================================
    function showToast(message, type) {
        type = type || "info";
        var container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            container.style.cssText = "position:fixed;bottom:20px;right:20px;z-index:10000;display:flex;flex-direction:column;gap:8px;max-width:350px";
            container.setAttribute("aria-live", "polite");
            container.setAttribute("role", "alert");
            document.body.appendChild(container);
        }
        var toast = document.createElement("div");
        var bg = type === "error" ? "#dc2626" : type === "success" ? "#16a34a" : "#3b82f6";
        toast.style.cssText = "background:" + bg + ";color:#fff;padding:10px 16px;border-radius:6px;font-size:0.85rem;box-shadow:0 4px 12px rgba(0,0,0,0.3);animation:slideIn 0.2s ease;cursor:pointer";
        toast.textContent = message;
        toast.onclick = function () { toast.remove(); };
        container.appendChild(toast);
        setTimeout(function () { if (toast.parentNode) toast.remove(); }, 5000);
    }

    function showSkeleton(containerId) {
        var container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '<div class="skeleton" style="height:80px;margin:4px 0"></div>'.repeat(3);
    }

    function hideSkeleton(containerId) {
        var container = document.getElementById(containerId);
        if (!container) return;
        container.querySelectorAll(".skeleton").forEach(function (el) { el.remove(); });
    }

    function toggleCollapsible(header) {
        var card = header.closest(".pro-card");
        if (!card) return;
        card.classList.toggle("collapsed");
        var body = card.querySelector(".pro-card-body");
        if (body) {
            body.style.display = card.classList.contains("collapsed") ? "none" : "";
        }
    }

    function updateWorkflowStep(step) {
        var stepsContainer = document.querySelector(".workflow-steps");
        if (!stepsContainer) return;
        var steps = stepsContainer.querySelectorAll(".workflow-step");
        steps.forEach(function (s, i) {
            var num = i + 1;
            s.classList.remove("active", "completed");
            s.querySelector(".workflow-circle").textContent = num;
            if (num === step) {
                s.classList.add("active");
            } else if (num < step) {
                s.classList.add("completed");
            }
        });
        stepsContainer.setAttribute("aria-label", "Workflow progress: step " + step + " of 4");
        stepsContainer.setAttribute("aria-valuenow", step);
    }

    // ── Modal accessibility (WCAG 2.1 §2.4.3 / §2.1.2) ──
    // Focus trap: when a modal opens, focus moves to its first focusable
    // element, Tab cycles inside it, Escape closes it, and closing restores
    // focus to the element that opened it. Handlers are stored per-modal so
    // nested/stacked modals don't clobber each other.
    var _modalTrapHandlers = {};
    var _modalReturnFocus = {};

    function _modalFocusable(modal) {
        var sel = "a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex='-1'])";
        var els = modal.querySelectorAll(sel);
        var out = [];
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            // Skip elements hidden by the app's .hidden/[hidden] conventions
            // (offsetParent-based checks are unusable: jsdom never
            // implements layout, so offsetParent is always null there).
            if (el.closest("[hidden], .hidden")) continue;
            out.push(el);
        }
        return out;
    }

    function _wireModalTrap(modalId) {
        var modal = document.getElementById(modalId);
        if (!modal || _modalTrapHandlers[modalId]) return;
        var handler = function (e) {
            if (modal.classList.contains("hidden")) return;
            if (e.key === "Escape") {
                closeModal(modalId);
                return;
            }
            if (e.key !== "Tab") return;
            var focusable = _modalFocusable(modal);
            if (!focusable.length) {
                e.preventDefault();
                return;
            }
            var first = focusable[0];
            var last = focusable[focusable.length - 1];
            if (e.shiftKey && document.activeElement === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        };
        modal.addEventListener("keydown", handler);
        _modalTrapHandlers[modalId] = handler;
    }

    function _unwireModalTrap(modalId) {
        var modal = document.getElementById(modalId);
        var handler = _modalTrapHandlers[modalId];
        if (modal && handler) modal.removeEventListener("keydown", handler);
        delete _modalTrapHandlers[modalId];
    }

    function openModal(modalId) {
        var modal = document.getElementById(modalId);
        if (!modal) return;
        _modalReturnFocus[modalId] = document.activeElement;
        modal.classList.remove("hidden");
        _wireModalTrap(modalId);
        var focusable = _modalFocusable(modal);
        if (focusable.length) {
            focusable[0].focus();
        } else if (!modal.hasAttribute("tabindex")) {
            modal.setAttribute("tabindex", "-1");
        }
        if (!focusable.length) modal.focus();
    }

    function closeModal(modalId) {
        var modal = document.getElementById(modalId);
        if (!modal) return;
        modal.classList.add("hidden");
        _unwireModalTrap(modalId);
        var prev = _modalReturnFocus[modalId];
        delete _modalReturnFocus[modalId];
        if (prev && prev.focus && document.contains(prev)) {
            prev.focus();
        }
    }

    function loadMissingKeys() {
        var els = document.querySelectorAll("[data-i18n]");
        els.forEach(function (el) {
            var key = el.getAttribute("data-i18n");
            if (!key) return;
            var current = el.textContent.trim();
            if (!current || current === key) {
                el.textContent = key;
            }
        });
        var placeholderEls = document.querySelectorAll("[data-i18n-placeholder]");
        placeholderEls.forEach(function (el) {
            var key = el.getAttribute("data-i18n-placeholder");
            if (!key) return;
            if (!el.getAttribute("placeholder") || el.getAttribute("placeholder") === key) {
                el.setAttribute("placeholder", key);
            }
        });
    }

    // ============================================================
    // Notification System
    // ============================================================
    var _notifications = [];
    var NOTIF_KEY = "kawkab_notifications";

    function _notifIcon(type) {
        return type === "error" ? "&#x274C;" :
               type === "success" ? "&#x2705;" :
               type === "warning" ? "&#x26A0;&#xFE0F;" :
               type === "info" ? "&#x2139;&#xFE0F;" : "&#x2139;&#xFE0F;";
    }

    function _notifLoad() {
        try {
            var raw = localStorage.getItem(NOTIF_KEY);
            if (raw) { _notifications = JSON.parse(raw); }
        } catch (e) { _notifications = []; }
    }

    function _notifSave() {
        try { localStorage.setItem(NOTIF_KEY, JSON.stringify(_notifications)); } catch (e) {}
    }

    function _notifUpdateBadge() {
        var badge = document.getElementById("notification-count");
        if (!badge) return;
        var unread = 0;
        for (var i = 0; i < _notifications.length; i++) {
            if (!_notifications[i].read) unread++;
        }
        if (unread > 0) {
            badge.textContent = unread > 99 ? "99+" : String(unread);
            badge.classList.remove("hidden");
        } else {
            badge.classList.add("hidden");
        }
    }

    function _notifRender() {
        var dropdown = document.getElementById("notification-dropdown");
        if (!dropdown) return;
        var list = dropdown.querySelector(".notification-list");
        if (!list) return;
        if (_notifications.length === 0) {
            list.innerHTML = '<div class="notification-empty">No notifications</div>';
            return;
        }
        var html = "";
        var start = Math.max(0, _notifications.length - 50);
        for (var i = _notifications.length - 1; i >= start; i--) {
            var n = _notifications[i];
            html += '<div class="notification-item' + (n.read ? "" : " unread") + '">' +
                '<span class="notification-icon">' + _notifIcon(n.type) + '</span>' +
                '<div class="notification-body">' +
                '<span class="notification-message">' + _escapeHtml(n.message) + '</span>' +
                '<span class="notification-time">' + (n.time || "") + '</span>' +
                '</div></div>';
        }
        list.innerHTML = html;
    }

    function _escapeHtml(s) {
        if (typeof s !== "string") return "";
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function addNotification(type, message) {
        type = type || "info";
        _notifications.push({
            type: type,
            message: message,
            time: new Date().toLocaleTimeString(),
            read: false,
        });
        if (_notifications.length > 100) {
            _notifications.splice(0, _notifications.length - 100);
        }
        _notifSave();
        _notifUpdateBadge();
    }

    function markAllNotificationsRead() {
        for (var i = 0; i < _notifications.length; i++) {
            _notifications[i].read = true;
        }
        _notifSave();
        _notifUpdateBadge();
        _notifRender();
    }

    function getUnreadCount() {
        var count = 0;
        for (var i = 0; i < _notifications.length; i++) {
            if (!_notifications[i].read) count++;
        }
        return count;
    }

    function initNotifications() {
        _notifLoad();
        _notifUpdateBadge();
        _notifRender();

        var bell = document.getElementById("notification-bell");
        var dropdown = document.getElementById("notification-dropdown");
        if (bell && dropdown) {
            bell.addEventListener("click", function (e) {
                e.stopPropagation();
                var isOpen = !dropdown.classList.contains("hidden");
                dropdown.classList.toggle("hidden");
                if (!isOpen) {
                    markAllNotificationsRead();
                }
            });
            document.addEventListener("click", function () {
                if (dropdown) dropdown.classList.add("hidden");
            });
            dropdown.addEventListener("click", function (e) {
                e.stopPropagation();
            });
        }

        var clearBtn = document.getElementById("notification-clear-all");
        if (clearBtn) {
            clearBtn.addEventListener("click", function () {
                _notifications = [];
                _notifSave();
                _notifUpdateBadge();
                _notifRender();
            });
        }
    }

    // Patch showToast to also create notifications
    var _origShowToast = showToast;
    showToast = function (message, type) {
        addNotification(type || "info", message);
        _origShowToast(message, type);
    };

    // ============================================================
    // Public API
    // ============================================================
    window.showToast = showToast;
    window.showSkeleton = showSkeleton;
    window.hideSkeleton = hideSkeleton;
    window.toggleCollapsible = toggleCollapsible;
    window.updateWorkflowStep = updateWorkflowStep;
    window.openModal = openModal;
    window.closeModal = closeModal;
    window.loadMissingKeys = loadMissingKeys;

    window.KawkabUI = window.KawkabUI || {};
    window.KawkabUI.addNotification = addNotification;
    window.KawkabUI.markAllNotificationsRead = markAllNotificationsRead;
    window.KawkabUI.getUnreadCount = getUnreadCount;
    window.KawkabUI.initNotifications = initNotifications;
})();
