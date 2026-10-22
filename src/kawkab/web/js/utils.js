window.KawkabUtils = window.KawkabUtils || {};

(function(KU) {
  KU.formatDate = function(dateStr) {
    if (!dateStr) return '';
    try {
      const d = new Date(dateStr);
      if (isNaN(d.getTime())) return dateStr;
      return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch(e) {
      return dateStr;
    }
  };

  KU.escapeHtml = function(str) {
    if (!str) return '';
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  };
  window.__kawkab = window.__kawkab || {};
  window.__kawkab.escapeHtml = KU.escapeHtml;

  // Resolve the QWebChannel bridge at call time. app.js assigns both
  // window.__kawkab.bridge and window.bridge once the channel connects;
  // window.kawkabBridge is the legacy alias some modules still probe.
  // Single source of truth so the split modules stop re-implementing it.
  KU.getBridge = function() {
    return (window.__kawkab && window.__kawkab.bridge) || window.bridge || window.kawkabBridge || null;
  };
  window.__kawkab.getBridge = KU.getBridge;

  KU.showToast = function(message, type, duration) {
    type = type || 'info';
    duration = duration || 3000;
    var toast = document.getElementById('toast-container');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'toast-container';
      toast.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px';
      document.body.appendChild(toast);
    }
    var el = document.createElement('div');
    el.className = 'toast toast-' + type;
    el.textContent = message;
    toast.appendChild(el);
    setTimeout(function() { el.remove(); }, duration);
  };

  KU.eventTypeIcon = function(type) {
    var icons = {
      GOAL: '\u26BD', SHOT: '\uD83C\uDFAF', PASS: '\u27A1\uFE0F', TACKLE: '\uD83D\uDED1', INTERCEPTION: '\u270B',
      DRIBBLE: '\uD83C\uDFC3', CORNER: '\uD83D\uDEA9', FREE_KICK: '\u26A1', THROW_IN: '\uD83D\uDCE5', CLEARANCE: '\uD83D\uDD04',
      CROSS: '\u2708\uFE0F', BLOCK: '\uD83D\uDEE1\uFE0F', CARRY: '\uD83C\uDFCB\uFE0F', DUEL: '\u2694\uFE0F', FOUL: '\uD83C\uDFE8',
      OFFSIDE: '\uD83D\uDEA9', HAND_BALL: '\u270B', YELLOW_CARD: '\uD83C\uDFE8', RED_CARD: '\uD83C\uDFE5',
      SUBSTITUTION: '\uD83D\uDD04', PENALTY: '\u26AA', BALL_OUT: '\uD83D\uDCE4', OUT_OF_PLAY: '\u23F8\uFE0F',
      SAVE: '\uD83E\uDDE4', GOAL_KICK: '\uD83E\uDD45'
    };
    return icons[type] || '\u2022';
  };

  KU.formatTime = function(seconds) {
    if (seconds == null || isNaN(seconds)) return '00:00';
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
  };

  KU.deepClone = function(obj) {
    return JSON.parse(JSON.stringify(obj));
  };

  KU.truncate = function(str, maxLen) {
    maxLen = maxLen || 30;
    if (!str || str.length <= maxLen) return str || '';
    return str.substring(0, maxLen - 3) + '...';
  };

  KU.now = function() {
    return new Date().toISOString();
  };
})(window.KawkabUtils);
