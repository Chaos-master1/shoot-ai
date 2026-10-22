describe('app-matchweek.js — KawkabMatchWeek', function () {
  var state;
  var bridge;

  function panelHtml() {
    return '' +
      '<section id="training-section" class="section">' +
      '<div id="matchweek-panel">' +
      '<span id="mw-stage"></span>' +
      '<button id="mw-refresh"></button>' +
      '<p id="mw-note"></p>' +
      '<div id="mw-diagnoses"></div>' +
      '<div id="mw-plan"></div>' +
      '<button id="mw-generate" class="hidden"></button>' +
      '<div id="mw-recheck"></div>' +
      '<div id="mw-provenance"></div>' +
      '</div>' +
      '<div id="training-plan-preview" class="hidden"><div id="training-plan-weeks"></div></div>' +
      '</section>';
  }

  function setState(payload) {
    state = payload;
  }

  beforeEach(function () {
    jest.resetModules();
    document.body.innerHTML = panelHtml();
    window.__kawkab = { currentMatchId: 1 };
    window.t = function (key) { return key; };
    window.showToast = jest.fn();
    state = null;
    bridge = {
      get_match_week_state: function (matchId, cb) {
        cb(JSON.stringify(state));
      },
      generate_training_plan: jest.fn(function (matchId, cb) {
        cb(JSON.stringify({ success: true, plan_id: 9, plan: {} }));
      })
    };
    window.KawkabUtils = {
      getBridge: function () { return bridge; }
    };
    require('../app-matchweek.js');
    window.KawkabMatchWeek.init();
  });

  function text(id) {
    return document.getElementById(id).textContent;
  }

  test('renders no_analysis with a select hint', async function () {
    setState({
      success: true, match_id: 1, stage: 'no_analysis',
      note: 'this match has no analysis events yet', diagnoses: [],
      plan: null, recheck: null, provenance: { calibrated: null }
    });
    await window.KawkabMatchWeek.refresh();
    expect(text('mw-stage')).toBe('no analysis');
    expect(text('mw-note')).toContain('no analysis events');
  });

  test('renders no_plan and offers the generate action', async function () {
    setState({
      success: true, match_id: 1, stage: 'no_plan', note: 'no plan yet',
      diagnoses: [], plan: null, recheck: null, provenance: {}
    });
    await window.KawkabMatchWeek.refresh();
    expect(document.getElementById('mw-generate').classList.contains('hidden')).toBe(false);
    expect(text('mw-plan')).toContain('No plan yet');
  });

  test('renders diagnoses from the persisted plan', async function () {
    setState({
      success: true, match_id: 1, stage: 'no_plan', note: '',
      diagnoses: [{ rule_id: 'r1', rule_name: 'Unforced Errors', confidence: 0.8 }],
      plan: null, recheck: null, provenance: {}
    });
    await window.KawkabMatchWeek.refresh();
    expect(text('mw-diagnoses')).toContain('Unforced Errors');
    expect(text('mw-diagnoses')).toContain('80%');
  });

  test('renders awaiting_matches without inventing a delta', async function () {
    setState({
      success: true, match_id: 1, stage: 'awaiting_matches', note: '0/2 post-plan matches recorded',
      diagnoses: [],
      plan: { id: 1, title: 'Plan', duration_weeks: 4, total_drills: 3, status: 'draft', weeks: [], priority_addressed: [] },
      recheck: {
        after_window_complete: false,
        rules: [{ rule_id: 'r1', rule_name: 'Errors', metric: 'turnovers', state: 'insufficient_sample', note: 'not enough matches', before: [], after: [] }]
      },
      provenance: { calibrated: false, after_window_complete: false, based_on_match_ids: [3], after_window: [] }
    });
    await window.KawkabMatchWeek.refresh();
    expect(text('mw-recheck')).toContain('Waiting for the next two matches');
    expect(text('mw-recheck')).toContain('insufficient sample');
    expect(text('mw-provenance')).toContain('Not calibrated');
  });

  test('renders a measured delta and its direction', async function () {
    setState({
      success: true, match_id: 1, stage: 'measured', note: '1 of 1 diagnosed metric(s) measured',
      diagnoses: [],
      plan: { id: 1, title: 'Plan', weeks: [], priority_addressed: [] },
      recheck: {
        after_window_complete: true,
        rules: [{
          rule_id: 'r1', rule_name: 'Errors', metric: 'turnovers',
          state: 'measured', mean_before: 6.5, mean_after: 2.5, delta: -4, improved: true,
          note: 'a directional signal to review, not a causal verdict'
        }]
      },
      provenance: { calibrated: true, after_window_complete: true, based_on_match_ids: [3], after_window: [4, 5] }
    });
    await window.KawkabMatchWeek.refresh();
    var recheck = text('mw-recheck');
    expect(recheck).toContain('6.5');
    expect(recheck).toContain('2.5');
    expect(recheck).toContain('-4');
    expect(recheck).toContain('improved');
    expect(recheck).toContain('not a causal verdict');
    expect(text('mw-provenance')).toContain('Calibrated');
  });

  test('surfaces an error instead of a fake stage', async function () {
    setState({ error: 'storage failure while checking this match' });
    await window.KawkabMatchWeek.refresh();
    expect(text('mw-stage')).toBe('error');
    expect(text('mw-note')).toContain('storage failure');
  });

  test('generating a plan calls the slot then refreshes', async function () {
    setState({
      success: true, match_id: 1, stage: 'no_plan', note: '',
      diagnoses: [], plan: null, recheck: null, provenance: {}
    });
    await window.KawkabMatchWeek.refresh();
    document.getElementById('mw-generate').click();
    expect(bridge.generate_training_plan).toHaveBeenCalledWith('1', expect.any(Function));
    expect(window.showToast).toHaveBeenCalled();
  });

  test('renders the persisted plan weeks', async function () {
    setState({
      success: true, match_id: 1, stage: 'awaiting_matches', note: '',
      diagnoses: [],
      plan: {
        id: 1, title: 'Plan', duration_weeks: 4, total_drills: 2, status: 'draft',
        priority_addressed: ['errors'],
        weeks: [{
          week_number: 1, theme: 'Foundation', primary_focus: 'build-up',
          expected_improvements: ['fewer turnovers'], re_test_focus: 'rondo',
          sessions: [{ day: 'MD-3', focus: 'rondo', drills: ['rondo_3v1'], total_duration_min: 20, intensity: 'medium' }]
        }]
      },
      recheck: { after_window_complete: false, rules: [] },
      provenance: {}
    });
    await window.KawkabMatchWeek.refresh();
    expect(text('training-plan-weeks')).toContain('Foundation');
    expect(text('training-plan-weeks')).toContain('rondo_3v1');
    expect(document.getElementById('training-plan-preview').classList.contains('hidden')).toBe(false);
  });
});
