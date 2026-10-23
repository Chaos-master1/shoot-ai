// The elite benchmark card is the only place the comparison reaches the
// coach, so it is exercised through the real render path: a stubbed bridge
// returning the pro-analytics report, a click on Load, and assertions on
// the rendered DOM.

describe('app-proanalytics.js — elite benchmark card', function () {
  var report;
  var seasonReport;

  function shell() {
    document.body.innerHTML =
      '<select id="proanalytics-match-select"><option value="1">Match 1</option></select>' +
      '<button id="proanalytics-load-btn">Load</button>' +
      '<button id="proanalytics-season-btn">Season</button>' +
      '<span id="proanalytics-status"></span>' +
      '<div id="proanalytics-content"></div>';
  }

  function benchmarkBlock(overrides) {
    var block = {
      data_available: true,
      match_name: 'Real Madrid vs Liverpool',
      population: {
        label: 'StatsBomb elite cohort',
        n_matches: 297,
        n_team_matches: 594,
        n_teams: 144,
        definitions: {
          shots: { definition: 'Shots recorded for the team in the match.' },
          ppda: {
            definition: 'Opponent passes allowed in their own 60% divided by our tackles, interceptions and fouls there.'
          },
          pass_completion_pct: { definition: 'Completed passes divided by attempted passes.' }
        },
        model_card: { available: true, brier: 0.0784, n_shots: 7518 },
        competitions: [{ name: 'Champions League', n_matches: 20, seasons: ['2017/2018'] }]
      },
      teams: [
        {
          side: 'home',
          team: 'Real Madrid',
          shots: 14,
          xg: 1.511,
          goals: 3,
          minutes: 90,
          metrics: {
            shots: metric(14, 65.2, 'above_median', 11.8, 0),
            shots_per_90: metric(14, 65.2, 'above_median', 11.8, 0),
            xg: metric(1.51, 67.0, 'above_median', 1.24, 2),
            xg_per_90: metric(1.51, 67.0, 'above_median', 1.24, 2),
            goals: metric(3, 88.9, 'top_quartile', 1.0, 0),
            ppda: metric(7.3, 55.0, 'above_median', 7.33, 2),
            pass_completion_pct: metric(84.2, 72.0, 'top_quartile', 77.6, 1)
          }
        },
        {
          side: 'away',
          team: 'Liverpool',
          shots: 14,
          xg: null,
          goals: null,
          minutes: 90,
          metrics: {
            shots: metric(14, 65.2, 'above_median', 11.8, 0),
            xg: {
              value: null,
              percentile: null,
              band: null,
              population: { n: 594 },
              comparable: false,
              reason: '2 of 14 shots lack the position features the model needs'
            },
            goals: {
              value: null,
              percentile: null,
              band: null,
              population: { n: 594 },
              comparable: false,
              reason: 'this match records no goals'
            }
          }
        }
      ],
      limitations: [
        { key: 'single_match', text: 'One match is 90 minutes of football.' }
      ]
    };
    Object.keys(overrides || {}).forEach(function (k) { block[k] = overrides[k]; });
    return block;
  }

  function metric(value, percentile, band, median, digits) {
    return {
      value: value,
      percentile: percentile,
      interval: [Math.max(0, percentile - 3), Math.min(100, percentile + 3)],
      band: band,
      population: { n: 594, median: median },
      comparable: true,
      reason: null,
      digits: digits
    };
  }

  function load() {
    document.getElementById('proanalytics-match-select').value = '1';
    document.getElementById('proanalytics-load-btn').click();
  }

  function content() {
    return document.getElementById('proanalytics-content').textContent;
  }

  beforeEach(function () {
    jest.resetModules();
    shell();
    window.__kawkab = {
      escapeHtml: function (s) {
        return String(s)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
      }
    };
    window.t = function (key) { return key; };
    window.showToast = jest.fn();
    report = {
      success: true,
      data_available: true,
      match_id: 1,
      n_events: 1200,
      blocks: { elite_benchmark: benchmarkBlock() }
    };
    global.bridge = {
      get_all_matches: function (cb) {
        cb(JSON.stringify({ matches: [{ id: 1, name: 'Real Madrid vs Liverpool' }] }));
      },
      get_pro_analytics_report: function (id, cb) { cb(JSON.stringify(report)); },
      get_season_pro_report: function (cb) { cb(JSON.stringify(seasonReport)); }
    };
    seasonReport = {
      success: true,
      data_available: true,
      n_matches: 4,
      blocks: {
        benchmark_trend: {
          data_available: true,
          team: 'Test Utd',
          n_matches: 4,
          n_matches_stored: 6,
          minutes: 360,
          metrics: {
            shots_per_90: metric(10, 62.0, 'above_median', 9.8, 1),
            xg_per_90: metric(1.2, 58.0, 'above_median', 1.15, 2),
            ppda: metric(7.4, 55.0, 'above_median', 7.33, 2)
          },
          trend: {
            available: true,
            metric: 'xg_per_90',
            split: 'first 2 vs last 2 matches',
            before: 1.0,
            after: 1.4,
            delta: 0.4,
            note: 'a direction to review, not a causal verdict'
          },
          block: { size: 4, draws: 2000, seed: 0, method: 'same-size blocks' },
          population: {
            label: 'StatsBomb elite cohort',
            n_matches: 297,
            n_team_matches: 594,
            definitions: {
              ppda: { definition: 'Opponent passes allowed in their own 60% divided by our tackles.' }
            }
          },
          limitations: [
            {
              key: 'matched_block_size',
              text: 'The percentile compares your N-match average against N-match blocks drawn from the cohort.'
            }
          ]
        }
      }
    };
    require('../app-proanalytics.js');
    window.initProAnalyticsWorkspace();
  });

  afterEach(function () {
    delete global.bridge;
  });

  test('states the population the numbers were measured against', function () {
    load();
    var text = content();
    expect(text).toContain('Elite Benchmark');
    expect(text).toContain('StatsBomb elite cohort');
    expect(text).toContain('594');
    expect(text).toContain('297');
    expect(text).toContain('144');
    expect(text).toContain('Champions League');
  });

  test('renders the percentile in words, not as a bare number', function () {
    load();
    var text = content();
    expect(text).toContain('14');
    expect(text).toContain('better than 65% of the cohort');
    expect(text).toContain('better than 67% of the cohort');
    expect(text).toContain('±3');
    expect(text).toContain('cohort median');
    expect(text).toContain('above the cohort median');
    expect(text).toContain('top 25% of the cohort');
  });

  test('splits metrics into output and style groups', function () {
    load();
    var text = content();
    expect(text).toContain('Output');
    expect(text).toContain('Style');
    expect(text).toContain('PPDA');
    expect(text).toContain('Pass completion');
    expect(text).toContain('better than 55% of the cohort');
  });

  test('every metric carries the definition it was measured with', function () {
    load();
    var ppda = document.querySelector(
      '[title="Opponent passes allowed in their own 60% divided by our tackles, interceptions and fouls there."]'
    );
    expect(ppda).not.toBeNull();
    expect(ppda.textContent).toBe('PPDA');
  });

  test('an unmeasurable metric shows its reason instead of a zero', function () {
    load();
    var text = content();
    expect(text).toContain('lack the position features the model needs');
    expect(text).toContain('this match records no goals');
    // the shots metric is measured, so its number is still there
    expect(text).toContain('better than 65% of the cohort');
  });

  test('publishes the model\u2019s own error beside the numbers', function () {
    load();
    var text = content();
    expect(text).toContain('Model check: Brier 0.078 on 7,518 shots (lower is better)');
  });

  test('carries the limitations with the answer', function () {
    load();
    expect(content()).toContain('One match is 90 minutes of football.');
    expect(content()).toContain('cannot tell you');
  });

  test('escapes a team name instead of injecting it', function () {
    report.blocks.elite_benchmark.teams[0].team = '<img src=x onerror=alert(1)>';
    load();
    expect(document.getElementById('proanalytics-content').querySelector('img')).toBeNull();
    expect(content()).toContain('<img src=x onerror=alert(1)>');
  });

  test('the season card states the matched block size', function () {
    document.getElementById('proanalytics-season-btn').click();
    var text = content();
    expect(text).toContain('Season Benchmark');
    expect(text).toContain('Test Utd');
    expect(text).toContain('4 matches');
    expect(text).toContain('360 minutes');
    expect(text).toContain('percentiles compare your 4-match block against 4-match blocks drawn from the cohort');
    expect(text).toContain('better than 62% of the cohort');
    expect(text).toContain('Trend across the block');
    expect(text).toContain('1.00 → 1.40 (+0.40)');
    expect(text).toContain('not a causal verdict');
  });

  test('the season card refuses an undertrained block', function () {
    seasonReport.blocks.benchmark_trend = {
      data_available: false,
      reason: 'only 2 stored match(es) for Test Utd — a block needs at least 3'
    };
    document.getElementById('proanalytics-season-btn').click();
    var text = content();
    expect(text).toContain('no data');
    expect(text).toContain('only 2 stored match(es)');
    expect(text).not.toContain('better than');
  });

  test('an unavailable block reports why, without inventing numbers', function () {
    report.blocks.elite_benchmark = {
      data_available: false,
      reason: 'no shots stored for this match — nothing to compare',
      population: null,
      limitations: null
    };
    load();
    var text = content();
    expect(text).toContain('no data');
    expect(text).toContain('no shots stored for this match');
    expect(text).not.toContain('better than');
  });
});
