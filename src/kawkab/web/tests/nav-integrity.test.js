/**
 * Navigation integrity contract: every sidebar data-section / bottom-nav
 * data-section must point at an element that actually exists in index.html.
 *
 * History: 23 sidebar links (Timeline, xG, xT, VAEP, Charts, Heatmap, Pass
 * Network, Momentum, Transitions, Finishing, Set Pieces, Phases, Report,
 * Game Plan, Compare, Shortlist, Contracts, Recruitment, Settings, Search,
 * Import, Data, Sandbox) pointed at section ids that never existed. The
 * sidebar click handler (index.html syncSidebarClicks) toggles hidden on
 * every .section that doesn't match — so a click on one of these links
 * hid EVERY section and left a blank app. The bottom-nav Settings button
 * had the same bug.
 */
const fs = require('fs');
const path = require('path');

const indexHtml = fs.readFileSync(
  path.join(__dirname, '..', 'index.html'),
  'utf8'
);

function sectionIds() {
  const out = [];
  const re = /<section[^>]*\bid="([a-z-]+-section)"/g;
  let m;
  while ((m = re.exec(indexHtml)) !== null) out.push(m[1]);
  return new Set(out);
}

function dataSectionTargets(selectorPrefix) {
  const out = [];
  const re = new RegExp('<(?:a|button)[^>]*class="' + selectorPrefix + '[^"]*"[^>]*data-section="([a-z-]+)"', 'g');
  let m;
  while ((m = re.exec(indexHtml)) !== null) out.push(m[1]);
  return out;
}

describe('nav integrity', () => {
  const sections = sectionIds();

  test('index.html has a healthy set of real sections', () => {
    expect(sections.size).toBeGreaterThanOrEqual(20);
  });

  test('every sidebar nav-item data-section target exists as a <section>', () => {
    const targets = dataSectionTargets('nav-item');
    expect(targets.length).toBeGreaterThan(0);
    const dead = targets.filter((t) => !sections.has(t));
    expect(dead).toEqual([]);
  });

  test('every bottom-nav data-section target exists as a <section>', () => {
    const targets = dataSectionTargets('bottom-nav-item');
    expect(targets.length).toBeGreaterThan(0);
    const dead = targets.filter((t) => !sections.has(t));
    expect(dead).toEqual([]);
  });

  test('the training section is reachable from the sidebar', () => {
    // It had a router registration but no nav target, so no click could
    // reach the Training Planner. One-directional link checks did not
    // catch it; this pins the door.
    expect(dataSectionTargets('nav-item')).toContain('training-section');
  });

  test('no nav link still points at a removed/never-built section id', () => {
    // These ids were never real <section> elements. If a future change
    // re-adds them as real sections, update this list deliberately.
    // (settings-section was re-added as a real section in the elite push;
    // shortlist/contracts/recruitment live inside scout-section + settings,
    // not as standalone sections.)
    const removedIds = [
      'timeline-section', 'xg-section', 'xt-section', 'vaep-section',
      'charts-section', 'heatmap-section', 'pass-network-section',
      'momentum-section', 'transitions-section', 'finishing-section',
      'set-piece-section', 'phases-section', 'tactics-report-section',
      'game-plan-section', 'player-compare-section', 'shortlist-section',
      'contracts-section', 'recruitment-section',
      'search-section', 'import-section', 'data-export-section',
      'tactical-sandbox-section',
    ];
    for (const id of removedIds) {
      expect(indexHtml).not.toContain('data-section="' + id + '"');
    }
  });
});
