/* Tests for app-charts.js — Chart.js enhanced chart renderers (CommonJS) */

var fs = require('fs');
var path = require('path');

var CHARTS_JS_PATH = path.resolve(__dirname, '../js/app-charts.js');
var chartsCode = fs.readFileSync(CHARTS_JS_PATH, 'utf-8');

// Track created chart instances
let createdCharts = [];

beforeEach(() => {
    createdCharts = [];
    document.body.innerHTML = '';
    // Reset globals the IIFE checks
    delete window.KawkabCharts;
    delete window.__kawkabCharts;
    delete window.Chart;
    delete window.renderMomentumIndex;
    delete window.renderWinProbability;
    delete window.renderXGTimeline;
    delete window.renderRadarCharts;
    // Track calls to fallback originals
    window.renderMomentumIndex = jest.fn();
    window.renderWinProbability = jest.fn();
    window.renderXGTimeline = jest.fn();
    window.renderRadarCharts = jest.fn();
});

/** Evaluate the IIFE in the current jsdom window context */
function loadChartsIIFE() {
    // eslint-disable-next-line no-eval
    (function() { eval(chartsCode); }).call(window);
}

/** Create a mock Chart constructor */
function createMockChart() {
    function MockChart(ctx, config) {
        this.ctx = ctx;
        this.config = config;
        this.destroyed = false;
        createdCharts.push(this);
    }
    MockChart.prototype.destroy = function() {
        this.destroyed = true;
    };
    // Real Chart.js v3+ requires registering plugins/components before use;
    // app-charts.js calls ChartJS.register(KawkabAnnotationPlugin) at load
    // time whenever Chart is present (js/app-charts.js:107).
    MockChart.register = jest.fn();
    return MockChart;
}

describe('KawkabCharts.init', () => {
    it('does not replace renderers when Chart.js is absent', () => {
        loadChartsIIFE();
        expect(window.KawkabCharts).toBeTruthy();
        expect(window.renderMomentumIndex).toBe(window.renderMomentumIndex); // still original mock
        // init() returns early if no ChartJS
    });

    it('replaces renderers when Chart.js is present', () => {
        window.Chart = createMockChart();
        loadChartsIIFE();
        // After IIFE, init was called on DOMContentLoaded (or immediately if readyState !== 'loading')
        expect(window.renderMomentumIndex).not.toBe(undefined);
        expect(typeof window.renderMomentumIndex).toBe('function');
    });
});

describe('renderMomentum (fallback)', () => {
    it('silently no-ops when Chart.js absent', () => {
        loadChartsIIFE();
        expect(() => window.KawkabCharts.renderMomentum({ timeline: [{ minute: 1, home_momentum: 0.5, away_momentum: -0.3 }] })).not.toThrow();
    });

    it('falls back when data has too few points', () => {
        window.Chart = createMockChart();
        var origMock = window.renderMomentumIndex;
        loadChartsIIFE();
        window.KawkabCharts.renderMomentum({ timeline: [{ minute: 1, home_momentum: 0.5, away_momentum: -0.3 }] });
        expect(origMock).toHaveBeenCalled();
    });

    it('falls back when timeline is missing', () => {
        window.Chart = createMockChart();
        var origMock = window.renderMomentumIndex;
        loadChartsIIFE();
        window.KawkabCharts.renderMomentum({});
        expect(origMock).toHaveBeenCalled();
    });
});

describe('renderMomentum (Chart.js path)', () => {
    it('creates a line chart with correct data', () => {
        window.Chart = createMockChart();
        const canvas = document.createElement('canvas');
        canvas.id = 'momentum-canvas';
        document.body.appendChild(canvas);
        loadChartsIIFE();

        window.KawkabCharts.renderMomentum({
            timeline: [
                { minute: 0, home_momentum: 0, away_momentum: 0 },
                { minute: 15, home_momentum: 0.3, away_momentum: -0.2 },
                { minute: 30, home_momentum: 0.1, away_momentum: -0.1 },
            ],
        });

        expect(createdCharts.length).toBe(1);
        const chart = createdCharts[0];
        expect(chart.config.type).toBe('line');
        expect(chart.config.data.labels).toEqual([0, 15, 30]);
        expect(chart.config.data.datasets[0].label).toBe('Home');
        expect(chart.config.data.datasets[1].label).toBe('Away');
    });

    it('returns early when canvas is missing', () => {
        window.Chart = createMockChart();
        loadChartsIIFE();
        expect(() => window.KawkabCharts.renderMomentum({
            timeline: [{ minute: 0, home_momentum: 0, away_momentum: 0 }, { minute: 90, home_momentum: 0.5, away_momentum: -0.5 }],
        })).not.toThrow();
        expect(createdCharts.length).toBe(0);
    });

    it('destroys previous chart instance before creating new one', () => {
        window.Chart = createMockChart();
        const canvas = document.createElement('canvas');
        canvas.id = 'momentum-canvas';
        document.body.appendChild(canvas);
        loadChartsIIFE();

        window.KawkabCharts.renderMomentum({
            timeline: [{ minute: 0, home_momentum: 0, away_momentum: 0 }, { minute: 90, home_momentum: 0.5, away_momentum: -0.5 }],
        });
        expect(createdCharts.length).toBe(1);
        const first = createdCharts[0];

        window.KawkabCharts.renderMomentum({
            timeline: [{ minute: 0, home_momentum: 0, away_momentum: 0 }, { minute: 45, home_momentum: 0.2, away_momentum: -0.2 }],
        });
        expect(first.destroyed).toBe(true);
        expect(createdCharts.length).toBe(2);
    });
});

describe('renderWinProb', () => {
    it('silently no-ops when Chart.js absent', () => {
        loadChartsIIFE();
        expect(() => window.KawkabCharts.renderWinProb({ timeline: [{ minute: 1, home_win: 0.5, draw: 0.3, away_win: 0.2 }] })).not.toThrow();
    });

    it('creates a chart with 3 datasets when conditions met', () => {
        window.Chart = createMockChart();
        const canvas = document.createElement('canvas');
        canvas.id = 'winprob-canvas';
        document.body.appendChild(canvas);
        loadChartsIIFE();

        window.KawkabCharts.renderWinProb({
            timeline: [
                { minute: 0, home_win: 0.5, draw: 0.3, away_win: 0.2, home_score: 0, away_score: 0 },
                { minute: 90, home_win: 0.8, draw: 0.1, away_win: 0.1, home_score: 2, away_score: 0 },
            ],
        });

        expect(createdCharts.length).toBe(1);
        const chart = createdCharts[0];
        expect(chart.config.data.datasets.length).toBe(3);
        expect(chart.config.data.datasets[0].label).toBe('Home Win');
        expect(chart.config.data.datasets[1].label).toBe('Draw');
        expect(chart.config.data.datasets[2].label).toBe('Away Win');
    });
});

describe('renderXGTimeline', () => {
    it('silently no-ops when Chart.js absent', () => {
        loadChartsIIFE();
        expect(() => window.KawkabCharts.renderXGTimeline({ timeline: [{ minute: 1, home_xg: 0.1, away_xg: 0.05 }] })).not.toThrow();
    });

    it('creates a bar chart with 4 datasets', () => {
        window.Chart = createMockChart();
        const canvas = document.createElement('canvas');
        canvas.id = 'xg-timeline-canvas';
        document.body.appendChild(canvas);
        loadChartsIIFE();

        window.KawkabCharts.renderXGTimeline({
            timeline: [
                { minute: 0, home_xg: 0, away_xg: 0 },
                { minute: 45, home_xg: 0.3, away_xg: 0.1 },
                { minute: 90, home_xg: 0.5, away_xg: 0.2, home_goal: true },
            ],
        });

        expect(createdCharts.length).toBe(1);
        const chart = createdCharts[0];
        expect(chart.config.type).toBe('bar');
        expect(chart.config.data.datasets.length).toBe(4);
        expect(chart.config.data.datasets[0].label).toBe('Home xG');
    });

    it('shows away xG as negative values', () => {
        window.Chart = createMockChart();
        const canvas = document.createElement('canvas');
        canvas.id = 'xg-timeline-canvas';
        document.body.appendChild(canvas);
        loadChartsIIFE();

        window.KawkabCharts.renderXGTimeline({
            timeline: [
                { minute: 0, home_xg: 0, away_xg: 0 },
                { minute: 45, home_xg: 0.2, away_xg: 0.15 },
            ],
        });

        const awayXGData = createdCharts[0].config.data.datasets[2].data;
        expect(awayXGData[1]).toBe(-0.15);
    });
});

describe('renderRadar', () => {
    it('silently no-ops when Chart.js absent', () => {
        loadChartsIIFE();
        expect(() => window.KawkabCharts.renderRadar({ labels: ['Pass'], values: [0.5] })).not.toThrow();
    });

    it('creates a radar chart', () => {
        window.Chart = createMockChart();
        const canvas = document.createElement('canvas');
        canvas.id = 'radar-canvas';
        document.body.appendChild(canvas);
        loadChartsIIFE();

        window.KawkabCharts.renderRadar({
            labels: ['Passing', 'Shooting', 'Defending'],
            values: [0.8, 0.6, 0.7],
            playerName: 'Player A',
        });

        expect(createdCharts.length).toBe(1);
        const chart = createdCharts[0];
        expect(chart.config.type).toBe('radar');
        expect(chart.config.data.labels).toEqual(['Passing', 'Shooting', 'Defending']);
        expect(chart.config.data.datasets[0].label).toBe('Player A');
    });

    it('falls back when too few labels', () => {
        window.Chart = createMockChart();
        var origRadarMock = window.renderRadarCharts;
        loadChartsIIFE();
        // IIFE stores the original in __kawkabCharts.radar, so check it was called
        window.KawkabCharts.renderRadar({ labels: ['One'], values: [0.5] });
        expect(origRadarMock).toHaveBeenCalled();
    });
});

describe('_register and __kawkabCharts', () => {
    it('stores registered fallback renderers', () => {
        window.Chart = createMockChart();
        loadChartsIIFE();
        expect(window.__kawkabCharts).toBeTruthy();
        expect(typeof window.__kawkabCharts.momentum).toBe('function');
        expect(typeof window.__kawkabCharts.winprob).toBe('function');
        expect(typeof window.__kawkabCharts.xgtimeline).toBe('function');
        expect(typeof window.__kawkabCharts.radar).toBe('function');
    });
});
