import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from fund_monitor import build_chart_time_axis, compute_metrics, svg_fr_chart
from signal_rules import signal_status, replay, weekly_bars
from three_index_report import historical_signal_events, kline_levels_chart, period_chart_html

NS = {'s': 'http://www.w3.org/2000/svg'}


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class PeriodPolishTests(unittest.TestCase):
    def test_initial_down_state_is_not_a_sell(self):
        close = np.linspace(200, 100, 500)
        daily = pd.DataFrame({'date': pd.bdate_range('2023-01-02', periods=500), 'close': close})
        sig = signal_status(daily)
        initial = replay(weekly_bars(daily)).events[0]
        self.assertEqual(initial['factor'], 0)
        self.assertEqual(historical_signal_events(daily, sig), [])

    def test_dense_labels_avoid_all_candles_triangles_labels_and_level_text(self):
        n = 156
        close = np.full(n, 100.0)
        low, high = close - 1, close + 1
        low[::3] = 96
        low[-1], high[-1] = 70, 130  # Leave space for labels, as in real charts.
        frame = compute_metrics(pd.DataFrame(dict(date=pd.date_range('2022-01-07', periods=n, freq='W-FRI'),
                                                  open=close - .3, close=close, high=high, low=low)))
        events = [dict(date=frame.date.iloc[i], direction='buy' if i % 2 else 'sell', factor=.5 if i % 2 else 0,
                       label='抄底' if i % 2 else '止损', reason='完整原因') for i in range(40, 80)]
        axis = build_chart_time_axis(frame.date)
        svg = ET.fromstring(kline_levels_chart(frame, 'K', [(105, '近一年最高 105', '#888888')], axis=axis, events=events))
        wicks = [e for e in svg.findall('s:line', NS) if e.get('stroke-width') == '1.1']
        bodies = [e for e in svg.findall('s:rect', NS) if e.get('opacity') == '0.9']
        candle_boxes = [(float(b.get('x')), float(w.get('y1')), float(b.get('x')) + float(b.get('width')), float(w.get('y2')))
                        for w, b in zip(wicks, bodies)]
        groups = svg.findall('s:g[@class="signal-marker"]', NS)
        triangles = []
        for g in groups:
            pts = [tuple(map(float, p.split(','))) for p in g.find('s:polygon', NS).get('points').split()]
            xs, ys = zip(*pts)
            triangles.append((min(xs), min(ys), max(xs), max(ys)))
            i = frame.index[frame.date == g.get('data-date')][0]
            gap = min(ys) - candle_boxes[i][3] if g.get('data-direction') == 'buy' else candle_boxes[i][1] - max(ys)
            self.assertGreaterEqual(gap, 6)
        boxes = []
        labels = svg.findall('.//s:text[@class="signal-label"]', NS)
        self.assertGreater(len(labels), 3)
        self.assertLess(len(labels), len(events))
        self.assertTrue(svg.findall('.//s:line[@class="signal-leader"]', NS))
        for text in labels:
            box = tuple(map(float, text.get('data-box').split(',')))
            self.assertEqual(text.get('text-anchor'), 'middle')
            self.assertIn(int(text.get('data-offset')), (0, 12, 24, 36))
            self.assertFalse(any(overlap(box, other) for other in candle_boxes + triangles + boxes))
            self.assertFalse(overlap(box, (948, 0, 1060, 380)))  # Right-hand level labels.
            boxes.append(box)

    def test_html_chrome_and_sticky_axis_are_separate_from_scrolling_plot(self):
        n = 50
        close = np.arange(n) + 100.
        frame = compute_metrics(pd.DataFrame(dict(date=pd.bdate_range('2025-01-01', periods=n),
                                                  open=close, close=close, high=close + 2, low=close - 2)))
        axis = build_chart_time_axis(frame.date)
        for svg, height, title, legend in [
            (kline_levels_chart(frame, 'K标题', [], axis=axis), 380, 'K标题', [('MA20', '#2563eb', 'line')]),
            (svg_fr_chart(frame, 'Fr标题', axis=axis, height=330), 330, 'Fr标题', [('Fr', '#2563eb', 'line')]),
        ]:
            html = period_chart_html(svg, title, legend, axis, height)
            root = ET.fromstring('<root>' + html + '</root>')
            self.assertEqual(root.find('header/h3').text, title)
            charts = root.findall('.//s:svg', NS)
            yaxis, plot = charts
            self.assertEqual(yaxis.get('class'), 'chart-axis')
            self.assertEqual(plot.get('class'), 'chart-plot')
            self.assertEqual(len(yaxis.findall('s:text', NS)), 5)
            self.assertEqual(yaxis.find('s:rect', NS).get('fill'), '#fff')
            self.assertFalse(plot.findall('.//s:text[@class="axis-label"]', NS))
            self.assertFalse(plot.findall('.//s:text[@class="chart-title"]', NS))
            self.assertFalse(plot.findall('.//s:g[@class="chart-legend"]', NS))
            self.assertEqual(plot.get('viewBox'), f'58 16 1002 {height - 16}')


if __name__ == '__main__':
    unittest.main()
