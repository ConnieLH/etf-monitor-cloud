import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from fund_monitor import build_chart_time_axis, compute_metrics, svg_fr_chart
from signal_rules import replay, signal_status, weekly_bars
from three_index_report import (aggregate_ohlc, csi_kline_from_perf, fr_display_frame,
                                historical_signal_events, kline_levels_chart,
                                period_chart_frames, period_chart_pair, read_kline)

NS = {"s": "http://www.w3.org/2000/svg"}


def history():
    dates = pd.bdate_range('2015-01-05', periods=1800)
    close = 100 + 18 * np.sin(np.arange(len(dates)) / 45) + np.arange(len(dates)) / 300
    return pd.DataFrame(dict(date=dates, open=close - .3, high=close + 2,
                             low=close - 2, close=close))


class FollowupTests(unittest.TestCase):
    def test_markers_match_full_replay_and_exclude_provisional_week(self):
        daily = history().iloc[:-2]  # Wednesday: final weekly candle is provisional.
        sig = signal_status(daily)
        self.assertFalse(sig['week_complete'])
        weekly = weekly_bars(daily)
        confirmed = weekly.iloc[:-1].reset_index(drop=True)
        expected, previous = [], 1.0
        for event in replay(confirmed).events:
            if event['factor'] != previous:
                expected.append((event['date'], 'buy' if event['factor'] > previous else 'sell', event['factor']))
            previous = event['factor']
        actual = historical_signal_events(daily, sig)
        self.assertEqual([(e['date'], e['direction'], e['factor']) for e in actual], expected)
        self.assertGreater(len(actual), 6)  # Does not use signal_status's last-six truncation.
        frame = period_chart_frames(daily)['week'].tail(156).reset_index(drop=True)
        svg = ET.fromstring(kline_levels_chart(frame, 'K', [], axis=build_chart_time_axis(frame.date), events=actual))
        markers = svg.findall('s:g[@class="signal-marker"]', NS)
        shown = [(d.strftime('%Y-%m-%d'), direction, factor) for d, direction, factor in expected if d in set(frame.date)]
        self.assertEqual([(m.get('data-date'), m.get('data-direction'), float(m.get('data-factor'))) for m in markers], shown)
        xs = build_chart_time_axis(frame.date).xs
        for marker in markers:
            i = frame.index[frame.date == marker.get('data-date')][0]
            triangle = marker.find('s:polygon', NS)
            points = [tuple(map(float, p.split(','))) for p in triangle.get('points').split()]
            self.assertAlmostEqual(points[0][0], xs[i], delta=.051)
            buy = marker.get('data-direction') == 'buy'
            self.assertEqual(triangle.get('fill'), '#dc2626' if buy else '#16a34a')
            self.assertEqual(points[0][1] < points[1][1], buy)

    def test_holiday_completed_week_uses_signal_status_cutoff(self):
        daily = history().iloc[:-2]
        sig = signal_status(daily, pd.DatetimeIndex(daily.date))
        self.assertTrue(sig['week_complete'])
        self.assertEqual(sig['week_date'], daily.date.iloc[-1])
        events = historical_signal_events(daily, sig)
        self.assertEqual(events[-1]['date'], replay(weekly_bars(daily)).events[-1]['date'])

    def test_reference_reading_uses_weekly_conclusion_and_week_has_no_banner(self):
        daily = history()
        sig = signal_status(daily)
        frames = period_chart_frames(daily)
        html = period_chart_pair('x', '测试', daily, {}, sig)
        self.assertEqual(html.count('class="reference-view"'), 4)
        self.assertEqual(html.count('决策以周K为准：' + read_kline({}, sig, frames['week'], period='week')), 2)
        self.assertEqual(html.count('（仅供参考）'), 2)
        week = html.split('<div class="period-view period-week">')[1].split('<div class="period-view period-month">')[0]
        self.assertNotIn('class="reference-view"', week)
        self.assertIn('▲▼是按规则历史上的买卖点', week)
        for key in ('day', 'month'):
            part = html.split(f'<div class="period-view period-{key}">')[1].split('<div class="period-view ')[0]
            self.assertNotIn('class="signal-marker"', part)

    def test_warmup_masks_first_26_without_shifting_axis_or_mutating_metrics(self):
        for freq in ('B', 'W-FRI', 'MS'):
            full = compute_metrics(pd.DataFrame({'date': pd.date_range('2020-01-01', periods=50, freq=freq), 'close': np.arange(50) + 100.0}))
            original = full.copy()
            shown = fr_display_frame(full, 40)
            self.assertTrue(shown.iloc[:16][['fr', 'fr_bar']].isna().all().all())
            self.assertTrue(shown.iloc[16:][['fr', 'fr_bar']].notna().all().all())
            axis = build_chart_time_axis(shown.date)
            svg = ET.fromstring(svg_fr_chart(shown, 'Fr', axis=axis))
            points = svg.find('s:polyline', NS).get('points').split()
            self.assertEqual(len(points), 24)
            self.assertAlmostEqual(float(points[0].split(',')[0]), axis.xs[16], delta=.051)
            pd.testing.assert_frame_equal(full, original)

    def test_close_only_history_is_display_only_and_does_not_invent_ohlc(self):
        raw = pd.DataFrame({'tradeDate':['20170103','20170104','20170203'], 'open':[100,None,102],
                            'high':[103,None,105], 'low':[99,None,101], 'close':[101,102,103]})
        original = csi_kline_from_perf(raw, '930955')
        extended = csi_kline_from_perf(raw, '930955', include_close_only=True)
        self.assertEqual(len(original), 2)
        self.assertEqual(len(extended), 3)
        month = aggregate_ohlc(extended, 'month')
        self.assertTrue(pd.isna(month.iloc[0]['high']))
        self.assertTrue(pd.isna(month.iloc[0]['low']))
        self.assertEqual(month.iloc[0]['close'], 102)

    def test_nearby_marker_labels_are_omitted_but_triangles_remain(self):
        frame = period_chart_frames(history())['week'].tail(156).reset_index(drop=True)
        events = [dict(date=d, direction='buy', factor=.5, label='抄底半仓', reason='测试') for d in frame.date.iloc[70:80]]
        svg = ET.fromstring(kline_levels_chart(frame, 'K', [], axis=build_chart_time_axis(frame.date), events=events))
        groups = svg.findall('s:g[@class="signal-marker"]', NS)
        self.assertEqual(len(groups), 10)
        self.assertLess(sum(g.find('s:text', NS) is not None for g in groups), 10)


if __name__ == '__main__':
    unittest.main()
