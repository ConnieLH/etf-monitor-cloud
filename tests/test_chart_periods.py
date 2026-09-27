import re
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from fund_monitor import build_chart_time_axis, compute_metrics, svg_fr_chart
from signal_rules import weekly_bars
from three_index_report import (aggregate_ohlc, CHART_PERIODS, kline_levels_chart,
                                period_chart_frames, period_chart_pair)


def prices(dates):
    close = np.arange(len(dates), dtype=float) + 100
    return pd.DataFrame({"date": dates, "open": close - 1, "high": close + 2,
                         "low": close - 3, "close": close})


class Inputs(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.radios = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "input":
            self.radios.append(dict(attrs))


class PeriodDataTests(unittest.TestCase):
    def test_holiday_short_weeks_and_month_boundary_use_actual_trading_day(self):
        frame = prices(pd.to_datetime(["2025-01-23", "2025-01-24", "2025-01-27",
                                       "2025-02-05", "2025-02-06", "2025-02-07"]))
        week = aggregate_ohlc(frame, "week")
        self.assertEqual(week.date.tolist(), pd.to_datetime(["2025-01-24", "2025-01-27", "2025-02-07"]).tolist())
        self.assertEqual(week.iloc[0][["open", "high", "low", "close"]].tolist(), [99, 103, 97, 101])
        self.assertEqual(week.date.tolist(), weekly_bars(frame).date.tolist())
        month = aggregate_ohlc(frame, "month")
        self.assertEqual(month.date.tolist(), pd.to_datetime(["2025-01-27", "2025-02-07"]).tolist())
        self.assertEqual(month.iloc[0][["open", "high", "low", "close"]].tolist(), [99, 104, 97, 102])
        self.assertEqual(month.iloc[1][["open", "high", "low", "close"]].tolist(), [102, 107, 100, 105])

    def test_fr_and_ma_are_calculated_before_display_window_is_taken(self):
        daily = prices(pd.bdate_range("2010-01-01", "2026-09-24"))
        frames = period_chart_frames(daily)
        weekly = weekly_bars(daily)
        np.testing.assert_array_equal(frames["week"].fr, weekly.fr)
        np.testing.assert_allclose(frames["week"].fr_bar, weekly.bar, rtol=0, atol=0, equal_nan=True)
        for period, (_, count, averages) in CHART_PERIODS.items():
            full = frames[period]
            shown = full.tail(count)
            start = len(full) - len(shown)
            self.assertEqual(len(shown), count)
            for n in averages:
                expected = full.close.iloc[start - n + 1:start + 1].mean()
                self.assertAlmostEqual(shown.iloc[0][f"ma_{n}"], expected)
            expected_fr = compute_metrics(full[["close"]])
            np.testing.assert_allclose(full.fr, expected_fr.fr, rtol=0, atol=1e-15)

    def test_extra_monthly_history_cannot_change_daily_or_weekly_signals(self):
        full = prices(pd.bdate_range("2010-01-01", "2026-09-24"))
        current = full[full.date >= "2020-01-01"]
        base = period_chart_frames(current)
        extended = period_chart_frames(current, full)
        for period in ("day", "week"):
            pd.testing.assert_frame_equal(base[period], extended[period])
        self.assertGreater(len(extended["month"]), len(base["month"]))


class AlignedSvgTests(unittest.TestCase):
    def test_dates_candle_centers_fr_points_and_ticks_match_in_every_period(self):
        frames = period_chart_frames(prices(pd.bdate_range("2010-01-01", "2026-09-24")))
        ns = {"s": "http://www.w3.org/2000/svg"}
        for period, (_, count, averages) in CHART_PERIODS.items():
            frame = frames[period].tail(count).reset_index(drop=True)
            axis = build_chart_time_axis(frame.date)
            candle = ET.fromstring(kline_levels_chart(frame, "K", [], axis=axis, ma_periods=averages))
            fr = ET.fromstring(svg_fr_chart(frame, "Fr", axis=axis))
            candle_x = [float(x.attrib["x1"]) for x in candle.findall("s:line", ns) if x.get("stroke-width") == "1.1"]
            fr_line = next(x for x in fr.findall("s:polyline", ns) if x.get("stroke") == "#2563eb")
            fr_x = [float(x.split(",")[0]) for x in fr_line.get("points").split()]
            self.assertEqual(candle_x, fr_x)
            bars = [x for x in fr.findall("s:rect", ns) if x.get("opacity") == "0.75" and float(x.get("x")) > 300]
            for x in bars:
                center = float(x.get("x")) + float(x.get("width")) / 2
                self.assertLess(min(abs(center - v) for v in axis.xs), 0.011)
            for chart in (candle, fr):
                rect = chart.find("s:rect", ns)
                self.assertEqual((rect.get("x"), rect.get("width")), ("58", "884"))
            def ticks(chart):
                return [(x.get("x"), x.text) for x in chart.findall("s:text", ns) if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", x.text or "")]
            self.assertEqual(ticks(candle), ticks(fr))

    def test_initial_missing_bar_keeps_its_fr_date_and_axis_slot(self):
        frame = compute_metrics(prices(pd.bdate_range("2025-01-01", periods=25)))
        axis = build_chart_time_axis(frame.date)
        chart = ET.fromstring(svg_fr_chart(frame, "Fr", axis=axis))
        ns = {"s": "http://www.w3.org/2000/svg"}
        line = chart.find("s:polyline", ns)
        self.assertEqual(len(line.get("points").split()), 25)
        self.assertTrue(line.get("points").startswith("58.0,"))
        with self.assertRaises(ValueError):
            svg_fr_chart(frame.iloc[1:], "Fr", axis=axis)


class PeriodControlsTests(unittest.TestCase):
    def test_month_threshold_default_week_and_independent_radio_groups(self):
        for months in (47, 48):
            frame = prices(pd.date_range("2020-01-01", periods=months, freq="MS"))
            html = period_chart_pair("000510", "中证A500", frame, {}, {})
            radios = Inputs(html).radios
            self.assertEqual([r["value"] for r in radios if "checked" in r], ["week"])
            month = next(r for r in radios if r["value"] == "month")
            self.assertEqual("disabled" in month, months < 48)
            self.assertEqual('title="历史数据不足"' in html, months < 48)
            self.assertNotIn("<script", html)
            other = Inputs(period_chart_pair("000688", "科创50", frame, {}, {})).radios
            self.assertTrue({r["name"] for r in radios}.isdisjoint(r["name"] for r in other))
            self.assertTrue({r["id"] for r in radios}.isdisjoint(r["id"] for r in other))


if __name__ == "__main__":
    unittest.main()
