import unittest

import numpy as np
import pandas as pd

from signal_rules import (
    DIVERGENCE_FACTOR,
    WARMUP_WEEKS,
    fr_state_label,
    replay,
    signal_status,
    week_is_complete,
)


def weekly(close, fr, bar=None):
    n = len(close)
    dates = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    fr = np.asarray(fr, dtype=float)
    if bar is None:
        bar = np.r_[np.nan, np.diff(fr) * 3]
    return pd.DataFrame({"date": dates, "close": np.asarray(close, dtype=float), "fr": fr, "bar": np.asarray(bar, dtype=float)})


class WeekCompleteTests(unittest.TestCase):
    def test_holiday_friday_makes_thursday_the_week_end(self):
        calendar = pd.DatetimeIndex(["2026-09-22", "2026-09-23", "2026-09-24", "2026-09-28"])
        self.assertTrue(week_is_complete(pd.Timestamp("2026-09-24"), calendar))

    def test_midweek_with_trading_days_left_is_incomplete(self):
        calendar = pd.DatetimeIndex(["2026-08-26", "2026-08-27", "2026-08-28"])
        self.assertFalse(week_is_complete(pd.Timestamp("2026-08-26"), calendar))

    def test_without_calendar_friday_counts_as_complete(self):
        self.assertTrue(week_is_complete(pd.Timestamp("2026-08-28"), None))
        self.assertFalse(week_is_complete(pd.Timestamp("2026-08-26"), None))


class RegimeTests(unittest.TestCase):
    def test_exit_on_first_week_below_zero_and_reentry_above(self):
        n = WARMUP_WEEKS + 10
        fr = [0.02] * (WARMUP_WEEKS + 3) + [-0.01] * 4 + [0.01] * 3
        st = replay(weekly(np.linspace(100, 110, n), fr))
        factors = [(e["factor"], e["reason"]) for e in st.events]
        self.assertEqual(factors[0][0], 1.0)
        self.assertEqual(factors[1][0], 0.0)
        self.assertIn("跌破0轴", factors[1][1])
        self.assertEqual(st.events[1]["date"], weekly(np.ones(n), fr)["date"].iloc[WARMUP_WEEKS + 3])
        self.assertEqual(factors[2][0], 1.0)
        self.assertEqual(st.regime, "up")


class DivergenceTests(unittest.TestCase):
    @staticmethod
    def scenario():
        n = WARMUP_WEEKS + 20
        close = np.full(n, 100.0)
        fr = np.full(n, -0.02)
        base = WARMUP_WEEKS
        # swing low at base+5, confirmed four weeks later
        close[base:base + 5] = [97, 96, 95, 94, 93]
        close[base + 5] = 90
        fr[base + 5] = -0.05
        close[base + 6:base + 14] = 93
        # new price low with weaker force, then the turn
        close[base + 14] = 88
        fr[base + 13:base + 16] = [-0.020, -0.030, -0.028]
        close[base + 15] = 89
        # decline re-accelerates: price and Fr both below the divergence low
        close[base + 16], fr[base + 16] = 87, -0.029
        close[base + 17], fr[base + 17] = 85, -0.060
        close[base + 18:] = 85
        fr[base + 18:] = -0.06
        bar = np.r_[np.nan, np.diff(fr) * 3]
        return weekly(close, fr, bar), base

    def test_bottom_divergence_buys_half_then_stops_out(self):
        frame, base = self.scenario()
        st = replay(frame)
        reasons = [(e["date"], e["factor"]) for e in st.events]
        buy = [e for e in st.events if e["factor"] == DIVERGENCE_FACTOR]
        self.assertEqual(len(buy), 1)
        self.assertEqual(buy[0]["date"], frame["date"].iloc[base + 15])
        stop = [e for e in st.events if "止损" in e["reason"]]
        self.assertEqual(len(stop), 1)
        self.assertEqual(stop[0]["date"], frame["date"].iloc[base + 17])
        self.assertEqual(st.factor, 0.0, reasons)

    def test_no_buy_when_force_also_makes_new_low(self):
        frame, base = self.scenario()
        frame.loc[base + 14, "fr"] = -0.07            # Fr below the swing low's Fr: no divergence
        frame["bar"] = np.r_[np.nan, np.diff(frame["fr"].to_numpy()) * 3]
        st = replay(frame)
        self.assertFalse(any(e["factor"] == DIVERGENCE_FACTOR for e in st.events))


class StateLabelTests(unittest.TestCase):
    def test_four_states(self):
        self.assertEqual(fr_state_label(0.01, 0.001), "极强，持股待涨")
        self.assertEqual(fr_state_label(0.01, -0.001), "强市中的回踩或震荡")
        self.assertEqual(fr_state_label(-0.01, 0.001), "弱市中的反弹或震荡")
        self.assertEqual(fr_state_label(-0.01, -0.001), "极弱，谨慎做多")


class SignalStatusTests(unittest.TestCase):
    def test_incomplete_week_is_provisional(self):
        dates = pd.bdate_range("2025-01-01", "2026-08-26")
        close = pd.Series(np.linspace(100, 140, len(dates)))
        status = signal_status(pd.DataFrame({"date": dates, "close": close}),
                               pd.DatetimeIndex(list(dates) + [pd.Timestamp("2026-08-27"), pd.Timestamp("2026-08-28")]))
        self.assertFalse(status["week_complete"])
        self.assertEqual(status["week_date"], pd.Timestamp("2026-08-21"))
        self.assertEqual(status["live_date"], pd.Timestamp("2026-08-26"))
        self.assertEqual(status["regime"], "up")
        self.assertEqual(status["factor"], 1.0)


if __name__ == "__main__":
    unittest.main()
