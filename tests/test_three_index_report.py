import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from three_index_report import (
    DIVIDEND_SPREAD_RULES,
    INDEXES,
    MIN_SPREAD_HISTORY_FOR_PERCENTILE,
    PE_PERCENTILE_RULES,
    cutoff_for_completed_day,
    dividend_status,
    drawdown_trigger,
    fetch_csi_index_data,
    fetch_csi_perf_history,
    history_ranges,
    index_start_date,
    investment_plan,
    latest_frame_date,
    merge_dividend_history,
    normalize_shanghai_now,
    resolve_data_freshness,
    run,
    upsert_rows,
)


def source_dates(price_date: str, turnover_date: str, valuation_date: str = "2026-08-24"):
    return {
        **{f"price_{code}": pd.Timestamp(price_date) for code in INDEXES},
        **{f"valuation_{code}": pd.Timestamp(valuation_date) for code in INDEXES},
        "turnover_shanghai": pd.Timestamp(turnover_date),
        "turnover_shenzhen": pd.Timestamp(turnover_date),
    }


class BeijingTimeTests(unittest.TestCase):
    def test_github_utc_time_is_converted_before_cutoff(self):
        now_utc = datetime(2026, 8, 26, 11, 4, tzinfo=timezone.utc)
        self.assertEqual(normalize_shanghai_now(now_utc).strftime("%Y-%m-%d %H:%M"), "2026-08-26 19:04")
        self.assertEqual(cutoff_for_completed_day(now_utc), pd.Timestamp("2026-08-26"))

    def test_naive_time_is_treated_as_beijing_time(self):
        self.assertEqual(
            cutoff_for_completed_day(datetime(2026, 8, 26, 17, 30)),
            pd.Timestamp("2026-08-26"),
        )
        self.assertEqual(
            cutoff_for_completed_day(datetime(2026, 8, 26, 15, 59)),
            pd.Timestamp("2026-08-25"),
        )


class CsiFetchTests(unittest.TestCase):
    def test_long_history_is_overlaid_with_current_year_query(self):
        requested = []

        def fake_get(_session, _url, *, params, **_kwargs):
            requested.append((params["startDate"], params["endDate"]))
            response = Mock()
            source = "overlay" if params["startDate"] == "20260101" else "long"
            response.json.return_value = {
                "data": [
                    {"tradeDate": params["startDate"], "source": source},
                    {"tradeDate": "20260826", "source": source},
                ]
            }
            return response

        with patch("three_index_report.get_with_retry", side_effect=fake_get), patch(
            "three_index_report.time.sleep"
        ):
            frame = fetch_csi_perf_history(
                Mock(),
                "000510",
                "2025-06-01",
                "2026-08-26",
            )

        self.assertEqual(requested, [("20250601", "20260826"), ("20260101", "20260826")])
        self.assertEqual(frame["tradeDate"].tolist(), ["20250601", "20260101", "20260826"])
        self.assertEqual(frame.loc[frame["tradeDate"] == "20260826", "source"].item(), "overlay")

    def test_price_and_pe_share_one_history_fetch(self):
        raw = pd.DataFrame(
            {
                "tradeDate": ["20260825", "20260826"],
                "open": [100.0, 101.0],
                "close": [101.0, 102.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "tradingVol": [1_000.0, 1_100.0],
                "tradingValue": [2.0, 2.2],
                "changePct": [1.0, 0.99],
                "change": [1.0, 1.0],
                "peg": [16.1, 16.2],
            }
        )
        with patch("three_index_report.fetch_csi_perf_history", return_value=raw) as fetch:
            price, valuation = fetch_csi_index_data(
                Mock(),
                "000510",
                "2020-01-01",
                "2026-08-26",
            )

        fetch.assert_called_once()
        self.assertEqual(price["date"].max(), pd.Timestamp("2026-08-26"))
        self.assertEqual(valuation["date"].max(), pd.Timestamp("2026-08-26"))
        self.assertEqual(valuation.iloc[-1]["pe_ttm"], 16.2)


class FreshnessTests(unittest.TestCase):
    def test_lagging_valuation_does_not_drag_price_cutoff_back(self):
        result = resolve_data_freshness(
            source_dates("2026-08-26", "2026-08-26", "2026-08-24"),
            pd.Timestamp("2026-08-26"),
            pd.Timestamp("2026-08-26"),
            {"status": "ok"},
        )
        self.assertEqual(result["expected_trading_day"], pd.Timestamp("2026-08-26"))
        self.assertTrue(result["all_required_sources_fresh"])

    def test_lagging_required_price_source_is_rejected(self):
        dates = source_dates("2026-08-26", "2026-08-26")
        dates["price_000688"] = pd.Timestamp("2026-08-25")
        result = resolve_data_freshness(
            dates,
            pd.Timestamp("2026-08-26"),
            pd.Timestamp("2026-08-26"),
            {"status": "ok"},
        )
        self.assertFalse(result["all_required_sources_fresh"])
        self.assertEqual(result["stale_required_sources"], {"price_000688": pd.Timestamp("2026-08-25")})

    def test_weekend_uses_last_observed_trading_day(self):
        result = resolve_data_freshness(
            source_dates("2026-08-28", "2026-08-28"),
            pd.Timestamp("2026-08-30"),
            pd.Timestamp("2026-08-28"),
            {"status": "ok"},
        )
        self.assertEqual(result["expected_trading_day"], pd.Timestamp("2026-08-28"))
        self.assertTrue(result["all_required_sources_fresh"])

    def test_calendar_failure_fails_closed_even_with_current_quotes(self):
        result = resolve_data_freshness(
            source_dates("2026-08-26", "2026-08-26"),
            pd.Timestamp("2026-08-26"),
            None,
            {"status": "degraded", "error": "offline"},
        )
        self.assertIsNone(result["expected_trading_day"])
        self.assertFalse(result["freshness_resolved"])
        self.assertFalse(result["all_required_sources_fresh"])

    def test_calendar_failure_cannot_let_all_old_sources_self_certify(self):
        result = resolve_data_freshness(
            source_dates("2026-08-25", "2026-08-25"),
            pd.Timestamp("2026-08-26"),
            None,
            {"status": "degraded", "error": "offline"},
        )
        self.assertFalse(result["freshness_resolved"])
        self.assertFalse(result["all_required_sources_fresh"])

    def test_monday_preclose_uses_friday_not_weekend_cap(self):
        now = datetime(2026, 8, 31, 15, 59)
        calendar_cap = cutoff_for_completed_day(now)
        frame = pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-28", "2026-08-31"])}
        )
        completed_quote = latest_frame_date(frame, "test_quote", calendar_cap)
        result = resolve_data_freshness(
            source_dates("2026-08-28", "2026-08-28"),
            calendar_cap,
            pd.Timestamp("2026-08-28"),
            {"status": "ok"},
        )
        self.assertEqual(calendar_cap, pd.Timestamp("2026-08-30"))
        self.assertEqual(completed_quote, pd.Timestamp("2026-08-28"))
        self.assertEqual(result["expected_trading_day"], pd.Timestamp("2026-08-28"))
        self.assertTrue(result["all_required_sources_fresh"])


class ReportIntegrationTests(unittest.TestCase):
    @staticmethod
    def price_frame():
        dates = pd.bdate_range(end="2026-08-26", periods=320)
        close = pd.Series(range(1000, 1320), dtype=float)
        return pd.DataFrame(
            {
                "date": dates,
                "open": close - 1,
                "close": close,
                "high": close + 3,
                "low": close - 3,
                "volume": 1_000_000.0,
                "amount": 10_000_000_000.0,
                "amplitude_pct": 0.6,
                "pct_change": 0.1,
                "change": 1.0,
                "turnover_pct": float("nan"),
            }
        )

    @staticmethod
    def valuation_frame():
        dates = pd.bdate_range(end="2026-08-24", periods=320)
        return pd.DataFrame(
            {
                "date": dates,
                "pe_ttm": [10 + index / 100 for index in range(len(dates))],
                "official_close": 1000.0,
            }
        )

    @staticmethod
    def turnover_frame():
        frame = ReportIntegrationTests.price_frame()
        return frame[["date", "open", "close", "high", "low", "volume", "amount"]]

    def test_full_report_keeps_current_charts_when_pe_lags(self):
        price = self.price_frame()
        valuation = self.valuation_frame()
        turnover = self.turnover_frame()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "three_index_report.fetch_csi_index_data",
            side_effect=lambda *_args: (price.copy(), valuation.copy()),
        ), patch(
            "three_index_report.fetch_tencent_kline", side_effect=lambda *_args: turnover.copy()
        ), patch(
            "three_index_report.latest_trade_calendar_day",
            return_value=(pd.Timestamp("2026-08-26"), {"status": "ok"}),
        ), patch(
            "three_index_report.official_turnover_crosscheck",
            return_value={"date": "2026-08-26", "available": False},
        ), patch(
            "three_index_report.fetch_csi_dividend_yield", side_effect=RuntimeError("offline")
        ), patch("three_index_report.time.sleep"):
            output = Path(temp_dir) / "output"
            snapshot = run(
                output,
                "2025-01-01",
                datetime(2026, 8, 26, 11, 4, tzinfo=timezone.utc),
            )

            self.assertEqual(snapshot["data_cutoff"], "2026-08-26")
            self.assertEqual(snapshot["generated_at"], "2026-08-26 19:04:00 Asia/Shanghai")
            self.assertTrue(snapshot["freshness"]["all_required_sources_fresh"])
            self.assertTrue(all(item["pe_lag_sessions"] == 2 for item in snapshot["indices"].values()))

            html = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn("今天", html)
            self.assertEqual(set(snapshot["weekly_signals"]), set(INDEXES))
            self.assertTrue(all(sig["regime"] in ("up", "down") for sig in snapshot["weekly_signals"].values()))
            self.assertIn("统一数据截止：2026-08-26（价格、趋势与成交）", html)
            self.assertIn("不会再把K线、均线、Fr或成交额裁回旧日", html)
            self.assertEqual(html.count(" 日K与均线</text>"), 3)
            self.assertEqual(html.count("周线Fr趋势动量（每根柱子是一周）"), 3)
            self.assertEqual(html.count("怎么看这张图"), 9)
            self.assertIn("分批进场进度", html)
            self.assertIn("以股票资金100万为例", html)
            self.assertNotIn("本期定投参考", html)

            diagnostics = json.loads((output / "source_freshness.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["source_latest_dates"]["valuation_000510"], "2026-08-24")
            self.assertEqual(diagnostics["expected_trading_day"], "2026-08-26")

    def test_stale_required_source_stops_deploy_but_keeps_diagnostics(self):
        price = self.price_frame()
        valuation = self.valuation_frame()
        turnover = self.turnover_frame()

        def index_data_for_code(_session, code, *_args):
            selected_price = price.iloc[:-1].copy() if code == "000688" else price.copy()
            return selected_price, valuation.copy()

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "three_index_report.fetch_csi_index_data", side_effect=index_data_for_code
        ), patch(
            "three_index_report.fetch_tencent_kline", side_effect=lambda *_args: turnover.copy()
        ), patch(
            "three_index_report.latest_trade_calendar_day",
            return_value=(pd.Timestamp("2026-08-26"), {"status": "ok"}),
        ), patch(
            "three_index_report.fetch_csi_dividend_yield", side_effect=RuntimeError("offline")
        ), patch("three_index_report.time.sleep"):
            output = Path(temp_dir) / "output"
            with self.assertRaisesRegex(RuntimeError, "price_000688=2026-08-25"):
                run(
                    output,
                    "2025-01-01",
                    datetime(2026, 8, 26, 11, 4, tzinfo=timezone.utc),
                )

            self.assertFalse((output / "report.html").exists())
            diagnostics = json.loads((output / "source_freshness.json").read_text(encoding="utf-8"))
            self.assertFalse(diagnostics["all_required_sources_fresh"])
            self.assertEqual(diagnostics["stale_required_sources"], {"price_000688": "2026-08-25"})

    def test_history_and_plan_are_saved_and_rerun_does_not_duplicate(self):
        price = self.price_frame()
        valuation = self.valuation_frame()
        turnover = self.turnover_frame()
        dividend = pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-25", "2026-08-26"]), "dividend_yield": [4.6, 4.7]}
        )
        bond = pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-24", "2026-08-25"]), "cn10y": [1.70, 1.72]}
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "three_index_report.fetch_csi_index_data",
            side_effect=lambda *_args: (price.copy(), valuation.copy()),
        ), patch(
            "three_index_report.fetch_tencent_kline", side_effect=lambda *_args: turnover.copy()
        ), patch(
            "three_index_report.latest_trade_calendar_day",
            return_value=(pd.Timestamp("2026-08-26"), {"status": "ok"}),
        ), patch(
            "three_index_report.official_turnover_crosscheck",
            return_value={"date": "2026-08-26", "available": False},
        ), patch(
            "three_index_report.fetch_csi_dividend_yield", side_effect=lambda *_args: dividend.copy()
        ), patch(
            "three_index_report.fetch_cn10y_yield", side_effect=lambda *_args: bond.copy()
        ), patch("three_index_report.time.sleep"):
            history = Path(temp_dir) / "history"
            now = datetime(2026, 8, 26, 11, 4, tzinfo=timezone.utc)
            snapshot = run(Path(temp_dir) / "out1", "2025-01-01", now, history_dir=history)
            run(Path(temp_dir) / "out2", "2025-01-01", now, history_dir=history)

            plan = snapshot["investment_plan"]["930955"]
            self.assertEqual(plan["basis"], "股息率 − 10年国债")
            # 4.70 - 1.72 = 2.98 个百分点 -> 第二档
            self.assertEqual(plan["multiplier"], 1.5)
            self.assertIn("PE百分位", snapshot["investment_plan"]["000510"]["basis"])

            daily = pd.read_csv(history / "daily_snapshots.csv", dtype={"code": str})
            self.assertEqual(len(daily), 3)
            self.assertEqual(sorted(daily["code"]), sorted(INDEXES))
            dividend_history = pd.read_csv(history / "dividend_yield_history.csv")
            self.assertEqual(len(dividend_history), 2)

            html = (Path(temp_dir) / "out1" / "report.html").read_text(encoding="utf-8")
            self.assertIn("股息率：4.70%", html)


class PlanRuleTests(unittest.TestCase):
    @staticmethod
    def summary(code, percentile, drawdown=-0.05):
        return {
            "code": code,
            "name": INDEXES[code]["name"],
            "role": INDEXES[code]["role"],
            "pe_percentile": percentile,
            "pe_ttm": 20.0,
            "pe_date": pd.Timestamp("2026-08-26"),
            "drawdown_250": drawdown,
        }

    def test_pe_rules_cover_whole_percentile_range(self):
        for code, rules in PE_PERCENTILE_RULES.items():
            uppers = [upper for upper, _, _ in rules]
            self.assertEqual(uppers, sorted(uppers), code)
            self.assertEqual(uppers[-1], 1.0, code)

    def test_a500_multiplier_follows_percentile(self):
        self.assertEqual(investment_plan(self.summary("000510", 0.10))["multiplier"], 2.0)
        self.assertEqual(investment_plan(self.summary("000510", 0.45))["multiplier"], 1.5)
        self.assertEqual(investment_plan(self.summary("000510", 0.75))["multiplier"], 1.0)
        self.assertEqual(investment_plan(self.summary("000510", 0.90))["multiplier"], 0.5)

    def test_star50_pauses_at_extreme_valuation(self):
        self.assertEqual(investment_plan(self.summary("000688", 0.95))["multiplier"], 0.0)

    def test_dividend_index_falls_back_to_pe_without_yield(self):
        plan = investment_plan(self.summary("930955", 0.20), {"available": False})
        self.assertEqual(plan["multiplier"], 1.5)
        self.assertIn("临时替代", plan["basis"])

    def test_dividend_spread_rules(self):
        lowers = [lower for lower, _, _ in DIVIDEND_SPREAD_RULES]
        self.assertEqual(lowers, sorted(lowers, reverse=True))
        for spread, expected in [(4.0, 2.0), (3.0, 1.5), (2.0, 1.0), (0.8, 0.5)]:
            dividend = {
                "available": True,
                "dividend_yield": 1.7 + spread,
                "cn10y": 1.7,
                "spread": spread,
                "date": pd.Timestamp("2026-08-26"),
            }
            plan = investment_plan(self.summary("930955", 0.9), dividend)
            self.assertEqual(plan["multiplier"], expected, spread)

    def test_drawdown_triggers(self):
        self.assertIsNone(drawdown_trigger(-0.10))
        self.assertIn("第一档", drawdown_trigger(-0.16))
        self.assertIn("第二档", drawdown_trigger(-0.30))
        self.assertIsNone(drawdown_trigger(None))


class HistoryTests(unittest.TestCase):
    def test_short_range_is_single_request_and_long_range_is_split(self):
        self.assertEqual(history_ranges("2022-01-01", "2026-08-26"), [("2022-01-01", "2026-08-26")])
        ranges = history_ranges("2017-05-26", "2026-08-26")
        self.assertEqual(ranges[0][0], "2017-05-26")
        self.assertEqual(ranges[-1][1], "2026-08-26")
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            self.assertEqual(pd.Timestamp(start) - pd.Timestamp(end), pd.Timedelta(days=1))

    def test_dividend_index_uses_its_own_earlier_start(self):
        self.assertEqual(index_start_date("930955", "2020-01-01"), "2017-05-26")
        self.assertEqual(index_start_date("000510", "2020-01-01"), "2020-01-01")

    def test_upsert_keeps_newest_value(self):
        old = pd.DataFrame({"date": ["2026-08-25"], "code": ["930955"], "spread": [2.0]})
        new = pd.DataFrame({"date": ["2026-08-25", "2026-08-26"], "code": ["930955", "930955"], "spread": [2.1, 2.2]})
        merged = upsert_rows(old, new, ["date", "code"])
        self.assertEqual(merged["spread"].tolist(), [2.1, 2.2])

    def test_dividend_uses_latest_bond_yield_on_or_before_date(self):
        dividend = pd.DataFrame({"date": pd.to_datetime(["2026-08-26"]), "dividend_yield": [4.7]})
        bond = pd.DataFrame({"date": pd.to_datetime(["2026-08-25", "2026-08-27"]), "cn10y": [1.72, 1.80]})
        merged = merge_dividend_history(pd.DataFrame(), dividend, bond, "930955")
        self.assertAlmostEqual(merged.iloc[0]["cn10y"], 1.72)
        self.assertAlmostEqual(merged.iloc[0]["spread"], 2.98)

    def test_dividend_percentile_needs_enough_history_and_fresh_data(self):
        dates = pd.bdate_range(end="2026-08-26", periods=MIN_SPREAD_HISTORY_FOR_PERCENTILE)
        history = pd.DataFrame(
            {
                "date": dates,
                "code": "930955",
                "dividend_yield": 4.5,
                "cn10y": 1.7,
                "spread": [2.0 + index / 100 for index in range(len(dates))],
            }
        )
        status = dividend_status(history, pd.Timestamp("2026-08-26"))
        self.assertTrue(status["available"])
        self.assertEqual(status["spread_percentile"], 1.0)

        short = dividend_status(history.tail(5), pd.Timestamp("2026-08-26"))
        self.assertIsNone(short["spread_percentile"])

        stale = dividend_status(history, pd.Timestamp("2026-10-01"))
        self.assertFalse(stale["available"])


if __name__ == "__main__":
    unittest.main()
