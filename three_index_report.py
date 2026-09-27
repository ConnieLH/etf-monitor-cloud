#!/usr/bin/env python3
"""Generate a self-contained monitoring report for three China A-share indices.

Price/trend and turnover must share the latest completed trading day. Valuation
is allowed to lag, but its own as-of date is always disclosed. It is a
monitoring aid, not a trading instruction.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
from datetime import datetime, time as dt_time
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from fund_monitor import (ChartTimeAxis, build_chart_time_axis, validate_chart_time_axis,
                          compute_metrics, fmt_num, fmt_pct, summarize, svg_fr_chart, svg_kline_chart)
from signal_rules import signal_status, weekly_bars


INDEXES: Dict[str, Dict[str, str]] = {
    "000510": {
        "name": "中证A500",
        "secid": "1.000510",
        # 红色/绿色在页面中表示涨跌，指数主题色避开这两种颜色。
        "color": "#2563eb",
        "role": "核心宽基",
        "launch_date": "2024-09-23",
        "history_start": "2020-01-01",
    },
    "000688": {
        "name": "科创50",
        "secid": "1.000688",
        "color": "#7c3aed",
        "role": "科技成长",
        "launch_date": "2020-07-23",
        "history_start": "2020-01-01",
    },
    "930955": {
        "name": "红利低波100",
        "secid": "2.930955",
        "color": "#d97706",
        "role": "防守红利",
        "launch_date": "2017-05-26",
        # 从发布日起抓数，估值百分位才能覆盖指数的完整实盘历史。
        "history_start": "2017-05-26",
    },
}

# ---------------------------------------------------------------------------
# 定投节奏规则：倍数 × 你自己设定的每月基准金额 = 本月参考投入。
# 每条为 (上限, 倍数, 说明)，按顺序匹配第一条满足“当前值 <= 上限”的规则。
# 想调整节奏，只改这里即可；测试会检查规则覆盖了 0~1 的全部区间。
# ---------------------------------------------------------------------------
PE_PERCENTILE_RULES: Dict[str, List[Tuple[float, float, str]]] = {
    "000510": [
        (0.30, 2.0, "估值处于历史低位"),
        (0.50, 1.5, "估值低于中位数"),
        (0.80, 1.0, "估值中性偏高"),
        (1.00, 0.5, "估值处于历史高位"),
    ],
    "000688": [
        (0.30, 1.5, "估值处于历史低位"),
        (0.60, 1.0, "估值中性"),
        (0.85, 0.5, "估值偏高"),
        (1.00, 0.0, "估值处于历史极高位，暂停或只保留最小金额"),
    ],
    # 红利低波100在取不到股息率时，退回用 PE 百分位。
    "930955": [
        (0.30, 1.5, "估值处于历史低位"),
        (0.60, 1.0, "估值中性"),
        (0.85, 0.75, "估值偏高"),
        (1.00, 0.5, "估值处于历史高位"),
    ],
}

# 红利低波100：股息率 − 10年国债收益率（百分点）。利差越大越便宜，按“当前值 >= 下限”匹配。
DIVIDEND_SPREAD_RULES: List[Tuple[float, float, str]] = [
    (3.5, 2.0, "股息率比国债高出很多，性价比突出"),
    (2.5, 1.5, "股息率明显高于国债"),
    (1.5, 1.0, "股息率仍高于国债，性价比一般"),
    (float("-inf"), 0.5, "股息率与国债的差距已明显收窄"),
]

# 距近250个交易日最高点的回撤，达到后提示额外加仓（在倍数之外单独执行）。
DRAWDOWN_TRIGGERS: List[Tuple[float, str]] = [
    (-0.25, "距近一年高点回撤超过25%：触发第二档额外加仓"),
    (-0.15, "距近一年高点回撤超过15%：触发第一档额外加仓"),
]
DRAWDOWN_WINDOW = 250
MIN_SPREAD_HISTORY_FOR_PERCENTILE = 60

EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
CSI_PERF_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
SZSE_REPORT_URL = "https://www.szse.cn/api/report/ShowReport"
TENCENT_KLINE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
DEFAULT_START = "2020-01-01"
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
CLOSE_CONFIRM_TIME = dt_time(16, 0)


def direct_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; ETFMonitor/1.0)",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: Dict[str, str],
    timeout: int = 30,
    attempts: int = 3,
) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as error:  # pragma: no cover - exercised by live failures
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"数据请求失败：{url}；{last_error}")


def fetch_em_kline(session: requests.Session, secid: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "secid": secid,
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "0",
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    payload = get_with_retry(session, EM_KLINE_URL, params=params).json()
    data = payload.get("data") or {}
    rows = data.get("klines") or []
    if not rows:
        raise RuntimeError(f"东方财富未返回 {secid} 的日线数据")
    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude_pct",
        "pct_change",
        "change",
        "turnover_pct",
    ]
    frame = pd.DataFrame([row.split(",") for row in rows], columns=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "close", "high", "low"])
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


HISTORY_CHUNK_YEARS = 5
HISTORY_SINGLE_REQUEST_YEARS = 6


def history_ranges(start: str, end: str) -> List[Tuple[str, str]]:
    """Split very long history queries so one request never spans too many years.

    Ranges up to ~6 years stay a single request (unchanged behaviour); longer
    ones, such as 红利低波100 from 2017, are cut into 5-year blocks.
    """

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts - start_ts <= pd.Timedelta(days=int(365.25 * HISTORY_SINGLE_REQUEST_YEARS)):
        return [(start, end)]
    ranges: List[Tuple[str, str]] = []
    block_start = start_ts
    while block_start <= end_ts:
        block_end = min(block_start + pd.DateOffset(years=HISTORY_CHUNK_YEARS) - pd.Timedelta(days=1), end_ts)
        ranges.append((block_start.strftime("%Y-%m-%d"), block_end.strftime("%Y-%m-%d")))
        block_start = block_end + pd.Timedelta(days=1)
    return ranges


def fetch_csi_perf_history(session: requests.Session, code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch long history, then overlay a fresh current-year query.

    CSI can serve a stale cached tail for a multi-year query while returning
    current data for the same endpoint with a recent start date. One current-
    year overlay fixes that cache pattern without multiplying request volume.
    """

    end_year = int(end[:4])
    overlay_start = max(pd.Timestamp(start), pd.Timestamp(f"{end_year}-01-01")).strftime("%Y-%m-%d")
    ranges = history_ranges(start, end)
    if overlay_start != start:
        ranges.append((overlay_start, end))
    frames: List[pd.DataFrame] = []
    for index, (range_start, range_end) in enumerate(ranges):
        params = {
            "indexCode": code,
            "startDate": range_start.replace("-", ""),
            "endDate": range_end.replace("-", ""),
        }
        payload = get_with_retry(session, CSI_PERF_URL, params=params).json()
        rows = payload.get("data") or []
        if rows:
            frames.append(pd.DataFrame(rows))
        if index + 1 < len(ranges):
            time.sleep(0.35)
    if not frames:
        raise RuntimeError(f"中证指数官网未返回 {code} 的日行情与估值数据")
    frame = pd.concat(frames, ignore_index=True)
    if "tradeDate" not in frame:
        raise RuntimeError(f"中证指数官网 {code} 返回字段不完整")
    frame["tradeDate"] = frame["tradeDate"].astype(str)
    # Long history is concatenated before the current-year overlay, so dedupe
    # before sorting to deterministically keep the overlay row for each date.
    frame = frame.drop_duplicates("tradeDate", keep="last")
    return frame.sort_values("tradeDate").reset_index(drop=True)


def csi_kline_from_perf(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """Convert CSI performance rows to the report's OHLC schema."""

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["tradeDate"], format="%Y%m%d", errors="coerce"),
            "open": pd.to_numeric(frame.get("open"), errors="coerce"),
            "close": pd.to_numeric(frame.get("close"), errors="coerce"),
            "high": pd.to_numeric(frame.get("high"), errors="coerce"),
            "low": pd.to_numeric(frame.get("low"), errors="coerce"),
            "volume": pd.to_numeric(frame.get("tradingVol"), errors="coerce"),
            "amount": pd.to_numeric(frame.get("tradingValue"), errors="coerce") * 1e8,
            "pct_change": pd.to_numeric(frame.get("changePct"), errors="coerce"),
            "change": pd.to_numeric(frame.get("change"), errors="coerce"),
        }
    )
    result = result.dropna(subset=["date", "open", "close", "high", "low"])
    prev_close = result["close"].shift(1)
    result["amplitude_pct"] = (result["high"] - result["low"]) / prev_close * 100
    result["turnover_pct"] = float("nan")
    return result.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_csi_kline(session: requests.Session, code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch official CSI daily prices with a current-year cache overlay."""

    return csi_kline_from_perf(fetch_csi_perf_history(session, code, start, end), code)


def fetch_tencent_kline(session: requests.Session, symbol: str, start: str, end: str) -> pd.DataFrame:
    """腾讯行情日K（用于上证综指/深证综指的成交额代理）。

    接口单次上限 320 根，按自然年分段请求；amount 字段单位为万元，换算为元。
    """
    start_year = int(start[:4])
    end_year = int(end[:4])
    bars: List[List] = []
    for year in range(start_year, end_year + 1):
        year_start = f"{year}-01-01" if year > start_year else start
        year_end = f"{year}-12-31" if year < end_year else end
        params_url = f"{TENCENT_KLINE_URL}?param={symbol},day,{year_start},{year_end},320,qfq"
        payload = get_with_retry(session, params_url, params={}).json()
        node = (payload.get("data") or {}).get(symbol) or {}
        rows = node.get("qfqday") or node.get("day") or []
        bars.extend(row for row in rows if isinstance(row, list) and len(row) >= 9)
        time.sleep(0.35)
    if not bars:
        raise RuntimeError(f"腾讯行情未返回 {symbol} 的日线数据")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([bar[0] for bar in bars], errors="coerce"),
            "open": pd.to_numeric([bar[1] for bar in bars], errors="coerce"),
            "close": pd.to_numeric([bar[2] for bar in bars], errors="coerce"),
            "high": pd.to_numeric([bar[3] for bar in bars], errors="coerce"),
            "low": pd.to_numeric([bar[4] for bar in bars], errors="coerce"),
            "volume": pd.to_numeric([bar[5] for bar in bars], errors="coerce"),
            "amount": pd.to_numeric([bar[8] for bar in bars], errors="coerce") * 1e4,
        }
    )
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def csi_pe_from_perf(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """Convert CSI performance rows to the report's rolling-PE schema."""

    if "tradeDate" not in frame or "peg" not in frame:
        raise RuntimeError(f"中证指数官网 {code} 返回字段不完整")
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["tradeDate"], format="%Y%m%d", errors="coerce"),
            "pe_ttm": pd.to_numeric(frame["peg"], errors="coerce"),
            "official_close": pd.to_numeric(frame.get("close"), errors="coerce"),
        }
    )
    result = result.dropna(subset=["date", "pe_ttm"])
    result = result[result["pe_ttm"] > 0]
    return result.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_csi_pe(session: requests.Session, code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch official CSI rolling PE with a current-year cache overlay."""

    return csi_pe_from_perf(fetch_csi_perf_history(session, code, start, end), code)


def fetch_csi_index_data(
    session: requests.Session,
    code: str,
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and parse price plus valuation with one cache-safe request sequence."""

    frame = fetch_csi_perf_history(session, code, start, end)
    return csi_kline_from_perf(frame, code), csi_pe_from_perf(frame, code)


def normalize_shanghai_now(now: datetime) -> datetime:
    """Return an Asia/Shanghai aware datetime.

    Naive datetimes are interpreted as Beijing time for backwards-compatible
    local calls. Aware datetimes (including GitHub runner UTC) are converted.
    """

    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI_TZ)
    return now.astimezone(SHANGHAI_TZ)


def cutoff_for_completed_day(now: datetime) -> pd.Timestamp:
    """Natural-day upper bound after converting *now* to Beijing time."""

    now_cn = normalize_shanghai_now(now)
    today = pd.Timestamp(now_cn.date())
    if now_cn.time() < CLOSE_CONFIRM_TIME:
        return today - pd.Timedelta(days=1)
    return today


def latest_frame_date(
    frame: pd.DataFrame,
    source_name: str,
    not_after: Optional[pd.Timestamp] = None,
) -> pd.Timestamp:
    if frame.empty or "date" not in frame:
        raise RuntimeError(f"数据源 {source_name} 未返回有效日期")
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if not_after is not None:
        dates = dates[dates <= pd.Timestamp(not_after).normalize()]
    latest = dates.max() if not dates.empty else pd.NaT
    if pd.isna(latest):
        suffix = f"（不晚于 {pd.Timestamp(not_after):%Y-%m-%d}）" if not_after is not None else ""
        raise RuntimeError(f"数据源 {source_name} 未返回有效日期{suffix}")
    return pd.Timestamp(latest).normalize()


TRADE_DATES: Optional[pd.DatetimeIndex] = None   # full calendar, used to tell whether a week is complete


def latest_trade_calendar_day(calendar_cap: pd.Timestamp) -> Tuple[Optional[pd.Timestamp], Dict[str, Any]]:
    """Read the A-share trade calendar; callers fail closed if it is unavailable."""

    global TRADE_DATES

    status: Dict[str, Any] = {
        "status": "degraded",
        "source": "akshare.tool_trade_date_hist_sina",
    }
    try:
        import akshare as ak

        calendar = ak.tool_trade_date_hist_sina()
        if calendar.empty:
            raise RuntimeError("交易日历为空")
        column = "trade_date" if "trade_date" in calendar else calendar.columns[0]
        dates = pd.to_datetime(calendar[column], errors="coerce").dropna()
        TRADE_DATES = pd.DatetimeIndex(dates).normalize()
        dates = dates[dates <= calendar_cap]
        if dates.empty:
            raise RuntimeError(f"交易日历没有不晚于 {calendar_cap:%Y-%m-%d} 的日期")
        latest = pd.Timestamp(dates.max()).normalize()
        status.update({"status": "ok", "latest_date": latest})
        return latest, status
    except Exception as error:  # pragma: no cover - depends on live calendar service
        status["error"] = str(error)
        return None, status


def resolve_data_freshness(
    source_completed_dates: Dict[str, pd.Timestamp],
    completed_calendar_cap: pd.Timestamp,
    calendar_date: Optional[pd.Timestamp],
    calendar_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the report day and identify required sources that are stale."""

    price_names = [f"price_{code}" for code in INDEXES]
    turnover_names = ["turnover_shanghai", "turnover_shenzhen"]
    required_names = price_names + turnover_names
    missing = [name for name in required_names if name not in source_completed_dates]
    if missing:
        raise RuntimeError(f"缺少必需数据源日期：{', '.join(missing)}")

    normalized = {
        name: pd.Timestamp(value).normalize()
        for name, value in source_completed_dates.items()
    }
    csi_common_day = min(normalized[name] for name in price_names)
    turnover_common_day = min(normalized[name] for name in turnover_names)
    # One current required source is enough to prove that lagging peers are
    # stale. The pre-close calendar cap still prevents accepting an intraday
    # bar as a completed session.
    observed_market_day = max(normalized[name] for name in required_names)
    calendar_cap = pd.Timestamp(completed_calendar_cap).normalize()

    if calendar_date is None:
        required_status = {
            name: {
                "latest_date": normalized[name],
                "fresh": None,
                "lag_calendar_days": None,
            }
            for name in required_names
        }
        return {
            "completed_calendar_cap": calendar_cap,
            "trade_calendar": calendar_status,
            "csi_price_common_day": csi_common_day,
            "turnover_common_day": turnover_common_day,
            "observed_market_day": observed_market_day,
            "expected_trading_day": None,
            "source_completed_dates": normalized,
            "required_sources": required_status,
            "stale_required_sources": {},
            "freshness_resolved": False,
            "all_required_sources_fresh": False,
            "freshness_error": "无法取得交易日历，不能证明最近完整交易日；已安全停止部署。",
        }

    calendar_day = pd.Timestamp(calendar_date).normalize()
    if calendar_day > calendar_cap:
        raise RuntimeError(
            f"交易日历日期 {calendar_day:%Y-%m-%d} 超过完整自然日上限 {calendar_cap:%Y-%m-%d}"
        )
    # The calendar catches the case where every quote source is stale. A newer
    # completed quote date can still advance the report if the calendar itself
    # has not refreshed yet.
    expected_day = max(calendar_day, observed_market_day)

    required_status = {
        name: {
            "latest_date": normalized[name],
            "fresh": normalized[name] >= expected_day,
            "lag_calendar_days": max((expected_day - normalized[name]).days, 0),
        }
        for name in required_names
    }
    stale = {
        name: item["latest_date"]
        for name, item in required_status.items()
        if not item["fresh"]
    }
    return {
        "completed_calendar_cap": pd.Timestamp(completed_calendar_cap).normalize(),
        "trade_calendar": calendar_status,
        "csi_price_common_day": csi_common_day,
        "turnover_common_day": turnover_common_day,
        "observed_market_day": observed_market_day,
        "expected_trading_day": expected_day,
        "source_completed_dates": normalized,
        "required_sources": required_status,
        "stale_required_sources": stale,
        "freshness_resolved": True,
        "all_required_sources_fresh": not stale,
    }


def percentile_rank(values: pd.Series, current: float) -> Optional[float]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    valid = valid[np.isfinite(valid)]
    if valid.empty or not math.isfinite(current):
        return None
    return float((valid <= current).mean())


def valuation_label(percentile: Optional[float]) -> str:
    if percentile is None:
        return "数据不足"
    if percentile <= 0.20:
        return "历史偏低"
    if percentile <= 0.40:
        return "相对偏低"
    if percentile <= 0.70:
        return "中性区间"
    if percentile <= 0.85:
        return "相对偏高"
    return "历史偏高"


def trend_label(frame: pd.DataFrame) -> str:
    last = frame.iloc[-1]
    close = float(last["close"])
    ma20 = float(last["ma_20"]) if pd.notna(last["ma_20"]) else close
    ma60 = float(last["ma_60"]) if pd.notna(last["ma_60"]) else close
    fr = float(last["fr"]) if pd.notna(last["fr"]) else 0.0
    if close > ma20 > ma60 and fr > 0:
        return "趋势偏强"
    if close < ma20 < ma60 and fr < 0:
        return "趋势偏弱"
    return "震荡观察"


def class_for_return(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "neutral"
    return "up" if value >= 0 else "down"


def finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def axis_ticks(length: int, count: int = 7) -> List[int]:
    if length <= 1:
        return [0]
    return sorted(set(np.linspace(0, length - 1, min(count, length)).astype(int).tolist()))


def svg_multi_line(
    series: List[Tuple[str, pd.Series, str]],
    dates: pd.Series,
    title: str,
    *,
    width: int = 1060,
    height: int = 380,
    baseline: Optional[float] = None,
    value_suffix: str = "",
) -> str:
    left, right, top, bottom = 66, 24, 54, 42
    chart_w, chart_h = width - left - right, height - top - bottom
    all_values: List[float] = []
    for _, values, _ in series:
        all_values.extend([float(v) for v in values if pd.notna(v) and math.isfinite(float(v))])
    if not all_values:
        return '<div class="empty">暂无足够数据</div>'
    ymin, ymax = min(all_values), max(all_values)
    if baseline is not None:
        ymin, ymax = min(ymin, baseline), max(ymax, baseline)
    padding = max((ymax - ymin) * 0.08, abs(ymax) * 0.01, 0.1)
    ymin, ymax = ymin - padding, ymax + padding

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * chart_h

    xs = np.linspace(left, left + chart_w, len(dates))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="20" y="26" font-size="17" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fff" stroke="#e5e7eb"/>',
    ]
    for frac in np.linspace(0, 1, 5):
        y = top + frac * chart_h
        value = ymax - frac * (ymax - ymin)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="6" y="{y + 4:.1f}" font-size="11" fill="#6b7280">{value:.1f}{escape(value_suffix)}</text>')
    if baseline is not None and ymin <= baseline <= ymax:
        y = sy(baseline)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#94a3b8" stroke-dasharray="5 4"/>')
    legend_x = left
    for name, _, color in series:
        parts.append(f'<line x1="{legend_x}" y1="42" x2="{legend_x + 18}" y2="42" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 24}" y="46" font-size="12" fill="#374151">{escape(name)}</text>')
        legend_x += 24 + len(name) * 14 + 34
    for _, values, color in series:
        points = []
        for x, value in zip(xs, values):
            if pd.notna(value) and math.isfinite(float(value)):
                points.append(f"{x:.1f},{sy(float(value)):.1f}")
        if points:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(points)}"/>')
    for index in axis_ticks(len(dates)):
        x = xs[index]
        label = pd.Timestamp(dates.iloc[index]).strftime("%Y-%m")
        if index == len(dates) - 1:
            label = pd.Timestamp(dates.iloc[index]).strftime("%Y-%m-%d")
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" font-size="11" fill="#6b7280" text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def normalized_performance_chart(frames: Dict[str, pd.DataFrame]) -> str:
    merged: Optional[pd.DataFrame] = None
    for code, frame in frames.items():
        item = frame[["date", "close"]].tail(252).rename(columns={"close": code})
        merged = item if merged is None else merged.merge(item, on="date", how="inner")
    if merged is None or merged.empty:
        return '<div class="empty">暂无足够数据</div>'
    series: List[Tuple[str, pd.Series, str]] = []
    for code, meta in INDEXES.items():
        normalized = merged[code] / float(merged[code].iloc[0]) * 100
        series.append((meta["name"], normalized, meta["color"]))
    return svg_multi_line(series, merged["date"], "近252个交易日：三指数归一化走势（起点=100）", baseline=100)


def relative_strength_chart(frames: Dict[str, pd.DataFrame]) -> str:
    merged = frames["000510"][["date", "close"]].rename(columns={"close": "base"})
    for code in ["000688", "930955"]:
        item = frames[code][["date", "close"]].rename(columns={"close": code})
        merged = merged.merge(item, on="date", how="inner")
    merged = merged.tail(252).reset_index(drop=True)
    growth = (merged["000688"] / merged["base"])
    defense = (merged["930955"] / merged["base"])
    growth = growth / float(growth.iloc[0]) * 100
    defense = defense / float(defense.iloc[0]) * 100
    return svg_multi_line(
        [
            ("科创50 / 中证A500", growth, INDEXES["000688"]["color"]),
            ("红利低波100 / 中证A500", defense, INDEXES["930955"]["color"]),
        ],
        merged["date"],
        "风格相对强弱（上行=相对中证A500更强，起点=100）",
        baseline=100,
    )


def pe_chart(frame: pd.DataFrame, name: str, color: str) -> str:
    plot = frame.tail(756).reset_index(drop=True)
    values = plot["pe_ttm"]
    q20, q50, q80 = (float(values.quantile(q)) for q in [0.2, 0.5, 0.8])
    return svg_multi_line(
        [
            ("滚动市盈率", values, color),
            ("20%分位线", pd.Series([q20] * len(plot)), "#16a34a"),
            ("中位数", pd.Series([q50] * len(plot)), "#64748b"),
            ("80%分位线", pd.Series([q80] * len(plot)), "#f59e0b"),
        ],
        plot["date"],
        f"{name}：滚动市盈率与历史区间",
    )


def build_turnover(shanghai: pd.DataFrame, shenzhen: pd.DataFrame) -> pd.DataFrame:
    left = shanghai[["date", "amount"]].rename(columns={"amount": "shanghai_amount"})
    right = shenzhen[["date", "amount"]].rename(columns={"amount": "shenzhen_amount"})
    frame = left.merge(right, on="date", how="inner")
    frame["total_amount"] = frame["shanghai_amount"] + frame["shenzhen_amount"]
    frame["ma20"] = frame["total_amount"].rolling(20).mean()
    frame["low_q20_250"] = frame["total_amount"].rolling(250, min_periods=120).quantile(0.20)
    frame["ratio_ma20"] = frame["total_amount"] / frame["ma20"]
    return frame.sort_values("date").reset_index(drop=True)


def turnover_chart(frame: pd.DataFrame, width: int = 1060, height: int = 380) -> str:
    plot = frame.tail(252).reset_index(drop=True)
    left, right, top, bottom = 66, 24, 54, 42
    chart_w, chart_h = width - left - right, height - top - bottom
    values = plot["total_amount"] / 1e12
    ma20 = plot["ma20"] / 1e12
    low_line = plot["low_q20_250"] / 1e12
    ymax = float(pd.concat([values, ma20]).max()) * 1.08
    ymin = 0.0

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * chart_h

    xs = np.linspace(left, left + chart_w, len(plot))
    zero_y = sy(0)
    bar_w = max(chart_w / max(len(plot), 1) * 0.72, 1.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<text x="20" y="26" font-size="17" font-weight="700" fill="#111827">沪深市场成交额代理口径（万亿元）</text>',
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fff" stroke="#e5e7eb"/>',
    ]
    for frac in np.linspace(0, 1, 5):
        y = top + frac * chart_h
        value = ymax - frac * (ymax - ymin)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" font-size="11" fill="#6b7280">{value:.2f}</text>')
    for x, value in zip(xs, values):
        y = sy(float(value))
        parts.append(f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{zero_y - y:.2f}" fill="#93c5fd" opacity="0.8"/>')
    for series, color, dash in [(ma20, "#f59e0b", ""), (low_line, "#16a34a", ' stroke-dasharray="5 4"')]:
        points = " ".join(f"{x:.1f},{sy(float(v)):.1f}" for x, v in zip(xs, series) if pd.notna(v))
        if points:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{points}"/>')
    parts.extend(
        [
            f'<rect x="{left}" y="36" width="12" height="10" fill="#93c5fd"/><text x="{left + 18}" y="46" font-size="12" fill="#374151">每日合计</text>',
            f'<line x1="{left + 95}" y1="42" x2="{left + 113}" y2="42" stroke="#f59e0b" stroke-width="3"/><text x="{left + 119}" y="46" font-size="12" fill="#374151">20日均值</text>',
            f'<line x1="{left + 205}" y1="42" x2="{left + 223}" y2="42" stroke="#16a34a" stroke-width="2" stroke-dasharray="5 4"/><text x="{left + 229}" y="46" font-size="12" fill="#374151">滚动250日20%分位</text>',
        ]
    )
    for index in axis_ticks(len(plot)):
        x = xs[index]
        label = plot.loc[index, "date"].strftime("%Y-%m")
        if index == len(plot) - 1:
            label = plot.loc[index, "date"].strftime("%Y-%m-%d")
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" font-size="11" fill="#6b7280" text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def official_turnover_crosscheck(date: pd.Timestamp) -> Dict[str, Any]:
    result: Dict[str, Any] = {"date": date.strftime("%Y-%m-%d"), "available": False}
    try:
        import akshare as ak

        sse = ak.stock_sse_deal_daily(date=date.strftime("%Y%m%d"))
        row = sse[sse["单日情况"].astype(str).str.strip() == "成交金额"].iloc[0]
        shanghai = float(str(row["股票"]).replace(",", "")) * 1e8
        session = direct_session()
        response = get_with_retry(
            session,
            SZSE_REPORT_URL,
            params={
                "SHOWTYPE": "xlsx",
                "CATALOGID": "1803_sczm",
                "TABKEY": "tab1",
                "txtQueryDate": date.strftime("%Y-%m-%d"),
            },
        )
        table = pd.read_excel(io.BytesIO(response.content))
        first_col = table.columns[0]
        amount_col = next(column for column in table.columns if "成交金额" in str(column))
        sz_row = table[table[first_col].astype(str).str.strip() == "股票"].iloc[0]
        shenzhen = float(str(sz_row[amount_col]).replace(",", ""))
        result.update(
            {
                "available": True,
                "sse_stock_amount": shanghai,
                "szse_stock_amount": shenzhen,
                "official_total": shanghai + shenzhen,
            }
        )
    except Exception as error:  # non-blocking provenance check
        result["error"] = str(error)
    return result


DIVIDEND_CODE = "930955"
DIVIDEND_HISTORY_FILE = "dividend_yield_history.csv"
DAILY_HISTORY_FILE = "daily_snapshots.csv"
DIVIDEND_MAX_STALE_DAYS = 14


def fetch_csi_dividend_yield(code: str) -> pd.DataFrame:
    """中证官方指数估值（近一段时间），取股息率。

    该接口只返回最近一段数据，所以需要每天把结果存进 history/ 逐步积累历史。
    股息率2 为“计算用股本”口径，与指数加权方式一致；缺失时退回股息率1。
    """

    import akshare as ak

    raw = ak.stock_zh_index_value_csindex(symbol=code)
    if raw is None or raw.empty:
        raise RuntimeError(f"中证指数估值接口未返回 {code} 的数据")
    column = next((name for name in ["股息率2", "股息率1"] if name in raw.columns), None)
    if column is None or "日期" not in raw.columns:
        raise RuntimeError(f"中证指数估值接口 {code} 缺少日期或股息率字段")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["日期"], errors="coerce"),
            "dividend_yield": pd.to_numeric(raw[column], errors="coerce"),
        }
    ).dropna()
    frame = frame[frame["dividend_yield"] > 0]
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_cn10y_yield(start: pd.Timestamp) -> pd.DataFrame:
    """中国10年期国债收益率（%），来自 AKShare 的中美国债收益率接口。"""

    import akshare as ak

    raw = ak.bond_zh_us_rate(start_date=pd.Timestamp(start).strftime("%Y%m%d"))
    column = "中国国债收益率10年"
    if raw is None or raw.empty or column not in raw.columns or "日期" not in raw.columns:
        raise RuntimeError("国债收益率接口未返回中国10年期数据")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["日期"], errors="coerce"),
            "cn10y": pd.to_numeric(raw[column], errors="coerce"),
        }
    ).dropna()
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def read_history(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"code": str})
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def write_history(path: Optional[Path], frame: pd.DataFrame, keys: List[str]) -> None:
    if path is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output = output.sort_values(keys).reset_index(drop=True)
    output.to_csv(path, index=False, encoding="utf-8")


def upsert_rows(existing: pd.DataFrame, new_rows: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Merge new rows into a history table; the newest value wins for a repeated key."""

    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    if combined.empty:
        return combined
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    if "code" in combined:
        combined["code"] = combined["code"].astype(str)
    combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values(keys).reset_index(drop=True)


def merge_dividend_history(
    existing: pd.DataFrame,
    dividend: pd.DataFrame,
    bond: pd.DataFrame,
    code: str,
) -> pd.DataFrame:
    """Attach the latest 10Y yield on or before each dividend date, then upsert."""

    if dividend.empty or bond.empty:
        return existing
    merged = pd.merge_asof(
        dividend.sort_values("date"),
        bond.sort_values("date"),
        on="date",
        direction="backward",
    ).dropna(subset=["cn10y"])
    merged["code"] = code
    merged["spread"] = merged["dividend_yield"] - merged["cn10y"]
    merged = merged[["date", "code", "dividend_yield", "cn10y", "spread"]]
    return upsert_rows(existing, merged, ["date", "code"])


def dividend_status(history: pd.DataFrame, cutoff: pd.Timestamp, code: str = DIVIDEND_CODE) -> Dict[str, Any]:
    """Latest dividend-yield spread not after *cutoff*, plus its historical percentile."""

    status: Dict[str, Any] = {"available": False, "code": code}
    if history.empty or "spread" not in history:
        status["reason"] = "尚未积累到股息率数据"
        return status
    rows = history[(history["code"].astype(str) == code) & (history["date"] <= cutoff)].dropna(subset=["spread"])
    if rows.empty:
        status["reason"] = "截止日前没有股息率数据"
        return status
    last = rows.iloc[-1]
    lag_days = int((pd.Timestamp(cutoff) - pd.Timestamp(last["date"])).days)
    status.update(
        {
            "date": pd.Timestamp(last["date"]),
            "dividend_yield": float(last["dividend_yield"]),
            "cn10y": float(last["cn10y"]),
            "spread": float(last["spread"]),
            "observations": int(len(rows)),
            "lag_days": lag_days,
            "spread_percentile": (
                percentile_rank(rows["spread"], float(last["spread"]))
                if len(rows) >= MIN_SPREAD_HISTORY_FOR_PERCENTILE
                else None
            ),
        }
    )
    if lag_days > DIVIDEND_MAX_STALE_DAYS:
        status["reason"] = f"股息率数据已滞后 {lag_days} 天"
        return status
    status["available"] = True
    return status


def update_dividend_history(history_dir: Optional[Path], cutoff: pd.Timestamp) -> Dict[str, Any]:
    """Fetch today's dividend yield and 10Y yield, store them, and report status.

    Non-blocking: any failure is recorded and the plan falls back to PE percentile.
    """

    path = history_dir / DIVIDEND_HISTORY_FILE if history_dir is not None else None
    history = read_history(path)
    fetch_error: Optional[str] = None
    try:
        dividend = fetch_csi_dividend_yield(DIVIDEND_CODE)
        bond = fetch_cn10y_yield(pd.Timestamp(cutoff) - pd.Timedelta(days=400))
        history = merge_dividend_history(history, dividend, bond, DIVIDEND_CODE)
        write_history(path, history, ["date", "code"])
    except Exception as error:  # pragma: no cover - depends on live services
        fetch_error = str(error)
    status = dividend_status(history, cutoff)
    if fetch_error:
        status["fetch_error"] = fetch_error
    return status


def rule_by_upper(rules: List[Tuple[float, float, str]], value: float) -> Tuple[float, str]:
    for upper, multiplier, text in rules:
        if value <= upper:
            return multiplier, text
    upper, multiplier, text = rules[-1]
    return multiplier, text


def rule_by_lower(rules: List[Tuple[float, float, str]], value: float) -> Tuple[float, str]:
    for lower, multiplier, text in rules:
        if value >= lower:
            return multiplier, text
    lower, multiplier, text = rules[-1]
    return multiplier, text


def drawdown_trigger(drawdown: Optional[float]) -> Optional[str]:
    if drawdown is None or not math.isfinite(drawdown):
        return None
    for threshold, text in DRAWDOWN_TRIGGERS:
        if drawdown <= threshold:
            return text
    return None


def investment_plan(summary: Dict[str, Any], dividend: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """本期定投倍数：由估值规则决定，回撤触发作为额外加仓提示单独给出。"""

    code = summary["code"]
    plan: Dict[str, Any] = {
        "code": code,
        "name": summary["name"],
        "role": summary["role"],
        "drawdown_250": summary.get("drawdown_250"),
        "drawdown_trigger": drawdown_trigger(summary.get("drawdown_250")),
    }
    if code == DIVIDEND_CODE and dividend and dividend.get("available"):
        multiplier, reason = rule_by_lower(DIVIDEND_SPREAD_RULES, float(dividend["spread"]))
        plan.update(
            {
                "basis": "股息率 − 10年国债",
                "basis_value": (
                    f"{dividend['dividend_yield']:.2f}% − {dividend['cn10y']:.2f}% = "
                    f"{dividend['spread']:.2f} 个百分点"
                ),
                "basis_date": dividend["date"],
                "multiplier": multiplier,
                "reason": reason,
            }
        )
        return plan

    percentile = summary.get("pe_percentile")
    if percentile is None:
        plan.update(
            {
                "basis": "PE百分位",
                "basis_value": "数据不足",
                "basis_date": summary.get("pe_date"),
                "multiplier": 1.0,
                "reason": "估值数据不足，按基准金额执行",
            }
        )
        return plan
    multiplier, reason = rule_by_upper(PE_PERCENTILE_RULES[code], float(percentile))
    basis = "PE百分位"
    if code == DIVIDEND_CODE:
        basis = "PE百分位（暂无股息率，临时替代）"
    plan.update(
        {
            "basis": basis,
            "basis_value": f"{fmt_pct(percentile, 1)}（PE {fmt_num(summary['pe_ttm'])}）",
            "basis_date": summary.get("pe_date"),
            "multiplier": multiplier,
            "reason": reason,
        }
    )
    return plan


def daily_history_rows(
    cutoff: pd.Timestamp,
    summaries: Dict[str, Dict[str, Any]],
    plans: Dict[str, Dict[str, Any]],
    dividend: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for code, summary in summaries.items():
        plan = plans[code]
        is_dividend = code == DIVIDEND_CODE and dividend.get("available")
        rows.append(
            {
                "date": cutoff,
                "code": code,
                "name": summary["name"],
                "close": summary["latest_close"],
                "pe_ttm": summary["pe_ttm"],
                "pe_date": summary["pe_date"].strftime("%Y-%m-%d"),
                "pe_percentile": summary["pe_percentile"],
                "drawdown_250": summary["drawdown_250"],
                "dividend_yield": dividend.get("dividend_yield") if is_dividend else None,
                "cn10y": dividend.get("cn10y") if is_dividend else None,
                "spread": dividend.get("spread") if is_dividend else None,
                "plan_basis": plan["basis"],
                "multiplier": plan["multiplier"],
                "drawdown_trigger": plan["drawdown_trigger"] or "",
                "weekly_regime": summary.get("weekly_regime") or "",
                "weekly_factor": summary.get("weekly_factor"),
                "weekly_stage": summary.get("weekly_stage") or "",
            }
        )
    return pd.DataFrame(rows)


def make_summary(
    code: str,
    price: pd.DataFrame,
    valuation: pd.DataFrame,
) -> Dict[str, Any]:
    if valuation.empty:
        raise RuntimeError(f"{INDEXES[code]['name']} 在行情截止日前没有可用估值数据")
    base = summarize(price, code, INDEXES[code]["name"], mode="index")
    launch = pd.Timestamp(INDEXES[code]["launch_date"])
    live_window = valuation[valuation["date"] >= launch]
    if live_window.empty:
        live_window = valuation
    latest_pe = float(valuation.iloc[-1]["pe_ttm"])
    pe_date = pd.Timestamp(valuation.iloc[-1]["date"])
    pe_lag_sessions = int((price["date"] > pe_date).sum())
    percentile = percentile_rank(live_window["pe_ttm"], latest_pe)
    last = price.iloc[-1]
    recent_high = float(price["close"].tail(DRAWDOWN_WINDOW).max())
    drawdown_250 = float(last["close"]) / recent_high - 1 if recent_high > 0 else None
    return {
        "code": code,
        "name": INDEXES[code]["name"],
        "role": INDEXES[code]["role"],
        "color": INDEXES[code]["color"],
        "start_date": base["start_date"],
        "end_date": base["end_date"],
        "latest_close": base["latest_close"],
        "return_1d": finite_or_none(last["ret_1d"]),
        "return_20d": finite_or_none(last["ret_20"]),
        "return_60d": finite_or_none(last["ret_60"]),
        "return_1y": base["return_1y"],
        "current_drawdown": base["current_drawdown"],
        "max_drawdown": base["max_drawdown"],
        "recent_high_250": recent_high,
        "drawdown_250": drawdown_250,
        "vol_20": base["vol_20"],
        "ma_20": base["ma_20"],
        "ma_60": base["ma_60"],
        "ma_120": base["ma_120"],
        "ma_250": base["ma_250"],
        "fr_latest": base["fr_latest"],
        "fr_bar_latest": base["fr_bar_latest"],
        "pe_ttm": latest_pe,
        "pe_date": pe_date,
        "pe_lag_sessions": pe_lag_sessions,
        "pe_percentile": percentile,
        "pe_label": valuation_label(percentile),
        "pe_window_start": live_window.iloc[0]["date"].strftime("%Y-%m-%d"),
        "pe_observations": int(len(live_window)),
        "trend_label": trend_label(price),
    }


def observation_text(summary: Dict[str, Any]) -> str:
    pe = summary["pe_percentile"]
    trend = summary["trend_label"]
    if pe is not None and pe <= 0.20 and trend == "趋势偏弱":
        return "估值位于历史偏低区间，但趋势仍弱，适合列入重点观察，不等同于立即买入信号。"
    if pe is not None and pe <= 0.40 and trend == "趋势偏强":
        return "估值相对不高且趋势偏强，条件较前期改善；仍应结合自身资金安排分散决策。"
    if pe is not None and pe >= 0.80:
        return "估值处在历史较高区间，长期资金更需要关注回撤风险与买入节奏。"
    return "估值与趋势暂未形成明显共振，维持常规观察即可。"


def pct_cell(value: Optional[float]) -> str:
    css = class_for_return(value)
    return f'<span class="{css}">{fmt_pct(value)}</span>'


def index_card(summary: Dict[str, Any]) -> str:
    percentile = fmt_pct(summary["pe_percentile"], 1)
    return f"""
    <article class="index-card" style="--accent:{summary['color']}">
      <div class="card-title"><span>{escape(summary['name'])}</span><small>{summary['code']} · {escape(summary['role'])}</small></div>
      <div class="big-number">{fmt_num(summary['latest_close'])}</div>
      <div class="card-grid">
        <div><label>当日</label>{pct_cell(summary['return_1d'])}</div>
        <div><label>近20日</label>{pct_cell(summary['return_20d'])}</div>
        <div><label>滚动PE<br><small>截至 {summary['pe_date'].strftime('%Y-%m-%d')}</small></label><b>{fmt_num(summary['pe_ttm'])}</b></div>
        <div><label>PE百分位</label><b>{percentile}</b></div>
      </div>
      <div class="tags"><span>{escape(summary['pe_label'])}</span><span>{escape(summary['trend_label'])}</span></div>
      <p>{escape(observation_text(summary))}</p>
    </article>"""


def detail_table(summary: Dict[str, Any]) -> str:
    rows = [
        ("最新点位", fmt_num(summary["latest_close"])),
        ("当日 / 20日 / 60日", f"{fmt_pct(summary['return_1d'])} / {fmt_pct(summary['return_20d'])} / {fmt_pct(summary['return_60d'])}"),
        ("近1年收益", fmt_pct(summary["return_1y"])),
        ("距近一年高点回撤", fmt_pct(summary["drawdown_250"])),
        (
            f"距区间高点 / 区间最大回撤（{summary['start_date']} 起）",
            f"{fmt_pct(summary['current_drawdown'])} / {fmt_pct(summary['max_drawdown'])}",
        ),
        ("20日年化波动", fmt_pct(summary["vol_20"])),
        ("MA20 / MA60", f"{fmt_num(summary['ma_20'])} / {fmt_num(summary['ma_60'])}"),
        ("MA120 / MA250", f"{fmt_num(summary['ma_120'])} / {fmt_num(summary['ma_250'])}"),
        ("Fr / BAR", f"{fmt_num(summary['fr_latest'], 4)} / {fmt_num(summary['fr_bar_latest'], 4)}"),
        (f"滚动PE / 百分位（截至 {summary['pe_date'].strftime('%Y-%m-%d')}）", f"{fmt_num(summary['pe_ttm'])} / {fmt_pct(summary['pe_percentile'], 1)}"),
        (
            "估值样本窗口",
            f"{summary['pe_window_start']} 起，约 {summary['pe_observations'] / 243:.1f} 年，"
            f"共 {summary['pe_observations']} 个观测"
            + ("（样本较短，百分位容易受近期行情影响）" if summary["pe_observations"] < 243 * 3 else ""),
        ),
    ]
    return '<table class="kv">' + "".join(f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in rows) + "</table>"


def format_multiplier(value: float) -> str:
    if value <= 0:
        return "暂停"
    return f"× {value:g}"


def plan_section(plans: Dict[str, Dict[str, Any]], dividend: Dict[str, Any]) -> str:
    rows = []
    for code in INDEXES:
        plan = plans[code]
        basis_date = plan.get("basis_date")
        date_text = f"截至 {pd.Timestamp(basis_date):%Y-%m-%d}" if basis_date is not None else ""
        trigger = plan.get("drawdown_trigger")
        trigger_html = (
            f'<span class="trigger">{escape(trigger)}</span>'
            if trigger
            else f'<span class="muted">未触发（{escape(fmt_pct(plan.get("drawdown_250")))}）</span>'
        )
        rows.append(
            f"""<tr>
              <td data-label="指数"><div><b>{escape(plan['name'])}</b><small>{escape(plan['role'])}</small></div></td>
              <td data-label="依据"><div>{escape(plan['basis'])}<small>{escape(plan['basis_value'])} {escape(date_text)}</small></div></td>
              <td data-label="本期倍数" class="mult"><div>{escape(format_multiplier(plan['multiplier']))}<small>{escape(plan['reason'])}</small></div></td>
              <td data-label="回撤加仓"><div>{trigger_html}</div></td>
            </tr>"""
        )
    dividend_note = ""
    if not dividend.get("available"):
        reason = dividend.get("reason") or dividend.get("fetch_error") or "暂无数据"
        dividend_note = (
            f'<p class="inline-warning">红利低波100暂用PE百分位代替股息率利差：{escape(str(reason))}。'
            "股息率历史会每天自动累积。</p>"
        )
    return f"""
  <section class="panel plan">
    <div class="section-heading"><div><span class="eyebrow">MONTHLY PLAN</span><h2>本期定投参考</h2></div>
    <div class="status"><b>倍数 × 你的每月基准金额</b><span>回撤加仓在倍数之外单独执行</span></div></div>
    <table class="plan-table">
      <thead><tr><th>指数</th><th>依据</th><th>本期倍数</th><th>回撤加仓</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    {dividend_note}
    <p class="muted small">规则写在脚本顶部的 PE_PERCENTILE_RULES、DIVIDEND_SPREAD_RULES、DRAWDOWN_TRIGGERS 中，可按自己的节奏修改。这是按预设规则计算的参考，不是投资建议。</p>
  </section>"""


WEEKLY_FACTOR_TEXT = {1.0: "持有（全部仓位）", 0.5: "半仓（底背离抄底）", 0.0: "空仓等待"}


def compute_weekly_signals(prices: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Weekly Fr risk switch per index; failures are reported, never block the report."""

    out: Dict[str, Dict[str, Any]] = {}
    for code in INDEXES:
        try:
            out[code] = signal_status(prices[code][["date", "close"]], TRADE_DATES)
        except Exception as error:  # pragma: no cover - defensive
            out[code] = {"error": str(error)}
    return out


def signal_section(signals: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for code, meta in INDEXES.items():
        sig = signals.get(code) or {}
        if "error" in sig or not sig:
            rows.append(f'<tr><td data-label="指数"><div><b>{escape(meta["name"])}</b></div></td>'
                        f'<td data-label="状态" colspan="4"><div class="muted">周线信号暂不可用：{escape(str(sig.get("error", "无数据")))}</div></td></tr>')
            continue
        up = sig["regime"] == "up"
        cycle = "上涨周期" if up else "下跌周期"
        since = f"{sig['since']:%Y-%m-%d} 那周起" if sig.get("since") is not None else ""
        badge = '<span class="sig-new">本周新信号</span>' if sig.get("new_signal") else ""
        position = WEEKLY_FACTOR_TEXT.get(float(sig["factor"]), f"{sig['factor']:.0%}")
        live = ""
        if not sig["week_complete"]:
            live = (f'<small>本周进行中（截至 {sig["live_date"]:%m-%d}）：Fr {sig["live_fr"]:+.4f}，'
                    f'BAR {sig["live_bar"]:+.4f}，以周收盘为准</small>')
        rows.append(
            f"""<tr class="{'sig-up' if up else 'sig-down'}">
              <td data-label="指数"><div><b>{escape(meta['name'])}</b><small>{escape(meta['role'])}</small></div></td>
              <td data-label="周线周期"><div><span class="cycle {'up' if up else 'down'}">{cycle}</span>{badge}<small>{escape(since)}</small></div></td>
              <td data-label="规则仓位"><div><b>{escape(position)}</b><small>{escape(sig['reason'])}</small></div></td>
              <td data-label="Fr / BAR"><div>{sig['fr']:+.4f} / {sig['bar']:+.4f}<small>{escape(sig['state_label'])}（周收盘 {sig['week_date']:%m-%d}）</small>{live}</div></td>
              <td data-label="下一步"><div>{escape(sig['stage'])}<small>{escape(sig['next'])}</small></div></td>
            </tr>"""
        )
    return f"""
  <section class="panel plan signals">
    <div class="section-heading"><div><span class="eyebrow">WEEKLY RISK SWITCH</span><h2>周线Fr买卖信号</h2></div>
    <div class="status"><b>跌破0轴即卖，上穿0轴买回</b><span>下跌周期里底背离拐点买半仓</span></div></div>
    <table class="plan-table">
      <thead><tr><th>指数</th><th>周线周期</th><th>规则仓位</th><th>Fr / BAR</th><th>下一步</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="muted small">周线取每周最后一个交易日收盘，信号只以完整的一周确认，下一个交易日执行。回测最大回撤：沪深300（2005–2026）由一直持有的 −72% 降到 −41%；三指数组合（2020–2026）由 −31% 降到 −22%，年化收益相近。抄底规则历史样本很少，只买半仓。规则见 signal_rules.py，不是投资建议。</p>
  </section>"""


EXAMPLE_TOTAL = 1_000_000          # 页面举例用的股票资金总额（不是真实资金）
TARGET_WEIGHTS = {"000510": 0.50, "930955": 0.35, "000688": 0.15}
VALUATION_CAP_PCT = 0.90           # PE 历史分位超过此值：最多持有目标的 2/3
VALUATION_CAP_SPREAD = 1.5         # 红利低波100：股息率−国债利差低于此值同样视为过贵


def wan(amount: float) -> str:
    value = amount / 10_000
    return f"{value:.1f}万".replace(".0万", "万")


def valuation_capped(code: str, summary: Dict[str, Any], dividend: Dict[str, Any]) -> Optional[str]:
    pct = summary.get("pe_percentile")
    if pct is not None and pct > VALUATION_CAP_PCT:
        return f"估值处于历史 {pct:.0%} 分位，最多持有目标仓位的 2/3"
    if code == DIVIDEND_CODE and dividend.get("available") and dividend["spread"] < VALUATION_CAP_SPREAD:
        return f"股息率只比国债高 {dividend['spread']:.2f} 个百分点，最多持有目标仓位的 2/3"
    return None


def decide(code: str, summary: Dict[str, Any], sig: Dict[str, Any], dividend: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the weekly signal into a plain-language instruction for one index."""

    target = EXAMPLE_TOTAL * TARGET_WEIGHTS[code]
    if not sig or "error" in sig:
        return {"level": "wait", "action": "信号暂不可用", "detail": "本次没有算出周线信号，先不操作。",
                "factor": None, "effective": None, "target": target, "now": 0.0, "cap": None}
    factor = float(sig["factor"])
    cap = valuation_capped(code, summary, dividend)
    effective = factor * (2 / 3 if cap else 1)
    now = target * effective
    last_reason = sig["events"][-1]["reason"] if sig.get("events") else ""
    if sig.get("new_signal"):
        level = "act"
        if factor == 0 and "止损" in last_reason:
            action = "卖出抄底的半仓（止损）"
        elif factor == 0:
            action = "卖出全部"
        elif factor == 0.5:
            action = "买入半仓"
        else:
            action = "买到满仓"
        detail = f"{last_reason}。下一个交易日按收盘净值操作。"
    elif factor >= 1:
        level = "hold"
        action = "持有"
        detail = "已持有的继续持有；还没建仓的，可以按规则仓位买入"
        if sig["bar"] < 0:
            detail += "，不过Fr正在回落，可以分两次买"
        detail += "。周线Fr收在0轴下方时卖出。"
    elif factor > 0:
        level = "hold"
        action = "持有半仓"
        detail = "等周线Fr重新上穿0轴再补到满仓；" + sig.get("next", "")
    else:
        level = "wait"
        action = "空仓等待"
        detail = sig.get("next", "")
    return {"level": level, "action": action, "detail": detail, "factor": factor, "effective": effective,
            "target": target, "now": now, "cap": cap}


def price_levels(summary: Dict[str, Any], sig: Dict[str, Any]) -> List[Tuple[float, str, str]]:
    high = summary.get("recent_high_250")
    levels: List[Tuple[float, str, str]] = []
    if high:
        levels.append((high, f"近一年最高 {high:.0f}", "#64748b"))
        levels.append((high * 0.85, f"回撤15% {high * 0.85:.0f}", "#d97706"))
    if sig and sig.get("regime") == "down" and sig.get("ref_close"):
        levels.append((sig["ref_close"], f"参照低点 {sig['ref_close']:.0f}", "#2563eb"))
    return levels


def kline_levels_chart(df: pd.DataFrame, title: str, levels: List[Tuple[float, str, str]],
                       width: int = 1060, height: int = 380, *,
                       axis: Optional[ChartTimeAxis] = None,
                       ma_periods: Tuple[int, int] = (20, 60)) -> str:
    """Candles and period moving averages with the shared pair's exact dates."""

    plot = (df if axis is not None else df.tail(250)).reset_index(drop=True)
    if axis is not None:
        validate_chart_time_axis(plot, axis)
        width = axis.width
    if len(plot) < 20:
        return f"<div>图表 {escape(title)} 无足够数据</div>"
    left, right, top, bottom = (axis.left if axis else 58), (axis.right if axis else 118), 34, 34
    cw, chh = width - left - right, height - top - bottom
    axis = axis or build_chart_time_axis(plot["date"], width, left, right)
    xs = axis.xs
    lo, hi = float(plot["low"].min()), float(plot["high"].max())
    for value, _, _ in levels:
        if lo * 0.8 <= value <= hi * 1.2:
            lo, hi = min(lo, value), max(hi, value)
    pad = (hi - lo) * 0.04 or 1.0
    lo, hi = lo - pad, hi + pad

    def sy(v: float) -> float:
        return top + (hi - v) / (hi - lo) * chh

    body_w = max(cw / len(plot) * 0.58, 1.2)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
             f'<text x="20" y="24" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
             f'<rect x="{left}" y="{top}" width="{cw}" height="{chh}" fill="#ffffff" stroke="#e5e7eb"/>']
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = top + frac * chh
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + cw}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="6" y="{y + 4:.1f}" font-size="11" fill="#6b7280">{hi - frac * (hi - lo):.0f}</text>')
    for x, (_, row) in zip(xs, plot.iterrows()):
        o, c, h, l = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])
        color = "#dc2626" if c >= o else "#16a34a"
        parts.append(f'<line x1="{x:.1f}" y1="{sy(h):.1f}" x2="{x:.1f}" y2="{sy(l):.1f}" stroke="{color}" stroke-width="1.1"/>')
        top_y = min(sy(o), sy(c))
        parts.append(f'<rect x="{x - body_w / 2:.2f}" y="{top_y:.2f}" width="{body_w:.2f}" height="{max(abs(sy(c) - sy(o)), 1.2):.2f}" fill="{color}" opacity="0.9"/>')
    for col, color in [(f"ma_{ma_periods[0]}", "#2563eb"), (f"ma_{ma_periods[1]}", "#f59e0b")]:
        pts = [f"{x:.1f},{sy(float(v)):.1f}" for x, v in zip(xs, plot[col]) if pd.notna(v)]
        if pts:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{" ".join(pts)}"/>')
    label_ys: List[float] = []
    for value, label, color in sorted(levels, key=lambda item: -item[0]):
        if not lo <= value <= hi:
            continue
        y = sy(value)
        ly = y + 4
        while any(abs(ly - other) < 13 for other in label_ys):
            ly += 13
        label_ys.append(ly)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + cw}" y2="{y:.1f}" stroke="{color}" stroke-width="1.2" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{left + cw + 6}" y="{ly:.1f}" font-size="11" fill="{color}">{escape(label)}</text>')
    last_x, last_close = xs[-1], float(plot["close"].iloc[-1])
    parts.append(f'<circle cx="{last_x:.1f}" cy="{sy(last_close):.1f}" r="3.5" fill="#111827"/>')
    for i in axis.ticks:
        label = plot.loc[i, "date"].strftime("%Y-%m") if i != len(plot) - 1 else plot.loc[i, "date"].strftime("%Y-%m-%d")
        parts.append(f'<text x="{xs[i]:.1f}" y="{height - 8}" font-size="11" fill="#6b7280" text-anchor="middle">{label}</text>')
    ly = 52
    parts.append(f'<rect x="{left}" y="{ly}" width="12" height="3" fill="#2563eb"/><text x="{left + 18}" y="{ly + 4}" font-size="12" fill="#374151">MA{ma_periods[0]}</text>')
    parts.append(f'<rect x="{left + 70}" y="{ly}" width="12" height="3" fill="#f59e0b"/><text x="{left + 88}" y="{ly + 4}" font-size="12" fill="#374151">MA{ma_periods[1]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_date_tick_positions_local(plot: pd.DataFrame, max_ticks: int = 8) -> List[int]:
    starts, last = [], None
    for i, dt in enumerate(plot["date"]):
        key = (dt.year, dt.month)
        if key != last:
            starts.append(i)
            last = key
    if len(starts) > max_ticks:
        idx = np.linspace(0, len(starts) - 1, max_ticks).astype(int)
        starts = [starts[i] for i in idx]
    return sorted(set(starts + [len(plot) - 1]))


MONTHLY_CHART_STARTS = {"000510": "2010-01-01"}

CHART_PERIODS = {
    "day": ("日", 250, (20, 60)),
    "week": ("周", 156, (20, 60)),
    "month": ("月", 120, (12, 36)),
}


def aggregate_ohlc(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    """Use actual last trading dates, including holiday-shortened W-SUN weeks."""
    if period not in ("week", "month"):
        raise ValueError("period must be week or month")
    frame = daily[["date", "open", "high", "low", "close"]].copy()
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame["period"] = frame["date"].dt.to_period("W-SUN" if period == "week" else "M")
    return frame.groupby("period", sort=True).agg(
        date=("date", "last"), open=("open", lambda values: values.iloc[0]),
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
    ).reset_index(drop=True)


def period_chart_frames(prices: pd.DataFrame,
                        monthly_prices: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Calculate on full history; only the rendering layer takes the last N bars."""
    daily = prices.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    weekly = compute_metrics(aggregate_ohlc(daily, "week"))
    signal_weekly = weekly_bars(daily)
    if weekly["date"].tolist() != signal_weekly["date"].tolist():
        raise ValueError("Weekly candles and signal dates differ")
    weekly["fr"] = signal_weekly["fr"].to_numpy()
    weekly["fr_bar"] = signal_weekly["bar"].to_numpy()
    monthly = compute_metrics(aggregate_ohlc(
        daily if monthly_prices is None else monthly_prices, "month"))
    for n in (12, 36):
        monthly[f"ma_{n}"] = monthly["close"].rolling(n).mean()
    return {"day": compute_metrics(daily), "week": weekly, "month": monthly}


def period_chart_pair(code: str, name: str, price: pd.DataFrame,
                      summary: Dict[str, Any], sig: Dict[str, Any],
                      monthly_prices: Optional[pd.DataFrame] = None) -> str:
    frames = period_chart_frames(price, monthly_prices)
    inputs, labels, panels = [], [], []
    for key, (unit, count, ma_periods) in CHART_PERIODS.items():
        disabled = key == "month" and len(frames[key]) < 48
        radio_id = f"period-{code}-{key}"
        inputs.append(f'<input class="period-toggle" type="radio" name="period-{code}" '
                      f'id="{radio_id}" value="{key}"'
                      + (' checked' if key == "week" else '')
                      + (' disabled' if disabled else '') + '>')
        labels.append(f'<label for="{radio_id}"'
                      + (' aria-disabled="true" title="历史数据不足"' if disabled else '')
                      + f'>{unit}K</label>')
        if disabled:
            continue
        frame = frames[key].tail(count).reset_index(drop=True)
        axis = build_chart_time_axis(frame["date"])
        how_k = [f"每根蜡烛是一{unit}：红色收盘比开盘高，绿色收盘比开盘低。",
                 f"蓝线是{ma_periods[0]}{unit}均线，橙线是{ma_periods[1]}{unit}均线，均线用完整历史计算。",
                 HOWTO_KLINE[2], HOWTO_KLINE[3]]
        if key == "week":
            how_fr = HOWTO_WEEKLY
            reading = read_weekly_fr(sig, frames[key])
        else:
            how_fr = [f"每根柱子代表一{unit}，与上面的{unit}K逐根对应。",
                      "蓝线是Fr，红柱表示Fr上升，绿柱表示Fr下降，虚线为0轴。",
                      "当前周期仅供观察；买卖信号始终使用已完成周线，不随视图切换。"]
            last = frame.iloc[-1] if len(frame) else None
            reading = (f"最新{unit}线Fr为 {last['fr']:+.4f}，BAR为 {last['fr_bar']:+.4f}。"
                       if last is not None else "暂无数据。")
            reading += "这里只展示当前周期动量，不产生买卖信号。"
        if key == "month":
            reading += "末月若尚未结束，K线和Fr均为截至最新交易日的暂定值。"
        panels.append(f'<div class="period-view period-{key}">'
                      f'<p class="muted small period-range">显示最近{count}{"个交易日" if key == "day" else unit}；'
                      f'实际 {len(frame)} 根，上下图共用时间轴。</p>'
                      '<div class="figure">'
                      f'<div class="chart">{kline_levels_chart(frame, name + " " + unit + "K与均线", price_levels(summary, sig), axis=axis, ma_periods=ma_periods)}</div>'
                      f'{howto_block(unit + "K", how_k, read_kline(summary, sig, frame, period=key))}</div>'
                      '<div class="figure">'
                      f'<div class="chart">{svg_fr_chart(frame, name + " " + unit + "线Fr趋势动量（每根柱子是一" + unit + "）", height=330, axis=axis)}</div>'
                      f'{howto_block(unit + "线Fr", how_fr, reading)}</div></div>')
    return (f'<div class="period-charts" role="group" aria-label="{escape(name)}图表周期">'
            + ''.join(inputs) + '<div class="period-tabs">' + ''.join(labels) + '</div>'
            + '<div class="period-views">' + ''.join(panels) + '</div></div>')


def weekly_fr_frame(prices: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly_bars(prices[["date", "close"]])
    return weekly.rename(columns={"bar": "fr_bar"})[["date", "fr", "fr_bar"]]


def read_kline(summary: Dict[str, Any], sig: Dict[str, Any], price: pd.DataFrame,
               period: str = "day") -> str:
    if price.empty:
        return "暂无数据。"
    unit, _, (fast, slow) = CHART_PERIODS[period]
    last = price.iloc[-1]
    close, ma20, ma60 = float(last["close"]), last.get(f"ma_{fast}"), last.get(f"ma_{slow}")
    if pd.notna(ma20) and pd.notna(ma60):
        if close < ma20 and close < ma60:
            trend = f"收盘 {close:.0f} 在{fast}{unit}线（{ma20:.0f}）和{slow}{unit}线（{ma60:.0f}）下方，短期和中期都在走弱。"
        elif close > ma20 and close > ma60:
            trend = f"收盘 {close:.0f} 在{fast}{unit}线（{ma20:.0f}）和{slow}{unit}线（{ma60:.0f}）上方，短期和中期都在走强。"
        else:
            trend = f"收盘 {close:.0f} 夹在{fast}{unit}线（{ma20:.0f}）和{slow}{unit}线（{ma60:.0f}）之间，处于震荡。"
    else:
        trend = f"收盘 {close:.0f}。"
    dd = summary.get("drawdown_250")
    high = summary.get("recent_high_250")
    text = trend
    if dd is not None and high:
        text += f"距近一年最高 {high:.0f} 下跌了 {-dd:.1%}"
        gap = 1 - high * 0.85 / close
        text += (f"，再跌 {gap:.1%} 碰到“回撤15%”线。" if gap > 0 else "，已经跌破“回撤15%”线。")
    if sig and sig.get("regime") == "down" and sig.get("ref_close"):
        ref = sig["ref_close"]
        if close > ref:
            text += f"蓝色虚线是抄底规则的参照低点 {ref:.0f}，周收盘跌破它才算“价新低”，还差 {1 - ref / close:.1%}。"
        else:
            text += f"周收盘已经低于参照低点 {ref:.0f}（价新低）。"
    return text


def read_weekly_fr(sig: Dict[str, Any], weekly: pd.DataFrame) -> str:
    if not sig or "error" in sig:
        return "周线信号暂不可用。"
    fr, bar = sig["fr"], sig["bar"]
    side = "0轴上方" if fr >= 0 else "0轴下方"
    color = "红柱" if bar >= 0 else "绿柱"
    since = f"，从 {sig['since']:%Y-%m-%d} 那周开始" if sig.get("since") is not None else ""
    recent = weekly["fr"].dropna().tail(4).tolist()
    trail = " → ".join(f"{v:+.4f}" for v in recent)
    text = (f"最新一周Fr是 {fr:+.4f}，在{side}，柱子是{color}，属于“{sig['state_label']}”。"
            f"现在是{'上涨' if sig['regime'] == 'up' else '下跌'}周期{since}。近4周Fr：{trail}。")
    if sig["regime"] == "up" and bar < 0:
        text += "Fr还在0轴上方但一直在往下走，如果跌破0轴就要卖出。"
    elif sig["regime"] == "down" and bar >= 0:
        text += "柱子转红说明下跌在减速，但只有Fr回到0轴上方才算进入上涨周期。"
    elif sig["regime"] == "down":
        text += "柱子是绿色，说明空方还在加强，继续等。"
    if not sig.get("week_complete", True):
        text += f"本周还没走完（数据截至 {sig['live_date']:%m-%d}），以周五收盘为准。"
    return text


def read_valuation(code: str, summary: Dict[str, Any], dividend: Dict[str, Any]) -> str:
    pct = summary.get("pe_percentile")
    years = summary["pe_observations"] / 243
    text = (f"最新滚动市盈率 {summary['pe_ttm']:.2f} 倍，比样本里 {pct:.0%} 的时间都高"
            if pct is not None else f"最新滚动市盈率 {summary['pe_ttm']:.2f} 倍")
    text += f"（样本从 {summary['pe_window_start']} 开始，约 {years:.1f} 年"
    text += "，偏短，参考价值有限）。" if years < 3 else "）。"
    if pct is not None:
        if pct > VALUATION_CAP_PCT:
            text += "已经进入最贵的一成区间，按规则最多持有目标仓位的 2/3。"
        elif pct >= 0.8:
            text += "偏贵，但还没到减仓线（90%）。"
        elif pct <= 0.2:
            text += "处在便宜的区间。"
        else:
            text += "处在中间区域。"
    if code == DIVIDEND_CODE and dividend.get("dividend_yield") is not None:
        text += (f"对红利指数更重要的是股息率：{dividend['dividend_yield']:.2f}%，"
                 f"比10年国债（{dividend['cn10y']:.2f}%）高 {dividend['spread']:.2f} 个百分点"
                 f"{'，相对债券很有吸引力' if dividend['spread'] >= 2.5 else ''}。")
    return text


HOWTO_KLINE = [
    "每根蜡烛是一天：红色收盘比开盘高，绿色收盘比开盘低。",
    "蓝线是20日均线（约一个月的平均价），橙线是60日均线（约三个月）。价格在两条线上方偏强，下方偏弱。",
    "灰色虚线是近一年最高价；橙色虚线是从那里下跌15%的位置；下跌周期里的蓝色虚线是抄底规则要跌破的“参照低点”。",
    "黑点是最新收盘价。",
]
HOWTO_WEEKLY = [
    "这张图是周线：每根柱子代表一周，比日线更慢、更稳，是买卖信号的依据。",
    "蓝线是Fr（多空力量）：在0轴上方是上涨周期，下方是下跌周期。跌破0轴就卖，重新站上0轴就买回。",
    "柱子是Fr比上一周的变化：红柱表示Fr在上升，绿柱表示Fr在下降。",
    "下跌周期里，如果价格跌破参照低点、Fr却没有更低（底背离），再出现第一根红柱，就是半仓抄底点。",
]
HOWTO_PE = [
    "彩色粗线是滚动市盈率（股价 ÷ 过去一年利润），越高越贵。",
    "三条横线是图中这段时间（约三年）的20%分位（绿）、中位数（灰）、80%分位（橙）：绿线以下算便宜，橙线以上算贵。",
    "估值只决定“最多买多少”，不决定什么时候买：分位超过90%时，最多只持有目标仓位的2/3。",
]


def howto_block(title: str, howto: List[str], reading: str) -> str:
    items = "".join(f"<li>{escape(item)}</li>" for item in howto)
    return (f'<div class="read"><div><h4>怎么看这张图</h4><ul>{items}</ul></div>'
            f'<div class="now"><h4>现在说明什么</h4><p>{escape(reading)}</p></div></div>')


def decision_box(decisions: Dict[str, Dict[str, Any]], cutoff: pd.Timestamp) -> str:
    acts = [code for code, d in decisions.items() if d["level"] == "act"]
    if acts:
        head = f"本周有 {len(acts)} 个新信号，需要操作"
        head_class = "act"
    else:
        head = "今天不需要操作"
        head_class = "calm"
    rows = []
    for code, meta in INDEXES.items():
        d = decisions[code]
        rows.append(f'<li class="lv-{d["level"]}"><span class="who">{escape(meta["name"])}</span>'
                    f'<span class="what">{escape(d["action"])}</span><span class="why">{escape(d["detail"])}</span></li>')
    return f"""
  <section class="panel decision">
    <span class="eyebrow">TODAY · {cutoff:%Y-%m-%d}</span>
    <h2 class="headline {head_class}">{escape(head)}</h2>
    <ul class="todo">{''.join(rows)}</ul>
    <p class="muted small">规则：周线Fr跌破0轴卖出，重新站上0轴买回；下跌周期里出现底背离拐点买半仓，价格和Fr同时再创新低就止损；估值进入最贵一成时最多持有2/3。信号以完整一周的收盘确认，下一个交易日执行。不是投资建议。</p>
  </section>"""


def progress_table(decisions: Dict[str, Dict[str, Any]], signals: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for code, meta in INDEXES.items():
        d, sig = decisions[code], signals.get(code) or {}
        target = d["target"]
        if not sig or "error" in sig:
            rows.append(f'<tr><td data-label="指数"><div><b>{escape(meta["name"])}</b></div></td><td colspan="4"><div class="muted">信号暂不可用</div></td></tr>')
            continue
        up = sig["regime"] == "up"
        factor = float(sig["factor"])
        if up:
            step1 = '<span class="done">已跳过</span><small>已进入上涨周期，直接满仓</small>'
            step2 = f'<span class="done">已触发</span><small>{sig["since"]:%Y-%m-%d} 那周周线Fr站上0轴</small>'
        else:
            if factor >= 0.5:
                step1 = f'<span class="done">已触发</span><small>{sig["since"]:%Y-%m-%d} 那周出现底背离拐点</small>'
            else:
                step1 = f'<span class="todo-pill">等待</span><small>{escape(sig.get("next", ""))}</small>'
            step2 = f'<span class="todo-pill">等待</span><small>等周线Fr从 {sig["fr"]:+.4f} 回到0轴以上</small>'
        cap = f'<small class="warn">{escape(d["cap"])}</small>' if d["cap"] else ""
        rows.append(f"""<tr>
          <td data-label="指数"><div><b>{escape(meta['name'])}</b><small>目标 {TARGET_WEIGHTS[code]:.0%}，例：{wan(target)}</small></div></td>
          <td data-label="周线周期"><div><span class="cycle {'up' if up else 'down'}">{'上涨周期' if up else '下跌周期'}</span><small>Fr {sig['fr']:+.4f}</small></div></td>
          <td data-label="第1步 半仓"><div>{step1}<small class="amt">例：{wan(target / 2)}</small></div></td>
          <td data-label="第2步 满仓"><div>{step2}<small class="amt">例：再买 {wan(target / 2)}</small></div></td>
          <td data-label="现在应持有"><div><b>{d['effective'] if d['effective'] is None else f"{d['effective']:.0%}"}</b><small>例：{wan(d['now'])}</small>{cap}</div></td>
        </tr>""")
    return f"""
  <section class="panel plan">
    <div class="section-heading"><div><span class="eyebrow">ENTRY PROGRESS</span><h2>分批进场进度</h2></div>
    <div class="status"><b>以股票资金100万为例</b><span>按你自己的资金等比例换算</span></div></div>
    <table class="plan-table">
      <thead><tr><th>指数</th><th>周线周期</th><th>第1步：半仓（底背离拐点）</th><th>第2步：满仓（周线Fr站上0轴）</th><th>现在应持有</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="muted small">目标比例：中证A500 50%、红利低波100 35%、科创50 15%。下跌周期里先等第1步；如果还没出现底背离就直接进入上涨周期，就跳过第1步一次买满。卖出不分步：周线Fr跌破0轴就全部卖出。</p>
  </section>"""


def index_section(code: str, meta: Dict[str, str], summary: Dict[str, Any], price: pd.DataFrame,
                  valuation: pd.DataFrame, sig: Dict[str, Any], decision: Dict[str, Any], dividend: Dict[str, Any],
                  monthly_prices: Optional[pd.DataFrame] = None) -> str:
    up = sig.get("regime") == "up" if sig else False
    pe_pct = summary.get("pe_percentile")
    pe_note = ""
    if summary["pe_lag_sessions"] > 0:
        pe_note = (f'<p class="inline-warning">估值数据截至 {summary["pe_date"]:%Y-%m-%d}，'
                   f'比行情晚 {summary["pe_lag_sessions"]} 个交易日。</p>')
    return f"""
  <section class="panel index-section" id="index-{code}">
    <div class="section-heading">
      <div><span class="eyebrow">{escape(meta['role'])}</span><h2>{escape(meta['name'])} <small>{code}</small></h2></div>
      <div class="status"><b><span class="cycle {'up' if up else 'down'}">{'上涨周期' if up else '下跌周期'}</span> {escape(decision['action'])}</b><span>按规则现在应持有 {'' if decision['effective'] is None else f"{decision['effective']:.0%}"}</span></div>
    </div>
    <div class="facts">
      <div><label>最新点位</label><b>{fmt_num(summary['latest_close'])}</b><small>{pct_cell(summary['return_1d'])} 当日</small></div>
      <div><label>距近一年最高</label><b>{fmt_pct(summary['drawdown_250'], 1)}</b><small>最高 {fmt_num(summary.get('recent_high_250'), 0)}</small></div>
      <div><label>周线Fr</label><b>{sig.get('fr', float('nan')):+.4f}</b><small>{escape(sig.get('state_label', ''))}</small></div>
      <div><label>估值分位</label><b>{fmt_pct(pe_pct, 0)}</b><small>滚动PE {fmt_num(summary['pe_ttm'])}</small></div>
    </div>
    {pe_note}
    {period_chart_pair(code, meta["name"], price, summary, sig, monthly_prices)}
    <div class="figure">
      <div class="chart">{pe_chart(valuation, meta['name'], meta['color'])}</div>
      {howto_block('估值', HOWTO_PE, read_valuation(code, summary, dividend))}
    </div>
    <details class="more"><summary>更多数据</summary>{detail_table(summary)}</details>
  </section>"""


def render_html(
    summaries: Dict[str, Dict[str, Any]],
    prices: Dict[str, pd.DataFrame],
    valuations: Dict[str, pd.DataFrame],
    turnover: pd.DataFrame,
    cutoff: pd.Timestamp,
    crosscheck: Dict[str, Any],
    generated_at: str,
    plans: Optional[Dict[str, Dict[str, Any]]] = None,
    dividend: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Dict[str, Any]]] = None,
    monthly_prices: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    dividend = dividend or {"available": False}
    if signals is None:
        signals = compute_weekly_signals(prices)
    decisions = {code: decide(code, summaries[code], signals.get(code) or {}, dividend) for code in INDEXES}
    lagged = [s for s in summaries.values() if s["pe_lag_sessions"] > 0]
    if lagged:
        lag_text = "；".join(f"{s['name']} PE 滞后 {s['pe_lag_sessions']} 个交易日" for s in lagged)
        valuation_notice = (f'<div class="data-warning"><b>估值日期提示：</b>{escape(lag_text)}。'
                            "估值只按各自标注日期使用，不会再把K线、均线、Fr或成交额裁回旧日。</div>")
    else:
        valuation_notice = ""
    sections = "".join(
        index_section(code, meta, summaries[code], prices[code], valuations[code], signals.get(code) or {}, decisions[code], dividend, (monthly_prices or {}).get(code))
        for code, meta in INDEXES.items()
    )
    crosscheck_text = ""
    if crosscheck.get("available"):
        difference = abs(float(turnover.iloc[-1]["total_amount"]) / float(crosscheck["official_total"]) - 1)
        crosscheck_text = f"成交额代理口径与交易所官方数据偏差 {difference:.2%}。"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>三指数投资决策报告 · {cutoff:%Y-%m-%d}</title>
<style>
:root{{--ink:#111827;--muted:#64748b;--line:#e2e8f0;--bg:#f1f5f9;--panel:#fff;--red:#dc2626;--green:#16803a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}}
.wrap{{max-width:1180px;margin:auto;padding:28px 18px 48px}} .hero{{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;border-radius:20px;padding:26px 30px;box-shadow:0 16px 40px rgba(15,23,42,.14)}}
.hero h1{{margin:3px 0 8px;font-size:30px;letter-spacing:.02em}} .hero p{{margin:0;color:#dbeafe}} .hero-meta{{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}} .hero-meta span{{background:rgba(255,255,255,.12);padding:6px 10px;border-radius:999px;font-size:13px}}
.data-warning{{margin:14px 0 0;padding:12px 15px;border-radius:12px;font-size:14px;border:1px solid #fbbf24;background:#fffbeb;color:#92400e}} .inline-warning{{border-left:3px solid #f59e0b;padding-left:9px;color:#92400e;font-size:14px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px;margin-top:18px;box-shadow:0 8px 26px rgba(15,23,42,.045)}} .section-heading{{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:14px}} h2{{margin:2px 0 0;font-size:23px}} h2 small{{font-size:14px;color:var(--muted);font-weight:500}} .eyebrow{{font-size:12px;color:#2563eb;font-weight:700;letter-spacing:.12em}}
.status{{text-align:right}} .status b,.status span{{display:block}} .status span{{font-size:13px;color:var(--muted)}} label{{color:var(--muted);font-size:13px}} .up{{color:var(--red);font-weight:700}} .down{{color:var(--green);font-weight:700}} .neutral{{color:var(--muted)}}
.decision .headline{{font-size:26px;margin:4px 0 12px}} .decision .headline.act{{color:#b45309}} .decision .headline.calm{{color:#1e3a8a}}
.todo{{list-style:none;margin:0 0 12px;padding:0;display:flex;flex-direction:column;gap:8px}} .todo li{{display:grid;grid-template-columns:110px 150px 1fr;gap:12px;align-items:baseline;padding:12px 14px;border-radius:12px;background:#f8fafc;border:1px solid var(--line)}}
.todo .who{{font-weight:700}} .todo .what{{font-weight:800;font-size:17px}} .todo .why{{color:#475569;font-size:14px}}
.todo li.lv-act{{background:#fff7ed;border-color:#fdba74}} .todo li.lv-act .what{{color:#b45309}} .todo li.lv-hold .what{{color:#b91c1c}} .todo li.lv-wait .what{{color:#475569}}
.cycle{{display:inline-block;border-radius:999px;padding:1px 9px;font-size:13px;font-weight:700}} .status span.cycle{{display:inline-block;margin-right:6px}} .cycle.up{{background:#fef2f2;color:#b91c1c}} .cycle.down{{background:#f0fdf4;color:#15803d}}
.plan-table{{width:100%;border-collapse:collapse;font-size:14px}} .plan-table th{{text-align:left;color:#475569;font-weight:600;padding:9px 10px;border-bottom:2px solid var(--line)}} .plan-table td{{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}} .plan-table small{{display:block;color:var(--muted);font-size:12px;margin-top:2px}} .plan-table small.amt{{color:#1e3a8a}} .plan-table small.warn{{color:#b45309}}
.done{{display:inline-block;background:#fef2f2;color:#b91c1c;border-radius:8px;padding:0 8px;font-size:13px;font-weight:700}} .todo-pill{{display:inline-block;background:#f1f5f9;color:#475569;border-radius:8px;padding:0 8px;font-size:13px;font-weight:700}}
.facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:6px}} .facts div{{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:10px 12px;display:flex;flex-direction:column}} .facts b{{font-size:20px;font-variant-numeric:tabular-nums}} .facts small{{color:var(--muted);font-size:12px}}
.figure{{margin-top:16px;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fff}} .chart{{padding:8px;overflow:hidden}} .chart svg{{display:block;width:100%;height:auto;min-width:640px}}
.read{{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--line)}} .read>div{{padding:12px 16px}} .read h4{{margin:0 0 6px;font-size:14px}} .read ul{{margin:0;padding-left:1.1em;color:#475569;font-size:13.5px;display:flex;flex-direction:column;gap:3px}} .read .now{{background:#f8fafc;border-left:1px solid var(--line)}} .read .now p{{margin:0;font-size:14px;color:#1f2937}}
details.more{{margin-top:14px}} details.more summary{{cursor:pointer;color:#2563eb;font-size:14px}} .kv{{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}} .kv th,.kv td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}} .kv th{{width:43%;color:#475569;font-weight:600}} .kv td{{font-variant-numeric:tabular-nums}}
.method{{font-size:14px;color:#475569}} .method li{{margin:6px 0}} .method summary{{cursor:pointer;font-weight:700;color:#111827}} .footer{{color:#64748b;font-size:12px;margin:20px 4px 0;text-align:center}} .muted{{color:var(--muted)}} .small{{font-size:12px}}
@media(max-width:820px){{.plan .section-heading{{flex-direction:column;align-items:flex-start}} .plan .status{{text-align:left}} .plan-table thead{{display:none}} .plan-table tr{{display:block;border-bottom:1px solid var(--line);padding:8px 0}} .plan-table td{{display:grid;grid-template-columns:92px 1fr;gap:10px;border:0;padding:5px 2px}} .plan-table td::before{{content:attr(data-label);color:var(--muted);font-size:12px;padding-top:2px}}
 .todo li{{grid-template-columns:1fr;gap:2px}} .facts{{grid-template-columns:1fr 1fr}} .read{{grid-template-columns:1fr}} .read .now{{border-left:0;border-top:1px solid var(--line)}} .hero{{padding:22px}} .hero h1{{font-size:25px}} .chart{{overflow-x:auto}} .chart svg{{width:760px;max-width:none}} .section-heading{{align-items:flex-start;flex-direction:column}} .status{{text-align:left}}}}
@media(max-width:480px){{.wrap{{padding:12px 10px 30px}} .hero,.panel{{border-radius:14px}} .panel{{padding:14px}} h2{{font-size:20px}} .decision .headline{{font-size:22px}}}}

.period-charts{{position:relative;margin-top:18px}} .period-tabs{{display:inline-flex;border:1px solid #cbd5e1;border-radius:9px;overflow:hidden}} .period-tabs label{{padding:8px 22px;cursor:pointer;background:#f8fafc;color:#475569;font-weight:600}} .period-tabs label+label{{border-left:1px solid #cbd5e1}} .period-tabs label[aria-disabled="true"]{{color:#94a3b8;background:#f1f5f9;cursor:not-allowed}}
.period-toggle{{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}} .period-view{{display:none}}
.period-toggle[value="day"]:checked ~ .period-views .period-day{{display:block}}
.period-toggle[value="day"]:checked ~ .period-tabs label[for$="-day"]{{background:#2563eb;color:white}}
.period-toggle[value="day"]:focus-visible ~ .period-tabs label[for$="-day"]{{outline:2px solid #111827;outline-offset:-3px}}
.period-toggle[value="week"]:checked ~ .period-views .period-week{{display:block}}
.period-toggle[value="week"]:checked ~ .period-tabs label[for$="-week"]{{background:#2563eb;color:white}}
.period-toggle[value="week"]:focus-visible ~ .period-tabs label[for$="-week"]{{outline:2px solid #111827;outline-offset:-3px}}
.period-toggle[value="month"]:checked ~ .period-views .period-month{{display:block}}
.period-toggle[value="month"]:checked ~ .period-tabs label[for$="-month"]{{background:#2563eb;color:white}}
.period-toggle[value="month"]:focus-visible ~ .period-tabs label[for$="-month"]{{outline:2px solid #111827;outline-offset:-3px}}
</style>
</head>
<body><main class="wrap">
  <header class="hero">
    <span class="eyebrow" style="color:#bfdbfe">DECISION REPORT</span>
    <h1>三指数投资决策报告</h1>
    <p>中证A500 · 红利低波100 · 科创50｜先看今天该做什么，再看每只指数的三张关键图</p>
    <div class="hero-meta"><span>统一数据截止：{cutoff:%Y-%m-%d}（价格、趋势与成交）</span><span>生成：{escape(generated_at)}</span></div>
  </header>
  {valuation_notice}
  {decision_box(decisions, cutoff)}
  {progress_table(decisions, signals)}
  {sections}
  <section class="panel method">
    <details><summary>规则、数据口径与限制</summary>
    <ul>
      <li><b>买卖规则：</b>周线取每周最后一个交易日收盘。周线Fr = (EMA12 − EMA26) ÷ EMA5；跌破0轴卖出，站上0轴买回；下跌周期里价格跌破参照低点（前后各4周内最低的周收盘）而Fr没有更低，出现第一根红柱时买半仓；之后价格和Fr同时再创新低则止损。规则写在 signal_rules.py。</li>
      <li><b>回测依据：</b>沪深300（2005–2026）最大回撤由一直持有的 −72% 降到 −41%；三指数组合（2020–2026）由 −31% 降到 −22%，年化收益相近。底背离信号历史样本很少（四只指数共9次），所以只买半仓。回测不含分红和申赎费用以外的成本。</li>
      <li><b>估值：</b>中证指数官网日频滚动市盈率，分位只从指数正式发布后算起；中证A500发布于2024年9月，样本短。红利低波100另看股息率减10年国债的利差。</li>
      <li><b>数据：</b>中证指数官网日行情；成交额为上证综指加深证综指的代理口径，只用于检查数据是否更新。{escape(crosscheck_text)}</li>
      <li><b>金额：</b>页面中的金额按股票资金100万举例，请按自己的资金等比例换算。</li>
    </ul>
    </details>
  </section>
  <footer class="footer">本页是按预设规则生成的参考，不是投资建议。数据可能因供应方修订而变化。</footer>
</main></body></html>"""


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"输出目录非空，为避免覆盖已停止：{path}")
    path.mkdir(parents=True, exist_ok=True)


def index_start_date(code: str, start: str) -> str:
    """Use the earlier of the CLI start date and the index's own history start."""

    return min(pd.Timestamp(start), pd.Timestamp(INDEXES[code]["history_start"])).strftime("%Y-%m-%d")


def run(
    output_dir: Path,
    start: str,
    now: datetime,
    history_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ensure_empty_output(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir()
    session = direct_session()
    now_cn = normalize_shanghai_now(now)
    request_end = now_cn.strftime("%Y-%m-%d")
    completed_limit = cutoff_for_completed_day(now_cn)
    generated_at = now_cn.strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")

    raw_prices: Dict[str, pd.DataFrame] = {}
    valuations: Dict[str, pd.DataFrame] = {}
    monthly_prices: Dict[str, pd.DataFrame] = {}
    for code in INDEXES:
        signal_start = index_start_date(code, start)
        # A single chunked fetch serves both views. Extra history is display-only:
        # the original dates still feed signals, summaries and valuation rankings.
        fetch_start = min(signal_start, MONTHLY_CHART_STARTS.get(code, signal_start))
        full_price, full_valuation = fetch_csi_index_data(session, code, fetch_start, request_end)
        monthly_prices[code] = full_price
        raw_prices[code] = full_price[full_price["date"] >= pd.Timestamp(signal_start)].copy()
        valuations[code] = full_valuation[full_valuation["date"] >= pd.Timestamp(signal_start)].copy()
        time.sleep(0.35)
    shanghai = fetch_tencent_kline(session, "sh000001", start, request_end)
    shenzhen = fetch_tencent_kline(session, "sz399106", start, request_end)

    source_frames: Dict[str, pd.DataFrame] = {
        **{
            f"price_{code}": frame
            for code, frame in raw_prices.items()
        },
        **{
            f"valuation_{code}": frame
            for code, frame in valuations.items()
        },
        "turnover_shanghai": shanghai,
        "turnover_shenzhen": shenzhen,
    }
    source_latest_dates: Dict[str, pd.Timestamp] = {
        name: latest_frame_date(frame, name)
        for name, frame in source_frames.items()
    }
    source_completed_dates: Dict[str, pd.Timestamp] = {
        name: latest_frame_date(frame, name, completed_limit)
        for name, frame in source_frames.items()
    }
    calendar_date, calendar_status = latest_trade_calendar_day(completed_limit)
    freshness = resolve_data_freshness(
        source_completed_dates,
        completed_limit,
        calendar_date,
        calendar_status,
    )
    freshness["source_latest_dates"] = source_latest_dates
    freshness["generated_at"] = generated_at
    freshness_path = output_dir / "source_freshness.json"
    freshness_path.write_text(
        json.dumps(json_ready(freshness), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("数据源日期检查：" + json.dumps(json_ready(freshness), ensure_ascii=False))
    if not freshness["freshness_resolved"]:
        raise RuntimeError(
            freshness["freshness_error"] + "详见 source_freshness.json。"
        )
    if not freshness["all_required_sources_fresh"]:
        stale_text = "；".join(
            f"{name}={pd.Timestamp(date).strftime('%Y-%m-%d')}"
            for name, date in freshness["stale_required_sources"].items()
        )
        expected_text = freshness["expected_trading_day"].strftime("%Y-%m-%d")
        raise RuntimeError(
            f"必需行情数据尚未全部更新到最近完整交易日 {expected_text}：{stale_text}。"
            "已停止部署旧报告，详见 source_freshness.json。"
        )
    cutoff = freshness["expected_trading_day"]

    prices: Dict[str, pd.DataFrame] = {}
    for code, frame in raw_prices.items():
        trimmed = frame[frame["date"] <= cutoff].copy().reset_index(drop=True)
        prices[code] = compute_metrics(trimmed)
        trimmed.to_csv(data_dir / f"{code}_price_daily.csv", index=False, encoding="utf-8-sig")
        monthly_prices[code] = monthly_prices[code][monthly_prices[code]["date"] <= cutoff].copy()
        monthly_prices[code].to_csv(data_dir / f"{code}_chart_price_daily.csv", index=False, encoding="utf-8-sig")
    for code, frame in valuations.items():
        valuations[code] = frame[frame["date"] <= cutoff].copy().reset_index(drop=True)
        if valuations[code].empty:
            raise RuntimeError(f"{INDEXES[code]['name']} 在 {cutoff:%Y-%m-%d} 前没有可用估值数据")
        valuations[code].to_csv(data_dir / f"{code}_valuation_daily.csv", index=False, encoding="utf-8-sig")
    shanghai = shanghai[shanghai["date"] <= cutoff]
    shenzhen = shenzhen[shenzhen["date"] <= cutoff]
    turnover = build_turnover(shanghai, shenzhen)
    turnover.to_csv(data_dir / "market_turnover_proxy_daily.csv", index=False, encoding="utf-8-sig")

    summaries = {code: make_summary(code, prices[code], valuations[code]) for code in INDEXES}
    freshness["valuation_dates"] = {
        code: summaries[code]["pe_date"] for code in INDEXES
    }
    freshness["valuation_lag_sessions"] = {
        code: summaries[code]["pe_lag_sessions"] for code in INDEXES
    }
    freshness_path.write_text(
        json.dumps(json_ready(freshness), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    crosscheck = official_turnover_crosscheck(cutoff)
    if crosscheck.get("available"):
        proxy = float(turnover.iloc[-1]["total_amount"])
        crosscheck["proxy_total"] = proxy
        crosscheck["absolute_relative_difference"] = abs(proxy / float(crosscheck["official_total"]) - 1)
    dividend = update_dividend_history(history_dir, cutoff)
    plans = {code: investment_plan(summaries[code], dividend) for code in INDEXES}
    signals = compute_weekly_signals(prices)
    for code, sig in signals.items():
        summaries[code]["weekly_regime"] = sig.get("regime")
        summaries[code]["weekly_factor"] = sig.get("factor")
        summaries[code]["weekly_stage"] = sig.get("stage")
    if history_dir is not None:
        daily_path = history_dir / DAILY_HISTORY_FILE
        daily = upsert_rows(
            read_history(daily_path),
            daily_history_rows(cutoff, summaries, plans, dividend),
            ["date", "code"],
        )
        write_history(daily_path, daily, ["date", "code"])
    report = render_html(
        summaries,
        prices,
        valuations,
        turnover,
        cutoff,
        crosscheck,
        generated_at,
        plans=plans,
        dividend=dividend,
        signals=signals,
        monthly_prices=monthly_prices,
    )
    (output_dir / "report.html").write_text(report, encoding="utf-8")

    snapshot = {
        "generated_at": generated_at,
        "data_cutoff": cutoff,
        "expected_trading_day": cutoff,
        "source_latest_dates": source_latest_dates,
        "valuation_dates": freshness["valuation_dates"],
        "freshness": freshness,
        "requested_start": start,
        "index_start_dates": {code: index_start_date(code, start) for code in INDEXES},
        "indices": summaries,
        "investment_plan": plans,
        "weekly_signals": signals,
        "dividend": dividend,
        "market_turnover": {
            "latest_total": float(turnover.iloc[-1]["total_amount"]),
            "latest_ma20": finite_or_none(turnover.iloc[-1]["ma20"]),
            "ratio_to_ma20": finite_or_none(turnover.iloc[-1]["ratio_ma20"]),
        },
        "official_crosscheck": crosscheck,
        "sources": {
            "price": "https://www.csindex.com.cn/csindex-home/perf/index-perf (CSI official)",
            "turnover_proxy": "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get (Tencent)",
            "valuation": CSI_PERF_URL,
            "turnover_method": "Shanghai Composite amount + Shenzhen Composite amount",
        },
    }
    (output_dir / "data_snapshot.json").write_text(
        json.dumps(json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# 三指数长期资金监控报告

- 数据截止：{cutoff.strftime('%Y-%m-%d')}
- 指数：中证A500（000510）、科创50（000688）、红利低波100（930955）
- 主报告：report.html
- 数据快照：data_snapshot.json
- 明细数据：data/ 目录
- 生成脚本：{Path(__file__).name}

报告中的价格、趋势与成交额使用已完整收盘的共同交易日，盘中数据会自动排除；估值按各指数页面标注的日期使用。页面为单文件自包含 HTML，可直接在手机或电脑浏览器打开。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return json_ready(snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成中证A500、科创50、红利低波100统一监控报告")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START,
        help="抓数起始日；若某指数在 INDEXES 中设置了更早的 history_start，则以更早者为准",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="历史数据目录（股息率与每日快照会追加到这里的 CSV）；不传则不保存历史",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.output_dir,
        arguments.start_date,
        datetime.now(SHANGHAI_TZ),
        history_dir=arguments.history_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
