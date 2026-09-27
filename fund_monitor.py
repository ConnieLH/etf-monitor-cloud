#!/usr/bin/env python3
"""
指数/基金 择时分析工具 — 基于 AKShare 数据，生成 HTML 可视化报告。

支持两种模式：
    - index 模式（默认）：分析 A 股指数，数据源为指数日 K 线（含 OHLCV）
    - fund  模式：分析开放式基金，数据源为累计净值走势（仅有每日净值，无 OHLCV）

数据口径说明：
    index 模式 — close/open/high/low 为指数真实的日 K 线价格
    fund  模式 — close 取自"累计净值"，open/high/low 均等于 close（基金无盘中价格）
                  因此 K 线图退化为净值走势线图，Fr/BAR 等指标仍基于 close 计算

运行方式：
    python3 fund_monitor.py [参数]

参数说明：
    --mode         分析模式：index（指数，默认）或 fund（基金）
    --code         代码，指数如 000001（上证指数）、000905（中证500）；基金如 007028
    --name         中文名称，用于报告标题显示，默认"上证指数"
    --start-date   分析起始日期，格式 YYYY-MM-DD，默认 2024-01-01
    --end-date     分析截止日期，格式 YYYY-MM-DD，默认取最新数据
    --output-dir   自定义输出目录，默认 output/{code}/

示例：
    # 分析上证指数（默认）
    python3 fund_monitor.py

    # 分析中证500，从2023年开始
    python3 fund_monitor.py --code 000905 --name 中证500 --start-date 2023-01-01

    # 分析基金007028
    python3 fund_monitor.py --mode fund --code 007028 --name 易方达中短期美元债A

    # 自定义输出目录
    python fund_monitor.py --mode fund --code 007028 --name 易方达中短期美元债A --output-dir reports/my_report

输出：
    - 控制台打印 JSON 格式的指标摘要
    - 在输出目录下生成 report.html（含走势图/K线图、Fr指标图、择时信号等）

依赖：
    pip install akshare numpy pandas
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Alert:
    level: str
    title: str
    detail: str


INDEX_SECID_MAP = {
    "000001": "1.000001",  # 上证指数
    "000905": "1.000905",  # 中证500
}


def fetch_index_kline(
    index_code: str,
    start_date: str = "2019-01-01",
    end_date: str = "2099-12-31",
) -> pd.DataFrame:
    import akshare as ak  # 延迟导入：报告脚本只用本文件的指标与画图函数

    last_err: Optional[Exception] = None
    raw_df: Optional[pd.DataFrame] = None

    primary_fetchers = [
        lambda: ak.index_zh_a_hist(
            symbol=index_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        ),
        lambda: ak.stock_zh_index_hist_csindex(
            symbol=index_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        ),
    ]

    for fetcher in primary_fetchers:
        for _ in range(2):
            try:
                raw_df = fetcher()
                if raw_df is not None and not raw_df.empty:
                    break
            except Exception as err:
                last_err = err
        if raw_df is not None and not raw_df.empty:
            break

    if raw_df is None:
        raise RuntimeError(f"AKShare 抓取指数 {index_code} K线失败：{last_err}")
    if raw_df.empty:
        raise RuntimeError(f"AKShare 未返回指数 {index_code} 的K线数据")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "成交金额": "amount",
        "振幅": "amplitude_pct",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover_pct",
    }
    df = raw_df.rename(columns=rename_map).copy()
    needed_cols = ["date", "open", "close", "high", "low", "volume", "amount"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"AKShare 返回字段缺失：{missing}")

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "amplitude_pct", "pct_change", "change", "turnover_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_fund_nav(
    fund_code: str,
    start_date: str = "2019-01-01",
    end_date: str = "2099-12-31",
) -> pd.DataFrame:
    """获取开放式基金累计净值走势。

    数据口径：
        - 调用 ak.fund_open_fund_info_em()，indicator="累计净值走势"
        - 返回字段：净值日期、累计净值
        - 累计净值已包含分红再投资，适合用于计算真实收益率
        - 基金无盘中价格，open/high/low 均填充为与 close（累计净值）相同的值
        - volume/amount 填 0（基金无成交量概念）
    """
    import akshare as ak

    last_err: Optional[Exception] = None
    raw_df: Optional[pd.DataFrame] = None
    for attempt in range(2):
        try:
            raw_df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
            break
        except Exception as err:
            last_err = err
    if raw_df is None:
        raise RuntimeError(f"AKShare 抓取基金 {fund_code} 净值失败：{last_err}")
    if raw_df.empty:
        raise RuntimeError(f"AKShare 未返回基金 {fund_code} 的净值数据")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw_df["净值日期"])
    df["close"] = pd.to_numeric(raw_df["累计净值"], errors="coerce")
    # 基金无盘中价格，open/high/low 统一取累计净值
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 0
    df["amount"] = 0

    # 按日期筛选
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    for n in [5, 20, 60, 120, 250]:
        out[f"ret_{n}"] = out["close"].pct_change(n)
    for n in [20, 60, 120, 250]:
        out[f"ma_{n}"] = out["close"].rolling(n).mean()

    out["cum_max"] = out["close"].cummax()
    out["drawdown"] = out["close"] / out["cum_max"] - 1
    out["vol_20"] = out["ret_1d"].rolling(20).std() * np.sqrt(250)
    out["vol_60"] = out["ret_1d"].rolling(60).std() * np.sqrt(250)

    out["ema_5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["ema_10"] = out["close"].ewm(span=10, adjust=False).mean()
    ema_12 = out["close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["close"].ewm(span=26, adjust=False).mean()
    out["fr1"] = ema_12 - ema_26
    out["fr"] = out["fr1"] / out["ema_5"]
    out["fr_bar"] = (out["fr"] - out["fr"].shift(1)) * 3
    return out


def latest_non_nan(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def fmt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v * 100:.{digits}f}%"


def fmt_num(v: Optional[float], digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:.{digits}f}"


def trailing_return(df: pd.DataFrame, trading_days: int) -> Optional[float]:
    if len(df) <= trading_days:
        return None
    return float(df["close"].iloc[-1] / df["close"].iloc[-trading_days - 1] - 1)


def range_position(df: pd.DataFrame, trading_days: int) -> Optional[float]:
    if len(df) < 2:
        return None
    tail = df.tail(min(trading_days, len(df)))
    low = float(tail["close"].min())
    high = float(tail["close"].max())
    current = float(tail["close"].iloc[-1])
    if high == low:
        return 1.0
    return (current - low) / (high - low)


def max_drawdown_info(df: pd.DataFrame) -> Dict[str, object]:
    idx = df["drawdown"].idxmin()
    mdd = float(df.loc[idx, "drawdown"])
    trough_date = df.loc[idx, "date"]
    peak_idx = df.loc[:idx, "close"].idxmax()
    peak_date = df.loc[peak_idx, "date"]
    return {
        "max_drawdown": mdd,
        "peak_date": peak_date.strftime("%Y-%m-%d"),
        "trough_date": trough_date.strftime("%Y-%m-%d"),
    }


def timing_signal(df: pd.DataFrame) -> Tuple[str, str]:
    last = df.iloc[-1]
    close = float(last["close"])
    ma20 = last.get("ma_20")
    ma60 = last.get("ma_60")
    fr = last.get("fr")
    fr_bar = last.get("fr_bar")
    dd = float(last["drawdown"])
    pos_1y = range_position(df, 252)

    if pd.notna(ma60) and close > float(ma60) and pd.notna(fr) and pd.notna(fr_bar) and float(fr) > 0 and float(fr_bar) > 0:
        return (
            "偏右侧买入/继续持有区",
            f"指数站上MA60（{fmt_num(float(ma60))}），Fr与BAR同步转强，趋势型资金可偏向继续持有或回踩分批买入。",
        )

    if dd <= -0.15 and pos_1y is not None and pos_1y <= 0.35 and pd.notna(fr_bar) and float(fr_bar) > 0:
        return (
            "偏左侧分批布局区",
            f"距历史高位回撤较深（{fmt_pct(dd)}），且近1年位置偏低（{fmt_pct(pos_1y)}），若BAR开始修复，可考虑分批左侧布局。",
        )

    if pos_1y is not None and pos_1y >= 0.9 and pd.notna(fr_bar) and float(fr_bar) < 0 and pd.notna(ma20) and close < float(ma20):
        return (
            "偏减仓/止盈观察区",
            f"指数处于近1年高位区（{fmt_pct(pos_1y)}），BAR回落且跌破MA20，适合观察是否分批止盈。",
        )

    return (
        "中性观察区",
        "当前位置更像震荡整理，适合继续观察是否重返MA60、以及Fr/BAR是否出现明确共振信号，再决定加仓或减仓。",
    )


def build_alerts(df: pd.DataFrame) -> List[Alert]:
    alerts: List[Alert] = []
    last = df.iloc[-1]
    current_dd = float(last["drawdown"])
    close = float(last["close"])
    ma20 = last.get("ma_20")
    ma60 = last.get("ma_60")
    ret20 = last.get("ret_20")
    ret60 = last.get("ret_60")
    vol20 = last.get("vol_20")
    vol60 = last.get("vol_60")
    fr = last.get("fr")
    fr_bar = last.get("fr_bar")

    if pd.notna(ma60) and close < float(ma60):
        alerts.append(Alert("warning", "跌破60日均线", f"当前收盘 {fmt_num(close)} 低于 MA60 {fmt_num(float(ma60))}"))
    else:
        alerts.append(Alert("good", "站上60日均线", f"当前收盘 {fmt_num(close)} 高于 MA60 {fmt_num(float(ma60))}" if pd.notna(ma60) else "MA60 数据不足"))

    if current_dd <= -0.15:
        alerts.append(Alert("warning", "回撤较深", f"当前距历史高点回撤 {fmt_pct(current_dd)}，更偏左侧观察区"))
    elif current_dd <= -0.08:
        alerts.append(Alert("info", "存在中等回撤", f"当前距历史高点回撤 {fmt_pct(current_dd)}"))
    else:
        alerts.append(Alert("good", "回撤可控", f"当前距历史高点回撤 {fmt_pct(current_dd)}"))

    if pd.notna(fr) and pd.notna(fr_bar):
        if float(fr) > 0 and float(fr_bar) > 0:
            alerts.append(Alert("good", "Fr 共振转强", f"Fr {fmt_num(float(fr), 4)}，BAR {fmt_num(float(fr_bar), 4)}，趋势偏强"))
        elif float(fr_bar) < 0:
            alerts.append(Alert("warning", "BAR回落", f"Fr {fmt_num(float(fr), 4)}，BAR {fmt_num(float(fr_bar), 4)}，短线动能走弱"))
        else:
            alerts.append(Alert("info", "Fr中性", f"Fr {fmt_num(float(fr), 4)}，BAR {fmt_num(float(fr_bar), 4)}"))

    if pd.notna(vol20) and pd.notna(vol60) and float(vol20) > float(vol60) * 1.25:
        alerts.append(Alert("warning", "短期波动放大", f"20日年化波动 {fmt_pct(float(vol20))}，高于60日年化波动 {fmt_pct(float(vol60))}"))
    else:
        alerts.append(Alert("info", "波动可控", f"20日/60日年化波动 {fmt_pct(float(vol20)) if pd.notna(vol20) else '--'} / {fmt_pct(float(vol60)) if pd.notna(vol60) else '--'}"))

    if pd.notna(ret20) and pd.notna(ret60):
        alerts.append(Alert("info", "短中期收益", f"近20日 {fmt_pct(float(ret20))}；近60日 {fmt_pct(float(ret60))}"))

    return alerts


def summarize(df: pd.DataFrame, index_code: str, index_name: str, mode: str = "index") -> Dict[str, object]:
    last = df.iloc[-1]
    first = df.iloc[0]
    mdd = max_drawdown_info(df)
    signal_title, signal_detail = timing_signal(df)
    summary = {
        "index_code": index_code,
        "index_name": index_name,
        "start_date": first["date"].strftime("%Y-%m-%d"),
        "end_date": last["date"].strftime("%Y-%m-%d"),
        "latest_close": float(last["close"]),
        "latest_open": float(last["open"]),
        "latest_high": float(last["high"]),
        "latest_low": float(last["low"]),
        "since_start_return": float(last["close"] / first["close"] - 1),
        "return_1m": trailing_return(df, 21),
        "return_3m": trailing_return(df, 63),
        "return_6m": trailing_return(df, 126),
        "return_1y": trailing_return(df, 252),
        "return_3y": trailing_return(df, 756),
        "vol_20": latest_non_nan(df["vol_20"]),
        "vol_60": latest_non_nan(df["vol_60"]),
        "current_drawdown": float(last["drawdown"]),
        "max_drawdown": mdd["max_drawdown"],
        "mdd_peak_date": mdd["peak_date"],
        "mdd_trough_date": mdd["trough_date"],
        "ma_20": latest_non_nan(df["ma_20"]),
        "ma_60": latest_non_nan(df["ma_60"]),
        "ma_120": latest_non_nan(df["ma_120"]),
        "ma_250": latest_non_nan(df["ma_250"]),
        "range_pos_1y": range_position(df, 252),
        "range_pos_3y": range_position(df, 756),
        "fr_latest": latest_non_nan(df["fr"]),
        "fr_bar_latest": latest_non_nan(df["fr_bar"]),
        "signal_title": signal_title,
        "signal_detail": signal_detail,
        "alerts": [a.__dict__ for a in build_alerts(df)],
        "mode": mode,
    }
    return summary


def build_date_tick_positions(plot_df: pd.DataFrame, max_ticks: int = 8) -> List[int]:
    if plot_df.empty:
        return []
    month_starts: List[int] = []
    last_key = None
    for i, dt in enumerate(plot_df["date"]):
        key = (dt.year, dt.month)
        if key != last_key:
            month_starts.append(i)
            last_key = key
    if not month_starts:
        return [0, len(plot_df) - 1]
    if len(month_starts) > max_ticks:
        idxs = np.linspace(0, len(month_starts) - 1, max_ticks).astype(int)
        month_starts = [month_starts[i] for i in idxs]
    last = len(plot_df) - 1
    # drop a month tick that would collide with the final date label
    month_starts = [i for i in month_starts if last - i >= max(len(plot_df) * 0.08, 1) or i == last]
    picks = sorted(set(month_starts + [last]))
    return picks


@dataclass(frozen=True)
class ChartTimeAxis:
    """One date sequence and plot geometry shared by an aligned chart pair."""

    dates: Tuple[pd.Timestamp, ...]
    xs: Tuple[float, ...]
    ticks: Tuple[int, ...]
    width: int
    left: int
    right: int


def build_chart_time_axis(dates, width: int = 1060, left: int = 58,
                          right: int = 118) -> ChartTimeAxis:
    dates = tuple(pd.Timestamp(value) for value in dates)
    frame = pd.DataFrame({"date": dates})
    return ChartTimeAxis(dates, tuple(np.linspace(left, width - right, len(dates))),
                         tuple(build_date_tick_positions(frame)), width, left, right)


def validate_chart_time_axis(frame: pd.DataFrame, axis: ChartTimeAxis) -> None:
    if tuple(pd.Timestamp(value) for value in frame["date"]) != axis.dates:
        raise ValueError("Chart dates must exactly match the shared time axis")


def svg_kline_chart(df: pd.DataFrame, title: str, mode: str = "index", width: int = 980, height: int = 360) -> str:
    plot_df = df.tail(220).reset_index(drop=True)
    if len(plot_df) < 20:
        return f"<div>图表 {title} 无足够数据</div>"

    left_pad, right_pad, top_pad, bottom_pad = 58, 20, 34, 34
    chart_w = width - left_pad - right_pad
    chart_h = height - top_pad - bottom_pad
    x_vals = np.linspace(left_pad, left_pad + chart_w, len(plot_df))

    low_min = float(plot_df["low"].min())
    high_max = float(plot_df["high"].max())
    if low_min == high_max:
        low_min -= 1.0
        high_max += 1.0

    def scale_y(v: float) -> float:
        return top_pad + (high_max - v) / (high_max - low_min) * chart_h

    body_w = max(chart_w / len(plot_df) * 0.58, 1.2)
    ma20 = plot_df["ma_20"].dropna().reset_index()
    ma60 = plot_df["ma_60"].dropna().reset_index()

    def line_points(series_df: pd.DataFrame) -> str:
        pts = []
        for _, row in series_df.iterrows():
            idx = int(row["index"])
            pts.append(f"{x_vals[idx]:.1f},{scale_y(float(row.iloc[1])):.1f}")
        return " ".join(pts)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="20" y="24" font-size="16" font-weight="700" fill="#111827">{title}</text>',
        f'<rect x="{left_pad}" y="{top_pad}" width="{chart_w}" height="{chart_h}" fill="#ffffff" stroke="#e5e7eb"/>',
    ]

    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = top_pad + frac * chart_h
        val = high_max - frac * (high_max - low_min)
        parts.append(f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + chart_w}" y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(f'<text x="6" y="{y+4:.1f}" font-size="11" fill="#6b7280">{val:.2f}</text>')

    if mode == "fund":
        # 基金模式：净值走势线图（无 OHLCV，K线无意义）
        nav_points = " ".join(
            f"{x:.1f},{scale_y(float(row['close'])):.1f}"
            for x, (_, row) in zip(x_vals, plot_df.iterrows())
        )
        parts.append(f'<polyline fill="none" stroke="#6366f1" stroke-width="2" points="{nav_points}"/>')
    else:
        # 指数模式：标准 K 线图
        for x, (_, row) in zip(x_vals, plot_df.iterrows()):
            open_, close_, high_, low_ = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])
            color = "#dc2626" if close_ >= open_ else "#16a34a"
            parts.append(f'<line x1="{x:.1f}" y1="{scale_y(high_):.1f}" x2="{x:.1f}" y2="{scale_y(low_):.1f}" stroke="{color}" stroke-width="1.1"/>')
            top = min(scale_y(open_), scale_y(close_))
            body_h = max(abs(scale_y(close_) - scale_y(open_)), 1.2)
            parts.append(f'<rect x="{x - body_w/2:.2f}" y="{top:.2f}" width="{body_w:.2f}" height="{body_h:.2f}" fill="{color}" opacity="0.9"/>')

    if not ma20.empty:
        parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="1.7" points="{line_points(ma20)}"/>')
    if not ma60.empty:
        parts.append(f'<polyline fill="none" stroke="#f59e0b" stroke-width="1.7" points="{line_points(ma60)}"/>')

    label_idx = build_date_tick_positions(plot_df, max_ticks=8)
    for i in label_idx:
        x = x_vals[i]
        label = plot_df.loc[i, "date"].strftime("%Y-%m") if i != len(plot_df) - 1 else plot_df.loc[i, "date"].strftime("%Y-%m-%d")
        parts.append(f'<line x1="{x:.1f}" y1="{top_pad + chart_h}" x2="{x:.1f}" y2="{top_pad + chart_h + 4}" stroke="#9ca3af" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-8}" font-size="11" fill="#6b7280" text-anchor="middle">{label}</text>')

    ly = 52
    parts.append(f'<rect x="{left_pad}" y="{ly}" width="12" height="3" fill="#2563eb"/>')
    parts.append(f'<text x="{left_pad + 18}" y="{ly+4}" font-size="12" fill="#374151">MA20</text>')
    parts.append(f'<rect x="{left_pad + 80}" y="{ly}" width="12" height="3" fill="#f59e0b"/>')
    parts.append(f'<text x="{left_pad + 98}" y="{ly+4}" font-size="12" fill="#374151">MA60</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_fr_chart(df: pd.DataFrame, title: str, width: int = 980, height: int = 320,
                 *, axis: Optional[ChartTimeAxis] = None) -> str:
    # Keep every supplied date in aligned mode, including an initial NaN BAR.
    plot_df = (df if axis is not None else df.dropna(subset=["fr", "fr_bar"]).tail(220)).reset_index(drop=True)
    if axis is not None:
        validate_chart_time_axis(plot_df, axis)
        width = axis.width
    if len(plot_df) < 20:
        return f"<div>图表 {title} 无足够数据</div>"

    left_pad, right_pad, top_pad, bottom_pad = (axis.left if axis else 58), (axis.right if axis else 20), 30, 34
    chart_w = width - left_pad - right_pad
    chart_h = height - top_pad - bottom_pad
    xs = axis.xs if axis else np.linspace(left_pad, left_pad + chart_w, len(plot_df))

    fr_vals = plot_df["fr"].astype(float).tolist()
    bar_vals = plot_df["fr_bar"].astype(float).tolist()
    finite = [v for v in fr_vals + bar_vals if math.isfinite(v)] + [0.0]
    ymin, ymax = min(finite), max(finite)
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0

    def scale_y(v: float) -> float:
        return top_pad + (ymax - v) / (ymax - ymin) * chart_h

    zero_y = scale_y(0.0)
    bar_w = max(chart_w / len(plot_df) * 0.72, 1.0)
    fr_points = " ".join(f"{x:.1f},{scale_y(v):.1f}" for x, v in zip(xs, fr_vals) if math.isfinite(v))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="20" y="24" font-size="16" font-weight="700" fill="#111827">{title}</text>',
        f'<rect x="{left_pad}" y="{top_pad}" width="{chart_w}" height="{chart_h}" fill="#ffffff" stroke="#e5e7eb"/>',
    ]

    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = top_pad + frac * chart_h
        val = ymax - frac * (ymax - ymin)
        parts.append(f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + chart_w}" y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(f'<text x="6" y="{y+4:.1f}" font-size="11" fill="#6b7280">{val:.3f}</text>')

    parts.append(f'<line x1="{left_pad}" y1="{zero_y:.1f}" x2="{left_pad + chart_w}" y2="{zero_y:.1f}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 4"/>')

    for x, v in zip(xs, bar_vals):
        if not math.isfinite(v):
            continue
        y = scale_y(v)
        color = "#dc2626" if v >= 0 else "#16a34a"
        top = min(y, zero_y)
        h = max(abs(zero_y - y), 1.0)
        parts.append(f'<rect x="{x - bar_w/2:.2f}" y="{top:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" opacity="0.75"/>')

    parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="1.6" points="{fr_points}"/>')

    label_idx = axis.ticks if axis else build_date_tick_positions(plot_df, max_ticks=8)
    for i in label_idx:
        x = xs[i]
        label = plot_df.loc[i, "date"].strftime("%Y-%m") if i != len(plot_df) - 1 else plot_df.loc[i, "date"].strftime("%Y-%m-%d")
        parts.append(f'<line x1="{x:.1f}" y1="{top_pad + chart_h}" x2="{x:.1f}" y2="{top_pad + chart_h + 4}" stroke="#9ca3af" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-8}" font-size="11" fill="#6b7280" text-anchor="middle">{label}</text>')

    ly = 52
    parts.append(f'<rect x="{left_pad}" y="{ly}" width="12" height="3" fill="#2563eb"/>')
    parts.append(f'<text x="{left_pad + 18}" y="{ly+4}" font-size="12" fill="#374151">Fr</text>')
    parts.append(f'<rect x="{left_pad + 70}" y="{ly-4}" width="12" height="12" fill="#dc2626" opacity="0.75"/>')
    parts.append(f'<text x="{left_pad + 88}" y="{ly+4}" font-size="12" fill="#374151">BAR增量</text>')
    parts.append(f'<rect x="{left_pad + 170}" y="{ly-4}" width="12" height="12" fill="#16a34a" opacity="0.75"/>')
    parts.append(f'<text x="{left_pad + 188}" y="{ly+4}" font-size="12" fill="#374151">BAR减量</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_table(rows: List[Tuple[str, str]]) -> str:
    trs = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="kv">{trs}</table>'


def render_html(summary: Dict[str, object], df: pd.DataFrame) -> str:
    mode = summary.get("mode", "index")
    is_fund = mode == "fund"
    type_label = "基金" if is_fund else "指数"

    overview_rows = [
        (type_label, f"{summary['index_name']}（{summary['index_code']}）"),
        ("分析区间", f"{summary['start_date']} ~ {summary['end_date']}"),
    ]
    if is_fund:
        # 基金只有累计净值，不展示开高低
        overview_rows.append(("最新累计净值", fmt_num(summary['latest_close'], 4)))
    else:
        overview_rows.append(("最新开/高/低/收", f"{fmt_num(summary['latest_open'])} / {fmt_num(summary['latest_high'])} / {fmt_num(summary['latest_low'])} / {fmt_num(summary['latest_close'])}"))
    overview_rows += [
        ("区间累计收益", fmt_pct(summary["since_start_return"])),
        ("近1月 / 3月 / 6月 / 1年 / 3年", " / ".join([
            fmt_pct(summary['return_1m']),
            fmt_pct(summary['return_3m']),
            fmt_pct(summary['return_6m']),
            fmt_pct(summary['return_1y']),
            fmt_pct(summary['return_3y']),
        ])),
        ("当前回撤 / 最大回撤", f"{fmt_pct(summary['current_drawdown'])} / {fmt_pct(summary['max_drawdown'])}"),
        ("最大回撤区间", f"{summary['mdd_peak_date']} → {summary['mdd_trough_date']}"),
        ("MA20 / MA60 / MA120 / MA250", f"{fmt_num(summary['ma_20'])} / {fmt_num(summary['ma_60'])} / {fmt_num(summary['ma_120'])} / {fmt_num(summary['ma_250'])}"),
        ("20日 / 60日年化波动", f"{fmt_pct(summary['vol_20'])} / {fmt_pct(summary['vol_60'])}"),
        ("近1年 / 近3年区间位置", f"{fmt_pct(summary['range_pos_1y'])} / {fmt_pct(summary['range_pos_3y'])}"),
        ("Fr / BAR 最新值", f"{fmt_num(summary['fr_latest'], 4)} / {fmt_num(summary['fr_bar_latest'], 4)}"),
    ]

    alert_html = "".join(
        f'<div class="alert {a["level"]}"><strong>{a["title"]}</strong><span>{a["detail"]}</span></div>'
        for a in summary["alerts"]
    )

    if is_fund:
        kline_chart = svg_kline_chart(df, f"{summary['index_name']} 累计净值走势 + 均线（近220日）", mode="fund")
    else:
        kline_chart = svg_kline_chart(df, f"{summary['index_name']} 日K + 均线（近220个交易日）", mode="index")
    fr_chart = svg_fr_chart(df, "Fr 指标 + BAR（近220日）")

    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{summary['index_name']} 择时观察</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
.container {{ max-width: 1120px; margin: 0 auto; }}
h1 {{ margin-bottom: 8px; }}
.sub {{ color: #6b7280; margin-bottom: 20px; }}
.card {{ background: white; border-radius: 14px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 2px 10px rgba(15,23,42,0.06); }}
.kv {{ width: 100%; border-collapse: collapse; }}
.kv th, .kv td {{ padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }}
.kv th {{ width: 260px; color: #374151; font-weight: 600; }}
.alerts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }}
.alert {{ border-radius: 12px; padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }}
.alert.good {{ background: #ecfdf5; color: #065f46; }}
.alert.info {{ background: #eff6ff; color: #1e3a8a; }}
.alert.warning {{ background: #fff7ed; color: #9a3412; }}
.signal {{ border-left: 5px solid #2563eb; background: #f8fbff; }}
.note {{ color: #4b5563; line-height: 1.7; }}
.footer {{ color: #6b7280; font-size: 12px; margin-top: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>{summary['index_name']}（{summary['index_code']}）择时观察</h1>
  <div class="sub">基于 AKShare {'基金累计净值' if is_fund else '指数日K线'}数据自动生成 | 模式：{type_label}</div>

  <div class="card signal">
    <h2>当前结论</h2>
    <p><strong>{summary['signal_title']}</strong></p>
    <p>{summary['signal_detail']}</p>
  </div>

  <div class="card">
    <h2>概览</h2>
    {html_table(overview_rows)}
  </div>

  <div class="card">
    <h2>监控提示</h2>
    <div class="alerts">{alert_html}</div>
  </div>

  <div class="card">{kline_chart}</div>
  <div class="card">{fr_chart}</div>

  <div class="card note">
    <h2>数据口径与使用说明</h2>
    {"<p><strong>基金模式口径：</strong>close 取自累计净值（含分红再投资），无盘中价格，open/high/low 均等于 close，K线图退化为净值走势线图。均线、Fr/BAR、波动率、回撤等指标均基于累计净值计算。</p>" if is_fund else "<p><strong>指数模式口径：</strong>数据为指数真实日K线（OHLCV），所有指标基于收盘价计算。</p>"}
    <p>1）核心目标是辅助判断当前更偏买入、持有还是减仓观察。</p>
    <p>2）上图看价格/净值结构和趋势位置（MA20/MA60）；下图看动能变化（Fr + BAR）。</p>
    <p>3）当前信号不代表机械交易指令，而是用于识别：趋势右侧区、左侧分批区、以及高位减仓观察区。</p>
  </div>

  <div class="footer">生成日期：{pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M UTC')}</div>
</div>
</body>
</html>
"""


def save_output(output_dir: Path, html: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="指数/基金 择时分析工具（HTML报告）")
    parser.add_argument("--mode", default="index", choices=["index", "fund"],
                        help="分析模式：index=指数（默认），fund=基金")
    parser.add_argument("--code", default="000001",
                        help="代码。指数如 000001（上证指数）、000905（中证500）；基金如 007028")
    parser.add_argument("--name", default=None,
                        help="中文名称，用于报告标题。默认：指数模式为'上证指数'，基金模式为基金代码")
    parser.add_argument("--start-date", default="2024-01-01", help="开始日期，YYYY-MM-DD")
    parser.add_argument("--end-date", default="2099-12-31", help="结束日期，YYYY-MM-DD")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 output/{code}/")
    args = parser.parse_args()

    if args.name is None:
        args.name = "上证指数" if args.mode == "index" and args.code == "000001" else args.code

    outdir = Path(args.output_dir) if args.output_dir else Path("output") / args.code

    if args.mode == "fund":
        df = fetch_fund_nav(args.code, args.start_date, args.end_date)
    else:
        df = fetch_index_kline(args.code, args.start_date, args.end_date)

    df = compute_metrics(df)
    summary = summarize(df, args.code, args.name, mode=args.mode)
    html = render_html(summary, df)
    output_path = save_output(outdir, html)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n输出HTML：{output_path.resolve()}")


if __name__ == "__main__":
    main()
