"""Weekly Fr risk switch and bottom-divergence buy rule.

Rules (validated in backtest_portfolio / backtest_book on CSI 300 since 2005
and the three-index portfolio since 2020):

* weekly bar  = the last trading day's close of each calendar week;
* Fr / BAR    = fund_monitor.compute_metrics on weekly closes
                (Fr = (EMA12 - EMA26) / EMA5, BAR = (Fr - Fr[-1]) * 3);
* up cycle    : weekly Fr >= 0 -> hold the full target weight;
* down cycle  : weekly Fr < 0  -> sell (exit immediately, the book's
                "sell actively" rule);
* bottom buy  : inside a down cycle, the weekly close breaks below the last
                confirmed swing low (lowest close within +-4 weeks) while Fr
                stays above that low's Fr, then BAR turns from green to red
                -> buy half;
* stop        : after a bottom buy, price and Fr both make new lows -> sell it.

Signals are confirmed only by a completed week. A week still in progress is
reported as provisional and never changes the confirmed state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fund_monitor import compute_metrics

SWING = 4
WARMUP_WEEKS = 30
DIVERGENCE_FACTOR = 0.5


def weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly closes (last trading day of each calendar week) with Fr/BAR."""

    frame = daily[["date", "close"]].dropna().sort_values("date")
    frame = frame.assign(week=frame["date"].dt.to_period("W-SUN"))
    weekly = frame.groupby("week", sort=True).tail(1).reset_index(drop=True)
    metrics = compute_metrics(weekly[["close"]].reset_index(drop=True))
    weekly["fr"] = metrics["fr"].to_numpy()
    weekly["bar"] = metrics["fr_bar"].to_numpy()
    return weekly[["date", "week", "close", "fr", "bar"]]


def week_is_complete(last_date: pd.Timestamp, trade_dates: Optional[pd.DatetimeIndex]) -> bool:
    """True when no later trading day exists in the same calendar week."""

    last_date = pd.Timestamp(last_date).normalize()
    week_end = last_date + pd.Timedelta(days=6 - last_date.weekday())
    if trade_dates is not None and len(trade_dates):
        later = trade_dates[(trade_dates > last_date) & (trade_dates <= week_end)]
        return len(later) == 0
    return last_date.weekday() >= 4


@dataclass
class SignalState:
    regime: Optional[str] = None            # "up" / "down"
    factor: float = 1.0                     # 1 full, 0.5 bottom-buy half, 0 out
    since: Optional[pd.Timestamp] = None    # week the current factor started
    reason: str = ""
    swing_lows: List[int] = field(default_factory=list)
    armed: Optional[Dict[str, int]] = None
    hold: Optional[Dict[str, float]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)


def replay(weekly: pd.DataFrame) -> SignalState:
    """Run the rules over completed weekly bars and return the final state."""

    close = weekly["close"].to_numpy(dtype=float)
    fr = weekly["fr"].to_numpy(dtype=float)
    bar = weekly["bar"].to_numpy(dtype=float)
    dates = list(weekly["date"])
    st = SignalState()

    def change(i: int, factor: float, reason: str) -> None:
        st.factor, st.since, st.reason = factor, dates[i], reason
        st.events.append({"date": dates[i], "factor": factor, "reason": reason})

    for i in range(len(weekly)):
        if i < WARMUP_WEEKS or np.isnan(fr[i]):
            continue
        j = i - SWING                       # a swing low is known SWING weeks after it
        if j >= SWING and close[j] == close[j - SWING:i + 1].min():
            st.swing_lows.append(j)
        regime = "up" if fr[i] >= 0 else "down"
        if st.regime is None:
            st.regime = regime
            change(i, 1.0 if regime == "up" else 0.0, "周线Fr在0轴上方" if regime == "up" else "周线Fr在0轴下方")
            continue
        if regime != st.regime:
            st.regime, st.armed, st.hold = regime, None, None
            if regime == "up":
                change(i, 1.0, "周线Fr上穿0轴，进入上涨周期")
            else:
                change(i, 0.0, "周线Fr跌破0轴，进入下跌周期")
            continue
        if regime != "down":
            continue
        if st.hold is not None:
            if close[i] < st.hold["close"] and fr[i] < st.hold["fr"]:
                st.hold = None
                change(i, 0.0, "抄底后价格和Fr同时再创新低，止损")
            continue
        if not st.swing_lows:
            continue
        ref = st.swing_lows[-1]
        if st.armed is None:
            # price new low + force weakening: arm, then wait for the turn
            if close[i] < close[ref] and fr[i] > fr[ref]:
                st.armed = {"ref": ref, "low": i}
            continue
        if close[i] < close[st.armed["low"]]:
            st.armed["low"] = i
        if fr[i] < fr[st.armed["ref"]]:
            st.armed = None                 # decline re-accelerated: wait for a new swing low
            continue
        if bar[i] > 0 and bar[i - 1] <= 0:
            low = st.armed["low"]
            st.hold = {"close": float(close[low]), "fr": float(fr[low])}
            st.armed = None
            change(i, DIVERGENCE_FACTOR, "底背离拐点：价新低、Fr未新低、BAR转红，买入半仓")
    return st


def fr_state_label(fr: float, bar: float) -> str:
    """The book's four market states from Fr position and BAR colour."""

    if fr >= 0:
        return "极强，持股待涨" if bar >= 0 else "强市中的回踩或震荡"
    return "弱市中的反弹或震荡" if bar >= 0 else "极弱，谨慎做多"


def signal_status(daily: pd.DataFrame, trade_dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """Current weekly signal for one index, ready for the report."""

    weekly = weekly_bars(daily)
    complete = week_is_complete(weekly["date"].iloc[-1], trade_dates)
    confirmed = weekly if complete else weekly.iloc[:-1]
    st = replay(confirmed.reset_index(drop=True))
    last = confirmed.iloc[-1]
    now = weekly.iloc[-1]
    status: Dict[str, Any] = {
        "week_date": pd.Timestamp(last["date"]),
        "week_complete": complete,
        "live_date": pd.Timestamp(now["date"]),
        "live_fr": float(now["fr"]),
        "live_bar": float(now["bar"]),
        "fr": float(last["fr"]),
        "bar": float(last["bar"]),
        "regime": st.regime,
        "factor": st.factor,
        "since": st.since,
        "reason": st.reason,
        "state_label": fr_state_label(float(last["fr"]), float(last["bar"])),
        "new_signal": bool(st.events and pd.Timestamp(st.events[-1]["date"]) == pd.Timestamp(last["date"])
                           and len(st.events) > 1),
        "events": [{"date": pd.Timestamp(e["date"]), "factor": e["factor"], "reason": e["reason"]} for e in st.events[-6:]],
    }
    closes = confirmed["close"].to_numpy(dtype=float)
    frs = confirmed["fr"].to_numpy(dtype=float)
    if st.regime == "down":
        if st.hold is not None:
            status["next"] = (f"持有抄底半仓；若周收盘跌破 {st.hold['close']:.0f} 且Fr低于 {st.hold['fr']:.4f}，止损")
            status["stage"] = "抄底半仓持有中"
        elif st.swing_lows:
            ref = st.swing_lows[-1]
            ref_close, ref_fr = float(closes[ref]), float(frs[ref])
            status.update({"ref_date": pd.Timestamp(confirmed["date"].iloc[ref]), "ref_close": ref_close, "ref_fr": ref_fr})
            new_low = closes[-1] < ref_close
            weaker = frs[-1] > ref_fr
            if st.armed is not None:
                status["stage"] = "价新低、力衰减已满足"
                status["next"] = "等周线BAR由绿转红，出现即买入半仓"
            elif not new_low:
                gap = 1 - ref_close / closes[-1]
                status["stage"] = "等待价新低"
                status["next"] = (f"再跌 {gap:.1%} 跌破 {ref_close:.0f}（{status['ref_date']:%Y-%m-%d} 低点），"
                                  f"同时Fr保持在 {ref_fr:.4f} 以上，再等BAR转红")
            elif not weaker:
                status["stage"] = "空方仍在加强"
                status["next"] = "价格已创新低，但Fr也创新低；等下一次反弹形成新的参照低点"
            else:
                status["stage"] = "价新低、力衰减已满足"
                status["next"] = "等周线BAR由绿转红，出现即买入半仓"
        else:
            status["stage"] = "等待形成参照低点"
            status["next"] = "等一次反弹后形成周线波段低点"
        status["exit_hint"] = "等周线Fr重新上穿0轴后全部买回"
    else:
        status["stage"] = "上涨周期"
        status["next"] = "持有；周线Fr收在0轴下方即卖出"
        status["exit_hint"] = f"Fr距0轴 {float(last['fr']):+.4f}"
    return status
