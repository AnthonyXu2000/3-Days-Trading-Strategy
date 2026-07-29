"""
短线反弹策略 - 每日盘后选股扫描脚本
====================================
数据源: Yahoo Finance chart API (直接用 requests 调用)，免费，无需 API Key
标的池: 标普500 + 纳斯达克100 成分股并集
用法: 收盘后运行 `python scan.py`，输出当日(T日)满足全部6条筛选条件的候选标的

依赖安装:
    pip install pandas numpy lxml requests

建议用 cron / Windows 计划任务 在每天美股收盘后 30-60 分钟自动运行本脚本
(建议延迟一点，避免数据源当天数据还没完全更新)
"""

import io
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests


warnings.filterwarnings("ignore")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"




# ========== 可调参数（先固定下来，方便回测；回测完再微调）==========
CONFIG = {
    "ema_lookback_days": 10,       # 均线"右高左左"对比基准 N（对比 N 天前的自己）
    "pullback_window": 10,         # 回调幅度观察窗口（交易日，T-10 到 T-1）
    "pullback_min_pct": 0.10,      # 最小回调幅度 10%
    "max_gain_pct": 0.08,          # 建仓日涨幅上限 8%
    "min_price": 20,               # 最低股价 $20
    "min_avg_volume": 5_000_000,   # 最近5日平均成交量下限
    "volume_window": 5,            # 成交量平均窗口
    "history_period": "2y",        # 下载历史数据长度（保证 EMA120 有足够 warm-up）
    "min_history_days": 300,       # 数据不足这个天数的标的直接跳过（EMA120 warm-up 不够）
}


def get_index_constituents():
    """从维基百科抓取标普500和纳指100成分股，取并集。

    注意：维基百科表格结构偶尔会变化，如果抓取失败，
    可以把 sp500 / nasdaq100 换成手动维护的静态列表作为 fallback。
    """
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    nasdaq100_url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
    # 维基百科会对无 User-Agent 的请求返回 403，需要伪装成浏览器请求
    headers = {"User-Agent": USER_AGENT}

    sp500_html = requests.get(sp500_url, headers=headers, timeout=30).text
    sp500 = pd.read_html(io.StringIO(sp500_html))[0]["Symbol"].tolist()

    nasdaq100_html = requests.get(nasdaq100_url, headers=headers, timeout=30).text
    nasdaq100 = None
    for t in pd.read_html(io.StringIO(nasdaq100_html)):
        if "Ticker" in t.columns:
            nasdaq100 = t["Ticker"].tolist()
            break
        if "Symbol" in t.columns:
            nasdaq100 = t["Symbol"].tolist()
            break

    universe = sorted(set(sp500) | set(nasdaq100 or []))
    # Yahoo Finance 符号格式差异处理，如 BRK.B -> BRK-B
    universe = [s.replace(".", "-") for s in universe]
    return universe


def compute_signals(df, cfg):
    """
    对单只标的的历史数据 (df: 按日期升序的 Open/High/Low/Close/Volume)
    判断最新一个交易日(T日 = 信号日)是否满足全部6条筛选条件。
    满足则返回结果 dict，否则返回 None。
    """
    if df is None or len(df) < cfg["min_history_days"]:
        return None

    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["EMA60"] = df["Close"].ewm(span=60, adjust=False).mean()
    df["EMA120"] = df["Close"].ewm(span=120, adjust=False).mean()

    N = cfg["ema_lookback_days"]
    if len(df) < 120 + N + cfg["pullback_window"] + 2:
        return None

    T_date = df.index[-1]      # 建仓日 / 信号日
    T1_date = df.index[-2]     # 前一交易日 T-1

    # ---- 条件1: 均线多头排列（用 T-1 收盘的 EMA 值判断）----
    ema30_t1 = df.loc[T1_date, "EMA30"]
    ema60_t1 = df.loc[T1_date, "EMA60"]
    ema120_t1 = df.loc[T1_date, "EMA120"]
    if not (ema30_t1 > ema60_t1 > ema120_t1):
        return None

    # "右高左低"：三条均线当前值都要高于 N 天前的自己
    idx_t1 = df.index.get_loc(T1_date)
    idx_t1_minus_n = idx_t1 - N
    if idx_t1_minus_n < 0:
        return None
    ema30_t1_n = df["EMA30"].iloc[idx_t1_minus_n]
    ema60_t1_n = df["EMA60"].iloc[idx_t1_minus_n]
    ema120_t1_n = df["EMA120"].iloc[idx_t1_minus_n]
    if not (ema30_t1 > ema30_t1_n and ema60_t1 > ema60_t1_n and ema120_t1 > ema120_t1_n):
        return None

    # ---- 条件2: 回调幅度（T-10 到 T-1 窗口，用最高价/最低价）----
    window = df.iloc[-(cfg["pullback_window"] + 1) : -1]
    if len(window) < cfg["pullback_window"]:
        return None
    hi_val = window["High"].max()
    lo_val = window["Low"].min()
    hi_pos = int(np.argmax(window["High"].values))
    lo_pos = int(np.argmin(window["Low"].values))
    pullback_pct = (hi_val - lo_val) / hi_val
    if pullback_pct < cfg["pullback_min_pct"]:
        return None
    if not (hi_pos < lo_pos):   # 高点必须严格早于低点
        return None

    # ---- 条件3: 建仓当天(T日)K线形态 ----
    open_t = df.loc[T_date, "Open"]
    close_t = df.loc[T_date, "Close"]
    low_t = df.loc[T_date, "Low"]
    close_t1 = df.loc[T1_date, "Close"]
    ema10_t = df.loc[T_date, "EMA10"]

    if not (close_t > open_t):
        return None
    day_gain = (close_t - close_t1) / close_t1
    if not (day_gain < cfg["max_gain_pct"]):
        return None
    if not (low_t < ema10_t):
        return None
    if not (close_t > ema10_t):
        return None

    # ---- 条件4: 价格门槛 ----
    if not (close_t > cfg["min_price"]):
        return None

    # ---- 条件5: 流动性门槛 ----
    avg_vol = df["Volume"].iloc[-cfg["volume_window"] :].mean()
    if not (avg_vol >= cfg["min_avg_volume"]):
        return None

    return {
        "signal_date": T_date.strftime("%Y-%m-%d"),
        "close": round(float(close_t), 2),
        "day_gain_pct": round(float(day_gain) * 100, 2),
        "pullback_pct": round(float(pullback_pct) * 100, 2),
        "avg_volume_5d": int(avg_vol),
        "T_low_stop_ref": round(float(low_t), 2),  # 次日买入的止损参照价(锚定T日最低价)
        "ema30": round(float(ema30_t1), 2),
        "ema60": round(float(ema60_t1), 2),
        "ema120": round(float(ema120_t1), 2),
    }

def download_history(session, ticker, cfg):
    """通过 Yahoo Finance chart API 直接用 requests 下载单只标的的历史日线数据。"""
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": cfg["history_period"], "interval": "1d"}
    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    result = (resp.json().get("chart") or {}).get("result")
    if not result:
        return None

    result = result[0]
    ts = result.get("timestamp")
    if not ts:
        return None
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s"),
    )
    df.index = df.index.tz_localize(None).normalize()
    return df.dropna(subset=["Close"])


def scan_universe(universe, cfg, max_workers=16):
    results = []
    print(f"标的池共 {len(universe)} 只，开始下载历史数据...")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(download_history, session, ticker, cfg): ticker for ticker in universe
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            done += 1
            if done % 100 == 0:
                print(f"  已处理 {done}/{len(universe)} ...")

            try:
                df = future.result()
                sig = compute_signals(df, cfg)
                if sig:
                    sig["ticker"] = ticker
                    results.append(sig)
            except Exception:
                continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    universe = get_index_constituents()
    df_result = scan_universe(universe, CONFIG)

    today_str = datetime.now().strftime("%Y-%m-%d")
    if df_result.empty:
        print(f"\n[{today_str}] 今日没有标的满足全部筛选条件。")
    else:
        cols = [
            "ticker", "signal_date", "close", "day_gain_pct",
            "pullback_pct", "avg_volume_5d", "T_low_stop_ref",
        ]
        df_result = df_result[cols].sort_values("pullback_pct", ascending=False)
        print(f"\n[{today_str}] 找到 {len(df_result)} 个候选标的：\n")
        print(df_result.to_string(index=False))

        out_file = f"candidates_{today_str}.csv"
        df_result.to_csv(out_file, index=False)
        print(f"\n结果已保存到 {out_file}")
        print("提醒：T_low_stop_ref 是这只标的的止损锚定价（信号日最低价）。")
        print("次日开盘价若已跌破该价，本次信号作废，不要买入。")
