#!/usr/bin/env python3
"""睡眠報告：技師判讀 vs YASA 自動判期，輸出單一 HTML 檔。

用法：
    python3 sleep_report.py                                   # 自動挑 data/ 裡有判讀檔的第一個
    python3 sleep_report.py --psg data/NW_GT028_P2_28C_170725H-28.edf
    python3 sleep_report.py --psg data/SC4001E0-PSG.edf --out report_sc4001.html

內容：hypnogram、各期時間與佔比（圓餅）、睡眠效率、入睡潛伏期、REM 潛伏期、
覺醒次數與時間、各期 EEG 功率頻譜。所有圖以 base64 內嵌，HTML 可以單獨帶走。

技師判讀來源同 staging.py：EDF 旁的同名 RemLogic .txt，或 Sleep-EDF 的 *-Hypnogram.edf。
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
from scipy.signal import welch               # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staging                               # noqa: E402
from readers import Recording                # noqa: E402

for _f in ("PingFang HK", "Heiti TC", "Arial Unicode MS"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

EPOCH = 30.0
STAGES = ["W", "N1", "N2", "N3", "REM"]
SLEEP = {"N1", "N2", "N3", "N4", "REM"}
COLOR = {"W": "#e0a300", "N1": "#3fa7d6", "N2": "#2563c9", "N3": "#6d3fd6",
         "N4": "#5227a8", "REM": "#e04f4f"}
PLOT_ORDER = ["W", "REM", "N1", "N2", "N3"]
BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("sigma", 12, 14), ("beta", 13, 32)]
INK, MUTED, LINE = "#1c2530", "#6b7785", "#e3e8ef"


# ---------------------------------------------------------------- 指標


def norm(stages):
    """把 N4 併進 N3（AASM），其餘原樣。"""
    return [("N3" if s == "N4" else s) for s in stages]


def metrics(stages, lights_off=None, lights_on=None):
    """標準睡眠參數。單位：分鐘。

    TIB 有 Lights Off/On 就用它，否則用整段記錄。
    SOL 從 TIB 起點算到第一個睡眠 epoch；REM 潛伏期從**入睡**算到第一個 REM
    （AASM 慣例，不是從熄燈算）。WASO 只算入睡之後、最後一次睡著之前的清醒。
    """
    s = norm(stages)
    n = len(s)
    lo = 0 if lights_off is None else max(0, min(n, lights_off))
    hi = n if lights_on is None else max(lo + 1, min(n, lights_on))
    win = s[lo:hi]

    sleep_idx = [i for i, x in enumerate(win) if x in SLEEP]
    m = dict(TIB=(hi - lo) * EPOCH / 60, TST=0.0, SE=np.nan, SOL=np.nan,
             REM_lat=np.nan, WASO=0.0, n_awake=0, SPT=np.nan,
             stage_min={k: 0.0 for k in STAGES}, stage_pct={k: 0.0 for k in STAGES},
             lights=(lo, hi))
    for x in win:
        if x in m["stage_min"]:
            m["stage_min"][x] += EPOCH / 60
    if not sleep_idx:
        return m

    on, off = sleep_idx[0], sleep_idx[-1]
    m["TST"] = len(sleep_idx) * EPOCH / 60
    m["SPT"] = (off - on + 1) * EPOCH / 60
    m["SE"] = m["TST"] / m["TIB"] * 100 if m["TIB"] else np.nan
    m["SOL"] = on * EPOCH / 60
    rem = [i for i, x in enumerate(win) if x == "REM" and i >= on]
    if rem:
        m["REM_lat"] = (rem[0] - on) * EPOCH / 60
    spt = win[on:off + 1]
    m["WASO"] = sum(1 for x in spt if x == "W") * EPOCH / 60
    m["n_awake"] = sum(1 for i, x in enumerate(spt)      # 連續 W 算一次覺醒
                       if x == "W" and (i == 0 or spt[i - 1] != "W"))
    for k in m["stage_min"]:                             # 佔比以 TST 為分母（W 以 TIB）
        base = m["TST"] if k != "W" else m["TIB"]
        m["stage_pct"][k] = m["stage_min"][k] / base * 100 if base else 0.0
    return m


# ---------------------------------------------------------------- 圖


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=145, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(alpha=.25, lw=.6, color=LINE)
    ax.set_axisbelow(True)


def fig_hypnogram(tracks, start_dt=None, lights=None):
    yv = {s: i for i, s in enumerate(PLOT_ORDER)}
    fig, axes = plt.subplots(len(tracks), 1, figsize=(13.5, 2.4 * len(tracks)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (label, sc) in zip(axes, tracks):
        s = norm(sc.stages)
        hrs = np.arange(len(s)) * EPOCH / 3600
        y = np.array([yv.get(x, np.nan) for x in s], dtype=float)
        ax.step(hrs, y, where="post", lw=1.0, color="#3b4657")
        for st in PLOT_ORDER:
            msk = np.array([x == st for x in s])
            if msk.any():
                ax.plot(hrs[msk], y[msk], ".", ms=2.6, color=COLOR[st])
        if lights and lights[0] is not None:
            for x, lb in ((lights[0], "lights off"), (lights[1], "lights on")):
                if x is not None:
                    ax.axvline(x * EPOCH / 3600, color="#2a9d8f", lw=1, ls="--", alpha=.8)
        ax.set_yticks(range(len(PLOT_ORDER)))
        ax.set_yticklabels(PLOT_ORDER)
        ax.invert_yaxis()
        ax.set_ylabel(label, fontsize=11, color=INK)
        ax.margins(x=.004)
        style(ax)
    axes[-1].set_xlabel("記錄開始後（小時）", fontsize=10, color=MUTED)
    if start_dt is not None:
        ticks = axes[-1].get_xticks()
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(
            [(start_dt + dt.timedelta(hours=float(t))).strftime("%H:%M") for t in ticks])
        axes[-1].set_xlabel("時鐘時間", fontsize=10, color=MUTED)
    fig.tight_layout()
    return b64(fig)


def fig_pies(tracks):
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.6 * len(tracks), 4.6), squeeze=False)
    for ax, (label, m) in zip(axes[0], tracks):
        vals = [m["stage_min"][k] for k in STAGES]
        tot = sum(vals) or 1
        keep = [(k, v) for k, v in zip(STAGES, vals) if v > 0]
        ax.pie([v for _, v in keep], labels=[k for k, _ in keep],
               colors=[COLOR[k] for k, _ in keep], startangle=90, counterclock=False,
               autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
               wedgeprops=dict(width=.42, edgecolor="white", linewidth=1.6),
               textprops=dict(fontsize=10, color=INK), pctdistance=.79)
        ax.text(0, .06, f"{tot / 60:.1f} h", ha="center", va="center",
                fontsize=17, color=INK, fontweight="bold")
        ax.text(0, -.16, "記錄總長", ha="center", va="center", fontsize=9, color=MUTED)
        ax.set_title(label, fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    return b64(fig)


def fig_stage_bars(tracks):
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    w = .38
    xs = np.arange(len(STAGES))
    for k, (label, m) in enumerate(tracks):
        ax.bar(xs + (k - (len(tracks) - 1) / 2) * w, [m["stage_min"][s] for s in STAGES],
               w, label=label, color=[COLOR[s] for s in STAGES],
               alpha=1 - .38 * k, edgecolor="white", lw=1)
        for x, s in zip(xs, STAGES):
            v = m["stage_min"][s]
            ax.text(x + (k - (len(tracks) - 1) / 2) * w, v + 2, f"{v:.0f}",
                    ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel("分鐘", fontsize=10, color=MUTED)
    if len(tracks) > 1:
        ax.legend(frameon=False, fontsize=9,
                  handles=[plt.Rectangle((0, 0), 1, 1, fc="#8a93a0", alpha=1 - .38 * k)
                           for k in range(len(tracks))],
                  labels=[t[0] for t in tracks])
    style(ax)
    fig.tight_layout()
    return b64(fig)


def stage_spectra(rec, stages, max_epochs=400):
    """每一期取 EEG，逐 epoch 算 Welch 再平均。回傳 (freqs, {期: psd}, 頻段功率)。"""
    eeg = next((n for n in rec.ch_names if "eeg" in n.lower()), None)
    if eeg is None:
        return None
    fs = rec.sfreqs[rec.ch_names.index(eeg)]
    s = norm(stages)
    out, bands = {}, {}
    rng = np.random.default_rng(0)
    freqs = None
    for st in STAGES:
        idx = [i for i, x in enumerate(s) if x == st]
        if len(idx) < 3:
            continue
        if len(idx) > max_epochs:                     # 太多就抽樣，圖形不受影響
            idx = sorted(rng.choice(idx, max_epochs, replace=False))
        psds = []
        for i in idx:
            y = rec.read([eeg], i * EPOCH, (i + 1) * EPOCH)[eeg][1]
            if len(y) < fs * 4:
                continue
            f, p = welch(y - y.mean(), fs, nperseg=int(fs * 4))
            psds.append(p)
            freqs = f
        if psds:
            out[st] = np.mean(psds, axis=0)
            tot = np.trapz(out[st][(freqs >= .5) & (freqs <= 45)],
                           freqs[(freqs >= .5) & (freqs <= 45)]) or 1
            bands[st] = {b: float(np.trapz(out[st][(freqs >= lo) & (freqs < hi)],
                                           freqs[(freqs >= lo) & (freqs < hi)]) / tot * 100)
                         for b, lo, hi in BANDS}
    return (freqs, out, bands, eeg) if out else None


def fig_spectra(freqs, psds, eeg_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    m = (freqs > 0.3) & (freqs <= 30)
    for st, p in psds.items():
        axes[0].semilogy(freqs[m], p[m], lw=1.5, color=COLOR[st], label=st)
        axes[1].plot(freqs[m], p[m] / (p[m].max() or 1), lw=1.5, color=COLOR[st], label=st)
    axes[0].set_ylabel(f"PSD（µV²/Hz）", fontsize=10, color=MUTED)
    axes[0].set_title("絕對功率（對數）", fontsize=11, color=INK)
    axes[1].set_ylabel("正規化", fontsize=10, color=MUTED)
    axes[1].set_title("波形比較（各自最大值 = 1）", fontsize=11, color=INK)
    for ax in axes:
        ax.set_xlabel("Hz", fontsize=10, color=MUTED)
        ax.legend(frameon=False, fontsize=9)
        for lo, hi, c in ((0.5, 4, "#6d3fd6"), (8, 13, "#3fa7d6"), (12, 14, "#2a9d8f")):
            ax.axvspan(lo, hi, color=c, alpha=.05)
        style(ax)
    fig.suptitle(f"各睡眠階段的 EEG 功率頻譜　（{eeg_name}）", fontsize=12, color=INK)
    fig.tight_layout()
    return b64(fig)


# ---------------------------------------------------------------- HTML

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f6f9;color:#1c2530;
     font:15px/1.65 -apple-system,"PingFang TC","Helvetica Neue",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:38px 22px 70px}
h1{font-size:25px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:#6b7785;font-size:13.5px;margin-bottom:26px}
.card{background:#fff;border:1px solid #e3e8ef;border-radius:13px;padding:22px 24px;
      margin-bottom:20px;box-shadow:0 1px 2px rgba(20,30,50,.04)}
.card>h2{font-size:15px;margin:0 0 16px;letter-spacing:.02em;color:#3b4657;
         text-transform:uppercase;font-weight:600}
img{width:100%;height:auto;display:block}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #eef1f6}
th:first-child,td:first-child{text-align:left}
thead th{color:#6b7785;font-weight:600;font-size:12.5px;text-transform:uppercase;
         letter-spacing:.04em;border-bottom:1.5px solid #e3e8ef}
tbody tr:last-child td{border-bottom:none}
td.d{color:#6b7785;font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:13px}
.kpi{background:#fff;border:1px solid #e3e8ef;border-radius:11px;padding:15px 17px}
.kpi .l{font-size:11.5px;color:#6b7785;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:26px;font-weight:650;margin-top:5px;letter-spacing:-.02em}
.kpi .u{font-size:12.5px;color:#6b7785;font-weight:400;margin-left:3px}
.kpi .c{font-size:12px;color:#6b7785;margin-top:3px}
.note{font-size:12.5px;color:#6b7785;margin-top:13px;line-height:1.6}
.tag{display:inline-block;padding:2px 8px;border-radius:5px;font-size:11.5px;
     background:#eef1f6;color:#3b4657;margin-left:7px;vertical-align:2px}
@media(prefers-color-scheme:dark){
 body{background:#0f1319;color:#e6ebf2}
 .card,.kpi{background:#161b23;border-color:#252c37;box-shadow:none}
 .card>h2,.kpi .l,.kpi .c,.sub,td.d,thead th{color:#8b95a4}
 th,td{border-color:#222932}.tag{background:#222932;color:#c3cbd6}
 img{border-radius:7px;background:#fff}
}
"""


def kpi(label, val, unit="", cmp_=None):
    v = "—" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:.1f}"
    c = ""
    if cmp_ is not None and not (isinstance(cmp_, float) and np.isnan(cmp_)):
        c = f"<div class='c'>自動判期 {cmp_:.1f}{unit}</div>"
    return (f"<div class='kpi'><div class='l'>{label}</div>"
            f"<div class='v'>{v}<span class='u'>{unit}</span></div>{c}</div>")


def table(rows, head):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(
        f"<td class='d'>{c}</td>" if i else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main():
    ap = argparse.ArgumentParser(description="睡眠報告（技師 vs 自動判期）")
    ap.add_argument("--psg", help="PSG 的 .edf / .raw")
    ap.add_argument("--out", help="輸出 HTML 路徑")
    ap.add_argument("--no-auto", action="store_true", help="不跑 YASA，只做技師判讀")
    ap.add_argument("--lights-off", type=float, default=None,
                    help="熄燈時間（記錄開始後幾分鐘）。判讀檔沒有 Lights Off 事件時，"
                         "TIB 預設從記錄起點算，SE 會偏低、SOL 會偏長——"
                         "廠商報表用的是另一個分析起點，兩者不會一致")
    ap.add_argument("--lights-on", type=float, default=None, help="開燈時間（分鐘）")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    path = args.psg
    if path is None:
        cands = sorted(f for f in os.listdir("data") if f.lower().endswith((".edf", ".raw"))
                       and "hypnogram" not in f.lower())
        path = next((os.path.join("data", f) for f in cands
                     if staging.find_technician_file(os.path.join("data", f))[0]), None)
        if path is None:
            sys.exit("data/ 裡找不到有技師判讀檔的紀錄，用 --psg 指定")
    print(f"讀取 {path}")
    rec = Recording(path)
    n30 = int(rec.duration // EPOCH)

    tech = staging.load_technician(path, rec.meas_date, n30)
    auto = None if args.no_auto else (staging.cached_auto(path) or staging.auto_stage(rec, path))
    if tech is None and auto is None:
        sys.exit("既沒有技師判讀也沒有自動判期")
    print(f"技師 {len(tech) if tech else 0} epoch　自動 {len(auto) if auto else 0} epoch")

    lo = hi = None
    lights_src = "整段記錄"
    if tech:
        for onset, ev in tech.lights:
            if "off" in ev.lower():
                lo = int(onset / EPOCH)
            elif "on" in ev.lower():
                hi = int(onset / EPOCH)
    if lo is not None or hi is not None:
        lights_src = "判讀檔的 Lights 事件"
    if args.lights_off is not None:
        lo, lights_src = int(args.lights_off * 60 / EPOCH), "手動指定"
    if args.lights_on is not None:
        hi, lights_src = int(args.lights_on * 60 / EPOCH), "手動指定"
    if lo is None:
        print("⚠️  判讀檔沒有 Lights Off 事件，TIB 從記錄起點算起——"
              "SE 會比廠商報表低、SOL 會比較長。可用 --lights-off 指定。")

    tracks = [(n, s) for n, s in (("技師判讀", tech), ("自動判期 (YASA)", auto)) if s]
    mets = [(n, metrics(s.stages, lo, hi)) for n, s in tracks]
    mt = dict(mets)

    print("繪圖…")
    img_hyp = fig_hypnogram(tracks, rec.meas_date, (lo, hi))
    img_pie = fig_pies(mets)
    img_bar = fig_stage_bars(mets)
    sp = stage_spectra(rec, (tech or auto).stages)
    img_sp = fig_spectra(sp[0], sp[1], sp[3]) if sp else None

    a = mets[0][1]
    b = mets[1][1] if len(mets) > 1 else {}
    g = (lambda k: b.get(k) if b else None)

    kpis = "".join([
        kpi("睡眠效率 SE", a["SE"], "%", g("SE")),
        kpi("總睡眠時間 TST", a["TST"], " 分", g("TST")),
        kpi("入睡潛伏期 SOL", a["SOL"], " 分", g("SOL")),
        kpi("REM 潛伏期", a["REM_lat"], " 分", g("REM_lat")),
        kpi("入睡後清醒 WASO", a["WASO"], " 分", g("WASO")),
        kpi("覺醒次數", float(a["n_awake"]), " 次",
            float(b["n_awake"]) if b else None),
    ])

    rows = []
    for k in STAGES:
        r = [k, f"{a['stage_min'][k]:.1f}", f"{a['stage_pct'][k]:.1f}%"]
        if b:
            r += [f"{b['stage_min'][k]:.1f}", f"{b['stage_pct'][k]:.1f}%",
                  f"{b['stage_min'][k] - a['stage_min'][k]:+.1f}"]
        rows.append(r)
    head = ["階段", "技師 (分)", "技師 %"] + (
        ["自動 (分)", "自動 %", "差異 (分)"] if b else [])

    cmp_html = ""
    ag = staging.agreement(tech, auto) if (tech and auto) else None
    if ag:
        po, kappa, n, cm, dis = ag
        labels = list(cm.keys())
        rows2 = [[r] + [str(cm[r][c]) for c in labels]
                 + [f"{cm[r][r] / max(sum(cm[r].values()), 1):.0%}"] for r in labels]
        cmp_html = f"""
        <div class='card'><h2>判期一致性<span class='tag'>技師 = 標準</span></h2>
        <div class='kpis' style='margin-bottom:18px'>
          {kpi("一致率", po * 100, "%")}{kpi("Cohen κ", kappa, "")}
          {kpi("比對 epoch", float(n), "")}{kpi("不一致", float(len(dis)), " 個")}
        </div>
        {table(rows2, ["技師＼自動"] + labels + ["逐期一致率"])}
        <div class='note'>N1 兩邊最難對上是常態——人類判讀者彼此在 N1 的一致度本來就最低。
        自動判期用來定位和篩檢，不取代技師判讀。</div></div>"""

    band_html = ""
    if sp and sp[2]:
        bh = ["階段"] + [b_[0] for b_ in BANDS]
        br = [[st] + [f"{sp[2][st][b_[0]]:.1f}%" for b_ in BANDS] for st in sp[2]]
        band_html = f"""<div class='card'><h2>各階段相對頻段功率</h2>{table(br, bh)}
        <div class='note'>以 0.5–45 Hz 為分母。sigma 12–14 Hz 與 alpha/beta 有重疊，
        沿用 TS 分析軟體的頻段定義。</div></div>"""

    src = f"技師：{tech.source}　" if tech else ""
    src += "自動：YASA" if auto else ""
    lights_txt = ""
    if lo is not None or hi is not None:
        lights_txt = (f"　熄燈–開燈："
                      f"{fmt(lo)}–{fmt(hi)}" if lo is not None else "")

    tib_txt = f"{fmt(lo or 0)}–{fmt(hi if hi is not None else n30)}，共 {a['TIB']:.1f} 分"
    html = f"""<!doctype html><html lang="zh-TW"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>睡眠報告 — {rec.name}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>睡眠報告</h1>
<div class="sub">{rec.name}　·　{rec.meas_date or '起始時間未知'}　·
記錄長度 {rec.duration / 3600:.2f} 小時　·　{src}{lights_txt}</div>

<div class="kpis" style="margin-bottom:20px">{kpis}</div>

<div class="card"><h2>Hypnogram</h2><img src="data:image/png;base64,{img_hyp}"></div>

<div class="card"><h2>各階段時間與佔比</h2>
<img src="data:image/png;base64,{img_pie}">
<img src="data:image/png;base64,{img_bar}" style="margin-top:14px">
<div style="margin-top:18px">{table(rows, head)}</div>
<div class="note">W 的百分比以 TIB（臥床時間）為分母，其餘各期以 TST（總睡眠時間）為分母。</div>
</div>

{cmp_html}

{"<div class='card'><h2>各階段 EEG 功率頻譜</h2><img src='data:image/png;base64," + img_sp + "'></div>" if img_sp else ""}
{band_html}

<div class="card"><h2>指標定義</h2>
<div class="note">
<b>TIB</b> 臥床時間：本報告用「{lights_src}」（{tib_txt}）。
判讀檔沒有 Lights Off 事件時就從記錄起點算，這會讓 <b>SE 偏低、SOL 偏長</b>，
和廠商報表的 SE_TIA／Sleep Latency 不會一致（廠商用的是另一個分析起點）；
要對齊就用 <code>--lights-off</code> 指定熄燈時間。<br>
<b>TST</b> 總睡眠時間 = 所有 N1/N2/N3/REM epoch。<br>
<b>SE</b> 睡眠效率 = TST ÷ TIB × 100%。<br>
<b>SOL</b> 入睡潛伏期 = TIB 起點到第一個睡眠 epoch。<br>
<b>REM 潛伏期</b> = <u>入睡</u>到第一個 REM epoch（AASM 慣例，不從熄燈算）。<br>
<b>WASO</b> = 入睡之後、最後一個睡眠 epoch 之前的清醒時間。<br>
<b>覺醒次數</b> = 睡眠期間內連續清醒段的段數（連續 W 算一次）。<br>
epoch 長度 30 秒；R&K 的 stage 4 依 AASM 併入 N3。
</div></div>
</div></body></html>"""

    out = args.out or f"sleep_report_{os.path.splitext(os.path.basename(path))[0]}.html"
    open(out, "w", encoding="utf-8").write(html)
    print(f"\n睡眠效率 {a['SE']:.1f}%　TST {a['TST']:.0f} 分　SOL {a['SOL']:.1f} 分　"
          f"REM 潛伏 {a['REM_lat']:.1f} 分　WASO {a['WASO']:.1f} 分　覺醒 {a['n_awake']} 次")
    print(f"報告 → {os.path.abspath(out)}")


def fmt(i):
    return "—" if i is None else f"{i * EPOCH / 3600:.2f} h"


if __name__ == "__main__":
    main()
