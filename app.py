"""EDF Viewer — 睡眠腦波檢視器 (Streamlit + MNE + Plotly)

    streamlit run /Users/appletina/edf_viewer/app.py

把 .edf 放進同目錄的 data/，或直接上傳。技師判期（RemLogic 匯出的同名 .txt，
或 Sleep-EDF 的 *-Hypnogram.edf）會自動載入；自動判期用 YASA，按鈕觸發後快取。
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epoch_metrics as emx  # noqa: E402
import sleep_report as rpt  # noqa: E402
import staging  # noqa: E402
from readers import Recording  # noqa: E402

st.set_page_config(page_title="EDF Viewer", layout="wide", page_icon="🧠")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
SIG_EXT = (".edf", ".raw", ".bdf")
JUMP_STAGES = ["W", "N1", "N2", "N3", "REM"]
SC = staging.STAGE_COLOR


@st.cache_resource(show_spinner="讀取檔案…")
def load_recording(path, mtime, size):
    return Recording(path)


@st.cache_data(show_spinner=False)
def load_tech(path, mtime, start, n_ep):
    return staging.load_technician(path, start, n_ep)


def list_data_files():
    try:
        return sorted(f for f in os.listdir(DATA_DIR)
                      if f.lower().endswith(SIG_EXT) and "hypnogram" not in f.lower())
    except OSError:
        return []


@st.cache_data(show_spinner=False, ttl=60)
def files_with_scoring(names):
    """哪些檔旁邊有技師判讀檔。163 個檔只要 0.05 秒，所以每次都掃。"""
    return {f for f in names
            if staging.find_technician_file(os.path.join(DATA_DIR, f))[0]}


def fmt_dur(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def colored(text, color, size="2.1rem", weight=700):
    return (f"<span style='font-size:{size};font-weight:{weight};color:{color}'>"
            f"{text}</span>")


# ---------------------------------------------------------------- sidebar：選檔

os.makedirs(DATA_DIR, exist_ok=True)
files = list_data_files()

src = st.sidebar.radio("檔案來源", ["data/ 目錄", "上傳檔案"], horizontal=True)
path = None
if src == "data/ 目錄":
    if files:
        scored = files_with_scoring(tuple(files))
        only = st.sidebar.checkbox(f"只列有技師判期的（{len(scored)}/{len(files)}）", False)
        shown = [f for f in files if f in scored] if only else files
        first = next((i for i, f in enumerate(shown) if f in scored), 0)
        path = os.path.join(DATA_DIR, st.sidebar.selectbox(
            "選擇檔案", shown, index=first,
            format_func=lambda f: ("✓ " if f in scored else "　　") + f))
    else:
        st.sidebar.info(f"data/ 是空的\n\n把 .edf 放進：\n`{DATA_DIR}`")
else:
    up = st.sidebar.file_uploader("上傳 .edf", type=["edf", "EDF", "bdf", "raw", "RAW"])
    if up is not None:
        tmp = os.path.join(tempfile.gettempdir(), f"edfviewer_{up.name}")
        if not os.path.exists(tmp) or os.path.getsize(tmp) != up.size:
            with open(tmp, "wb") as fh:
                fh.write(up.getbuffer())
        path = tmp

if not path:
    st.caption("← 從左側選一個 .edf 檔，或上傳一個檔案。")
    st.stop()

try:
    stt = os.stat(path)
    rec = load_recording(path, stt.st_mtime, stt.st_size)
except Exception as e:  # noqa: BLE001
    st.error(f"開檔失敗：{e}")
    st.stop()

# ---------------------------------------------------------------- sidebar：設定

WIN_OPTS = [5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400]


def fmt_win(s):
    if s < 60:
        return f"{s} 秒"
    if s < 3600:
        return f"{s // 60} 分"
    return f"{s / 3600:g} 小時"


epoch_len = st.sidebar.select_slider("視窗長度", WIN_OPTS, value=30, format_func=fmt_win,
                                     help="拉到 1 小時就會把整整一小時的波形壓進畫面")
picks = st.sidebar.multiselect("頻道", rec.ch_names, default=rec.ch_names[:min(4, len(rec.ch_names))])
n_epochs = max(1, int(np.ceil(rec.duration / epoch_len)))

# ---- 判期 ----
st.sidebar.divider()
n_ep30 = int(rec.duration // staging.EPOCH)
try:
    tech = load_tech(path, stt.st_mtime, rec.meas_date, n_ep30)
except Exception as e:  # noqa: BLE001
    st.sidebar.warning(f"技師判期讀取失敗：{e}")
    tech = None
auto = staging.cached_auto(path)


@st.cache_data(show_spinner=False)
def load_metrics(path, mtime, start):
    return emx.load(path, start)


try:
    metrics, metrics_why = load_metrics(path, stt.st_mtime, rec.meas_date)
except Exception as e:  # noqa: BLE001
    metrics, metrics_why = None, f"讀取失敗：{e}"

st.sidebar.markdown("**睡眠分期**")
st.sidebar.caption(f"技師：{tech.source if tech else '— 找不到判讀檔'}")
st.sidebar.caption(f"自動：{'YASA（已快取）' if auto else '— 尚未執行'}")
st.sidebar.markdown("**逐-epoch 生理指標**")
st.sidebar.caption(f"{metrics.source}（{metrics_why}）" if metrics else f"— {metrics_why}")
if st.sidebar.button("執行自動判期（YASA）", width="stretch"):
    bar = st.sidebar.progress(0.0, "讀取訊號…")
    try:
        auto = staging.auto_stage(rec, path, progress=lambda f: bar.progress(f, "讀取訊號…"))
        bar.progress(1.0, "完成")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.sidebar.error(f"自動判期失敗：{e}")

tracks = [t for t in (("技師", tech), ("自動", auto)) if t[1] is not None]
jump_src = st.sidebar.radio("分期跳轉依據", [n for n, _ in tracks] or ["—"], horizontal=True,
                            disabled=not tracks)
show_hyp = st.sidebar.checkbox("顯示 hypnogram", True, disabled=not tracks)
show_ar = st.sidebar.checkbox("標出 arousal", True,
                              disabled=not (tech and tech.arousals))

st.sidebar.divider()
view = st.sidebar.radio("檢視", ["波形", "睡眠報告"], horizontal=True)
robust = st.sidebar.checkbox("Y 軸穩健縮放（忽略極端尖峰）", True)
plot_h = st.sidebar.slider("每頻道高度 (px)", 90, 320, 170, 10)
wide = st.sidebar.checkbox("固定紙速（畫布變寬、橫向捲動）", False,
                           help="勾掉＝整個視窗壓進螢幕寬，多長的視窗都一次看完；"
                                "勾起來＝波形不被壓扁，但要左右捲")
paper = st.sidebar.slider(
    "紙速 (px / 秒)", 8, 80, 36, 2,
    help="每秒佔多少像素。30 秒 × 36 ≈ 1080 px（一個螢幕寬）") if wide else None

# 目前 epoch 長度下，每個 epoch 對應的分期
def stage_series(sc):
    if sc is None:
        return None
    return [sc.at(i * epoch_len) for i in range(n_epochs)]


s_tech, s_auto = stage_series(tech), stage_series(auto)
s_jump = s_tech if (jump_src == "技師" and s_tech) else s_auto

# ---------------------------------------------------------------- epoch 狀態

if st.session_state.get("_file") != path:
    st.session_state._file = path
    st.session_state.epoch = 0
st.session_state.setdefault("epoch", 0)
prev_len = st.session_state.get("_epoch_len")
if prev_len is not None and prev_len != epoch_len:      # 改 epoch 長度時保留時間位置
    st.session_state.epoch = int(st.session_state.epoch * prev_len / epoch_len)
st.session_state._epoch_len = epoch_len
st.session_state.epoch = int(np.clip(st.session_state.epoch, 0, n_epochs - 1))
st.session_state.epoch_box = st.session_state.epoch + 1


def _goto(i):
    i = int(np.clip(i, 0, n_epochs - 1))
    st.session_state.epoch = i
    st.session_state.epoch_box = i + 1


def _step(d):
    _goto(st.session_state.epoch + d)


def _box_changed():
    _goto(int(st.session_state.epoch_box) - 1)


def _find_next(pred):
    cur = st.session_state.epoch
    for i in range(cur + 1, n_epochs):
        if pred(i):
            return _goto(i)
    for i in range(0, cur + 1):                        # 找不到就從頭繞一圈
        if pred(i):
            return _goto(i)
    st.session_state._notfound = True


def _next_stage(target):
    if s_jump:
        _find_next(lambda i: s_jump[i] == target)


def _next_disagree():
    if s_tech and s_auto:
        _find_next(lambda i: s_tech[i] in staging.STAGES and s_auto[i] in staging.STAGES
                   and s_tech[i] != s_auto[i])


# ---------------------------------------------------------------- 主畫面

fs_txt = " / ".join(f"{v:g}" for v in sorted(set(rec.sfreqs)))
when = rec.meas_date.strftime("%Y-%m-%d %H:%M:%S") if rec.meas_date else "未知"
st.caption(
    f"**{rec.name}** · {len(rec.ch_names)} 頻道 · {fs_txt} Hz · "
    f"時長 {fmt_dur(rec.duration)} · 錄製時間 {when}"
)

nav = st.columns([1, 1, 1, 1, 1.6, .5, 1, 1, 1, 1, 1, 1], vertical_alignment="bottom")
nav[0].button("−100", on_click=_step, args=(-100,), width="stretch")
nav[1].button("−1", on_click=_step, args=(-1,), width="stretch")
nav[2].button("+1", on_click=_step, args=(1,), width="stretch")
nav[3].button("+100", on_click=_step, args=(100,), width="stretch")
nav[4].number_input("Epoch", 1, n_epochs, step=1, key="epoch_box",
                    on_change=_box_changed, label_visibility="collapsed")
nav[5].markdown(f"<div style='padding-top:.45rem;color:#888;font-size:.8rem'>/ {n_epochs}</div>",
                unsafe_allow_html=True)
for k, s in enumerate(JUMP_STAGES):
    nav[6 + k].button(s, on_click=_next_stage, args=(s,), width="stretch",
                      disabled=not s_jump, help=f"跳到下一個 {s}（依{jump_src}）")
nav[11].button("≠", on_click=_next_disagree, width="stretch",
               disabled=not (s_tech and s_auto), help="跳到下一個兩邊判讀不一致的 epoch")

ep = st.session_state.epoch
t0 = ep * epoch_len
t1 = min(rec.duration, t0 + epoch_len)
if st.session_state.pop("_notfound", None):
    st.toast("整份記錄裡找不到符合的 epoch")

clock = ""
if rec.meas_date is not None:
    clock = (rec.meas_date + dt.timedelta(seconds=t0)).strftime("%H:%M:%S") + " · "
bits = []
if s_tech:
    bits.append("<span style='font-size:.8rem;color:#888'>技師 </span>"
                + colored(s_tech[ep] or "—", SC.get(s_tech[ep], "#8a8a8a")))
if s_auto:
    c = ""
    if auto.confidence:
        i30 = int(t0 // staging.EPOCH)
        if i30 < len(auto.confidence):
            c = f"<span style='font-size:.75rem;color:#888'> {auto.confidence[i30]:.0%}</span>"
    bits.append("<span style='font-size:.8rem;color:#888'>自動 </span>"
                + colored(s_auto[ep] or "—", SC.get(s_auto[ep], "#8a8a8a")) + c)
if s_tech and s_auto and s_tech[ep] and s_auto[ep]:
    ok = s_tech[ep] == s_auto[ep]
    bits.append(colored("✓" if ok else "✗", "#3a9d5d" if ok else "#d64545", "1.5rem"))
bits.append(f"<span style='font-size:.85rem;color:#888'>{clock}{fmt_dur(t0)}–{fmt_dur(t1)}</span>")
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:1.4rem;line-height:1.3'>"
    + "".join(f"<div>{b}</div>" for b in bits) + "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- 逐-epoch 指標

if metrics is not None:
    show_metrics = st.sidebar.checkbox("顯示逐-epoch 生理指標", True)
    ln_as_linear = st.sidebar.checkbox("ln 欄位換算成線性值", False,
                                       help="delta/theta/alpha/beta/TP/HF/LF/EOG/EMG "
                                            "在廠商檔裡是 ln 值，勾起來會顯示 exp() 後的數值")
else:
    show_metrics = ln_as_linear = False

if show_metrics:
    row = metrics.at(t0)
    if row is None:
        st.caption(f"這個 epoch 超出廠商指標檔的範圍（該檔只到 {metrics.n} 格）。")
    else:
        cols = [c for c in metrics.columns if row.get(c) is not None]
        per_line = 9
        for k in range(0, len(cols), per_line):
            cs = st.columns(min(per_line, len(cols) - k))
            for col, name in zip(cs, cols[k:k + per_line]):
                v = row[name]
                if ln_as_linear and metrics.is_ln(name):
                    v, unit = np.exp(v), metrics.units[name].replace("ln(", "").rstrip(")")
                else:
                    unit = metrics.units[name]
                txt = f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}"
                col.metric(name, txt, help=f"{metrics.note(name)}（{unit}）".strip("（）"))
        st.caption(f"來源：{metrics.source} —— 廠商軟體算好的每 30 秒指標，"
                   f"與既有分析同一把尺。缺值（`-`）的欄位不顯示。")

# ---------------------------------------------------------------- hypnogram

if show_hyp and not tech:
    st.caption("這個檔旁邊沒有技師判讀檔，所以沒有技師那條線。"
               "第一階段（清醒）記錄本來就沒有判期；第二階段選單裡有 ✓ 的才有。")

if show_hyp and tracks:
    hf = go.Figure()
    order = ["W", "REM", "N1", "N2", "N3", "N4"]
    ymap = {s: i for i, s in enumerate(order)}
    for name, sc, dash, width, op in (("技師", tech, None, 1.6, 1.0),
                                      ("自動", auto, "dot", 1.4, .75)):
        if sc is None:
            continue
        x = np.arange(len(sc)) * staging.EPOCH / 3600
        y = [ymap.get(s, None) for s in sc.stages]
        hf.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line_shape="hv",
                                line=dict(width=width, dash=dash), opacity=op,
                                connectgaps=False,
                                hovertemplate="%{customdata}<extra>" + name + "</extra>",
                                customdata=[s or "—" for s in sc.stages]))
    if show_ar and tech and tech.arousals:
        ax = [o / 3600 for o, _ in tech.arousals]
        hf.add_trace(go.Scatter(x=ax, y=[-0.55] * len(ax), mode="markers", name="arousal",
                                marker=dict(symbol="line-ns-open", size=7, color="#d64545",
                                            line_width=1.2),
                                hovertemplate="arousal<extra></extra>"))
    # 透明的點擊熱區：鋪滿整張圖，這樣點空白處也能跳
    hit_x = np.arange(max(len(t) for t in (tech.stages if tech else [],
                                           auto.stages if auto else []))) * staging.EPOCH / 3600
    ys = [-0.9, 0, 1, 2, 3, 4, 5]
    hf.add_trace(go.Scattergl(
        x=np.repeat(hit_x, len(ys)), y=np.tile(ys, len(hit_x)), mode="markers",
        marker=dict(size=16, color="rgba(0,0,0,0)"), hoverinfo="none", showlegend=False,
        selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
        name=""))
    hf.add_vline(x=t0 / 3600, line_color="#2a9d8f", line_width=2)
    hf.update_layout(height=175, margin=dict(l=72, r=16, t=6, b=28),
                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                     legend=dict(orientation="h", y=1.22, x=0, font_size=11),
                     xaxis_title=None, hovermode="x unified")
    hf.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=order,
                    autorange="reversed", tickfont_size=10, showgrid=True,
                    gridcolor="rgba(128,128,128,.15)", range=[len(order) - .5, -1])
    hf.update_xaxes(title_text="小時", showgrid=True, gridcolor="rgba(128,128,128,.15)",
                    title_font_size=10, tickfont_size=10)
    sel = st.plotly_chart(hf, use_container_width=True, key="hypno_click",
                          on_select="rerun", selection_mode="points",
                          config={"displaylogo": False})
    pts = (sel or {}).get("selection", {}).get("points", [])
    if pts:                                   # 點 hypnogram → 下方波形跳到那個時間
        tgt = int(np.clip(float(pts[0]["x"]) * 3600 // epoch_len, 0, n_epochs - 1))
        if tgt != ep:
            st.session_state.epoch = tgt      # epoch_box 由下一輪腳本開頭同步
            st.rerun()
    st.caption("點 hypnogram 任一處，下方波形就會跳到那個時間點。")

# ---------------------------------------------------------------- 睡眠報告


def png(b64s):
    import base64 as _b
    return _b.b64decode(b64s)


@st.cache_data(show_spinner="計算各階段功率頻譜…")
def cached_spectra(path, mtime, stages):
    return rpt.stage_spectra(load_recording(path, mtime, os.stat(path).st_size), list(stages))


def show_report():
    tracks = [(n, s) for n, s in (("技師判讀", tech), ("自動判期 (YASA)", auto)) if s]
    if not tracks:
        st.info("這個檔沒有技師判讀，也還沒跑自動判期——側欄按「執行自動判期（YASA）」就能產生報告。")
        return
    lo = hi = None
    if tech:
        for onset, ev in tech.lights:
            if "off" in ev.lower():
                lo = int(onset / staging.EPOCH)
            elif "on" in ev.lower():
                hi = int(onset / staging.EPOCH)
    mets = [(n, rpt.metrics(s.stages, lo, hi)) for n, s in tracks]
    a = mets[0][1]
    b = mets[1][1] if len(mets) > 1 else None

    def m_(label, key, unit="", fmt="{:.1f}"):
        v = a[key]
        txt = "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v) + unit
        d = None
        if b is not None:
            bv = b[key]
            if not (isinstance(bv, float) and np.isnan(bv)) and not (
                    isinstance(v, float) and np.isnan(v)):
                d = f"{bv - v:+.1f}{unit}"
        return label, txt, d

    items = [m_("睡眠效率 SE", "SE", "%"), m_("總睡眠時間 TST", "TST", " 分"),
             m_("入睡潛伏期 SOL", "SOL", " 分"), m_("REM 潛伏期", "REM_lat", " 分"),
             m_("入睡後清醒 WASO", "WASO", " 分"),
             m_("覺醒次數", "n_awake", " 次", "{:.0f}")]
    cols = st.columns(len(items))
    for c, (lab, val, d) in zip(cols, items):
        c.metric(lab, val, d, delta_color="off",
                 help="下方小字為「自動判期 − 技師判讀」的差" if d else None)
    if lo is None and tech:
        st.caption("判讀檔沒有 Lights Off 事件，TIB 從記錄起點算 —— SE 會比廠商報表低、SOL 偏長。")

    st.markdown("**Hypnogram**")
    st.image(png(rpt.fig_hypnogram(tracks, rec.meas_date, (lo, hi))), width="stretch")

    st.markdown("**各階段時間與佔比**")
    st.image(png(rpt.fig_pies(mets)), width="stretch")
    st.image(png(rpt.fig_stage_bars(mets)), width="stretch")
    rows = []
    for k in rpt.STAGES:
        r = {"階段": k, "技師 (分)": round(a["stage_min"][k], 1),
             "技師 %": round(a["stage_pct"][k], 1)}
        if b:
            r |= {"自動 (分)": round(b["stage_min"][k], 1),
                  "自動 %": round(b["stage_pct"][k], 1),
                  "差異 (分)": round(b["stage_min"][k] - a["stage_min"][k], 1)}
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("W 的百分比以 TIB 為分母，其餘各期以 TST 為分母。")

    if metrics is not None:
        st.markdown("**逐-epoch 生理指標趨勢**")
        default = [c for c in ("delta", "alpha", "LF/HF", "EMG", "ACT") if c in metrics.columns]
        picks_m = st.multiselect("指標", metrics.columns, default=default or metrics.columns[:3],
                                 key="metric_trend")
        if picks_m:
            hrs = np.arange(metrics.n) * emx.EPOCH / 3600
            mf = make_subplots(rows=len(picks_m), cols=1, shared_xaxes=True,
                               vertical_spacing=0.03,
                               subplot_titles=[f"{c}（{metrics.units[c]}）" for c in picks_m])
            for r, c in enumerate(picks_m, start=1):
                y = metrics.series(c)
                if ln_as_linear and metrics.is_ln(c):
                    y = np.exp(y)
                mf.add_trace(go.Scattergl(x=hrs, y=y, mode="lines", line=dict(width=0.9),
                                          name=c, connectgaps=False), row=r, col=1)
                mf.update_yaxes(tickfont_size=9, row=r, col=1)
            mf.add_vline(x=t0 / 3600, line_color="#2a9d8f", line_width=2, line_dash="dot")
            mf.update_layout(height=150 * len(picks_m) + 40, showlegend=False,
                             margin=dict(l=60, r=16, t=28, b=36),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            mf.update_annotations(font_size=11)
            mf.update_xaxes(title_text="小時", row=len(picks_m), col=1,
                            showgrid=True, gridcolor="rgba(128,128,128,.15)")
            st.plotly_chart(mf, use_container_width=True, config={"displaylogo": False})
            st.caption(f"來源 {metrics.source}，共 {metrics.n} 格；綠虛線 = 目前檢視位置。"
                       f"缺值處線會斷開。")

    st.markdown("**各階段 EEG 功率頻譜**")
    sp = cached_spectra(path, stt.st_mtime, tuple((tech or auto).stages))
    if sp is None:
        st.info("找不到 EEG 通道，畫不了頻譜。")
    else:
        freqs, psds, bands, eeg_name = sp
        st.image(png(rpt.fig_spectra(freqs, psds, eeg_name)), width="stretch")
        st.dataframe(pd.DataFrame(
            [{"階段": s} | {b_[0]: round(bands[s][b_[0]], 1) for b_ in rpt.BANDS}
             for s in bands]), width="stretch", hide_index=True)
        st.caption("相對頻段功率 (%)，以 0.5–45 Hz 為分母。"
                   "sigma 12–14 Hz 與 alpha/beta 有重疊，沿用 TS 分析軟體的定義。")


if view == "睡眠報告":
    show_report()
    st.stop()

# ---------------------------------------------------------------- 波形

if not picks:
    st.info("請在左側選擇至少一個頻道。")
    st.stop()

if t1 - t0 >= 1800:
    with st.spinner(f"讀取 {(t1 - t0) / 3600:.1f} 小時的訊號…"):
        data = rec.read(picks, t0, t1)
else:
    data = rec.read(picks, t0, t1)

# X 軸用真實時鐘時間（21:09:10）；沒有記錄起始時間的檔才退回「秒」
use_clock = rec.meas_date is not None
base = np.datetime64(rec.meas_date) if use_clock else None
grid_s = max(1, round(epoch_len / 30))


def xaxis(t):
    return base + (np.asarray(t) * 1e9).astype("timedelta64[ns]") if use_clock else t


def envelope(t, y, target):
    """把資料壓成 min/max 包絡。

    壓縮顯示時，60000 個點塞進約 1100 px，直接丟給瀏覽器不但慢，尖峰還會被
    隨機抽掉。改成每個像素格取該格的極大與極小值成對輸出，時間軸照樣鋪滿
    整個視窗，波形的上下包絡也完整保留。
    """
    n = len(y)
    if n <= target * 2:
        return t, y
    blk = int(np.ceil(n / target))
    m = (n // blk) * blk
    yb = y[:m].reshape(-1, blk)
    tb = t[:m].reshape(-1, blk)
    lo_i, hi_i = yb.argmin(1), yb.argmax(1)
    first_is_min = lo_i < hi_i
    y1 = np.where(first_is_min, yb.min(1), yb.max(1))
    y2 = np.where(first_is_min, yb.max(1), yb.min(1))
    t1_ = tb[np.arange(len(tb)), np.minimum(lo_i, hi_i)]
    t2_ = tb[np.arange(len(tb)), np.maximum(lo_i, hi_i)]
    tt = np.empty(len(y1) * 2); yy = np.empty(len(y1) * 2)
    tt[0::2], tt[1::2] = t1_, t2_
    yy[0::2], yy[1::2] = y1, y2
    if m < n:                                   # 尾巴不足一格的殘料照原樣接上
        tt, yy = np.append(tt, t[m:]), np.append(yy, y[m:])
    return tt, yy


fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=True, vertical_spacing=0.015)
px_wide = int((t1 - t0) * paper) if paper else 1150      # 畫布大概有多少像素寬
for r, n in enumerate(picks, start=1):
    t, y = data[n]
    i = rec.ch_names.index(n)
    t, y = envelope(t, y, px_wide)
    fig.add_trace(go.Scattergl(
        x=xaxis(t), y=y, mode="lines", name=n, line=dict(width=0.9),
        hovertemplate=("%{x|%H:%M:%S.%L}　%{y:.2f}<extra>" + n + "</extra>") if use_clock
        else ("%{x:.3f}s　%{y:.2f}<extra>" + n + "</extra>")),
        row=r, col=1)
    if robust and len(data[n][1]):
        raw_y = data[n][1]                      # 量程用原始資料算，不用包絡
        if len(raw_y) > 200_000:                # 長視窗抽樣，百分位數差不到 0.1%
            raw_y = raw_y[::len(raw_y) // 200_000]
        m = float(np.median(raw_y))
        s = float(np.percentile(np.abs(raw_y - m), 99.5)) or 1.0
        fig.update_yaxes(range=[m - 1.15 * s, m + 1.15 * s], row=r, col=1)
    fig.update_yaxes(title_text=f"{n}<br>{rec.units[i]}", title_font_size=11,
                     tickfont_size=9, row=r, col=1, zeroline=True,
                     zerolinecolor="rgba(128,128,128,.3)", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.18)",
                     dtick=grid_s * 1000 if use_clock else grid_s,
                     tickformat="%H:%M:%S" if use_clock else None, row=r, col=1)

if show_ar and tech:
    for o, d in tech.arousals:
        if t0 - 1 < o < t1:
            x0, x1 = xaxis([o, min(o + max(d, 1), t1)])
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(214,69,69,.13)",
                          line_width=0, annotation_text="arousal", annotation_font_size=9,
                          annotation_position="top left")

fig.update_layout(height=max(420, plot_h * len(picks)), showlegend=False,
                  margin=dict(l=72, r=16, t=8, b=34), hovermode="x unified", dragmode="pan",
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(title_text="時鐘時間" if use_clock else "秒", row=len(picks), col=1)
cfg = {"scrollZoom": True, "displaylogo": False,
       "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
if paper is None:
    st.plotly_chart(fig, use_container_width=True, config=cfg)
else:
    # 固定紙速：畫布寬度 = 視窗秒數 × px/秒，放不下就讓它橫向捲動
    st.markdown("<style>div[data-testid='stPlotlyChart'],div.stPlotlyChart"
                "{overflow-x:auto;overflow-y:hidden}</style>", unsafe_allow_html=True)
    fig.update_layout(width=int((t1 - t0) * paper) + 88)
    st.plotly_chart(fig, use_container_width=False, config=cfg)
    st.caption(f"紙速 {paper} px/秒　·　畫布寬 {int((t1 - t0) * paper) + 88} px"
               f"　·　視窗 {t1 - t0:g} 秒")

# ---------------------------------------------------------------- 統計 / 判期比對

with st.expander("頻道統計", expanded=False):
    st.dataframe(pd.DataFrame([{
        "頻道": n, "單位": rec.units[rec.ch_names.index(n)],
        "取樣率 (Hz)": f"{rec.sfreqs[rec.ch_names.index(n)]:g}",
        "最大值": round(float(np.max(data[n][1])), 3) if len(data[n][1]) else np.nan,
        "最小值": round(float(np.min(data[n][1])), 3) if len(data[n][1]) else np.nan,
        "平均值": round(float(np.mean(data[n][1])), 3) if len(data[n][1]) else np.nan,
        "標準差": round(float(np.std(data[n][1])), 3) if len(data[n][1]) else np.nan,
    } for n in picks]), width="stretch", hide_index=True)

cmp = staging.agreement(tech, auto)
if cmp:
    po, kappa, n, matrix, dis = cmp
    with st.expander(f"判期比對　一致率 {po:.1%}　Cohen κ {kappa:.3f}　"
                     f"（{n} 個 epoch，{len(dis)} 個不一致）", expanded=False):
        lbl = list(matrix.keys())
        df = pd.DataFrame([[matrix[r][c] for c in lbl] for r in lbl], index=lbl, columns=lbl)
        df.insert(0, "技師小計", df.sum(axis=1))
        df["逐期一致率"] = [f"{matrix[r][r] / max(sum(matrix[r].values()), 1):.0%}" for r in lbl]
        st.dataframe(df, width="stretch")
        st.caption("列＝技師判讀，欄＝YASA 自動判期。N1 兩邊最難對上是常態——"
                   "人類判讀者之間在 N1 的一致度本來就最低。自動判期是輔助定位用的，"
                   "不能取代技師判讀。")
