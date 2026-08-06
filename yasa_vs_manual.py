#!/usr/bin/env python3
"""YASA 自動睡眠分期 vs 技師人工判讀。

用法：
    python3 yasa_vs_manual.py                       # 自動找 data/ 裡的 PSG + Hypnogram
    python3 yasa_vs_manual.py --psg data/SC4002E0-PSG.edf
    python3 yasa_vs_manual.py --no-crop             # 不裁掉記錄前後的清醒段
    python3 yasa_vs_manual.py --emg "EMG submental" # 額外加 EMG（Sleep-EDF 不建議）

預設只用 EEG + EOG。

需要：mne、yasa、scikit-learn、matplotlib（yasa 用 `pip install yasa` 裝）

輸出：
    <out>/hypnogram_compare.png    人工 vs YASA 兩條 hypnogram ＋ 不一致標記
    <out>/confusion_matrix.png     混淆矩陣（次數 ＋ 列標準化）
    <out>/epoch_stages.csv         逐 epoch 的兩邊判讀
    <out>/metrics.txt              各項指標
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import mne                               # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import yasa                              # noqa: E402
from sklearn.metrics import (            # noqa: E402
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
)

mne.set_log_level("ERROR")

# AASM 五期。Sleep-EDF 沿用 R&K，有 stage 4 —— 依 AASM 併進 N3。
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLOR = {"W": "#e0a300", "N1": "#3fa7d6", "N2": "#2563c9",
               "N3": "#6d3fd6", "REM": "#e04f4f"}
PLOT_ORDER = ["W", "REM", "N1", "N2", "N3"]          # 由上到下，睡眠圖的慣例
EPOCH = 30.0

ANN2STAGE = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3",    # R&K S4 併入 N3
    "Sleep stage R": "REM",
}


# ---------------------------------------------------------------- 資料


def find_pair(psg=None, hypno=None, data_dir="data"):
    """沒指定就自己在 data/ 裡配對 *-PSG.edf 與 *-Hypnogram.edf。"""
    if psg is None:
        cands = sorted(glob.glob(os.path.join(data_dir, "*-PSG.edf")))
        if not cands:
            sys.exit(f"找不到 PSG 檔，請把 *-PSG.edf 放進 {data_dir}/ 或用 --psg 指定")
        pref = [c for c in cands if "SC4002" in os.path.basename(c)]
        psg = (pref or cands)[0]
    if hypno is None:
        base = os.path.basename(psg).split("-")[0][:6]    # SC4002E0 → SC4002
        cands = [f for f in glob.glob(os.path.join(os.path.dirname(psg), "*.edf"))
                 if "hypnogram" in os.path.basename(f).lower()
                 and os.path.basename(f).startswith(base)]
        if not cands:
            sys.exit(f"找不到對應的 Hypnogram（前綴 {base}），請用 --hypno 指定")
        hypno = cands[0]
    return psg, hypno


def pick_channel(names, *keywords):
    for kw in keywords:
        for n in names:
            if kw.lower() in n.lower():
                return n
    return None


def manual_stages(hypno_path, n_epochs):
    """把人工標註攤成每 30 秒一格。未標註或 Movement time/? 留 None。"""
    ann = mne.read_annotations(hypno_path)
    out = [None] * n_epochs
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        st = ANN2STAGE.get(str(desc).strip())
        if st is None:
            continue
        i0 = int(round(onset / EPOCH))
        for k in range(int(round(dur / EPOCH))):
            if 0 <= i0 + k < n_epochs:
                out[i0 + k] = st
    return out


def sleep_window(stages, pad_min=30):
    """回傳「第一個非 W」到「最後一個非 W」再各外擴 pad_min 分鐘的 epoch 範圍。

    Sleep-EDF 的記錄前後各掛了好幾小時的清醒段，不裁掉的話 accuracy 會被
    一大片 W 灌水（兩邊都判 W 當然一致），看不出真正的判期能力。
    """
    idx = [i for i, s in enumerate(stages) if s and s != "W"]
    if not idx:
        return 0, len(stages)
    pad = int(pad_min * 60 / EPOCH)
    return max(0, idx[0] - pad), min(len(stages), idx[-1] + pad + 1)


# ---------------------------------------------------------------- 繪圖


def plot_hypnograms(manual, auto, disagree, out_png, title, conf=None):
    yv = {s: i for i, s in enumerate(PLOT_ORDER)}
    hrs = np.arange(len(manual)) * EPOCH / 3600

    has_conf = conf is not None
    n_rows = 4 if has_conf else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 0.7] + ([1.2] if has_conf else []),
                                              hspace=0.18))

    for ax, series, label in ((axes[0], manual, "Manual (technician)"),
                              (axes[1], auto, "YASA (automatic)")):
        y = np.array([yv[s] if s in yv else np.nan for s in series], dtype=float)
        ax.step(hrs, y, where="post", lw=1.1, color="#333")
        for s in PLOT_ORDER:                        # 每一期塗上自己的顏色
            m = np.array([x == s for x in series])
            ax.plot(hrs[m], y[m], ".", ms=2.2, color=STAGE_COLOR[s])
        ax.set_yticks(range(len(PLOT_ORDER)))
        ax.set_yticklabels(PLOT_ORDER, fontsize=9)
        ax.invert_yaxis()
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=.18, lw=.6)
        ax.margins(x=0.005)

    axes[2].fill_between(hrs, 0, disagree.astype(float), step="post",
                         color="#d64545", lw=0)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Disagree", fontsize=9)
    axes[2].grid(alpha=.18, lw=.6)

    if has_conf:
        axes[3].plot(hrs, conf, lw=.8, color="#2a9d8f")
        axes[3].axhline(0.5, ls="--", lw=.7, color="#999")
        axes[3].set_ylim(0, 1)
        axes[3].set_ylabel("YASA\nconfidence", fontsize=9)
        axes[3].grid(alpha=.18, lw=.6)

    axes[-1].set_xlabel("Time from recording start (h)", fontsize=10)
    fig.suptitle(title, fontsize=11, y=.985)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, labels, out_png, title):
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row == 0, 1, row))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mat, fmt, ttl, vmax in ((axes[0], cm, "d", "Counts", cm.max()),
                                    (axes[1], norm, ".0%", "Row-normalised (recall)", 1.0)):
        im = ax.imshow(mat if fmt == "d" else norm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("YASA")
        ax.set_ylabel("Manual")
        ax.set_title(ttl, fontsize=10)
        thr = vmax * 0.55
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9,
                        color="white" if (norm[i, j] if fmt != "d" else v) > thr else "#222")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程


def main():
    ap = argparse.ArgumentParser(description="YASA 自動判期 vs 人工判讀")
    ap.add_argument("--psg", help="PSG 的 .edf")
    ap.add_argument("--hypno", help="人工判讀的 *-Hypnogram.edf")
    ap.add_argument("--eeg", help="EEG 通道名（預設自動挑）")
    ap.add_argument("--eog", help="EOG 通道名（預設自動挑）")
    ap.add_argument("--emg", default=None,
                    help="要一併餵給 YASA 的 EMG 通道名。預設不用——Sleep-EDF 的 "
                         "'EMG submental' 是 1 Hz 的整流包絡不是原始 EMG，"
                         "餵進去 accuracy 從 78.5%% 掉到 62.2%%")
    ap.add_argument("--out", default="yasa_report", help="輸出資料夾")
    ap.add_argument("--no-crop", action="store_true",
                    help="不裁掉記錄前後的清醒段（預設會裁，否則 accuracy 被 W 灌水）")
    ap.add_argument("--pad-min", type=float, default=30, help="睡眠段前後保留幾分鐘")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    psg, hypno = find_pair(args.psg, args.hypno)
    os.makedirs(args.out, exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    say(f"PSG      : {psg}")
    say(f"人工判讀 : {hypno}")

    # ---- 1. 讀 EEG / EOG ----
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
    eeg = args.eeg or pick_channel(raw.ch_names, "eeg fpz", "eeg", "c3", "c4")
    eog = args.eog if args.eog is not None else pick_channel(raw.ch_names, "eog")
    emg = args.emg or None                      # EMG 要明確指定才用，理由見 --emg 說明
    eog = eog or None
    if eeg is None:
        sys.exit("找不到 EEG 通道，用 --eeg 指定")
    keep = [c for c in (eeg, eog, emg) if c]
    raw.pick(keep)
    say(f"通道     : EEG={eeg}　EOG={eog}　EMG={emg or '不使用'}")
    if emg:                                     # 低取樣率的通道多半是包絡，不是原始訊號
        try:
            from readers import EDFFile
            native = {c.name: c.fs for c in EDFFile(os.path.realpath(psg)).channels}
            if native.get(emg, 999) < 20:
                say(f"⚠️  {emg} 原生取樣率只有 {native[emg]:g} Hz，"
                    f"很可能是整流包絡而非原始 EMG，會拖累判期")
        except Exception:
            pass
    say(f"取樣率   : {raw.info['sfreq']:g} Hz　長度 {raw.n_times / raw.info['sfreq'] / 3600:.2f} h")

    # ---- 2. YASA 自動判期 ----
    sls = yasa.SleepStaging(raw, eeg_name=eeg, eog_name=eog, emg_name=emg)
    hyp = sls.predict()
    auto = [{"WAKE": "W", "W": "W", "R": "REM", "REM": "REM"}.get(str(s), str(s))
            for s in np.asarray(hyp.hypno)]
    conf = np.asarray(hyp.proba).max(axis=1)
    say(f"YASA     : {len(auto)} 個 epoch，平均信心 {conf.mean():.3f}")

    # ---- 3. 人工標註 ----
    manual = manual_stages(hypno, len(auto))
    say(f"人工     : {sum(s is not None for s in manual)} 個 epoch 有標註")

    # ---- 裁掉前後清醒段 ----
    lo, hi = (0, len(auto)) if args.no_crop else sleep_window(manual, args.pad_min)
    if not args.no_crop:
        say(f"分析範圍 : epoch {lo}–{hi}"
            f"（{lo * EPOCH / 3600:.2f}–{hi * EPOCH / 3600:.2f} h，"
            f"睡眠段前後各留 {args.pad_min:g} 分鐘）")

    m_win, a_win, c_win = manual[lo:hi], auto[lo:hi], conf[lo:hi]
    valid = [i for i, (m, a) in enumerate(zip(m_win, a_win)) if m in STAGES and a in STAGES]
    y_true = [m_win[i] for i in valid]
    y_pred = [a_win[i] for i in valid]
    if not y_true:
        sys.exit("兩邊沒有可比對的 epoch")

    # ---- 4. hypnogram 對比圖 ----
    dis = np.array([(m in STAGES and a in STAGES and m != a)
                    for m, a in zip(m_win, a_win)])
    f1 = os.path.join(args.out, "hypnogram_compare.png")
    plot_hypnograms(m_win, a_win, dis, f1,
                    f"{os.path.basename(psg)}　Manual vs YASA", conf=c_win)

    # ---- 5. accuracy / kappa ----
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=STAGES)
    say()
    say(f"比對 epoch 數    : {len(y_true)}（不一致 {int(dis.sum())} 個）")
    say(f"Accuracy         : {acc:.4f}  ({acc:.1%})")
    say(f"Cohen's Kappa    : {kappa:.4f}")

    if not args.no_crop:                      # 順便報一個沒裁的版本，避免誤讀
        v2 = [i for i in range(len(auto)) if manual[i] in STAGES and auto[i] in STAGES]
        a2 = accuracy_score([manual[i] for i in v2], [auto[i] for i in v2])
        k2 = cohen_kappa_score([manual[i] for i in v2], [auto[i] for i in v2], labels=STAGES)
        say(f"（未裁全長對照   : accuracy {a2:.1%}、kappa {k2:.3f}，"
            f"n={len(v2)}——W 佔太多，數字會虛高）")

    # ---- 6. 混淆矩陣 ----
    labels = [s for s in STAGES if s in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f2 = os.path.join(args.out, "confusion_matrix.png")
    plot_confusion(cm, labels, f2, f"{os.path.basename(psg)}　acc {acc:.1%}　κ {kappa:.3f}")
    say()
    say("混淆矩陣（列＝人工，欄＝YASA）")
    say(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    # ---- 7. precision / recall / F1 ----
    rep = classification_report(y_true, y_pred, labels=labels, digits=3,
                                output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.loc[labels + ["macro avg", "weighted avg"]]
    df.columns = ["precision", "recall", "f1-score", "support"]
    df["support"] = df["support"].astype(int)
    say()
    say("各分期 precision / recall / F1")
    say(df.round(3).to_string())

    pd.DataFrame({"epoch": range(lo, hi),
                  "time_h": np.arange(lo, hi) * EPOCH / 3600,
                  "manual": m_win, "yasa": a_win,
                  "yasa_conf": np.round(c_win, 4),
                  "agree": [None if not (m in STAGES and a in STAGES) else (m == a)
                            for m, a in zip(m_win, a_win)]}
                 ).to_csv(os.path.join(args.out, "epoch_stages.csv"), index=False)
    open(os.path.join(args.out, "metrics.txt"), "w").write("\n".join(log) + "\n")

    say()
    say(f"輸出 → {os.path.abspath(args.out)}/")
    for f in ("hypnogram_compare.png", "confusion_matrix.png",
              "epoch_stages.csv", "metrics.txt"):
        say(f"    {f}")


if __name__ == "__main__":
    main()
