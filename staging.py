"""睡眠分期：技師判讀（RemLogic / Sleep-EDF）與自動判期（YASA）。

兩條軌道都統一成「每 30 秒一格」的 stage 陣列，格子對齊記錄起點，
未判讀的格子是 None，方便直接比對。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import numpy as np

EPOCH = 30.0
STAGES = ["W", "N1", "N2", "N3", "N4", "REM"]
STAGE_COLOR = {
    "W": "#e0a300", "N1": "#3fa7d6", "N2": "#2563c9", "N3": "#6d3fd6",
    "N4": "#5227a8", "REM": "#e04f4f", "M": "#8a8a8a", "?": "#8a8a8a", None: "#8a8a8a",
}
# 畫 hypnogram 時的縱軸順序（上到下：W → REM → N1 → N2 → N3）
STAGE_Y = {"W": 0, "REM": 1, "N1": 2, "N2": 3, "N3": 4, "N4": 5}

_NORM = {
    "w": "W", "wake": "W", "sleep stage w": "W", "sleep-s0": "W",
    "n1": "N1", "1": "N1", "s1": "N1", "sleep stage 1": "N1", "sleep-s1": "N1",
    "n2": "N2", "2": "N2", "s2": "N2", "sleep stage 2": "N2", "sleep-s2": "N2",
    "n3": "N3", "3": "N3", "s3": "N3", "sleep stage 3": "N3", "sleep-s3": "N3",
    "n4": "N4", "4": "N4", "sleep stage 4": "N4", "sleep-s4": "N4",
    "r": "REM", "rem": "REM", "sleep stage r": "REM", "sleep-rem": "REM",
    "movement time": "M", "sleep stage ?": "?",
}


def norm_stage(s):
    return _NORM.get(str(s).strip().lower(), str(s).strip())


class Scoring:
    """一份判期結果。`stages` 是 list，索引 = 第幾個 30 秒 epoch。"""

    def __init__(self, stages, source, kind, arousals=None, lights=None, confidence=None):
        self.stages = stages
        self.source = source          # 檔名或 "YASA"
        self.kind = kind              # "tech" | "auto"
        self.arousals = arousals or []
        self.lights = lights or []
        self.confidence = confidence  # 自動判期每 epoch 的信心值

    def __len__(self):
        return len(self.stages)

    def at(self, t):
        """指定秒數落在哪一期。"""
        i = int(t // EPOCH)
        return self.stages[i] if 0 <= i < len(self.stages) else None

    def counts(self):
        c = {}
        for s in self.stages:
            if s:
                c[s] = c.get(s, 0) + 1
        return c


# ---------------------------------------------------------------- 技師判讀


def _decode(raw):
    for enc in ("big5", "cp950", "utf-8-sig", "utf-8", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", "replace")


def find_technician_file(path):
    """找同名的判讀輸出檔。data/ 裡是 symlink，所以要先解到真實路徑。"""
    real = os.path.realpath(path)
    stem = os.path.splitext(real)[0]
    d = os.path.dirname(real)
    base = os.path.basename(stem)

    for ext in (".txt", ".TXT", ".Txt"):
        p = stem + ext
        if os.path.exists(p) and b"RemLogic" in open(p, "rb").read(64):
            return p, "remlogic"
    # Sleep-EDF：SC4002E0-PSG.edf ↔ SC4002EC-Hypnogram.edf
    pref = base.split("-")[0]
    best, best_n = None, 0
    try:
        for f in os.listdir(d):
            if "hypnogram" not in f.lower() or not f.lower().endswith(".edf"):
                continue
            n = len(os.path.commonprefix([pref.lower(), f.split("-")[0].lower()]))
            if n > best_n:
                best, best_n = os.path.join(d, f), n
    except OSError:
        pass
    if best and best_n >= 5:
        return best, "annotations"
    return None, None


def parse_remlogic(path, start_datetime, n_epochs_max=None):
    """RemLogic Event Export：每列是 `分期 <TAB> hh:mm:ss <TAB> 事件 <TAB> 秒數`。

    時間是牆上時鐘（跨午夜會繞回），用記錄起始時間換算成相對秒數。
    """
    lines = _decode(open(path, "rb").read()).splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("Sleep Stage")), None)
    if hdr is None:
        raise ValueError("找不到 RemLogic 的資料表頭")

    stages, arousals, lights = {}, [], []
    prev, day = None, 0
    for line in lines[hdr + 1:]:
        c = line.split("\t")
        if len(c) < 3 or not c[1].strip():
            continue
        try:
            hh, mm, ss = (int(x) for x in c[1].split(":"))
        except ValueError:
            continue
        secs = hh * 3600 + mm * 60 + ss
        if prev is not None and secs < prev:
            day += 1                      # 跨午夜
        prev = secs
        onset = (dt.datetime.combine(start_datetime.date(), dt.time(hh, mm, ss))
                 + dt.timedelta(days=day) - start_datetime).total_seconds()
        ev = c[2].strip()
        dur = float(c[3]) if len(c) > 3 and c[3].strip() else EPOCH
        st = norm_stage(ev)
        if st in STAGES or st in ("M", "?"):
            i0 = int(round(onset / EPOCH))
            for k in range(max(1, int(round(dur / EPOCH)))):
                if i0 + k >= 0:
                    stages[i0 + k] = st
        elif ev.lower().startswith("arousal"):
            arousals.append((onset, dur))
        elif ev.lower().startswith("lights"):
            lights.append((onset, ev))

    n = (max(stages) + 1) if stages else 0
    if n_epochs_max:
        n = min(n, n_epochs_max)
    arr = [stages.get(i) for i in range(n)]
    return Scoring(arr, os.path.basename(path), "tech", arousals, lights)


def parse_annotations(path, n_epochs_max=None):
    """Sleep-EDF 的 *-Hypnogram.edf。"""
    import mne
    ann = mne.read_annotations(path)
    stages = {}
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        st = norm_stage(desc)
        i0 = int(round(onset / EPOCH))
        for k in range(max(1, int(round(dur / EPOCH)))):
            if i0 + k >= 0:
                stages[i0 + k] = st
    n = (max(stages) + 1) if stages else 0
    if n_epochs_max:
        n = min(n, n_epochs_max)
    return Scoring([stages.get(i) for i in range(n)], os.path.basename(path), "tech")


def load_technician(path, start_datetime, n_epochs_max=None):
    f, kind = find_technician_file(path)
    if f is None:
        return None
    if kind == "remlogic":
        if start_datetime is None:
            return None
        return parse_remlogic(f, start_datetime, n_epochs_max)
    return parse_annotations(f, n_epochs_max)


# ---------------------------------------------------------------- 自動判期

CACHE_DIR = os.path.expanduser("~/.edf_viewer_cache")


def _cache_key(path):
    real = os.path.realpath(path)
    stt = os.stat(real)
    h = hashlib.md5(f"{real}|{stt.st_size}|{int(stt.st_mtime)}|yasa1".encode()).hexdigest()
    return os.path.join(CACHE_DIR, h + ".json")


def cached_auto(path):
    p = _cache_key(path)
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            return Scoring(d["stages"], "YASA", "auto", confidence=d.get("conf"))
        except Exception:
            return None
    return None


def _save_auto(path, sc):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        json.dump({"stages": sc.stages, "conf": sc.confidence}, open(_cache_key(path), "w"))
    except Exception:
        pass


def auto_stage(rec, path, progress=None):
    """YASA 自動判期。

    先把 EEG/EOG/EMG 各自用原生取樣率讀出來、重取樣到 100 Hz 再交給 YASA，
    避免經過「先升頻到 500 Hz 再降回 100 Hz」那一手。
    """
    import mne
    import yasa
    from scipy.signal import resample_poly

    def pick(*keys):
        for k in keys:
            for n in rec.ch_names:
                if k.lower() in n.lower():
                    return n
        return None

    eeg, eog, emg = pick("eeg", "c3", "c4", "fpz"), pick("eog", "loc", "e1"), pick("emg", "chin")
    if eeg is None:
        raise ValueError("找不到 EEG 通道，無法自動判期")

    names = [n for n in (eeg, eog, emg) if n]
    fs_out = 100.0
    chunks = {n: [] for n in names}
    step = 600.0
    n_step = int(np.ceil(rec.duration / step))
    for k in range(n_step):
        a, b = k * step, min(rec.duration, (k + 1) * step)
        data = rec.read(names, a, b)
        for n in names:
            y = data[n][1]
            fs = rec.sfreqs[rec.ch_names.index(n)]
            g = np.gcd(int(round(fs)), int(fs_out))
            chunks[n].append(resample_poly(y, int(fs_out) // g, int(round(fs)) // g))
        if progress:
            progress((k + 1) / n_step)

    sig = [np.concatenate(chunks[n]) for n in names]
    m = min(len(s) for s in sig)
    arr = np.vstack([s[:m] for s in sig])
    if names[0] == eeg and rec.units[rec.ch_names.index(eeg)] in ("uV", "µV"):
        arr = arr * 1e-6                       # MNE / YASA 都吃伏特

    types = ["eeg"] + (["eog"] if eog else []) + (["emg"] if emg else [])
    info = mne.create_info([n for n in names], fs_out, types, verbose="ERROR")
    raw = mne.io.RawArray(arr, info, verbose="ERROR")

    sls = yasa.SleepStaging(raw, eeg_name=eeg, eog_name=eog, emg_name=emg)
    hyp = sls.predict()
    stages = [norm_stage(s) for s in np.asarray(hyp.hypno).astype(str)]
    conf = [float(x) for x in np.asarray(hyp.proba).max(axis=1)]
    sc = Scoring(stages, "YASA", "auto", confidence=conf)
    _save_auto(path, sc)
    return sc


# ---------------------------------------------------------------- 比對


def agreement(tech, auto):
    """回傳 (一致率, kappa, 重疊 epoch 數, 混淆矩陣 dict, 不一致的 epoch 索引)。"""
    if tech is None or auto is None:
        return None
    n = min(len(tech), len(auto))
    idx = [i for i in range(n) if tech.stages[i] in STAGES and auto.stages[i] in STAGES]
    if not idx:
        return None
    a = [tech.stages[i] for i in idx]
    b = [auto.stages[i] for i in idx]
    same = sum(1 for x, y in zip(a, b) if x == y)
    labels = [s for s in STAGES if s in set(a) | set(b)]
    cm = {r: {c: 0 for c in labels} for r in labels}
    for x, y in zip(a, b):
        cm[x][y] += 1
    # Cohen's kappa
    tot = len(a)
    po = same / tot
    pe = sum((sum(cm[r].values()) / tot) * (sum(cm[x][r] for x in labels) / tot) for r in labels)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    disagree = [i for i in idx if tech.stages[i] != auto.stages[i]]
    return po, kappa, tot, cm, disagree
