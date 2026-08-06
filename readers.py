"""訊號檔讀取器：EDF / EDF+ / BDF 與 K&Y-SS001 (.RAW)。

兩種讀取器都是「惰性讀取」：只把畫面上要看的時間窗從硬碟撈出來，
所以開一個 21 小時的全夜檔跟開 10 分鐘的清醒檔一樣快。
每個通道都保留原生取樣率（EEG 125 Hz、ECG 500 Hz 不會被硬拉成同一個 fs）。
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import struct

import numpy as np

# ---------------------------------------------------------------- 共用介面


class Channel:
    def __init__(self, name, fs, unit, phys_min, phys_max, dig_min, dig_max, kind=None):
        self.name = name
        self.fs = float(fs)
        self.unit = unit
        self.phys_min, self.phys_max = float(phys_min), float(phys_max)
        self.dig_min, self.dig_max = float(dig_min), float(dig_max)
        self.kind = kind or guess_kind(name)

    @property
    def gain(self):
        span = self.dig_max - self.dig_min
        return (self.phys_max - self.phys_min) / span if span else 1.0

    def to_physical(self, dig):
        return (dig.astype(np.float64) - self.dig_min) * self.gain + self.phys_min


_KIND_PATTERNS = [
    ("EEG", r"eeg|c3|c4|o1|o2|f3|f4|fp|fz|cz|pz"),
    ("EOG", r"eog|loc|roc|e1|e2"),
    ("EMG", r"emg|chin"),
    ("ECG", r"ecg|ekg"),
    ("RESP", r"resp|flow|thor|abdo|snore|nasal"),
    ("SpO2", r"spo2|sao2|sat|pleth"),
    ("TEMP", r"temp|skin"),
    ("ACC", r"^[xyz]$|acc|actigraph|act"),
]


def guess_kind(name):
    n = name.strip().lower()
    for kind, pat in _KIND_PATTERNS:
        if re.search(pat, n):
            return kind
    return "OTHER"


class SignalFile:
    """共同介面：channels / duration / start_datetime / read()"""

    path: str
    fmt: str
    channels: list
    duration: float
    start_datetime = None
    meta: dict

    @property
    def ch_names(self):
        return [c.name for c in self.channels]

    def read(self, idx, tmin, tmax):
        """回傳 (t0_實際, fs, 物理量陣列)。實際起點會對齊到內部區塊邊界。"""
        raise NotImplementedError

    def read_window(self, idx, tmin, tmax, pad=0.0):
        """取 [tmin-pad, tmax+pad] 的資料，回傳 (時間軸, 值, fs, 前置pad樣本數)。"""
        a = max(0.0, tmin - pad)
        b = min(self.duration, tmax + pad)
        t0, fs, y = self.read(idx, a, b)
        t = t0 + np.arange(len(y)) / fs
        keep = (t >= a - 1e-9) & (t < b - 1e-9)
        t, y = t[keep], y[keep]
        n_pre = int(np.searchsorted(t, tmin))
        return t, y, fs, n_pre


# ---------------------------------------------------------------- EDF / BDF


class EDFFile(SignalFile):
    """寬容版 EDF/EDF+/BDF 讀取器。

    這批 KY1_12 轉出的 EDF 在 header 有格式瑕疵（pyedflib 會直接拒絕開檔），
    這裡只按位置解析欄位、不做合規檢查，所以照樣讀得出來。
    """

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        h = self._fh.read(256)
        if len(h) < 256:
            raise ValueError("檔案太短，不是 EDF")
        self.is_bdf = h[0:1] == b"\xff" or b"BIOSEMI" in h[1:8]
        self.fmt = "BDF" if self.is_bdf else "EDF"
        self.bps = 3 if self.is_bdf else 2

        self.patient = _s(h[8:88])
        self.recording = _s(h[88:168])
        self.start_datetime = _parse_edf_datetime(_s(h[168:176]), _s(h[176:184]))
        self.header_bytes = _int(h[184:192], 256)
        self.reserved = _s(h[192:236])
        n_records = _int(h[236:244], -1)
        self.rec_dur = _float(h[244:252], 1.0) or 1.0
        ns = _int(h[252:256], 0)
        if ns <= 0:
            raise ValueError("EDF header 沒有有效的通道數")

        hh = self._fh.read(self.header_bytes - 256)

        def block(off, w):
            return [_s(hh[off + w * i: off + w * (i + 1)]) for i in range(ns)]

        o = 0
        labels = block(o, 16); o += 16 * ns
        transducers = block(o, 80); o += 80 * ns
        units = block(o, 8); o += 8 * ns
        pmin = [_float(x, 0.0) for x in block(o, 8)]; o += 8 * ns
        pmax = [_float(x, 1.0) for x in block(o, 8)]; o += 8 * ns
        dmin = [_float(x, -32768.0) for x in block(o, 8)]; o += 8 * ns
        dmax = [_float(x, 32767.0) for x in block(o, 8)]; o += 8 * ns
        o += 80 * ns  # prefiltering
        spr = [_int(x, 0) for x in block(o, 8)]

        self.samples_per_record = spr
        self.rec_samples = sum(spr)
        self.rec_bytes = self.rec_samples * self.bps
        self._offsets = np.cumsum([0] + spr)

        size = os.path.getsize(path)
        avail = max(0, (size - self.header_bytes) // self.rec_bytes) if self.rec_bytes else 0
        self.n_records = avail if n_records <= 0 else min(n_records, avail)
        self.duration = self.n_records * self.rec_dur

        self.channels, self._picks = [], []
        for i in range(ns):
            if labels[i].lower().replace(" ", "") in ("edfannotations", "bdfannotations"):
                continue
            if spr[i] <= 0:
                continue
            lo, hi = (pmin[i], pmax[i]) if pmax[i] != pmin[i] else (dmin[i], dmax[i])
            self.channels.append(
                Channel(labels[i], spr[i] / self.rec_dur, units[i] or "a.u.",
                        lo, hi, dmin[i], dmax[i])
            )
            self._picks.append(i)
            self.channels[-1].transducer = transducers[i]

        self.meta = {
            "格式": self.fmt,
            "受測者欄": self.patient,
            "記錄欄": self.recording,
            "record 長度 (s)": self.rec_dur,
            "record 數": self.n_records,
            "header bytes": self.header_bytes,
        }

    def read(self, idx, tmin, tmax):
        ci = self._picks[idx]
        ch = self.channels[idx]
        n = self.samples_per_record[ci]
        r0 = max(0, int(np.floor(tmin / self.rec_dur)))
        r1 = min(self.n_records, int(np.ceil(tmax / self.rec_dur)))
        r1 = max(r1, r0 + 1)
        if r0 >= self.n_records:
            return 0.0, ch.fs, np.zeros(0)
        self._fh.seek(self.header_bytes + r0 * self.rec_bytes)
        buf = self._fh.read((r1 - r0) * self.rec_bytes)
        nrec = len(buf) // self.rec_bytes
        if nrec == 0:
            return 0.0, ch.fs, np.zeros(0)
        s0, s1 = self._offsets[ci], self._offsets[ci] + n
        if self.bps == 2:
            arr = np.frombuffer(buf[: nrec * self.rec_bytes], dtype="<i2")
            arr = arr.reshape(nrec, self.rec_samples)[:, s0:s1].reshape(-1)
        else:
            raw = np.frombuffer(buf[: nrec * self.rec_bytes], dtype=np.uint8)
            raw = raw.reshape(nrec, self.rec_samples, 3)[:, s0:s1, :].reshape(-1, 3)
            arr = (raw[:, 0].astype(np.int32)
                   | (raw[:, 1].astype(np.int32) << 8)
                   | (raw[:, 2].astype(np.int8).astype(np.int32) << 16))
        return r0 * self.rec_dur, ch.fs, ch.to_physical(arr)


def _s(b):
    return b.decode("latin1", "replace").strip()


def _int(b, default):
    try:
        return int(float(_s(b) if isinstance(b, bytes) else b))
    except Exception:
        return default


def _float(b, default):
    try:
        return float(_s(b) if isinstance(b, bytes) else b)
    except Exception:
        return default


def _parse_edf_datetime(d, t):
    try:
        dd, mm, yy = [int(x) for x in re.split(r"[.\-/]", d)[:3]]
        hh, mi, ss = [int(x) for x in re.split(r"[.:\-]", t)[:3]]
        year = 2000 + yy if yy < 85 else 1900 + yy
        return _dt.datetime(year, mm, dd, hh, mi, ss)
    except Exception:
        return None


# ---------------------------------------------------------------- K&Y RAW

# 已知機型組態。量程取自原廠 convert(EDF) 產生的 EDF header（digital 0..4095），
# 全部 186 個 EDF 的單位與 physical min/max 都一致，只有各通道取樣率有兩種配置。
KY_MONTAGE_8 = [
    ("EMG", "uV", -900.0, 900.0),
    ("EOG", "uV", -900.0, 900.0),
    ("EEG", "uV", -450.0, 450.0),
    ("ECG", "mV", -3.6, 3.6),
    ("Temp", "V", 0.0, 1.8),
    ("X", "G", -3.6, 3.6),
    ("Y", "G", -3.6, 3.6),
    ("Z", "G", -3.6, 3.6),
]
KY_CONFIGS = {
    8: KY_MONTAGE_8,                        # PSG / 清醒記錄
    1: [("ECG", "mV", -3.6, 3.6)],          # Holter 單導程
}


class KYRawFile(SignalFile):
    """K&Y-SS001 (.RAW) — KY1_12 記錄器原始檔。

    格式（逆向工程，已對照原廠 convert(EDF) 輸出逐點驗證 100% 相符）::

        0x000  b'K&Y-SS001\\x1a'      magic
        0x00e  'KY1_12'              機型字串
        0x022  uint16                通道數 nch
        0x026  17 × pascal string    取樣率；[0] = 多工基頻，[1..nch] = 各通道
        ...    uint16 type、nch bytes gain code
        0x200  資料：little-endian uint16，值 = digital(0..4095) << 4（低 4 bit 未使用）

    資料以基頻為節拍多工：第 i 個 frame 依通道宣告順序，輸出所有滿足
    ``i % (基頻 / 該通道取樣率) == 0`` 的通道各一個樣本。
    """

    MAGIC = b"K&Y-SS001"
    DATA_OFFSET = 512

    def __init__(self, path):
        self.path = path
        self.fmt = "K&Y RAW"
        head = open(path, "rb").read(self.DATA_OFFSET)
        if not head.startswith(self.MAGIC):
            raise ValueError("不是 K&Y-SS001 RAW 檔")
        self.device = head[14:20].split(b"\x00")[0].decode("latin1")
        self.version = struct.unpack("<H", head[34:36])[0]
        nch = struct.unpack("<H", head[36:38])[0]

        rates, o = [], 38
        for _ in range(17):
            ln = head[o]
            rates.append(_float(head[o + 1: o + 1 + ln], 0.0))
            o += 1 + ln
        self.type = struct.unpack("<H", head[o: o + 2])[0]
        self.gain_codes = list(head[o + 2: o + 2 + nch])

        if not (0 < nch <= 16) or rates[0] <= 0:
            raise ValueError(f"RAW header 異常：nch={nch} base={rates[0]}")
        self.base_rate = rates[0]
        ch_rates = rates[1: nch + 1]
        if any(r <= 0 for r in ch_rates):
            raise ValueError("RAW header 取樣率異常")

        cfg = KY_CONFIGS.get(nch)
        if cfg is None:  # 未知組態：先給通用名稱，之後可在介面上改
            cfg = [(f"CH{i + 1}", "a.u.", 0.0, 4095.0) for i in range(nch)]
            self.known_config = False
        else:
            self.known_config = True
        self.channels = [
            Channel(nm, ch_rates[i], un, lo, hi, 0.0, 4095.0)
            for i, (nm, un, lo, hi) in enumerate(cfg)
        ]

        # 多工樣板
        self._decim = [int(round(self.base_rate / r)) for r in ch_rates]
        g = 1
        for d in self._decim:
            g = _lcm(g, d)
        self.group_frames = g
        slots = [c for i in range(g) for c in range(nch) if i % self._decim[c] == 0]
        self.slots = slots
        self._cols = [np.array([i for i, s in enumerate(slots) if s == c]) for c in range(nch)]
        self._slot_len = len(slots)

        size = os.path.getsize(path)
        self._mm = np.memmap(path, dtype="<u2", mode="r", offset=self.DATA_OFFSET)
        self.n_groups = len(self._mm) // self._slot_len
        self.group_dur = self.group_frames / self.base_rate
        self.duration = self.n_groups * self.group_dur

        self.start_datetime = _ky_start_datetime(path)
        self.meta = {
            "格式": "K&Y-SS001 RAW",
            "機型": self.device,
            "version": self.version,
            "type": self.type,
            "多工基頻 (Hz)": self.base_rate,
            "通道數": nch,
            "gain code": self.gain_codes,
            "檔案大小 (MB)": round(size / 1e6, 1),
            "通道組態": "已知機型" if self.known_config else "未知組態（名稱為暫定）",
        }

    def read(self, idx, tmin, tmax):
        ch = self.channels[idx]
        g0 = max(0, int(np.floor(tmin / self.group_dur)))
        g1 = min(self.n_groups, int(np.ceil(tmax / self.group_dur)))
        g1 = max(g1, g0 + 1)
        if g0 >= self.n_groups:
            return 0.0, ch.fs, np.zeros(0)
        blk = self._mm[g0 * self._slot_len: g1 * self._slot_len]
        n = len(blk) // self._slot_len
        if n == 0:
            return 0.0, ch.fs, np.zeros(0)
        dig = np.asarray(blk[: n * self._slot_len].reshape(n, self._slot_len)[:, self._cols[idx]]).reshape(-1) >> 4
        return g0 * self.group_dur, ch.fs, ch.to_physical(dig)


def _lcm(a, b):
    from math import gcd
    return a * b // gcd(a, b)


def _ky_start_datetime(path):
    """RAW header 沒有時間戳；依序試 同名 EDF → 同名 TXT(`t HH:MM:SS`) → 檔案時間。"""
    stem = os.path.splitext(path)[0]
    for ext in (".EDF", ".edf"):
        if os.path.exists(stem + ext):
            try:
                h = open(stem + ext, "rb").read(256)
                dt = _parse_edf_datetime(_s(h[168:176]), _s(h[176:184]))
                if dt:
                    return dt
            except Exception:
                pass
    mt = _dt.datetime.fromtimestamp(os.path.getmtime(path))
    for ext in (".TXT", ".txt"):
        if os.path.exists(stem + ext):
            try:
                first = open(stem + ext, "rb").read(200).decode("latin1", "replace")
                m = re.search(r"t\s*(\d{1,2}):(\d{2}):(\d{2})", first)
                if m:
                    hh, mm, ss = (int(x) for x in m.groups())
                    return _dt.datetime.combine(mt.date(), _dt.time(hh % 24, mm, ss))
            except Exception:
                pass
    return mt


# ---------------------------------------------------------------- dispatcher


def open_signal_file(path):
    with open(path, "rb") as f:
        magic = f.read(16)
    if magic.startswith(KYRawFile.MAGIC):
        return KYRawFile(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".raw":
        return KYRawFile(path)
    return EDFFile(path)


# ---------------------------------------------------------------- 統一介面

UV_TYPES = {"eeg", "eog", "emg", "ecg", "seeg", "ecog"}


class Recording:
    """把 MNE Raw 與 K&Y .RAW 包成同一個介面。"""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        if path.lower().endswith(".raw"):
            self._ky = KYRawFile(path)
            self.kind = "ky"
            self.ch_names = self._ky.ch_names
            self.sfreqs = [c.fs for c in self._ky.channels]
            self.units = [c.unit for c in self._ky.channels]
            self.duration = self._ky.duration
            self.meas_date = self._ky.start_datetime
        else:
            import mne
            reader = mne.io.read_raw_bdf if path.lower().endswith(".bdf") else mne.io.read_raw_edf
            self._raw = reader(path, preload=False, verbose="ERROR")
            # 混合取樣率的 EDF，MNE 惰性讀取只有在視窗起點對齊 data record 邊界時
            # 才和原始資料一致，否則低取樣率通道的升頻相位會跑掉（實測差到 48 µV）。
            # 所以記下 record 長度，讀取時一律往前對齊到邊界再裁掉多的。
            try:
                h = open(path, "rb").read(256)
                self._rec_dur = _float(h[244:252], 0.0) or 0.0
            except Exception:
                self._rec_dur = 0.0
            self.kind = "mne"
            self.ch_names = list(self._raw.ch_names)
            fs = float(self._raw.info["sfreq"])
            self.sfreqs = [fs] * len(self.ch_names)
            types = self._raw.get_channel_types()
            self.units = ["µV" if t in UV_TYPES else "a.u." for t in types]
            self._scale = [1e6 if t in UV_TYPES else 1.0 for t in types]
            self.duration = self._raw.n_times / fs
            md = self._raw.info.get("meas_date")
            self.meas_date = md.replace(tzinfo=None) if md is not None else None

    def read(self, names, t0, t1):
        """回傳 {通道: (時間軸, 值)}，只從硬碟撈這段。"""
        out = {}
        if self.kind == "ky":
            for n in names:
                i = self.ch_names.index(n)
                start, fs, y = self._ky.read(i, t0, t1)
                t = start + np.arange(len(y)) / fs
                m = (t >= t0) & (t < t1)
                out[n] = (t[m], y[m])
        else:
            fs = self.sfreqs[0]
            a_req, b = int(round(t0 * fs)), int(round(t1 * fs))
            b = min(b, self._raw.n_times)
            snap = int(round(self._rec_dur * fs)) if self._rec_dur > 0 else 1
            if snap > 1:                      # 頭尾都要對齊到 record 邊界
                a = (a_req // snap) * snap
                b_read = min(self._raw.n_times, -(-b // snap) * snap)
            else:
                a, b_read = a_req, b
            drop = a_req - a
            picks = [self.ch_names.index(n) for n in names]
            data = self._raw.get_data(picks=picks, start=a, stop=b_read)[:, drop:drop + (b - a_req)]
            t = a_req / fs + np.arange(data.shape[1]) / fs
            for k, n in enumerate(names):
                out[n] = (t, data[k] * self._scale[self.ch_names.index(n)])
        return out


