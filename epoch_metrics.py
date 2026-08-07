"""廠商（TS.EXE / K&Y）逐-epoch 生理指標 .TXT 的解析。

每個 session 資料夾裡除了 RemLogic 的判讀輸出，還有一份廠商軟體算好的
每 30 秒指標檔（例如 `170725H.TXT`），第一欄是 epoch 序號，時間從記錄起點算起。

    ; .\\170725H.txt  c0:00:00 t22:12:36
    ; No.  Time   MPF    RR    TP    HF    LF LF/HF   LF% delta theta alpha  beta ...
    ;       min    Hz  unit ln(ms^2) ...
        1 0:00:00 9.477  1024 9.163 6.767 8.466 1.699 84.95 5.443 3.333 3.943 ...

用這份檔的理由：它就是既有分析與已發表結果所用的那把尺。自己從原始訊號重算
會得到不同的數值（廠商與自算的 LF/HF 實測差約 4.9 倍），兩者不可混用。
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

EPOCH = 30.0

# 表頭寫 uV^2，但實測值會出現負數 —— 這幾欄實際上是 ln(µV²)，要看線性值得取 exp。
LN_COLS = {"TP", "HF", "LF", "VLF", "delta", "theta", "alpha", "beta",
           "Sigma", "alleeg", "EOG", "EMG"}

# 欄位說明：廠商演算法與欄位定義見 use.bpp
NOTES = {
    "MPF": "平均頻率（mean power frequency）",
    "RR": "取自 RR 間期的頻譜 fft_mean，不是時域平均心搏間期",
    "TP": "總功率 0.003–0.4 Hz",
    "HF": "高頻 0.15–0.4 Hz",
    "LF": "低頻 0.04–0.15 Hz",
    "VLF": "極低頻 0.003–0.04 Hz",
    "LF/HF": "LF 與 HF 的比值（不等於交感／副交感平衡）",
    "LF%": "LF / (LF+HF) × 100",
    "delta": "0.5–4 Hz",
    "theta": "4–8 Hz",
    "alpha": "8–13 Hz",
    "beta": "13–32 Hz",
    "Sigma": "12–14 Hz（與 alpha/beta 重疊）",
    "EMG": "肌電 RMS",
    "ACT": "體動量（加速規）",
    "PST": "皮膚溫度，已套廠商校正公式",
}


class EpochMetrics:
    """一份廠商逐-epoch 指標檔。"""

    def __init__(self, path, rows, cols, units, start_clock):
        self.path = path
        self.source = os.path.basename(path)
        self.columns = cols                    # 不含 No./Time
        self.units = units                     # {欄位: 單位字串}
        self.start_clock = start_clock         # 檔頭的 t hh:mm:ss
        self._rows = rows                      # {epoch index: {欄位: float|None}}
        self.n = (max(rows) + 1) if rows else 0

    def at(self, t):
        """指定秒數落在哪一格，回傳該格的指標 dict（沒有就 None）。"""
        return self._rows.get(int(t // EPOCH))

    def series(self, col):
        """整份記錄的某一欄，缺值為 nan。"""
        return np.array([(self._rows.get(i) or {}).get(col, np.nan)
                         for i in range(self.n)], dtype=float)

    def is_ln(self, col):
        return col in LN_COLS

    def note(self, col):
        return NOTES.get(col, "")


def _decode(raw):
    for enc in ("big5", "cp950", "utf-8-sig", "utf-8", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", "replace")


_SESSION_TOKEN = re.compile(r"\d{6,8}[A-Za-z]?")


def _tokens(name):
    """從檔名抽出場次代號（170725H、22082916 之類）。"""
    return {m.group(0).upper() for m in _SESSION_TOKEN.finditer(name)}


def _header_clock(path):
    head = _decode(open(path, "rb").read(200))
    m = re.search(r"\bt(\d{1,2}):(\d{2}):(\d{2})", head)
    return (int(m.group(1)) % 24, int(m.group(2)), int(m.group(3))) if m else None


def find_vendor_txt(path, start_datetime=None):
    """找出這個 EDF 對應的廠商指標檔，回傳 (路徑, 判定理由)；找不到回傳 (None, 理由)。

    第一階段的資料夾裡，一位受測者的 8 個溫度場次共用同一層目錄，而且不見得
    每一場都有 TXT —— 只取「目錄裡第一個」會把別場的生理數據貼到這場的波形旁，
    而且無聲無息。所以配對要有正面證據：
      1. 檔名裡的場次代號（170725H / 22082916）相符
      2. TXT 表頭的 `t hh:mm:ss` 與 EDF 起始時刻相符（誤差 ≤ 60 秒）
    兩者都拿不到證據時，寧可回報「配不到」，不猜。
    """
    real = os.path.realpath(path)
    d = os.path.dirname(real)
    stem = os.path.splitext(os.path.basename(real))[0]
    want = _tokens(stem)

    cands = []
    for p in sorted(set(glob.glob(os.path.join(d, "*.TXT"))
                        + glob.glob(os.path.join(d, "*.txt")))):
        head = open(p, "rb").read(400)
        if b"RemLogic" in head:                       # 那是判讀輸出，不是指標檔
            continue
        if b"MPF" in head and b"No." in head:
            cands.append(p)
    if not cands:
        return None, "資料夾裡沒有廠商指標檔"

    scored = []
    for p in cands:
        name = os.path.splitext(os.path.basename(p))[0]
        tok_ok = bool(want & _tokens(name))
        clock_ok = None
        if start_datetime is not None:
            hc = _header_clock(p)
            if hc is not None:
                clock_ok = abs((hc[0] * 3600 + hc[1] * 60 + hc[2])
                               - (start_datetime.hour * 3600 + start_datetime.minute * 60
                                  + start_datetime.second)) <= 60
        if clock_ok is False:                         # 時鐘明確對不上 → 直接排除
            continue
        scored.append((2 * tok_ok + (1 if clock_ok else 0), tok_ok, clock_ok, p))

    scored.sort(reverse=True)
    if not scored or scored[0][0] == 0:
        return None, f"目錄裡有 {len(cands)} 個指標檔，但沒有一個對得上這場記錄"

    _, tok_ok, clock_ok, best = scored[0]
    why = "場次代號相符" if tok_ok else ""
    if clock_ok:
        why = (why + "、" if why else "") + "起始時刻相符"
    return best, why


def parse(path):
    lines = _decode(open(path, "rb").read()).splitlines()

    start_clock = None
    hdr_i = None
    for i, l in enumerate(lines[:10]):
        m = re.search(r"\bt(\d{1,2}:\d{2}:\d{2})", l)
        if m:
            start_clock = m.group(1)
        if l.lstrip(";").split()[:2] == ["No.", "Time"]:
            hdr_i = i
    if hdr_i is None:
        raise ValueError("找不到 `; No. Time ...` 表頭")

    names = lines[hdr_i].lstrip(";").split()          # No. Time MPF RR ...
    unit_row = (lines[hdr_i + 1].lstrip(";").split()
                if hdr_i + 1 < len(lines) and lines[hdr_i + 1].lstrip().startswith(";")
                else [])
    cols = names[2:]                                  # 去掉 No. 與 Time
    # 單位列少了 No. 那格，對齊到 Time 之後
    units = {c: (unit_row[k + 1] if k + 1 < len(unit_row) else "")
             for k, c in enumerate(cols)}
    for c in cols:                                    # 表頭單位不可靠，以實際尺度為準
        if c in LN_COLS and not units[c].startswith("ln"):
            units[c] = f"ln({units[c] or 'µV²'})"

    rows = {}
    for l in lines[hdr_i + 1:]:
        f = l.split()
        if len(f) < 3 or not f[0].isdigit():
            continue
        idx = int(f[0]) - 1                           # No. 從 1 起算
        vals = {}
        for c, v in zip(cols, f[2:]):
            try:
                vals[c] = float(v)
            except ValueError:
                vals[c] = None                        # 廠商用 '-' 表示缺值
        rows[idx] = vals
    return EpochMetrics(path, rows, cols, units, start_clock)


def load(path, start_datetime=None):
    """回傳 (EpochMetrics|None, 說明字串)。"""
    p, why = find_vendor_txt(path, start_datetime)
    if p is None:
        return None, why
    m = parse(p)
    m.match_reason = why
    return m, why
