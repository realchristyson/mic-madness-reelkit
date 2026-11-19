#!/usr/bin/env python3
"""ReelKit: turn a raw performance reel + the song MP3 into a finished Instagram reel.

  python3 reelkit.py check
  python3 reelkit.py sync     --video V --song S [--near SEC]
  python3 reelkit.py captions --video V --song S --offset O --lyrics L.txt
  python3 reelkit.py preview  --video V --hook "TEXT 🔥" [--captions C.json] [--pos top|center]
  python3 reelkit.py render   --video V --song S --offset O --hook "TEXT 🔥" --font A|B|C
                              [--captions C.json] [--pos top|center] [--hook-end SEC] --out OUT.mp4

Rules baked in: the video's own sound is muted, the song is lined up to the video,
the picture is never trimmed, the warm grade goes on last and sits UNDER the text.
"""
import argparse, json, os, re, shutil, subprocess, sys, urllib.request, difflib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "reelkit_cache")
os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------- tools
def ffmpeg_bin():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("NO_FFMPEG: run  pip install imageio-ffmpeg  and try again")

FF = None
def ff(args, capture=False):
    global FF
    FF = FF or ffmpeg_bin()
    r = subprocess.run([FF, "-hide_banner", "-v", "error", "-y"] + args,
                       capture_output=True)
    if r.returncode != 0:
        sys.exit("FFMPEG_ERROR: " + r.stderr.decode(errors="ignore")[-1500:])
    return r.stdout if capture else None

def probe(path):
    """Duration, display size, HDR flag. Works without ffprobe."""
    global FF
    FF = FF or ffmpeg_bin()
    err = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True).stderr.decode(errors="ignore")
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    hdr = bool(re.search(r"arib-std-b67|smpte2084|bt2020", err))
    has_v = "Video:" in err
    return {"dur": dur, "hdr": hdr, "has_video": has_v, "raw": err}

def frame(path, t, width=None):
    vf = ["-vf", f"scale={width}:-2"] if width else []
    png = ff(["-ss", f"{t:.3f}", "-i", path, "-frames:v", "1"] + vf + ["-f", "image2pipe", "-vcodec", "png", "-"], capture=True)
    from io import BytesIO
    return Image.open(BytesIO(png)).convert("RGB")

def out_size(path):
    im = frame(path, 0.0)
    W, H = im.size
    if W > 1080:
        H = int(round(H * 1080 / W / 2) * 2); W = 1080
    return W - W % 2, H - H % 2

def load_audio(path, sr=22050, start=None, dur=None):
    a = []
    if start is not None: a += ["-ss", f"{max(0, start):.3f}"]
    a += ["-i", path]
    if dur is not None: a += ["-t", f"{dur:.3f}"]
    raw = ff(a + ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture=True)
    return np.frombuffer(raw, dtype=np.float32).copy()

# ---------------------------------------------------------------- features
SR, NFFT, HOP = 22050, 2048, 512
FPS = SR / HOP

def spec(x, nfft=NFFT, hop=HOP):
    if len(x) < nfft: x = np.pad(x, (0, nfft - len(x)))
    win = np.hanning(nfft).astype(np.float32)
    n = 1 + (len(x) - nfft) // hop
    out = np.empty((n, nfft // 2 + 1), np.float32)
    for i in range(0, n, 2000):
        j = min(n, i + 2000)
        idx = np.arange(i, j)[:, None] * hop + np.arange(nfft)[None, :]
        out[i:j] = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    return out

def features(x, nfft=NFFT, hop=HOP, sr=SR):
    S = spec(x, nfft, hop)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    L = np.log1p(100 * S)
    band = (f > 250) & (f < 5000)            # phone speakers have no bass
    flux = np.maximum(0, np.diff(L[:, band], axis=0, prepend=L[:1, band])).sum(1)
    cb = (f > 80) & (f < 5000)
    pc = (np.round(12 * np.log2(f[cb] / 440.0) + 69) % 12).astype(int)
    chroma = np.zeros((S.shape[0], 12), np.float32)
    for k in range(12):
        chroma[:, k] = S[:, cb][:, pc == k].sum(1)
    chroma /= (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9)
    return flux.astype(np.float32), chroma

def _xcorr(a, b):
    """sum_t a[t]*b[t+L] for every L in 0..len(b)-len(a)."""
    n = len(a) + len(b)
    nf = 1 << (n - 1).bit_length()
    r = np.fft.irfft(np.fft.rfft(b, nf) * np.conj(np.fft.rfft(a, nf)), nf)
    return r[: len(b) - len(a) + 1]

def pearson_scan(clip, song):
    """Pearson correlation of clip (m,d) against every window of song (n,d)."""
    if clip.ndim == 1: clip, song = clip[:, None], song[:, None]
    m = len(clip)
    c = clip - clip.mean(0)
    num = sum(_xcorr(c[:, k], song[:, k]) for k in range(clip.shape[1]))
    cs = np.vstack([np.zeros((1, song.shape[1])), np.cumsum(song, 0)])
    cs2 = np.vstack([np.zeros((1, song.shape[1])), np.cumsum(song ** 2, 0)])
    s1 = cs[m:] - cs[:-m]; s2 = cs2[m:] - cs2[:-m]
    var_s = (s2 - s1 ** 2 / m).sum(1)
    var_c = (c ** 2).sum()
    return num / np.sqrt(np.maximum(var_s * var_c, 1e-12))

def find_peaks(score, sep, k=3):
    s = score.copy(); out = []
    for _ in range(k):
        i = int(np.argmax(s))
        if not np.isfinite(s[i]) or s[i] <= -1: break
        out.append(i)
        s[max(0, i - sep): i + sep + 1] = -np.inf
    return out


# ---------------------------------------------------------------- Shazam-style landmarks
def peaks2d(L, ft=10, tt=5, per_sec=90):
    """Spectral peaks: local maxima in a (2ft+1)x(2tt+1) box, strongest per_sec per second."""
    m = L.copy()
    for ax, r in ((0, tt), (1, ft)):
        out = m.copy()
        for k in range(1, r + 1):
            out = np.maximum(out, np.roll(m, k, ax)); out = np.maximum(out, np.roll(m, -k, ax))
        m = out
    loc = (L == m) & (L > np.median(L) + 1.0)
    t, f = np.nonzero(loc)
    v = L[t, f]
    keep = max(1, int(per_sec * L.shape[0] / FPS))
    o = np.argsort(-v)[:keep]
    o = o[np.argsort(t[o])]
    return t[o], f[o]

def hashes(x):
    S = spec(x)
    fr = np.fft.rfftfreq(NFFT, 1 / SR)
    b0, b1 = np.searchsorted(fr, 300), np.searchsorted(fr, 5000)
    L = np.log1p(100 * S[:, b0:b1])
    t, f = peaks2d(L)
    H = []
    for i in range(len(t)):
        n = 0
        for j in range(i + 1, min(len(t), i + 40)):
            dt = t[j] - t[i]
            if dt < 1: continue
            if dt > 45: break
            if abs(int(f[j]) - int(f[i])) > 80: continue
            H.append(((int(f[i]) << 16) | (int(f[j]) << 6) | int(dt), int(t[i])))
            n += 1
            if n >= 6: break
    return H

def landmark_offsets(clip, song, k=3):
    idx = {}
    for h, tt in hashes(song):
        idx.setdefault(h, []).append(tt)
    deltas = []
    for h, tc in hashes(clip):
        for ts in idx.get(h, ()):
            deltas.append(ts - tc)
    if not deltas: return []
    d = np.array(deltas)
    lo = d.min(); hist = np.bincount(d - lo)
    sm = hist + np.roll(hist, 1) + np.roll(hist, -1)
    out = []
    for _ in range(k):
        i = int(np.argmax(sm))
        if sm[i] <= 0: break
        out.append(((i + lo) / FPS, int(sm[i])))
        sm[max(0, i - int(FPS)): i + int(FPS) + 1] = 0
    return out

# ---------------------------------------------------------------- sync
def cmd_sync(a):
    info = probe(a.video)
    clip = load_audio(a.video)
    song = load_audio(a.song)
    D, Ds = len(clip) / SR, len(song) / SR
    if D < 2: sys.exit("VIDEO_TOO_SHORT")
    if np.sqrt(np.mean(clip ** 2)) < 1e-4:
        print(json.dumps({"status": "no_song_audio", "reason": "the video has no sound",
                          "video_seconds": round(D, 2)})); return
    cf, cc = features(clip); sf, sc = features(song)
    # pad the song with "silence" so the clip may start before the song or run past it
    pad = int(0.4 * len(cf))
    sfp = np.concatenate([np.full(pad, sf.mean()), sf, np.full(pad, sf.mean())])
    scp = np.vstack([np.tile(sc.mean(0), (pad, 1)), sc, np.tile(sc.mean(0), (pad, 1))])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    so = pearson_scan(z(cf), z(sfp)); sh = pearson_scan(cc, scp)
    score = 0.55 * so + 0.45 * sh
    lags = (np.arange(len(score)) - pad) / FPS
    if a.near is not None:
        score = np.where(np.abs(lags - a.near) <= 10, score, -np.inf)
    peaks = find_peaks(score, int(1.5 * FPS), 3)
    lmk = landmark_offsets(clip, song)
    if a.near is not None: lmk = [c for c in lmk if abs(c[0] - a.near) <= 10]
    lm_strong = len(lmk) > 0 and lmk[0][1] >= 12 and (len(lmk) < 2 or lmk[0][1] >= 2.5 * lmk[1][1])
    if lm_strong:   # landmarks win: put their best spot first
        li = int(round(lmk[0][0] * FPS)) + pad
        peaks = [li] + [p for p in peaks if abs(p - li) > 1.5 * FPS][:2]
    # fine refine with a sharper onset envelope
    fh = 128; ffps = SR / fh
    cfe, _ = features(clip, 1024, fh); sfe, _ = features(song, 1024, fh)
    def refine(lag, span=0.12, ref=None, seg=None):
        ref = cfe if ref is None else ref
        lo = int((lag - span) * ffps); hi = int((lag + span) * ffps) + 1
        best, bl = -2, lag
        padn = len(ref) + abs(lo) + 10
        sp = np.concatenate([np.full(padn, sfe.mean()), sfe, np.full(padn, sfe.mean())])
        rz = z(ref); spz = z(sp)
        for L in range(lo, hi):
            s0 = L + padn
            w = spz[s0: s0 + len(rz)]
            if len(w) < len(rz): continue
            wz = w - w.mean()
            r = float((rz * wz).sum() / (np.sqrt((wz ** 2).sum() * (rz ** 2).sum()) + 1e-9))
            if r > best: best, bl = r, L / ffps
        return bl, best
    cands = []
    for i in peaks:
        lag, r = refine(lags[i])
        cands.append({"offset": round(lag, 3), "score": round(float(score[i]), 3), "fine_r": round(r, 3)})
    top = cands[0]
    margin = top["score"] - (cands[1]["score"] if len(cands) > 1 else 0)
    # window check: does every part of the video agree with the same offset? (catches cuts)
    win, step = int(5 * ffps), int(2.5 * ffps); agree = tot = 0; bad = []
    for s0 in range(0, max(1, len(cfe) - win + 1), step):
        seg = cfe[s0: s0 + win]
        if seg.std() < 1e-6: continue
        tot += 1
        loc, r = refine(top["offset"] + s0 / ffps, span=0.6, ref=seg)
        loc -= s0 / ffps
        if abs(loc - top["offset"]) <= 0.045 and r > 0.15: agree += 1
        else: bad.append(round(s0 / ffps, 1))
    agree_pct = agree / max(1, tot)
    if lm_strong and lmk[0][1] >= 25: status = "locked"
    elif top["score"] >= 0.30 and margin >= 0.06 and agree_pct >= 0.6: status = "locked"
    elif lm_strong or top["score"] >= 0.2: status = "check"
    else: status = "no_song_audio"
    print(json.dumps({"status": status, "offset": top["offset"], "video_seconds": round(D, 2),
                      "song_seconds": round(Ds, 2), "margin": round(margin, 3),
                      "windows_agree_pct": round(100 * agree_pct), "windows_off_at_seconds": bad,
                      "candidates": cands, "landmarks": [[round(o, 3), n] for o, n in lmk], "hdr": info["hdr"]}, indent=1))

# ---------------------------------------------------------------- captions
def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'")).strip("'")

MODEL_URLS = ["https://github.com/realchristyson/mic-madness-reelkit/releases/download/model-v1/w2v_base_int8.onnx",
              "https://raw.githubusercontent.com/realchristyson/mic-madness-reelkit/main/model/w2v_base_int8.onnx"]
LABELS = ['-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y',
          'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z']

def ctc_greedy(lp):
    g = lp.argmax(-1); out = []; prev = -1
    for t in g:
        if t != prev and t != 0: out.append(LABELS[t])
        prev = t
    return "".join(out).replace("|", " ").lower()

def ctc_align(lp, tokens, star_lp=np.log(0.5)):
    """CTC Viterbi forced alignment. tokens: label ids, -1 = wildcard that soaks up extra audio."""
    T = lp.shape[0]; L = len(tokens); S = 2 * L + 1; NEG = -1e30
    emit = np.empty((T, L), np.float32)
    for j, tk in enumerate(tokens):
        emit[:, j] = star_lp if tk == -1 else lp[:, tk]
    skip = np.zeros(S, bool)
    for s in range(3, S, 2):
        skip[s] = tokens[(s - 1) // 2] != tokens[(s - 3) // 2] or tokens[(s - 1) // 2] == -1
    def e(t):
        r = np.empty(S); r[0::2] = lp[t, 0]; r[1::2] = emit[t]; return r
    dp = np.full(S, NEG); et = e(0); dp[0], dp[1] = et[0], et[1]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        m1 = np.concatenate([[NEG], dp[:-1]])
        m2 = np.where(skip, np.concatenate([[NEG, NEG], dp[:-2]]), NEG)
        st = np.vstack([dp, m1, m2]); arg = st.argmax(0); bp[t] = arg
        dp = st[arg, np.arange(S)] + e(t)
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    spans = [[None, None, 0.0, 0] for _ in tokens]
    for t in range(T - 1, -1, -1):
        if s % 2:
            j = (s - 1) // 2; sp = spans[j]
            sp[1] = t if sp[1] is None else sp[1]; sp[0] = t
            sp[2] += float(np.exp(emit[t, j])); sp[3] += 1
        s -= int(bp[t, s])
    return spans

NUMW = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
        "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve",
        "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
        "80": "eighty", "90": "ninety", "100": "hundred"}

def spoken(w):
    """Letters the aligner can use for one lyric word ('95' -> 'ninetyfive')."""
    s = w.lower().replace("’", "'").replace("&", "and").replace("$", "dollars")
    def num(m):
        n = m.group(0)
        if n in NUMW: return NUMW[n]
        if len(n) == 2: return NUMW.get(n[0] + "0", "") + NUMW.get(n[1], "")
        return "".join(NUMW[c] for c in n)
    s = re.sub(r"\d+", num, s)
    return re.sub(r"[^a-z']", "", s).strip("'")

def get_model(path=None):
    p = path or os.path.join(CACHE, "w2v_base_int8.onnx")
    if not os.path.exists(p) or os.path.getsize(p) < 50e6:
        if path: sys.exit("MODEL_NOT_FOUND: " + path)
        print("downloading the lyric timing model (95 MB, one time)...", file=sys.stderr)
        for u in MODEL_URLS:
            try:
                fetch(u, p)
                if os.path.getsize(p) > 50e6: break
            except Exception as e:
                print("  model download failed from", u, str(e)[:120], file=sys.stderr)
        if not os.path.exists(p) or os.path.getsize(p) < 50e6: raise RuntimeError("could not download model")
    return p

def cmd_captions(a):
    try:
        import onnxruntime as ort
    except Exception:
        print(json.dumps({"status": "no_aligner", "fix": "pip install onnxruntime"})); return
    D = probe(a.video)["dur"]
    raw = open(a.lyrics, encoding="utf-8", errors="ignore").read()
    raw = re.sub(r"\[[^\]]*\]|\(\s*x\s*\d+\s*\)|\(\s*\d+\s*x\s*\)", " ", raw, flags=re.I)  # [Chorus], (x2)
    words = [w for w in re.split(r"\s+", raw) if spoken(w)]
    if not words:
        print(json.dumps({"status": "no_lyrics"})); return
    pre = 1.0; seg0 = a.offset - pre
    x = load_audio(a.song, 16000, max(0.0, seg0), D + 2 * pre + min(0.0, seg0))
    if seg0 < 0: x = np.concatenate([np.zeros(int(-seg0 * 16000), np.float32), x])
    x = (x - x.mean()) / (x.std() + 1e-7)
    try:
        sess = ort.InferenceSession(get_model(a.model), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(json.dumps({"status": "no_aligner", "error": "model: " + str(e)[:200]})); return
    lps = []
    for i in range(0, len(x), 16000 * 30):           # 30 s chunks keep memory low
        c = x[i: i + 16000 * 30]
        if len(c) >= 800: lps.append(sess.run(None, {"audio": c[None]})[0][0])
    lp = np.concatenate(lps)
    fps = lp.shape[0] / (len(x) / 16000)
    dic = {c: i for i, c in enumerate(LABELS)}
    # 1) quick read of what's sung, to find WHICH part of the lyrics this clip is
    heard = ctc_greedy(lp)
    sp = [spoken(w) for w in words]
    lyr = " ".join(sp)
    w_at = []
    for i, s in enumerate(sp): w_at += [i] * (len(s) + 1)
    n = max(10, len(heard)); best = (0.0, 0)
    for s0 in range(0, max(1, len(lyr) - n // 2), 4):
        r = difflib.SequenceMatcher(None, heard, lyr[s0: s0 + n], autojunk=False).ratio()
        if r > best[0]: best = (r, s0)
    if best[0] < 0.3:
        print(json.dumps({"status": "lyrics_dont_match", "heard": heard[:300], "match": round(best[0], 2)})); return
    w0 = max(0, w_at[best[1]] - 6)
    w1 = min(len(words) - 1, w_at[min(len(w_at) - 1, best[1] + int(n * 1.25))] + 6)
    # 2) force-align that part, with a wildcard at each end
    toks, tw = [-1], [-1]
    for i in range(w0, w1 + 1):
        for c in sp[i].upper(): toks.append(dic[c]); tw.append(i)
        if i < w1: toks.append(dic["|"]); tw.append(-2)
    toks.append(-1); tw.append(-1)
    res = {}
    for (f0, f1, sc, cnt), w in zip(ctc_align(lp, toks), tw):
        if w < 0 or f0 is None: continue
        r = res.setdefault(w, [f0, f1, 0.0, 0]); r[0] = min(r[0], f0); r[1] = max(r[1], f1); r[2] += sc; r[3] += cnt
    t = {}
    for w, (f0, f1, sc, cnt) in res.items():
        st, en, conf = f0 / fps - pre, (f1 + 1) / fps - pre, sc / max(1, cnt)
        if en - st < 0.05 or conf < a.min_conf: continue   # squashed or unsure: leave it blank
        if st < -0.05 or st >= D: continue
        t[w] = (st, en)
    order = sorted(t); caps = []; i = 0
    while i < len(order):
        k = order[i]; s, e = t[k]; txt = words[k]
        nxt = t[order[i + 1]][0] if i + 1 < len(order) else None
        if nxt is not None and nxt - s < 0.22 and order[i + 1] == k + 1 and (not caps or len(caps[-1]["text"].split()) == 1):
            txt += " " + words[order[i + 1]]; e = t[order[i + 1]][1]; i += 1
            nxt = t[order[i + 1]][0] if i + 1 < len(order) else None
        end = min(nxt if nxt is not None else D, max(e, s + 0.25) + 0.5, D)
        disp = re.sub(r"[^\w'’ ?!-]", "", txt).lower().strip()
        if disp: caps.append({"text": disp, "start": round(max(0, s), 3), "end": round(end, 3)})
        i += 1
    out = a.out or os.path.splitext(a.video)[0] + "_captions.json"
    json.dump(caps, open(out, "w"), indent=1)
    sung = [k for k in range(w0, w1 + 1) if k in t]
    print(json.dumps({"status": "ok" if best[0] >= 0.45 and len(caps) >= 3 else "check",
                      "lyric_match": round(best[0], 2), "captions": len(caps), "file": out,
                      "timeline": " | ".join(f'{c["start"]:.1f} {c["text"]}' for c in caps)}, indent=1))

# ---------------------------------------------------------------- fonts + emoji
FONTS = {  # name, fontsource package, weight, size at 1080 wide, line pitch
    "A": ("CLEAN", "montserrat", 800, 60, 1.12),
    "B": ("HEAVY", "anton", 400, 76, 1.08),
    "C": ("ROUNDED", "nunito", 900, 66, 1.10),
}
CAP_FONT = ("montserrat", 800)

def fetch(u, p):
    with urllib.request.urlopen(u, timeout=120) as r, open(p, "wb") as f:
        f.write(r.read())

def font_file(pkg, weight):
    p = os.path.join(CACHE, f"{pkg}-{weight}.woff")
    if os.path.exists(p) and os.path.getsize(p) > 5000: return p
    urls = [f"https://cdn.jsdelivr.net/npm/@fontsource/{pkg}@5/files/{pkg}-latin-{weight}-normal.woff",
            f"https://unpkg.com/@fontsource/{pkg}@5/files/{pkg}-latin-{weight}-normal.woff"]
    for u in urls:
        try:
            fetch(u, p)
            if os.path.getsize(p) > 5000: return p
        except Exception:
            pass
    try:  # npm registry tarball (allowed in most sandboxes)
        import tarfile, io
        meta = json.load(urllib.request.urlopen(f"https://registry.npmjs.org/@fontsource/{pkg}/latest", timeout=20))
        tb = urllib.request.urlopen(meta["dist"]["tarball"], timeout=60).read()
        with tarfile.open(fileobj=io.BytesIO(tb)) as tf:
            m = tf.extractfile(f"package/files/{pkg}-latin-{weight}-normal.woff").read()
        open(p, "wb").write(m); return p
    except Exception:
        pass
    for f in ["/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]:
        if os.path.exists(f):
            print(f"WARNING: could not download {pkg}; using {f}", file=sys.stderr); return f
    import matplotlib
    return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")

def load_font(pkg, weight, size):
    f = font_file(pkg, weight)
    idx = 1 if f.endswith("HelveticaNeue.ttc") else 0
    return ImageFont.truetype(f, size, index=idx)

EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\U0001F1E6-\U0001F1FF]+")

def emoji_img(e, px):
    cps = [f"{ord(c):x}" for c in e]
    if "200d" not in cps: cps = [c for c in cps if c != "fe0f"]
    name = "-".join(cps)
    apple = "/System/Library/Fonts/Apple Color Emoji.ttc"
    if os.path.exists(apple):
        try:
            f = ImageFont.truetype(apple, 160)
            im = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((10, 10), e, font=f, embedded_color=True)
            im = im.crop(im.getbbox())
            return im.resize((round(px * im.width / im.height), px), Image.LANCZOS)
        except Exception:
            pass
    p = os.path.join(CACHE, f"tw-{name}.png")
    if not os.path.exists(p):
        for u in [f"https://raw.githubusercontent.com/jdecked/twemoji/v15.1.0/assets/72x72/{name}.png",
                  f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{name}.png",
                  f"https://unpkg.com/@twemoji/api@15.1.0/dist/72x72/{name}.png"]:
            try: fetch(u, p); break
            except Exception: pass
    if os.path.exists(p) and os.path.getsize(p) > 100:
        return Image.open(p).convert("RGBA").resize((px, px), Image.LANCZOS)
    print("WARNING: emoji not available, left off", file=sys.stderr)
    return None

def split_two(text, font, maxw):
    ws = text.split()
    if font.getlength(text) <= maxw or len(ws) < 2: return [text]
    best = None
    for i in range(1, len(ws)):
        l1, l2 = " ".join(ws[:i]), " ".join(ws[i:])
        w = max(font.getlength(l1), font.getlength(l2))
        if best is None or w < best[0]: best = (w, [l1, l2])
    return best[1]

def hook_png(hook, key, W):
    k = W / 1080
    _, pkg, wt, size, pitch = FONTS[key]
    emo = "".join(EMOJI_RE.findall(hook)).strip()
    txt = EMOJI_RE.sub("", hook).upper().strip()
    txt = re.sub(r"\s+", " ", txt)
    size = int(size * k); maxw = 962 * k - 2 * 52 * k
    while True:
        f = load_font(pkg, wt, size)
        eimg = emoji_img(emo, int(size * 0.95)) if emo else None
        ew = (eimg.width + size * 0.25) if eimg else 0
        lines = split_two(txt + (" XX" if eimg else ""), f, maxw)
        lines = [l[:-3] if l.endswith(" XX") else l for l in lines]
        widths = [f.getlength(l) + (ew if i == len(lines) - 1 else 0) for i, l in enumerate(lines)]
        if len(lines) <= 2 and max(widths) <= maxw or size < 30 * k: break
        size = int(size * 0.93)
    asc, desc = f.getmetrics()
    capb = f.getbbox("H"); caph = capb[3] - capb[1]
    lh = int(size * pitch)
    padx, pady = int(52 * k), int(38 * k)
    bw = int(max(widths) + 2 * padx); bh = int(lh * (len(lines) - 1) + caph + 2 * pady)
    im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=int(16 * k), fill=(255, 255, 255, 255))
    for i, l in enumerate(lines):
        lw_ = widths[i]; x = (bw - lw_) / 2
        ytop = pady + i * lh
        d.text((x, ytop - capb[1]), l, font=f, fill=(0, 0, 0, 255))
        if eimg is not None and i == len(lines) - 1:
            ex = x + f.getlength(l) + size * 0.25
            ey = ytop + caph / 2 - eimg.height / 2
            im.alpha_composite(eimg, (int(ex), int(ey)))
    return im

def caption_png(text, W, band_h):
    k = W / 1080
    f = load_font(*CAP_FONT, int(84 * k))
    while f.getlength(text) > 0.86 * W and f.size > 30: f = load_font(*CAP_FONT, int(f.size * 0.92))
    im = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
    bb = f.getbbox(text); tw = bb[2] - bb[0]; th = bb[3] - bb[1]
    x = (W - tw) / 2 - bb[0]; y = (band_h - th) / 2 - bb[1]
    for dy, blur, op in [(5 * k, 6 * k, 0.8), (0, 17 * k, 0.4)]:
        sh = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((x, y + dy), text, font=f, fill=(0, 0, 0, int(255 * op)))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(im).text((x, y), text, font=f, fill=(255, 255, 255, 255))
    return im

def hook_y(pos, H, hook_h):
    """pos: 'top', 'center', or a number 0-1 = where the TOP of the pill sits (0.13 = top)."""
    if pos == "top": return int(0.13 * H)
    if pos == "center": return int(0.5 * H - hook_h / 2)
    return int(min(max(float(pos), 0.125), 0.85) * H)

def cap_center(pos, H):
    if pos == "top": return int(0.66 * H)
    if pos == "center": return int(0.74 * H)
    return int(0.66 * H) if float(pos) < 0.45 else int((min(max(float(pos), 0.125), 0.85) - 0.07) * H)

WARM = (1.035, 1.0, 0.94)   # Levels: red input white 1/1.035, blue output white 0.94

def warm_img(im):
    a = np.asarray(im).astype(np.float32)
    a *= np.array(WARM, np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

# ---------------------------------------------------------------- preview
def cmd_preview(a):
    W, H = out_size(a.video)
    D = probe(a.video)["dur"]
    t = a.at if a.at is not None else min(1.0, D / 2)
    base = warm_img(frame(a.video, t, W).resize((W, H)))
    sample = "your lyrics here"
    if a.captions and os.path.exists(a.captions):
        caps = json.load(open(a.captions))
        if caps: sample = caps[min(len(caps) - 1, 2)]["text"]
    shots = []
    for key in "ABC":
        im = base.copy().convert("RGBA")
        hp = hook_png(a.hook, key, W)
        im.alpha_composite(hp, ((W - hp.width) // 2, hook_y(a.pos, H, hp.height)))
        band = int(0.2 * H); cp = caption_png(sample, W, band)
        im.alpha_composite(cp, (0, cap_center(a.pos, H) - band // 2))
        p = os.path.join(a.outdir, f"font_{key}.png"); im.convert("RGB").save(p)
        shots.append(im.convert("RGB"))
    tw = 360; th = int(tw * H / W)
    sheet = Image.new("RGB", (tw * 3 + 40, th + 90), (18, 18, 22))
    d = ImageDraw.Draw(sheet); lf = load_font("montserrat", 800, 30)
    for i, (key, im) in enumerate(zip("ABC", shots)):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (10 + i * (tw + 10), 10))
        lab = f"{key}  {FONTS[key][0]}"
        d.text((10 + i * (tw + 10) + (tw - lf.getlength(lab)) / 2, th + 30), lab, font=lf, fill=(255, 255, 255))
    p = os.path.join(a.outdir, "font_options.png"); sheet.save(p, quality=92)
    print(json.dumps({"sheet": p, "full_size": [os.path.join(a.outdir, f"font_{k}.png") for k in "ABC"]}))

# ---------------------------------------------------------------- render
def cmd_render(a):
    info = probe(a.video); D = info["dur"]
    W, H = out_size(a.video)
    work = os.path.join(CACHE, "render"); shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    hp = hook_png(a.hook, a.font, W)
    band = int(0.2 * H)
    canvas = Image.new("RGBA", (W, max(hp.height, 2)), (0, 0, 0, 0))
    canvas.alpha_composite(hp, ((W - hp.width) // 2, 0))
    hook_path = os.path.join(work, "hook.png"); canvas.save(hook_path)
    # caption track: a slideshow of PNG bands with exact durations
    caps = json.load(open(a.captions)) if a.captions else []
    blank = os.path.join(work, "blank.png"); Image.new("RGBA", (W, band), (0, 0, 0, 0)).save(blank)
    lines, tcur = [], 0.0
    for i, c in enumerate(caps):
        s, e = max(c["start"], tcur), min(c["end"], D)
        if e - s < 0.04: continue
        if s > tcur: lines += [f"file '{blank}'", f"duration {s - tcur:.3f}"]
        p = os.path.join(work, f"c{i:04d}.png"); caption_png(c["text"], W, band).save(p)
        lines += [f"file '{p}'", f"duration {e - s:.3f}"]; tcur = e
    lines += [f"file '{blank}'", f"duration {max(0.1, D - tcur + 0.5):.3f}", f"file '{blank}'"]
    lst = os.path.join(work, "caps.txt"); open(lst, "w").write("\n".join(lines))
    # picture chain: (HDR -> SDR) -> scale -> warm (last grade, under the text) -> hook -> captions
    v = "[0:v]"
    chain = []
    if info["hdr"]:
        filt = ff(["-filters"], capture=True).decode(errors="ignore")
        if " zscale " in filt:
            chain.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
        else:
            print("WARNING: HDR video and this ffmpeg can't tone-map; colors may look flat", file=sys.stderr)
    chain += [f"scale={W}:{H}:flags=lanczos", "format=yuv420p",
              f"colorchannelmixer=rr={WARM[0]}:gg={WARM[1]}:bb={WARM[2]}"]
    hend = f":enable='lte(t,{a.hook_end})'" if a.hook_end else ""
    fc = (f"{v}{','.join(chain)}[g];"
          f"[2:v]format=rgba[h];[g][h]overlay=x=0:y={hook_y(a.pos, H, hp.height)}{hend}[gh];"
          f"[3:v]format=rgba,setpts=PTS-STARTPTS[c];[gh][c]overlay=x=0:y={cap_center(a.pos, H) - band // 2}:eof_action=pass,format=yuv420p[v];")
    off = a.offset
    if off >= 0:
        fc += f"[1:a]atrim=start={off:.3f},asetpts=PTS-STARTPTS,"
    else:
        fc += f"[1:a]adelay={int(-off * 1000)}:all=1,"
    fade = max(0, D - 0.4)
    fc += f"aresample=48000,apad,atrim=duration={D:.3f},afade=t=in:d=0.02,afade=t=out:st={fade:.3f}:d=0.4[a]"
    ff(["-i", a.video, "-i", a.song, "-loop", "1", "-i", hook_path, "-f", "concat", "-safe", "0", "-i", lst,
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{D:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", a.out])
    # verify: length, and the song in the new file lines up with the original video's sound
    od = probe(a.out)["dur"]
    res = {"file": a.out, "video_seconds": round(D, 2), "output_seconds": round(od, 2),
           "length_ok": abs(od - D) < 0.15}
    room = load_audio(a.video); new = load_audio(a.out)
    if np.sqrt(np.mean(room ** 2)) > 1e-4:
        rf, _ = features(room, 1024, 128); nf, _ = features(new, 1024, 128)
        n = min(len(rf), len(nf)); lag_frames = int(0.5 * SR / 128)
        z = lambda q: (q - q.mean()) / (q.std() + 1e-9)
        best = max(range(-lag_frames, lag_frames + 1),
                   key=lambda L: float((z(rf[max(0, -L): n - max(0, L)]) * z(nf[max(0, L): n - max(0, -L)])).mean()))
        res["sync_error_ms"] = round(1000 * best * 128 / SR)
        res["sync_ok"] = abs(res["sync_error_ms"]) <= 45
    sheet = os.path.splitext(a.out)[0] + "_check.jpg"
    ts = [D * f for f in (0.05, 0.35, 0.65, 0.95)]
    ims = [frame(a.out, t, 270) for t in ts]
    sh = Image.new("RGB", (sum(i.width for i in ims) + 50, ims[0].height + 20), (0, 0, 0))
    x = 10
    for im in ims: sh.paste(im, (x, 10)); x += im.width + 10
    sh.save(sheet, quality=90); res["check_sheet"] = sheet
    print(json.dumps(res, indent=1))

def cmd_check(a):
    out = {"ffmpeg": ffmpeg_bin(), "python": sys.version.split()[0]}
    try: import onnxruntime; out["onnxruntime"] = True
    except Exception: out["onnxruntime"] = False
    for key in "ABC":
        out[f"font_{key}"] = os.path.basename(font_file(FONTS[key][1], FONTS[key][2]))
    out["emoji"] = emoji_img("🔥", 64) is not None
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    P = argparse.ArgumentParser(); sp = P.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("check")
    s = sp.add_parser("sync"); s.add_argument("--video", required=True); s.add_argument("--song", required=True); s.add_argument("--near", type=float)
    s = sp.add_parser("captions"); s.add_argument("--video", required=True); s.add_argument("--song", required=True)
    s.add_argument("--offset", type=float, required=True); s.add_argument("--lyrics", required=True); s.add_argument("--min-conf", type=float, default=0.15); s.add_argument("--model"); s.add_argument("--out")
    s = sp.add_parser("preview"); s.add_argument("--video", required=True); s.add_argument("--hook", required=True)
    s.add_argument("--captions"); s.add_argument("--pos", default="top"); s.add_argument("--at", type=float); s.add_argument("--outdir", default=".")
    s = sp.add_parser("render"); s.add_argument("--video", required=True); s.add_argument("--song", required=True)
    s.add_argument("--offset", type=float, required=True); s.add_argument("--hook", required=True); s.add_argument("--font", default="A", choices=list("ABC"))
    s.add_argument("--captions"); s.add_argument("--pos", default="top"); s.add_argument("--hook-end", type=float); s.add_argument("--out", required=True)
    a = P.parse_args()
    {"check": cmd_check, "sync": cmd_sync, "captions": cmd_captions, "preview": cmd_preview, "render": cmd_render}[a.cmd](a)
