"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ENHANCED SIGN LANGUAGE RECOGNITION SYSTEM  v2.0                     ║
║  Features: TTS • Word Prediction • Save Conversation • Split-Panel UI        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Keyboard Shortcuts:
  ESC  → Exit
  C    → Clear current sentence
  V    → Speak sentence aloud (TTS)
  S    → Save conversation to timestamped .txt file
  1-5  → Select word prediction
"""

import math
import os
import queue
import re
import threading
import time
import traceback
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

try:
    from cvzone.HandTrackingModule import HandDetector
except ImportError:
    raise ImportError("cvzone not found. Run: pip install cvzone")

try:
    from keras.models import load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        raise ImportError("Keras/TensorFlow not found. Run: pip install tensorflow")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not found — TTS disabled. Run: pip install pyttsx3")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "cnn8grps_rad1_model.h5"
HOLD_FRAMES  = 15          # frames a gesture must be held to confirm
CANVAS_SIZE  = 400         # white canvas for skeleton rendering
OFFSET       = 29          # bounding-box padding
CAM_W        = 640         # webcam capture width
CAM_H        = 480         # webcam capture height

# UI dimensions
LEFT_W       = 640         # left panel (camera feed)
RIGHT_W      = 420         # right panel
UI_H         = 700         # total window height
PANEL_PAD    = 18          # inner padding

# Colour palette  (BGR)
C_BG         = (18,  18,  22)
C_PANEL      = (28,  28,  36)
C_CARD       = (38,  38,  50)
C_ACCENT     = (0,  200,  100)    # green — letters
C_ORANGE     = (0,  165,  255)    # orange — special gestures
C_BLUE       = (220, 180,  60)    # gold — predictions
C_WHITE      = (240, 240, 245)
C_GREY       = (110, 110, 130)
C_RED        = (60,   60, 220)
C_BORDER     = (55,   55,  75)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO    = cv2.FONT_HERSHEY_PLAIN


# ─────────────────────────────────────────────────────────────────────────────
#  WORD PREDICTION MODULE
# ─────────────────────────────────────────────────────────────────────────────
COMMON_WORDS = """
the be to of and a in that have it for not on with he as you do at
this but his by from they we say her she or an will my one all would
there their what so up out if about who get which go me when make can
like time no just him know take people into year your good some could
them see other than then now look only come its over think also back
after use two how our work first well way even new want because any
these give day most us hello thank yes okay please help good morning
afternoon evening sorry understand speak language hand sign name
""".split()

_word_freq = Counter(COMMON_WORDS)

def predict_words(prefix: str, n: int = 5) -> list[str]:
    """Return up to n word completions for the given prefix (case-insensitive)."""
    if not prefix:
        return []
    p = prefix.lower()
    matches = [(w, f) for w, f in _word_freq.items() if w.startswith(p)]
    matches.sort(key=lambda x: -x[1])
    return [w for w, _ in matches[:n]]


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT-TO-SPEECH MODULE  (fixed: persistent worker thread + queue)
# ─────────────────────────────────────────────────────────────────────────────
_tts_queue:   queue.Queue = queue.Queue()
_tts_active:  bool        = False   # True while engine is speaking
_tts_thread:  threading.Thread | None = None


def _tts_worker():
    global _tts_active

    if not TTS_AVAILABLE:
        return

    while True:
        text = _tts_queue.get()
        if text is None:
            break

        try:
            _tts_active = True

            import pyttsx3 as _p3
            if hasattr(_p3, "_activeEngines"):
                _p3._activeEngines.clear()

            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.95)

            # ── KEY FIX ──────────────────────────────────────────────────────
            # The engine internally keeps an utterance queue (_agenda).
            # Even after stop(), leftover items can survive into the next
            # init() if the same underlying driver is reused.
            # Wipe it before say() so it always speaks the FULL new sentence.
            if hasattr(engine, "_agenda"):
                engine._agenda.clear()  # pyttsx3 internal utterance list

            engine.say(text)
            engine.runAndWait()

            try:
                engine.stop()
            except Exception:
                pass
            del engine

        except Exception as e:
            print(f"[TTS error] {e}")
        finally:
            _tts_active = False


def _init_tts():
    """Start the TTS worker thread once."""
    global _tts_thread
    if TTS_AVAILABLE and (_tts_thread is None or not _tts_thread.is_alive()):
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="tts-worker")
        _tts_thread.start()


def speak_text(text: str):
    if not TTS_AVAILABLE or not text.strip():
        return

    # Put the FULL text every time; worker always speaks what it dequeues.
    # No draining — draining can race with the worker picking up a partial item.
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
        except Exception:
            break

    _tts_queue.put(text)


def tts_is_active() -> bool:
    """Return True while the TTS engine is speaking."""
    return _tts_active


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE CONVERSATION MODULE
# ─────────────────────────────────────────────────────────────────────────────
conversation_log: list[tuple[str, str]] = []   # [(timestamp, sentence), …]

def save_conversation(filename: str | None = None) -> str:
    """Write conversation log to a .txt file. Returns the filename used."""
    if filename is None:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{ts}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("═" * 60 + "\n")
        f.write("  SIGN LANGUAGE RECOGNITION — Conversation Log\n")
        f.write(f"  Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("═" * 60 + "\n\n")
        for ts, sentence in conversation_log:
            f.write(f"[{ts}]  {sentence}\n")
        f.write(f"\n── Total entries: {len(conversation_log)} ──\n")
    return filename


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def filled_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """Draw a filled rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    # four corners
    cv2.circle(img, (x1+r, y1+r), r, color, thickness)
    cv2.circle(img, (x2-r, y1+r), r, color, thickness)
    cv2.circle(img, (x1+r, y2-r), r, color, thickness)
    cv2.circle(img, (x2-r, y2-r), r, color, thickness)
    # three rectangles
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)


def put_text_wrapped(img, text, x, y, font, scale, color, thickness, max_w):
    """Draw text, truncating with '…' if it exceeds max_w pixels."""
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw <= max_w:
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return
    # binary search for fitting length
    lo, hi = 1, len(text)
    while lo < hi:
        mid  = (lo + hi + 1) // 2
        seg  = text[:mid] + "…"
        (w, _), _ = cv2.getTextSize(seg, font, scale, thickness)
        if w <= max_w:
            lo = mid
        else:
            hi = mid - 1
    cv2.putText(img, text[:lo] + "…", (x, y), font, scale, color, thickness, cv2.LINE_AA)


def section_header(panel, text, y, width):
    """Draw a subtle section label."""
    cv2.putText(panel, text.upper(), (PANEL_PAD, y),
                FONT, 0.38, C_GREY, 1, cv2.LINE_AA)
    cv2.line(panel, (PANEL_PAD + len(text)*7 + 4, y - 4),
             (width - PANEL_PAD, y - 4), C_BORDER, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  UI BUILDER — RIGHT PANEL
# ─────────────────────────────────────────────────────────────────────────────
def build_right_panel(
        current_letter, hold_count, hold_frames,
        sentence, predictions,
        status_msg, tts_active, save_flash
) -> np.ndarray:

    w, h = RIGHT_W, UI_H
    panel = np.full((h, w, 3), C_BG, dtype=np.uint8)

    # ── Title ────────────────────────────────────────────────────────────────
    filled_rounded_rect(panel, (PANEL_PAD, 10), (w - PANEL_PAD, 52), C_PANEL, 8)
    cv2.putText(panel, "SIGN  LANGUAGE  ASSISTANT",
                (PANEL_PAD + 8, 36), FONT, 0.48, C_ACCENT, 1, cv2.LINE_AA)

    # ── Hold progress bar ────────────────────────────────────────────────────
    by = 70
    section_header(panel, "gesture hold", by, w)
    bar_y1, bar_y2 = by + 8, by + 28
    filled_rounded_rect(panel, (PANEL_PAD, bar_y1), (w - PANEL_PAD, bar_y2), C_CARD, 6)
    ratio   = min(hold_count / max(hold_frames, 1), 1.0)
    bar_end = PANEL_PAD + int((w - 2*PANEL_PAD) * ratio)
    cl_str  = str(current_letter)
    b_color = C_ORANGE if cl_str in ('Space', 'Backspace', 'Next') else C_ACCENT
    if bar_end > PANEL_PAD + 10:
        filled_rounded_rect(panel, (PANEL_PAD + 2, bar_y1 + 2),
                            (bar_end, bar_y2 - 2), b_color, 5)
    pct_str = f"{int(ratio*100)}%"
    cv2.putText(panel, pct_str, (w - PANEL_PAD - 40, bar_y2 - 4),
                FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)

    # ── Current letter (large card) ───────────────────────────────────────────
    lc_y1, lc_y2 = 112, 230
    filled_rounded_rect(panel, (PANEL_PAD, lc_y1), (w - PANEL_PAD, lc_y2), C_PANEL, 12)
    section_header(panel, "detected gesture", lc_y1 + 18, w)
    disp  = cl_str if cl_str else "—"
    color = C_ORANGE if cl_str in ('Space', 'Backspace', 'Next') else (
            C_RED if not cl_str else C_ACCENT)
    # large letter
    scale = 3.8 if len(disp) == 1 else (1.6 if len(disp) <= 5 else 1.0)
    (lw, lh), _ = cv2.getTextSize(disp, FONT, scale, 4)
    lx = (w - lw) // 2
    ly = lc_y1 + (lc_y2 - lc_y1 + lh) // 2
    cv2.putText(panel, disp, (lx, ly), FONT, scale, color, 4, cv2.LINE_AA)

    # ── Sentence card ─────────────────────────────────────────────────────────
    sc_y1, sc_y2 = 242, 360
    filled_rounded_rect(panel, (PANEL_PAD, sc_y1), (w - PANEL_PAD, sc_y2), C_PANEL, 12)
    section_header(panel, "sentence", sc_y1 + 18, w)

    sentence_str = ''.join(sentence)
    # wrap into two lines
    max_chars_line = (w - 2*PANEL_PAD - 10) // 13
    if len(sentence_str) <= max_chars_line:
        line1, line2 = sentence_str, ""
    else:
        line1 = sentence_str[-max_chars_line * 2: -max_chars_line] or ""
        line2 = sentence_str[-max_chars_line:]

    s_color = C_WHITE if sentence_str else C_GREY
    s_disp1 = line1 if sentence_str else "(start signing…)"
    cv2.putText(panel, s_disp1, (PANEL_PAD + 6, sc_y1 + 50),
                FONT, 0.70, s_color, 1, cv2.LINE_AA)
    if line2:
        cv2.putText(panel, line2, (PANEL_PAD + 6, sc_y1 + 82),
                    FONT, 0.70, C_WHITE, 1, cv2.LINE_AA)

    # blinking cursor
    if int(time.time() * 2) % 2 == 0:
        cx = PANEL_PAD + 6 + len(line2 or s_disp1) * 13
        cy_base = (sc_y1 + 82) if line2 else (sc_y1 + 50)
        cv2.line(panel, (cx, cy_base - 18), (cx, cy_base + 2), C_ACCENT, 2)

    # ── Word predictions ──────────────────────────────────────────────────────
    wp_y1, wp_y2 = 372, 510
    filled_rounded_rect(panel, (PANEL_PAD, wp_y1), (w - PANEL_PAD, wp_y2), C_PANEL, 12)
    section_header(panel, "word predictions  (press 1-5)", wp_y1 + 18, w)

    if predictions:
        slot_h = (wp_y2 - wp_y1 - 30) // len(predictions)
        for i, word in enumerate(predictions[:5]):
            sy  = wp_y1 + 28 + i * slot_h
            sx1 = PANEL_PAD + 6
            sx2 = w - PANEL_PAD - 6
            filled_rounded_rect(panel, (sx1, sy), (sx2, sy + slot_h - 4), C_CARD, 6)
            cv2.putText(panel, f"{i+1}", (sx1 + 8, sy + slot_h - 10),
                        FONT, 0.55, C_ORANGE, 1, cv2.LINE_AA)
            cv2.putText(panel, word, (sx1 + 28, sy + slot_h - 10),
                        FONT, 0.65, C_BLUE, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "no suggestions yet",
                    (PANEL_PAD + 10, wp_y1 + 60), FONT, 0.55, C_GREY, 1, cv2.LINE_AA)

    # ── Status / flash messages ───────────────────────────────────────────────
    st_y1, st_y2 = 522, 570
    filled_rounded_rect(panel, (PANEL_PAD, st_y1), (w - PANEL_PAD, st_y2), C_CARD, 8)
    st_color = C_ORANGE if "Saved" in status_msg or "Speaking" in status_msg else C_GREY
    cv2.putText(panel, status_msg[:52],
                (PANEL_PAD + 8, st_y1 + 30), FONT, 0.50, st_color, 1, cv2.LINE_AA)

    # TTS indicator
    if tts_active:
        cv2.circle(panel, (w - PANEL_PAD - 10, st_y1 + 22), 7, C_ACCENT, -1)
        cv2.putText(panel, "speaking", (w - PANEL_PAD - 82, st_y1 + 28),
                    FONT, 0.40, C_ACCENT, 1, cv2.LINE_AA)

    # ── Keyboard shortcuts cheatsheet ─────────────────────────────────────────
    ks_y = 585
    shortcuts = [
        ("ESC", "Quit"),
        ("C",   "Clear"),
        ("V",   "Speak"),
        ("S",   "Save"),
    ]
    slot_w = (w - 2*PANEL_PAD) // len(shortcuts)
    for i, (key, label) in enumerate(shortcuts):
        sx = PANEL_PAD + i * slot_w
        filled_rounded_rect(panel, (sx, ks_y), (sx + slot_w - 4, ks_y + 38), C_CARD, 6)
        cv2.putText(panel, key,   (sx + 8, ks_y + 16), FONT, 0.55, C_ORANGE, 1, cv2.LINE_AA)
        cv2.putText(panel, label, (sx + 8, ks_y + 32), FONT, 0.40, C_GREY,   1, cv2.LINE_AA)

    # ── Gesture legend ────────────────────────────────────────────────────────
    gl_y = 632
    cv2.putText(panel, "GESTURES:", (PANEL_PAD, gl_y + 14),
                FONT, 0.40, C_GREY, 1, cv2.LINE_AA)
    legends = [
        ("Space",     "IY up"),
        ("Backspace", "palm away"),
        ("Next",      "thumb across"),
    ]
    for i, (g, desc) in enumerate(legends):
        gx = PANEL_PAD + 78 + i * 112
        filled_rounded_rect(panel, (gx, gl_y), (gx + 106, gl_y + 22), C_CARD, 4)
        cv2.putText(panel, g,    (gx + 4, gl_y + 10), FONT, 0.32, C_ORANGE, 1, cv2.LINE_AA)
        cv2.putText(panel, desc, (gx + 4, gl_y + 20), FONT, 0.30, C_GREY,   1, cv2.LINE_AA)

    # ── Frame divider line on left edge ──────────────────────────────────────
    cv2.line(panel, (0, 0), (0, h), C_BORDER, 2)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
#  UI BUILDER — LEFT PANEL (camera feed wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def build_left_panel(frame, current_letter, sentence) -> np.ndarray:
    h_target = UI_H
    w_target = LEFT_W

    # resize camera frame
    fh, fw = frame.shape[:2]
    scale  = min(w_target / fw, (h_target - 80) / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))

    panel = np.full((h_target, w_target, 3), C_BG, dtype=np.uint8)

    # title bar
    filled_rounded_rect(panel, (0, 0), (w_target, 48), C_PANEL, 0)
    cv2.putText(panel, "◉  LIVE  FEED", (16, 32),
                FONT, 0.65, C_ACCENT, 1, cv2.LINE_AA)
    ts_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, ts_str, (w_target - 90, 32),
                FONT, 0.55, C_GREY, 1, cv2.LINE_AA)

    # paste camera frame
    y_off = 54 + (h_target - 54 - nh) // 2
    x_off = (w_target - nw) // 2
    panel[y_off:y_off+nh, x_off:x_off+nw] = resized

    # overlay: predicted letter badge
    cl_str = str(current_letter)
    if cl_str:
        badge_color = C_ORANGE if cl_str in ('Space','Backspace','Next') else C_ACCENT
        filled_rounded_rect(panel, (x_off + 8, y_off + 8),
                            (x_off + 90, y_off + 55), (0,0,0), 8)
        scale = 2.0 if len(cl_str) == 1 else 0.8
        (lw, _), _ = cv2.getTextSize(cl_str, FONT, scale, 3)
        cv2.putText(panel, cl_str,
                    (x_off + 8 + (82 - lw)//2, y_off + 46),
                    FONT, scale, badge_color, 3, cv2.LINE_AA)

    # border around frame
    cv2.rectangle(panel, (x_off - 2, y_off - 2),
                  (x_off + nw + 2, y_off + nh + 2), C_BORDER, 2)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
#  HAND-SKELETON DRAWING (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def draw_skeleton(white, pts, os, os1):
    for t in range(0,  4): cv2.line(white, (pts[t][0]+os,   pts[t][1]+os1),  (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
    for t in range(5,  8): cv2.line(white, (pts[t][0]+os,   pts[t][1]+os1),  (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
    for t in range(9, 12): cv2.line(white, (pts[t][0]+os,   pts[t][1]+os1),  (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
    for t in range(13,16): cv2.line(white, (pts[t][0]+os,   pts[t][1]+os1),  (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
    for t in range(17,20): cv2.line(white, (pts[t][0]+os,   pts[t][1]+os1),  (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)
    cv2.line(white, (pts[5][0]+os,  pts[5][1]+os1),  (pts[9][0]+os,  pts[9][1]+os1),  (0,255,0), 3)
    cv2.line(white, (pts[9][0]+os,  pts[9][1]+os1),  (pts[13][0]+os, pts[13][1]+os1), (0,255,0), 3)
    cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0,255,0), 3)
    cv2.line(white, (pts[0][0]+os,  pts[0][1]+os1),  (pts[5][0]+os,  pts[5][1]+os1),  (0,255,0), 3)
    cv2.line(white, (pts[0][0]+os,  pts[0][1]+os1),  (pts[17][0]+os, pts[17][1]+os1), (0,255,0), 3)
    for i in range(21):
        cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0,0,255), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  GESTURE CLASSIFICATION (identical rules from original)
# ─────────────────────────────────────────────────────────────────────────────
def classify_gesture(ch1, ch2, ch3, pts):  # noqa: C901
    """Apply all correction rules and return final letter / special token."""

    def d(a, b):
        return math.sqrt((pts[a][0]-pts[b][0])**2 + (pts[a][1]-pts[b][1])**2)

    pl = [ch1, ch2]

    # ── Group correction rules (verbatim from original) ───────────────────────
    l = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],
         [6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],
         [1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],
         [5,4],[7,0],[7,5],[7,2]]
    if pl in l:
        if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]:
            ch1 = 0

    l = [[2,2],[2,1]]
    if pl in l:
        if pts[5][0] < pts[4][0]: ch1 = 0

    l = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]; pl = [ch1,ch2]
    if pl in l:
        if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0]>pts[4][0]:
            ch1 = 2

    l = [[6,0],[6,6],[6,2]]; pl = [ch1,ch2]
    if pl in l:
        if d(8,16) < 52: ch1 = 2

    l = [[1,4],[1,5],[1,6],[1,3],[1,0]]; pl = [ch1,ch2]
    if pl in l:
        if pts[6][1]>pts[8][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1] and pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
            ch1 = 3

    l = [[4,6],[4,1],[4,5],[4,3],[4,7]]; pl = [ch1,ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]: ch1 = 3

    l = [[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]]; pl = [ch1,ch2]
    if pl in l:
        if pts[2][1]+15 < pts[16][1]: ch1 = 3

    l = [[6,4],[6,1],[6,2]]; pl = [ch1,ch2]
    if pl in l:
        if d(4,11) > 55: ch1 = 4

    l = [[1,4],[1,6],[1,1]]; pl = [ch1,ch2]
    if pl in l:
        if d(4,11)>50 and (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
            ch1 = 4

    l = [[3,6],[3,4]]; pl = [ch1,ch2]
    if pl in l:
        if pts[4][0] < pts[0][0]: ch1 = 4

    l = [[2,2],[2,5],[2,4]]; pl = [ch1,ch2]
    if pl in l:
        if pts[1][0] < pts[12][0]: ch1 = 4

    l = [[3,6],[3,5],[3,4]]; pl = [ch1,ch2]
    if pl in l:
        if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[10][1]:
            ch1 = 5

    l = [[3,2],[3,1],[3,6]]; pl = [ch1,ch2]
    if pl in l:
        if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]:
            ch1 = 5

    l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]; pl = [ch1,ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]: ch1 = 5

    l = [[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]; pl = [ch1,ch2]
    if pl in l:
        if pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
            ch1 = 5

    l = [[5,7],[5,2],[5,6]]; pl = [ch1,ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]: ch1 = 7

    l = [[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]]; pl = [ch1,ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]: ch1 = 7

    l = [[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]; pl = [ch1,ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]: ch1 = 7

    l = [[0,4],[0,2],[0,3],[0,1],[0,6]]; pl = [ch1,ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]: ch1 = 6

    l = [[7,2]]; pl = [ch1,ch2]
    if pl in l:
        if pts[18][1] < pts[20][1]: ch1 = 6

    l = [[2,1],[2,2],[2,6],[2,7],[2,0]]; pl = [ch1,ch2]
    if pl in l:
        if d(8,16) > 50: ch1 = 6

    l = [[4,6],[4,2],[4,1],[4,4]]; pl = [ch1,ch2]
    if pl in l:
        if d(4,11) < 60: ch1 = 6

    l = [[1,4],[1,6],[1,0],[1,2]]; pl = [ch1,ch2]
    if pl in l:
        if pts[5][0]-pts[4][0]-15 > 0: ch1 = 6

    l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[6,3],[6,4],[7,5]]; pl = [ch1,ch2]
    if pl in l:
        if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
            ch1 = 1

    l = [[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],[4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5]]; pl = [ch1,ch2]
    if pl in l:
        if pts[6][1]<pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
            ch1 = 1

    l = [[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]]; pl = [ch1,ch2]
    if pl in l:
        if pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
            ch1 = 1

    l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]; pl = [ch1,ch2]
    if pl in l:
        if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[2][0]<pts[0][0] and pts[4][1]>pts[14][1]:
            ch1 = 1

    l = [[4,1],[4,2],[4,4]]; pl = [ch1,ch2]
    if pl in l:
        if d(4,11)<50 and (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
            ch1 = 1

    l = [[3,4],[3,0],[3,1],[3,5],[3,6]]; pl = [ch1,ch2]
    if pl in l:
        if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[2][0]<pts[0][0] and pts[14][1]<pts[4][1]:
            ch1 = 1

    l = [[6,6],[6,4],[6,1],[6,2]]; pl = [ch1,ch2]
    if pl in l:
        if pts[5][0]-pts[4][0]-15 < 0: ch1 = 1

    l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],[7,6],[7,7]]; pl = [ch1,ch2]
    if pl in l:
        if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]:
            ch1 = 1

    l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]; pl = [ch1,ch2]
    if pl in l:
        if pts[4][0]<pts[5][0]+15 and (pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]):
            ch1 = 7

    l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]; pl = [ch1,ch2]
    if pl in l:
        if (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[14][1]:
            ch1 = 1

    fg = 13
    l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]; pl = [ch1,ch2]
    if pl in l:
        if not(pts[0][0]+fg<pts[8][0] and pts[0][0]+fg<pts[12][0] and pts[0][0]+fg<pts[16][0] and pts[0][0]+fg<pts[20][0]) \
        and not(pts[0][0]>pts[8][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) \
        and d(4,11) < 50:
            ch1 = 1

    l = [[5,0],[5,5],[0,1]]; pl = [ch1,ch2]
    if pl in l:
        if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1]:
            ch1 = 1

    # ── Subgroup → Letter ─────────────────────────────────────────────────────
    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0]<pts[6][0] and pts[4][0]<pts[10][0] and pts[4][0]<pts[14][0] and pts[4][0]<pts[18][0]:
            ch1 = 'A'
        if pts[4][0]>pts[6][0] and pts[4][0]<pts[10][0] and pts[4][0]<pts[14][0] and pts[4][0]<pts[18][0] and pts[4][1]<pts[14][1] and pts[4][1]<pts[18][1]:
            ch1 = 'T'
        if pts[4][1]>pts[8][1] and pts[4][1]>pts[12][1] and pts[4][1]>pts[16][1] and pts[4][1]>pts[20][1]:
            ch1 = 'E'
        if pts[4][0]>pts[6][0] and pts[4][0]>pts[10][0] and pts[4][0]>pts[14][0] and pts[4][1]<pts[18][1]:
            ch1 = 'M'
        if pts[4][0]>pts[6][0] and pts[4][0]>pts[10][0] and pts[4][1]<pts[18][1] and pts[4][1]<pts[14][1]:
            ch1 = 'N'

    if ch1 == 2: ch1 = 'C' if d(12,4) > 42 else 'O'
    if ch1 == 3: ch1 = 'G' if d(8,12)  > 72 else 'H'
    if ch1 == 7: ch1 = 'Y' if d(8,4)   > 42 else 'J'
    if ch1 == 4: ch1 = 'L'
    if ch1 == 6: ch1 = 'X'
    if ch1 == 5:
        if pts[4][0]>pts[12][0] and pts[4][0]>pts[16][0] and pts[4][0]>pts[20][0]:
            ch1 = 'Z' if pts[8][1] < pts[5][1] else 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if pts[6][1]>pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]: ch1 = 'B'
        if pts[6][1]>pts[8][1]  and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]: ch1 = 'D'
        if pts[6][1]<pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]: ch1 = 'F'
        if pts[6][1]<pts[8][1]  and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]: ch1 = 'I'
        if pts[6][1]>pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]<pts[20][1]: ch1 = 'W'
        if pts[6][1]>pts[8][1]  and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1] and pts[4][1]<pts[9][1]: ch1 = 'K'
        if (d(8,12)-d(6,10))<8  and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]): ch1 = 'U'
        if (d(8,12)-d(6,10))>=8 and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[9][1]: ch1 = 'V'
        if pts[8][0]>pts[12][0] and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]): ch1 = 'R'

    # ── Special gestures ──────────────────────────────────────────────────────
    if (pts[6][1] > pts[8][1] + 10 and
            pts[10][1] < pts[12][1] and
            pts[14][1] < pts[16][1] and
            pts[18][1] > pts[20][1] + 10):
        ch1 = 'Space'

    if (pts[4][0] < pts[5][0] - 20 and
            pts[6][1] < pts[8][1] and
            pts[10][1] < pts[12][1] and
            pts[14][1] < pts[16][1] and
            pts[18][1] < pts[20][1] and
            pts[4][1] < pts[0][1]):
        ch1 = 'Next'

    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and
            pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0] and
            pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and
            pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]):
        ch1 = 'Backspace'

    return ch1


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Init model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("        Place cnn8grps_rad1_model.h5 in the same directory.")
        return

    print("[INFO] Loading CNN model …")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded.")

    # white canvas
    white_blank = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8) * 255
    cv2.imwrite("white.jpg", white_blank)

    # camera
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    hd  = HandDetector(maxHands=1)
    hd2 = HandDetector(maxHands=1)

    # Start persistent TTS worker thread
    _init_tts()

    # ── State ─────────────────────────────────────────────────────────────────
    sentence   : list[str] = []
    current    : str       = ''
    hold_count : int       = 0
    predictions: list[str] = []
    status_msg : str       = "Ready — show a hand gesture to begin"
    save_flash : float     = 0.0   # timestamp of last save

    print("[INFO] System ready. Window: 'Sign Language Recognition'")
    print("       Shortcuts: ESC=quit  C=clear  V=speak  S=save  1-5=pick word")

    while True:
        try:
            ret, frame = capture.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            result = hd.findHands(frame, draw=False, flipType=True)
            hands  = result[0] if isinstance(result, tuple) else result

            ch1 = ''

            if hands:
                hand     = hands[0]
                x, y, w, h = hand['bbox']

                y1 = max(0, y - OFFSET);    y2 = min(frame.shape[0], y + h + OFFSET)
                x1 = max(0, x - OFFSET);    x2 = min(frame.shape[1], x + w + OFFSET)
                crop = frame[y1:y2, x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, y2), C_ACCENT[::-1], 2)

                white = cv2.imread("white.jpg")

                result2 = hd2.findHands(crop, draw=False, flipType=True)
                handz   = result2[0] if isinstance(result2, tuple) else result2

                if handz:
                    hand2 = handz[0]
                    pts   = hand2['lmList']

                    os_x  = ((CANVAS_SIZE - w) // 2) - 15
                    os_y  = ((CANVAS_SIZE - h) // 2) - 15

                    draw_skeleton(white, pts, os_x, os_y)
                    cv2.imshow("Hand Skeleton", white)

                    # CNN
                    white_input = white.reshape(1, CANVAS_SIZE, CANVAS_SIZE, 3)
                    prob = np.array(model.predict(white_input, verbose=0)[0], dtype='float32')
                    c1   = int(np.argmax(prob)); prob[c1] = 0
                    c2   = int(np.argmax(prob)); prob[c2] = 0
                    c3   = int(np.argmax(prob))

                    ch1 = classify_gesture(c1, c2, c3, pts)

                    # overlay on frame
                    label_color = C_ORANGE[::-1] if ch1 in ('Space','Next','Backspace') else (0, 0, 255)
                    cv2.putText(frame, str(ch1), (30, 80),
                                FONT, 2.2, label_color, 3, cv2.LINE_AA)

                    # hold-to-confirm
                    if ch1 == current:
                        hold_count += 1
                    else:
                        current    = ch1
                        hold_count = 0

                    if hold_count >= HOLD_FRAMES:
                        if ch1 == 'Space':
                            sentence.append(' ')
                            status_msg = "Space added"
                        elif ch1 == 'Backspace':
                            if sentence:
                                sentence.pop()
                            status_msg = "Backspace"
                        elif ch1 == 'Next':
                            # commit current word to conversation log
                            joined = ''.join(sentence).strip()
                            if joined:
                                ts = datetime.now().strftime("%H:%M:%S")
                                conversation_log.append((ts, joined))
                                status_msg = f"Logged: {joined[:30]}"
                        elif isinstance(ch1, str):
                            sentence.append(ch1)
                            status_msg = f"Added: {ch1}"

                        current    = ''
                        hold_count = 0

                    # word predictions from last partial word
                    full_text = ''.join(sentence)
                    parts     = re.split(r'\s+', full_text)
                    last_part = parts[-1] if parts else ''
                    predictions = predict_words(last_part, 5)

            else:
                ch1        = ''
                current    = ''
                hold_count = 0
                status_msg = "No hand detected"

            # ── Compose split-panel display ───────────────────────────────────
            # Use tts_is_active() instead of checking the (now-removed) lock
            active_tts  = tts_is_active()
            save_active = (time.time() - save_flash) < 2.0

            left  = build_left_panel(frame, ch1, sentence)
            right = build_right_panel(
                ch1, hold_count, HOLD_FRAMES,
                sentence, predictions,
                status_msg, active_tts, save_active
            )

            # ensure equal height
            lh, rh = left.shape[0], right.shape[0]
            if lh != rh:
                th = max(lh, rh)
                if lh < th:
                    pad       = np.full((th - lh, left.shape[1], 3), C_BG, dtype=np.uint8)
                    left      = np.vstack([left, pad])
                else:
                    pad       = np.full((th - rh, right.shape[1], 3), C_BG, dtype=np.uint8)
                    right     = np.vstack([right, pad])

            display = np.hstack([left, right])
            cv2.imshow("Sign Language Recognition", display)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == 27:            # ESC — quit
                break

            elif key == ord('c'):    # C — clear
                sentence.clear()
                status_msg = "Sentence cleared"

            elif key == ord('v'):    # V — speak
                text = ''.join(sentence).strip()
                if text:
                    speak_text(text)
                    status_msg = f"Speaking: {text[:40]}…" if len(text)>40 else f"Speaking: {text}"
                else:
                    status_msg = "Nothing to speak"

            elif key == ord('s'):    # S — save
                if conversation_log or sentence:
                    # also log current unsaved sentence
                    joined = ''.join(sentence).strip()
                    if joined:
                        ts = datetime.now().strftime("%H:%M:%S")
                        conversation_log.append((ts, joined))
                    fname = save_conversation()
                    save_flash = time.time()
                    status_msg = f"Saved → {fname}"
                    print(f"[INFO] Conversation saved to {fname}")
                else:
                    status_msg = "Nothing to save yet"

            elif ord('1') <= key <= ord('5'):   # 1-5 — pick prediction
                idx = key - ord('1')
                if idx < len(predictions):
                    word = predictions[idx]
                    # replace last partial word
                    full_text = ''.join(sentence)
                    parts     = re.split(r'(\s+)', full_text)
                    # remove last non-space token
                    while parts and parts[-1].strip():
                        parts.pop()
                    new_text  = ''.join(parts) + word + ' '
                    sentence  = list(new_text)
                    status_msg = f"Word selected: {word}"

        except Exception:
            print(traceback.format_exc())

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[INFO] Exiting …")
    if conversation_log:
        fname = save_conversation()
        print(f"[INFO] Auto-saved conversation to {fname}")

    # Gracefully stop TTS worker
    _tts_queue.put(None)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()