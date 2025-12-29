#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preserve_style_translate.py (FULL REGENERATED — multiline vertical cutoff fix)

Key updates folded in:
1) Multiline text placement is now STRICTLY constrained to the mask bbox/bands using true
   glyph bounding boxes (PIL textbbox with stroke_width). If a multiline render would be
   cut off, the renderer automatically SHRINKS the font size until it fully fits.
2) Group font consistency is removed (each region chooses its own optimal font size).
3) Mask bands are stored in MASK-LOCAL coordinates (0..mask_h), preventing double shifting.
4) Wrapping measurements include stroke_width to avoid edge clipping.
5) Still: no mask expansion; full-res OCR; debug overlays; two-pass erase.

Example:
  python preserve_style_translate.py \
    --input "images/**/*.jpg" \
    --outdir out_preserve \
    --summary summary.json \
    --schema chart_translate_schema.json \
    --ocr-backend rapid \
    --translate-backend gemini \
    --gcp-project "ai-agent-461123" --gcp-location us-central1 --gemini-model gemini-2.0-flash-001 \
    --inpaint-backend lama --lama-device auto \
    --allow-extra-line --max-extra-lines 1 --min-good-font-px 14 \
    --debug-dir debug_out
"""

from __future__ import annotations

import os
import re
import sys
import io
import json
import glob
import time
import random
import string
import argparse
import warnings
import contextlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Best-effort quieting of some noisy libs (some will still print)
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

warnings.filterwarnings("ignore", message=r".*No ccache found.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*deprecated.*", category=DeprecationWarning)

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Regex / constants
# -----------------------------
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DIGIT_RE = re.compile(r"\d")

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class OCRToken:
    text: str
    score: float
    poly: np.ndarray  # (4,2) int32

@dataclass
class TextRegion:
    id: str
    tokens: List[OCRToken]
    bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1
    mask: np.ndarray                # uint8 HxW 0/255 (NO expansion)
    zh_text: str
    line_count: int
    group_key: str = ""             # kept for debug only; no longer used for font consistency

    # translation + rendering
    en_lines: Optional[List[str]] = None
    en_text: str = ""

    # geometry within region bbox (local coords)
    mask_bbox_local: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (mx0,my0,mx1,my1) within bbox
    mask_bands_raw: List[Tuple[int, int]] = None  # list of (y0,y1) in MASK-LOCAL coords [0..mask_h)

@dataclass
class FileResult:
    path: str
    status: str
    reason: str = ""
    chinese_tokens: int = 0
    total_tokens: int = 0
    chart_gate_pass: bool = False
    regions: int = 0
    out_path: str = ""
    time_ms: int = 0
    ocr_backend: str = ""
    inpaint_backend: str = ""
    lama_device: str = ""

# -----------------------------
# Basic utils
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def is_image_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

def expand_inputs(inputs: List[str]) -> List[str]:
    out: List[str] = []
    for item in inputs:
        item = item.strip()
        if not item:
            continue
        if any(ch in item for ch in ["*", "?", "["]) or ("**" in item):
            matches = glob.glob(item, recursive=True)
            for m in matches:
                if os.path.isdir(m):
                    for root, _, files in os.walk(m):
                        for f in files:
                            p = os.path.join(root, f)
                            if is_image_file(p):
                                out.append(p)
                else:
                    if is_image_file(m):
                        out.append(m)
            continue

        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for f in files:
                    p = os.path.join(root, f)
                    if is_image_file(p):
                        out.append(p)
        else:
            if is_image_file(item):
                out.append(item)

    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            uniq.append(p)
            seen.add(ap)
    return uniq

def safe_imread(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return bgr

def random_id(prefix: str = "r") -> str:
    return prefix + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))

def contains_chinese(s: str) -> bool:
    return bool(CJK_RE.search(s or ""))

def clip_poly(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    p = poly.astype(np.int32).copy()
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    return p

def poly_to_bbox(poly: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    x0 = clamp(int(xs.min()), 0, w - 1)
    y0 = clamp(int(ys.min()), 0, h - 1)
    x1 = clamp(int(xs.max()) + 1, 1, w)
    y1 = clamp(int(ys.max()) + 1, 1, h)
    if x1 <= x0: x1 = min(w, x0 + 1)
    if y1 <= y0: y1 = min(h, y0 + 1)
    return (x0, y0, x1, y1)

def bbox_union(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def bbox_area(bb: Tuple[int,int,int,int]) -> int:
    return max(0, bb[2]-bb[0]) * max(0, bb[3]-bb[1])

def iou_bbox(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1-x0) * max(0, y1-y0)
    if inter <= 0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return inter / max(1, ua)

def mask_coverage_ratio(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    return float(m.sum()) / float(m.size)

def sanitize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Ensure mask is uint8, single-channel, and same HxW as target."""
    th, tw = target_hw
    if mask is None:
        return np.zeros((th, tw), dtype=np.uint8)

    m = mask
    if m.ndim == 3:
        if m.shape[2] == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        elif m.shape[2] == 4:
            m = cv2.cvtColor(m, cv2.COLOR_BGRA2GRAY)
        else:
            m = m[:, :, 0]

    if m.shape[0] != th or m.shape[1] != tw:
        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)

    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)

    m = ((m > 0).astype(np.uint8) * 255)
    return m

def median_rgb(pixels_rgb: np.ndarray) -> Tuple[int,int,int]:
    if pixels_rgb.size == 0:
        return (255,255,255)
    med = np.median(pixels_rgb.reshape(-1,3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))

def rel_luminance(rgb: Tuple[int,int,int]) -> float:
    def f(c: float) -> float:
        c = c / 255.0
        return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    r,g,b = rgb
    R,G,B = f(r),f(g),f(b)
    return 0.2126*R + 0.7152*G + 0.0722*B

def contrast_ratio(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    L1 = rel_luminance(a)
    L2 = rel_luminance(b)
    lo, hi = (min(L1,L2), max(L1,L2))
    return (hi + 0.05) / (lo + 0.05)

def save_mask_overlay_debug(
    bgr: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    fill_alpha: float = 0.20,
    outline_thickness: int = 2
) -> None:
    """Overlay mask on image (white fill + red outline)."""
    if bgr is None or mask is None:
        return
    h, w = bgr.shape[:2]
    m = sanitize_mask(mask, (h, w))
    m_bin = (m > 0).astype(np.uint8) * 255

    out = bgr.copy().astype(np.float32)
    white = np.full_like(out, 255.0)
    a = float(max(0.0, min(1.0, fill_alpha)))

    idx = (m_bin > 0)
    out[idx] = out[idx] * (1.0 - a) + white[idx] * a
    out = np.clip(out, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), int(outline_thickness))
    cv2.imwrite(out_path, out)

def residual_stroke_mask(bgr_roi: np.ndarray) -> np.ndarray:
    """
    Return a uint8 mask (0/255) of leftover 'ink' inside ROI.
    Works by comparing to a local background estimate (median blur).
    """
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

    # Local background estimate
    bg = cv2.medianBlur(gray, 21)
    diff = cv2.absdiff(gray, bg)

    # Adaptive threshold: pick a high percentile of diff
    t = int(np.percentile(diff, 92))
    t = max(12, min(t, 60))

    m = (diff >= t).astype(np.uint8) * 255

    # Clean up mask: close small gaps, remove tiny specks
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # Optional: drop very small connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 25:  # tune
            out[labels == i] = 255
    return out

def detect_residual_in_green_boxes(
    bgr: np.ndarray,
    regions: list,
    residual_max_cc_area: int,
) -> np.ndarray:
    """
    Detect leftover 'ink' pixels *strictly* inside each region's green bbox:
      global_green = (x0+mx0, y0+my0, x0+mx1, y0+my1)
    Uses a local background estimate (median blur) + abs diff threshold.
    Returns full-size uint8 mask (0/255).
    """
    H, W = bgr.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    for r in regions:
        x0, y0, x1, y1 = map(int, r.bbox)
        mx0, my0, mx1, my1 = map(int, r.mask_bbox_local)

        gx0 = max(0, min(W, x0 + mx0))
        gx1 = max(0, min(W, x0 + mx1))
        gy0 = max(0, min(H, y0 + my0))
        gy1 = max(0, min(H, y0 + my1))
        if gx1 <= gx0 or gy1 <= gy0:
            continue

        roi_g = gray[gy0:gy1, gx0:gx1]
        if roi_g.size == 0:
            continue

        # Local background estimate inside ROI
        # (kernel size chosen to preserve flat background while removing thin strokes)
        ksz = 21
        if min(roi_g.shape[0], roi_g.shape[1]) < 64:
            ksz = 11
        bg = cv2.medianBlur(roi_g, ksz)

        diff = cv2.absdiff(roi_g, bg)

        # Adaptive threshold: high percentile of local diff
        t = int(np.percentile(diff, 92))
        t = max(12, min(t, 70))
        m = (diff >= t).astype(np.uint8) * 255

        # Light cleanup: close gaps but DO NOT dilate outward
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

        # Remove huge components (avoid wiping large graphics)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area <= int(residual_max_cc_area):
                keep[labels == i] = 255

        out[gy0:gy1, gx0:gx1] = cv2.bitwise_or(out[gy0:gy1, gx0:gx1], keep)

    return out

# -----------------------------
# OCR Backends
# -----------------------------
class BaseOCR:
    def name(self) -> str:
        raise NotImplementedError
    def run(self, bgr: np.ndarray, img_path: Optional[str]=None) -> List[OCRToken]:
        raise NotImplementedError

class PaddleOCRBackend(BaseOCR):
    """
    Compatible with PaddleOCR variants; avoids unsupported kwargs like cls=...
    """
    def __init__(self, lang: str="ch", device: Optional[str]=None, suppress_init_logs: bool=True, suppress_run_logs: bool=True) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise RuntimeError("paddleocr not installed. pip install paddleocr") from e

        import inspect
        self._suppress_run_logs = bool(suppress_run_logs)

        kwargs: Dict[str, Any] = {}
        sig = inspect.signature(PaddleOCR.__init__)

        def has_param(name: str) -> bool:
            return name in sig.parameters
        if 0:
            if has_param("lang"):
                kwargs["lang"] = lang
            if has_param("use_textline_orientation"):
                kwargs["use_textline_orientation"] = False
            elif has_param("use_angle_cls"):
                kwargs["use_angle_cls"] = False
            if has_param("show_log"):
                kwargs["show_log"] = False
            if device and has_param("device"):
                kwargs["device"] = device

        for k, v in [
            ("use_doc_orientation_classify", False),
            ("use_doc_unwarping", False),
            ("use_doc_align", False),
            ("use_doc_preprocess", False),
            ("lang", lang),
            ("use_textline_orientation", False),
            ("use_angle_cls", False),
            ("show_log", False),
            ("device", device)
        ]:
            if has_param(k):
                kwargs[k] = v

        if suppress_init_logs:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                self._ocr = PaddleOCR(**kwargs)
        else:
            self._ocr = PaddleOCR(**kwargs)

        self._has_predict = hasattr(self._ocr, "predict")

    def name(self) -> str:
        return "paddle"

    def _normalize_v2(self, res: Any, w: int, h: int) -> List[OCRToken]:
        if res is None:
            return []
        inner = res[0] if (isinstance(res, list) and len(res)==1 and isinstance(res[0], list)) else res
        tokens: List[OCRToken] = []
        if not isinstance(inner, list):
            return tokens

        for item in inner:
            try:
                box = np.array(item[0], dtype=np.int32)
                text = item[1][0] if isinstance(item[1], (list, tuple)) else ""
                score = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 1.0
                if box.shape != (4,2):
                    bb = poly_to_bbox(box, w, h)
                    x0,y0,x1,y1 = bb
                    box = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.int32)
                tokens.append(OCRToken(text=str(text), score=float(score), poly=box))
            except Exception:
                continue
        return tokens

    def _normalize_v3_obj(self, obj: Any, w: int, h: int) -> List[OCRToken]:
        res = None
        if hasattr(obj, "res"):
            try:
                res = getattr(obj, "res")
            except Exception:
                res = None
        elif isinstance(obj, dict):
            res = obj.get("res", obj)
        if not isinstance(res, dict):
            return []

        rec_texts = res.get("rec_texts", None)
        rec_scores = res.get("rec_scores", None)
        rec_polys = res.get("rec_polys", None) or res.get("dt_polys", None)

        if rec_texts is None or rec_polys is None:
            return []
        if rec_scores is None:
            rec_scores = [1.0] * len(rec_texts)

        try:
            rec_polys = np.array(rec_polys, dtype=np.int32)
        except Exception:
            return []

        tokens: List[OCRToken] = []
        for i, t in enumerate(list(rec_texts)):
            text = "" if t is None else str(t)
            try:
                score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
            except Exception:
                score = 1.0
            poly = rec_polys[i]
            if poly.shape != (4,2):
                bb = poly_to_bbox(poly, w, h)
                x0,y0,x1,y1 = bb
                poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.int32)
            tokens.append(OCRToken(text=text, score=score, poly=poly.astype(np.int32)))
        return tokens

    def run(self, bgr: np.ndarray, img_path: Optional[str]=None) -> List[OCRToken]:
        h, w = bgr.shape[:2]

        def _call(fn):
            if not self._suppress_run_logs:
                return fn()
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                return fn()

        if self._has_predict and img_path:
            try:
                out = _call(lambda: self._ocr.predict(img_path))
                tokens: List[OCRToken] = []
                for obj in out:
                    tokens.extend(self._normalize_v3_obj(obj, w, h))
                if tokens:
                    return tokens
            except Exception:
                pass

        out = _call(lambda: self._ocr.ocr(bgr))
        return self._normalize_v2(out, w, h)

class RapidOCRBackend(BaseOCR):
    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception as e:
            raise RuntimeError("rapidocr-onnxruntime not installed. pip install rapidocr-onnxruntime") from e
        self._ocr = RapidOCR()

    def name(self) -> str:
        return "rapid"

    def run(self, bgr: np.ndarray, img_path: Optional[str]=None) -> List[OCRToken]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res, _ = self._ocr(rgb)
        tokens: List[OCRToken] = []
        if not res:
            return tokens
        for item in res:
            if len(item) < 3:
                continue
            box = np.array(item[0], dtype=np.int32)
            text = str(item[1]) if item[1] is not None else ""
            score = float(item[2]) if item[2] is not None else 0.0
            if box.shape != (4,2):
                continue
            tokens.append(OCRToken(text=text, score=score, poly=box))
        return tokens

class PaddleOCRBackend2(BaseOCR):
    """
    PaddleOCR backend with:
      - --paddle-enable-hpi
      - --paddlex-config YAML support
      - version-safe disabling of doc transforms (keeps coords aligned)
      - --paddle-rec-batch N YAML override in-memory (writes temp YAML)
    """
    def __init__(
        self,
        lang: str = "ch",
        device: Optional[str] = None,
        suppress_init_logs: bool = True,
        suppress_run_logs: bool = True,
        enable_hpi: bool = False,
        paddlex_config: Optional[str] = None,
        rec_batch_override: Optional[int] = None,
    ) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise RuntimeError("paddleocr not installed. pip install paddleocr") from e

        import inspect
        self._PaddleOCR = PaddleOCR
        self._suppress_run_logs = bool(suppress_run_logs)

        sig = inspect.signature(PaddleOCR.__init__)

        def has_param(name: str) -> bool:
            return name in sig.parameters

        def add_if_supported(k: str, v: Any) -> None:
            if has_param(k):
                kwargs[k] = v

        # -----------------------------
        # Build kwargs (version-safe)
        # -----------------------------
        kwargs: Dict[str, Any] = {}

        # language
        add_if_supported("lang", lang)

        # device (varies by versions)
        # prefer explicit args; otherwise let Paddle decide
        if device:
            # common: device='gpu'/'cpu'
            add_if_supported("device", device)
            # older: use_gpu=True/False
            if has_param("use_gpu"):
                kwargs["use_gpu"] = (device.lower() in ("gpu", "cuda"))
        else:
            # If user didn't specify, do nothing (Paddle decides).
            pass

        # HPI (PaddleOCR 3.x)
        add_if_supported("enable_hpi", bool(enable_hpi))

        # Disable optional doc transforms / orientation (IMPORTANT for bbox alignment + speed)
        # These flags are version-dependent. We set them if supported.
        add_if_supported("use_doc_orientation_classify", False)
        add_if_supported("use_doc_unwarping", False)
        add_if_supported("use_doc_align", False)
        add_if_supported("use_doc_preprocess", False)
        # Textline orientation / angle cls (can be slow + shifts sometimes)
        add_if_supported("use_textline_orientation", False)
        add_if_supported("use_angle_cls", False)

        # reduce logs if supported
        add_if_supported("show_log", False)

        # -----------------------------
        # Handle paddlex_config + YAML override for rec batch
        # -----------------------------
        effective_cfg = paddlex_config

        if rec_batch_override is not None:
            effective_cfg = self._make_overridden_paddlex_yaml(
                base_cfg_path=paddlex_config,
                init_kwargs_for_export=kwargs,
                rec_batch=int(rec_batch_override),
                suppress_init_logs=suppress_init_logs,
            )

        # pass paddlex_config if supported
        if effective_cfg and has_param("paddlex_config"):
            kwargs["paddlex_config"] = effective_cfg

        # -----------------------------
        # Init OCR (suppress init logs)
        # -----------------------------
        if suppress_init_logs:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                self._ocr = PaddleOCR(**kwargs)
        else:
            self._ocr = PaddleOCR(**kwargs)

    def name(self) -> str:
        return "paddle"

    def _make_overridden_paddlex_yaml(
        self,
        base_cfg_path: Optional[str],
        init_kwargs_for_export: Dict[str, Any],
        rec_batch: int,
        suppress_init_logs: bool,
    ) -> str:
        """
        Load YAML (from base_cfg_path or exported default), override TextRecognition.batch_size,
        and write to a temp file. Returns temp yaml path.
        """
        # Need YAML parser
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required for --paddle-rec-batch. pip install pyyaml") from e

        rec_batch = int(max(1, rec_batch))

        # 1) Get a base YAML path
        if base_cfg_path and os.path.exists(base_cfg_path):
            cfg_path = base_cfg_path
        else:
            # Export default config to a temp yaml
            # This may download models / init pipeline once; still worth it for batch speed later.
            tmp0 = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
            tmp0.close()
            cfg_path = tmp0.name

            # Build a minimal pipeline to export YAML
            # IMPORTANT: don't pass paddlex_config here (we're generating it)
            export_kwargs = dict(init_kwargs_for_export)
            export_kwargs.pop("paddlex_config", None)

            # Some versions may not have export_paddlex_config_to_yaml; fail cleanly
            if suppress_init_logs:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    pipe = self._PaddleOCR(**export_kwargs)
            else:
                pipe = self._PaddleOCR(**export_kwargs)

            if not hasattr(pipe, "export_paddlex_config_to_yaml"):
                raise RuntimeError(
                    "Your PaddleOCR version does not support export_paddlex_config_to_yaml(). "
                    "Please provide --paddlex-config PaddleOCR.yaml instead."
                )
            pipe.export_paddlex_config_to_yaml(cfg_path)

        # 2) Load YAML dict
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise RuntimeError(f"Invalid paddlex_config yaml: {cfg_path}")

        # 3) Override recognition batch in-memory (robust to key casing)
        def norm(s: str) -> str:
            return (s or "").lower().replace("_", "").replace("-", "")

        def find_key(d: dict, candidates: List[str]) -> Optional[str]:
            if not isinstance(d, dict):
                return None
            candn = [norm(c) for c in candidates]
            for k in d.keys():
                if norm(str(k)) in candn:
                    return k
            return None

        smk = find_key(cfg, ["SubModules", "sub_modules", "submodules"])
        if smk is None or not isinstance(cfg.get(smk), dict):
            # some configs might store modules elsewhere, but SubModules is the normal one
            raise RuntimeError("Cannot find 'SubModules' in paddlex_config YAML to override batch_size.")

        sub = cfg[smk]

        # TextRecognition module
        trk = find_key(sub, ["TextRecognition", "text_recognition", "textrecognition"])
        if trk and isinstance(sub.get(trk), dict):
            tr = sub[trk]
            bsk = find_key(tr, ["batch_size", "batchsize"])
            tr[bsk or "batch_size"] = rec_batch
        else:
            raise RuntimeError("Cannot find 'TextRecognition' module in paddlex_config YAML.")

        # If TextLineOrientation exists, batch it too (optional)
        tok = find_key(sub, ["TextLineOrientation", "text_line_orientation", "textlineorientation"])
        if tok and isinstance(sub.get(tok), dict):
            to = sub[tok]
            bsk = find_key(to, ["batch_size", "batchsize"])
            to[bsk or "batch_size"] = rec_batch

        # 4) Write overridden YAML to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        tmp.close()
        out_path = tmp.name

        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        return out_path

    def _normalize_v2(self, res: Any, w: int, h: int) -> List[OCRToken]:
        if res is None:
            return []
        inner = res[0] if (isinstance(res, list) and len(res) == 1 and isinstance(res[0], list)) else res
        tokens: List[OCRToken] = []
        if not isinstance(inner, list):
            return tokens

        for item in inner:
            try:
                box = np.array(item[0], dtype=np.int32)
                text = item[1][0] if isinstance(item[1], (list, tuple)) else ""
                score = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 1.0
                if box.shape != (4, 2):
                    bb = poly_to_bbox(box, w, h)
                    x0, y0, x1, y1 = bb
                    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
                tokens.append(OCRToken(text=str(text), score=float(score), poly=box))
            except Exception:
                continue
        return tokens

    def run(self, bgr: np.ndarray, img_path: Optional[str] = None) -> List[OCRToken]:
        h, w = bgr.shape[:2]

        def _call(fn):
            if not self._suppress_run_logs:
                return fn()
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                return fn()

        # Prefer .ocr(...) for consistent coords
        if img_path:
            out = _call(lambda: self._ocr.ocr(img_path))
        else:
            out = _call(lambda: self._ocr.ocr(bgr))
        return self._normalize_v2(out, w, h)

def build_ocr(backend: str, paddle_lang: str, paddle_device: Optional[str], suppress_init_logs: bool) -> BaseOCR:
    backend = (backend or "auto").lower()
    if backend == "rapid":
        return RapidOCRBackend()
    if backend == "paddle":
        return PaddleOCRBackend(lang=paddle_lang, device=paddle_device, suppress_init_logs=suppress_init_logs, suppress_run_logs=True)
    if backend == "auto":
        try:
            return PaddleOCRBackend(lang=paddle_lang, device=paddle_device, suppress_init_logs=suppress_init_logs, suppress_run_logs=True)
        except Exception:
            return RapidOCRBackend()
    raise ValueError(f"Unknown --ocr-backend: {backend}")

# -----------------------------
# OCR variants (NO downscale)
# -----------------------------
def preprocess_variants(bgr: np.ndarray, augment: int) -> List[np.ndarray]:
    out = [bgr]
    if augment >= 2:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab2 = cv2.merge([cl,a,b])
        out.append(cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR))
    if augment >= 3:
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        out.append(cv2.filter2D(bgr, -1, k))
    return out

def merge_tokens_by_iou(tokens_list: List[List[OCRToken]], w: int, h: int, iou_thr: float=0.75) -> List[OCRToken]:
    flat: List[OCRToken] = []
    for lst in tokens_list:
        flat.extend(lst)
    flat.sort(key=lambda t: t.score, reverse=True)

    kept: List[OCRToken] = []
    kept_bbs: List[Tuple[int,int,int,int]] = []
    for t in flat:
        bb = poly_to_bbox(t.poly, w, h)
        ok = True
        for kbb in kept_bbs:
            if iou_bbox(bb, kbb) >= iou_thr:
                ok = False
                break
        if ok:
            kept.append(t)
            kept_bbs.append(bb)
    return kept

# -----------------------------
# Schema + gating
# -----------------------------
def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def cluster_1d(values: List[float], tol: float) -> List[List[float]]:
    if not values:
        return []
    values = sorted(values)
    groups: List[List[float]] = [[values[0]]]
    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups

def ocr_gate_is_chart(tokens: List[OCRToken], schema: Dict[str,Any]) -> Tuple[bool, Dict[str,Any]]:
    texts = [t.text.strip() for t in tokens if t.text and t.text.strip()]
    total = len(texts)
    chinese = sum(1 for x in texts if contains_chinese(x))
    digits = sum(1 for x in texts if DIGIT_RE.search(x))
    joined = " ".join(texts)
    lower = joined.lower()

    kw = schema.get("keywords", {}) if isinstance(schema.get("keywords", {}), dict) else {}
    strong_kw = kw.get("strong_chart_keywords", []) or []
    eng_kw = kw.get("english_chart_keywords", []) or []

    strong_hits = sum(1 for k in strong_kw if (k in joined or k.lower() in lower))
    eng_hits = sum(1 for k in eng_kw if k.lower() in lower)

    cols = rows = 0
    table_like = False
    if len(tokens) >= 6:
        centers = []
        for t in tokens:
            bb = poly_to_bbox(t.poly, 10**9, 10**9)
            cx = (bb[0] + bb[2]) * 0.5
            cy = (bb[1] + bb[3]) * 0.5
            centers.append((cx, cy))
        xs = sorted([c[0] for c in centers])
        ys = sorted([c[1] for c in centers])
        x_tol = max(10.0, (xs[-1]-xs[0]) * 0.06) if xs else 10.0
        y_tol = max(10.0, (ys[-1]-ys[0]) * 0.06) if ys else 10.0
        cols = len(cluster_1d(xs, x_tol))
        rows = len(cluster_1d(ys, y_tol))
        table_like = (cols >= 3 and rows >= 3 and total >= 9)

    pass_gate = table_like or (strong_hits >= 2) or ((digits >= 6 and total >= 10) and (strong_hits >= 1 or eng_hits >= 2))
    info = {
        "total_texts": total,
        "chinese_texts": chinese,
        "digit_texts": digits,
        "strong_kw_hits": strong_hits,
        "eng_kw_hits": eng_hits,
        "cols": cols,
        "rows": rows,
        "table_like": table_like
    }
    return bool(pass_gate), info

# -----------------------------
# Build Chinese mask + regions (NO expansion)
# -----------------------------
def estimate_line_count(tokens: List[OCRToken], w: int, h: int) -> int:
    if not tokens:
        return 1
    bbs = [poly_to_bbox(t.poly, w, h) for t in tokens]
    heights = [max(1, bb[3]-bb[1]) for bb in bbs]
    med_h = float(np.median(heights)) if heights else 16.0
    ys = [((bb[1]+bb[3])*0.5) for bb in bbs]
    y_tol = max(6.0, med_h * 0.55)
    clusters = cluster_1d(ys, y_tol)
    return max(1, len(clusters))

def build_chinese_mask_and_tokens(
    tokens: List[OCRToken],
    shape_hw: Tuple[int,int],
    max_single_poly_area_ratio: float
) -> Tuple[np.ndarray, List[OCRToken], List[OCRToken]]:
    h, w = shape_hw
    mask = np.zeros((h,w), dtype=np.uint8)
    zh: List[OCRToken] = []
    non: List[OCRToken] = []
    img_area = float(w * h)

    for t in tokens:
        if not t.text:
            continue
        if contains_chinese(t.text):
            poly = clip_poly(t.poly, w, h)
            bb = poly_to_bbox(poly, w, h)
            if bbox_area(bb) > int(max_single_poly_area_ratio * img_area):
                continue
            zh.append(OCRToken(text=t.text, score=t.score, poly=poly))
            cv2.fillPoly(mask, [poly], 255)  # NO expansion
        else:
            non.append(t)

    mask = sanitize_mask(mask, (h, w))
    return mask, zh, non

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def build_regions_from_chinese_tokens(
    zh_tokens: List[OCRToken],
    img_hw: Tuple[int,int],
    gap_px: int = 8,
    iou_thr: float = 0.03
) -> List[TextRegion]:
    h, w = img_hw
    if not zh_tokens:
        return []

    bbs = [poly_to_bbox(t.poly, w, h) for t in zh_tokens]
    heights = [max(1, bb[3]-bb[1]) for bb in bbs]
    med_h = float(np.median(heights)) if heights else 18.0
    gap = max(gap_px, int(med_h * 0.35))

    uf = UnionFind(len(zh_tokens))

    def close(bb1, bb2) -> bool:
        x0 = max(bb1[0], bb2[0]); y0 = max(bb1[1], bb2[1])
        x1 = min(bb1[2], bb2[2]); y1 = min(bb1[3], bb2[3])
        inter = max(0, x1-x0) * max(0, y1-y0)
        if inter > 0:
            return True
        dx = max(0, max(bb1[0]-bb2[2], bb2[0]-bb1[2]))
        dy = max(0, max(bb1[1]-bb2[3], bb2[1]-bb1[3]))
        return (dx <= gap and dy <= gap)

    for i in range(len(zh_tokens)):
        for j in range(i+1, len(zh_tokens)):
            bb1, bb2 = bbs[i], bbs[j]
            if close(bb1, bb2):
                if iou_bbox(bb1, bb2) >= iou_thr or close(bb1, bb2):
                    uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(zh_tokens)):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    regions: List[TextRegion] = []
    for idxs in groups.values():
        bb = bbs[idxs[0]]
        region_mask = np.zeros((h,w), dtype=np.uint8)
        toks: List[OCRToken] = []
        for k in idxs:
            bb = bbox_union(bb, bbs[k])
            toks.append(zh_tokens[k])
            poly = clip_poly(zh_tokens[k].poly, w, h)
            cv2.fillPoly(region_mask, [poly], 255)

        region_mask = sanitize_mask(region_mask, (h,w))
        toks_sorted = sorted(toks, key=lambda t: (poly_to_bbox(t.poly, w, h)[1], poly_to_bbox(t.poly, w, h)[0]))
        zh_text = "\n".join([t.text.strip() for t in toks_sorted if t.text and t.text.strip()]).strip()
        lc = estimate_line_count(toks_sorted, w, h)

        regions.append(TextRegion(
            id=random_id("cell_"),
            tokens=toks_sorted,
            bbox=bb,
            mask=region_mask,
            zh_text=zh_text,
            line_count=lc,
            mask_bbox_local=(0,0,0,0),
            mask_bands_raw=[]
        ))

    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return regions

# -----------------------------
# Mask-tight bbox + mask bands per region
# -----------------------------
def mask_bbox_local(mask_crop: np.ndarray) -> Tuple[int,int,int,int]:
    ys, xs = np.where(mask_crop > 0)
    h, w = mask_crop.shape[:2]
    if len(xs) == 0:
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)

def mask_line_bands(mask_crop: np.ndarray, min_row_ink_ratio: float=0.02, merge_gap: int=3, min_band_h: int=6) -> List[Tuple[int,int]]:
    h, w = mask_crop.shape[:2]
    if h <= 0 or w <= 0:
        return []
    proj = (mask_crop > 0).sum(axis=1)
    thr = max(1, int(min_row_ink_ratio * w))
    has = proj >= thr

    bands: List[List[int]] = []
    y = 0
    while y < h:
        if not has[y]:
            y += 1
            continue
        y0 = y
        while y < h and has[y]:
            y += 1
        y1 = y
        bands.append([y0, y1])

    merged: List[List[int]] = []
    for b in bands:
        if not merged:
            merged.append(b)
            continue
        if b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)

    out = [(int(b[0]), int(b[1])) for b in merged if (b[1]-b[0]) >= min_band_h]
    return out

def merge_bands_to_k(bands: List[Tuple[int,int]], k: int) -> List[Tuple[int,int]]:
    if k <= 0:
        return []
    b = [list(x) for x in bands]
    while len(b) > k:
        gaps = []
        for i in range(len(b)-1):
            gap = b[i+1][0] - b[i][1]
            gaps.append((gap, i))
        gaps.sort(key=lambda x: x[0])
        _, i = gaps[0]
        b[i][1] = b[i+1][1]
        del b[i+1]
    return [(int(x[0]), int(x[1])) for x in b]

def uniform_bands(mask_h: int, k: int) -> List[Tuple[int,int]]:
    if k <= 0:
        return []
    mask_h = max(1, int(mask_h))
    base = mask_h // k
    rem = mask_h % k
    bands = []
    y = 0
    for i in range(k):
        hh = base + (1 if i < rem else 0)
        y0, y1 = y, y + max(1, hh)
        bands.append((y0, y1))
        y = y1
    bands[-1] = (bands[-1][0], mask_h)
    return bands

def select_bands_for_k(raw: List[Tuple[int,int]], k: int, mask_h: int) -> List[Tuple[int,int]]:
    if k <= 1:
        return [(0, mask_h)]
    raw = [(max(0,y0), min(mask_h,y1)) for (y0,y1) in (raw or []) if y1 > y0]
    raw.sort(key=lambda x: x[0])

    if len(raw) == k:
        return raw
    if len(raw) > k:
        return merge_bands_to_k(raw, k)
    return uniform_bands(mask_h, k)

def compute_region_mask_geometry(regions: List[TextRegion], img_hw: Tuple[int,int]) -> None:
    H, W = img_hw
    for r in regions:
        x0,y0,x1,y1 = r.bbox
        x0 = clamp(x0,0,W); x1 = clamp(x1,0,W)
        y0 = clamp(y0,0,H); y1 = clamp(y1,0,H)
        if x1 <= x0 or y1 <= y0:
            r.mask_bbox_local = (0,0,0,0)
            r.mask_bands_raw = []
            continue

        crop = r.mask[y0:y1, x0:x1]
        crop = sanitize_mask(crop, crop.shape[:2])
        mx0,my0,mx1,my1 = mask_bbox_local(crop)
        r.mask_bbox_local = (mx0,my0,mx1,my1)

        sub = crop[my0:my1, mx0:mx1] if (mx1>mx0 and my1>my0) else crop
        raw = mask_line_bands(sub, min_row_ink_ratio=0.02, merge_gap=3, min_band_h=6)

        # IMPORTANT FIX:
        # store bands in MASK-LOCAL coords [0..mask_h) (do NOT shift by my0)
        r.mask_bands_raw = raw if raw else []

# -----------------------------
# Group keys (debug only; no longer used for fonts)
# -----------------------------
def cluster_sizes(regions: List[TextRegion], rel_tol: float=0.15) -> Dict[str, int]:
    sizes = []
    for r in regions:
        x0,y0,x1,y1 = r.bbox
        sizes.append((r.id, x1-x0, y1-y0))

    clusters: List[Tuple[float,float,List[str]]] = []

    def similar(w1,h1,w2,h2) -> bool:
        dw = abs(w1-w2)/max(1.0, max(w1,w2))
        dh = abs(h1-h2)/max(1.0, max(h1,h2))
        return (dw <= rel_tol and dh <= rel_tol)

    for rid, w, h in sizes:
        placed = False
        for i, (cw,ch, ids) in enumerate(clusters):
            if similar(w,h,cw,ch):
                new_w = (cw*len(ids) + w) / (len(ids)+1)
                new_h = (ch*len(ids) + h) / (len(ids)+1)
                clusters[i] = (new_w, new_h, ids + [rid])
                placed = True
                break
        if not placed:
            clusters.append((float(w), float(h), [rid]))

    id_to_cluster: Dict[str,int] = {}
    for ci, (_,_, ids) in enumerate(clusters):
        for rid in ids:
            id_to_cluster[rid] = ci
    return id_to_cluster

def assign_row_col_clusters(regions: List[TextRegion]) -> Tuple[Dict[str,int], Dict[str,int]]:
    if not regions:
        return {}, {}

    centers = []
    heights = []
    widths = []
    for r in regions:
        x0,y0,x1,y1 = r.bbox
        centers.append((r.id, (x0+x1)*0.5, (y0+y1)*0.5))
        heights.append(max(1, y1-y0))
        widths.append(max(1, x1-x0))

    med_h = float(np.median(heights)) if heights else 18.0
    med_w = float(np.median(widths)) if widths else 80.0

    y_tol = max(8.0, med_h*0.55)
    x_tol = max(8.0, med_w*0.20)

    ys = sorted([(cy, rid) for rid, _, cy in centers])
    row_id: Dict[str,int] = {}
    cur = 0
    last_y = None
    for cy, rid in ys:
        if last_y is None or abs(cy - last_y) > y_tol:
            cur += 1
            last_y = cy
        row_id[rid] = cur

    xs = sorted([(cx, rid) for rid, cx, _ in centers])
    col_id: Dict[str,int] = {}
    cur = 0
    last_x = None
    for cx, rid in xs:
        if last_x is None or abs(cx - last_x) > x_tol:
            cur += 1
            last_x = cx
        col_id[rid] = cur

    return row_id, col_id

def build_group_keys(regions: List[TextRegion]) -> None:
    if not regions:
        return
    size_cluster = cluster_sizes(regions)
    row_cluster, col_cluster = assign_row_col_clusters(regions)

    row_map: Dict[Tuple[int,int], List[str]] = {}
    col_map: Dict[Tuple[int,int], List[str]] = {}

    for r in regions:
        sc = size_cluster[r.id]
        rc = row_cluster[r.id]
        cc = col_cluster[r.id]
        row_map.setdefault((rc, sc), []).append(r.id)
        col_map.setdefault((cc, sc), []).append(r.id)

    for r in regions:
        sc = size_cluster[r.id]
        rc = row_cluster[r.id]
        cc = col_cluster[r.id]
        rk = (rc, sc)
        ck = (cc, sc)
        if len(row_map.get(rk, [])) >= 2:
            r.group_key = f"row{rc}_sz{sc}"
        elif len(col_map.get(ck, [])) >= 2:
            r.group_key = f"col{cc}_sz{sc}"
        else:
            r.group_key = f"cell_{r.id}_sz{sc}"

# -----------------------------
# Translation (Gemini Vertex) + fit hints
# -----------------------------
def _require_real_project(project: str) -> None:
    bad = {None, "", "YOUR_PROJECT", "default"}
    if project is None or project.strip() in bad:
        raise SystemExit(
            "Vertex AI project is not set (or still 'YOUR_PROJECT'). "
            "Pass --gcp-project with a real Project ID."
        )

def estimate_fit_hint_for_region(
    box_w: int,
    box_h: int,
    target_lines: int,
    max_lines: int,
    min_font_px: int,
    max_font_px: int
) -> Dict[str,Any]:
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))

    denom = (target_lines * 1.15 + max(0, target_lines-1) * 0.18)
    est_font = int(box_h / max(1e-6, denom))
    est_font = int(max(min_font_px, min(max_font_px, est_font)))

    inner_w = int(box_w * 0.86)
    cpl = int(max(6, inner_w / max(1.0, 0.55 * est_font)))
    max_total = int(cpl * max(1, max_lines))

    return {
        "box_w": box_w,
        "box_h": box_h,
        "target_lines": int(target_lines),
        "max_lines": int(max_lines),
        "est_font_px": int(est_font),
        "max_chars_per_line": int(cpl),
        "max_total_chars": int(max_total),
        "notes": "Keep translation concise. Prefer short words/abbreviations. Insert line breaks if needed."
    }

class GeminiTranslator:
    def __init__(self, project: str, location: str, model: str, schema: Dict[str,Any], temperature: float=0.0, api_key: Optional[str]=None) -> None:
        self.project = project
        self.location = location
        self.model = model
        self.schema = schema
        self.temperature = float(temperature)
        self.api_key = api_key

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as e:
            raise RuntimeError("google-genai not installed. pip install google-genai") from e

        self._types = types
        
        # Use API Key if provided, otherwise use VertexAI
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            _require_real_project(project)
            self._client = genai.Client(vertexai=True, project=project, location=location)

    def _extract_json(self, s: str) -> Dict[str,Any]:
        s = (s or "").strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            return json.loads(s[i:j+1])
        raise RuntimeError("Gemini returned non-JSON or unparsable JSON.")

    def translate_regions(self, regions: List[TextRegion], min_font_px: int, max_font_px: int, max_extra_lines: int, max_retries: int=3) -> Dict[str,List[str]]:
        items = []
        for r in regions:
            target = int(max(1, r.line_count))
            max_lines = int(max(1, target + max(0, max_extra_lines)))

            mx0,my0,mx1,my1 = r.mask_bbox_local
            mask_w = max(1, mx1 - mx0)
            mask_h = max(1, my1 - my0)
            hint = estimate_fit_hint_for_region(mask_w, mask_h, target, max_lines, min_font_px, max_font_px)

            items.append({
                "id": r.id,
                "zh": r.zh_text,
                "fit_hint": hint,
            })

        tr = self.schema.get("translation", {}) if isinstance(self.schema.get("translation", {}), dict) else {}
        sys_prompt = tr.get("system_prompt", "").strip()

        fit_rules = (
            "You translate Chinese size-chart labels into concise English.\n"
            "IMPORTANT FIT RULES:\n"
            "- Respect fit_hint.max_chars_per_line and fit_hint.max_total_chars.\n"
            "- Prefer short words / standard abbreviations if needed (e.g., 'Bust', 'Hip', 'Len', 'Sleeve').\n"
            "- Return en_lines with line breaks to fit inside the area.\n"
            "- en_lines length MUST be between fit_hint.target_lines and fit_hint.max_lines.\n"
            "- Avoid adding extra explanation not present in the source.\n"
            "Output strictly valid JSON.\n"
        )
        if sys_prompt:
            sys_prompt = sys_prompt + "\n\n" + fit_rules
        else:
            sys_prompt = fit_rules

        input_json = json.dumps({"items": items}, ensure_ascii=False, indent=2)
        user_prompt = (
            "Translate the following items. For each item, output:\n"
            "{ id: string, en_lines: string[] }\n"
            "Return JSON: { items: [ ... ] }\n\n"
            f"INPUT_JSON:\n{input_json}\n"
        )

        types = self._types
        for attempt in range(1, max_retries+1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Content(role="user", parts=[types.Part(text=f"{sys_prompt}\n\n{user_prompt}")])
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json"
                    )
                )
                data = self._extract_json(resp.text or "")
                out: Dict[str,List[str]] = {}
                for it in data.get("items", []):
                    rid = it.get("id", "")
                    lines = it.get("en_lines", [])
                    if not rid or not isinstance(lines, list):
                        continue
                    out[rid] = [str(x) for x in lines]
                return out
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(0.6 * attempt)
        return {}

def normalize_lines(lines: List[str], target_lines: int, max_lines: int) -> List[str]:
    lines = [("" if l is None else str(l)) for l in (lines or [])]
    tmp: List[str] = []
    for l in lines:
        tmp.extend(l.split("\n"))
    lines = [x.strip() for x in tmp if x is not None]

    if len(lines) < target_lines:
        lines = lines + [""] * (target_lines - len(lines))

    if len(lines) > max_lines:
        head = lines[:max_lines-1]
        tail = " ".join(lines[max_lines-1:]).strip()
        lines = head + [tail]

    if len(lines) < target_lines:
        lines = lines + [""] * (target_lines - len(lines))
    return lines

# -----------------------------
# Inpainting backends (size-safe)
# -----------------------------
class BaseInpaint:
    def name(self) -> str:
        raise NotImplementedError
    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class OpenCVInpaint(BaseInpaint):
    def __init__(self, radius: int=3) -> None:
        self.radius = int(radius)
    def name(self) -> str:
        return "opencv"
    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = sanitize_mask(mask, bgr.shape[:2])
        out = cv2.inpaint(bgr, m, self.radius, cv2.INPAINT_TELEA)
        if out.shape[:2] != bgr.shape[:2]:
            out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out

class LamaInpaint(BaseInpaint):
    def __init__(self, device: str="auto") -> None:
        self.device = device
        self._lama = None
        self._torch = None
        try:
            import torch  # type: ignore
            from simple_lama_inpainting import SimpleLama  # type: ignore
            self._torch = torch
            dev = self._select_device(device)
            self.device = dev
            try:
                self._lama = SimpleLama(device=dev)
            except Exception:
                self.device = "cpu"
                self._lama = SimpleLama(device="cpu")
        except Exception as e:
            raise RuntimeError("LaMa backend requires: pip install simple-lama-inpainting torch") from e

    def _select_device(self, device: str) -> str:
        device = (device or "auto").lower()
        torch = self._torch
        if torch is None:
            return "cpu"
        if device in ["cpu", "cuda", "mps"]:
            if device == "cuda" and getattr(torch, "cuda", None) and torch.cuda.is_available():
                return "cuda"
            if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            if device == "cpu":
                return "cpu"
            return "cpu"
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def name(self) -> str:
        return "lama"

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = sanitize_mask(mask, bgr.shape[:2])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        mm = Image.fromarray(m)
        out = self._lama(img, mm)  # type: ignore
        out_rgb = np.array(out)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        if out_bgr.shape[:2] != bgr.shape[:2]:
            out_bgr = cv2.resize(out_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out_bgr

def build_inpainter(name: str, lama_device: str) -> Tuple[Optional[BaseInpaint], str]:
    name = (name or "opencv").lower()
    if name == "none":
        return None, ""
    if name == "opencv":
        return OpenCVInpaint(radius=3), ""
    if name == "lama":
        lama = LamaInpaint(device=lama_device)
        return lama, lama.device
    raise ValueError(f"Unknown --inpaint-backend: {name}")

# -----------------------------
# TWO-PASS ERASE (mask-safe + size-safe)
# -----------------------------
def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a binary mask WITHOUT expanding the outer boundary.
    IMPORTANT: No final inversion (prevents turning mask all-white).
    """
    m = (mask > 0).astype(np.uint8) * 255

    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(padded)
    ff = inv.copy()
    flood_mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 0)

    holes = ff[1:-1, 1:-1]
    filled = cv2.bitwise_or(m, holes)
    return filled

def detect_residual_ink_mask(
    bgr: np.ndarray,
    base_mask: np.ndarray,
    regions: List[TextRegion],
    ring_dilate: int = 4,
    deltaE_thr: float = 12.0,
    grad_thr: float = 40.0,
    max_cc_area: int = 2000
) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = sanitize_mask(base_mask, (h, w))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ring_dilate+1, 2*ring_dilate+1))
    dil = cv2.dilate(mask, k, iterations=1)
    ring = cv2.bitwise_and(dil, cv2.bitwise_not(mask))

    residual = np.zeros((h, w), dtype=np.uint8)

    for r in regions:
        x0, y0, x1, y1 = r.bbox
        pad = max(2, int(0.02 * max(x1-x0, y1-y0)))
        x0p = clamp(x0 - pad, 0, w); x1p = clamp(x1 + pad, 0, w)
        y0p = clamp(y0 - pad, 0, h); y1p = clamp(y1 + pad, 0, h)
        if x1p <= x0p or y1p <= y0p:
            continue

        roi_mask = mask[y0p:y1p, x0p:x1p]
        roi_ring = ring[y0p:y1p, x0p:x1p]
        roi_lab  = lab[y0p:y1p, x0p:x1p]
        roi_grad = grad[y0p:y1p, x0p:x1p]

        bg_pixels = roi_lab[roi_ring > 0]
        if bg_pixels.size == 0:
            bg_pixels = roi_lab[roi_mask == 0]
        if bg_pixels.size == 0:
            continue

        bg_med = np.median(bg_pixels.reshape(-1, 3), axis=0)
        d = np.abs(roi_lab - bg_med[None, None, :]).sum(axis=2)

        cand = ((d >= float(deltaE_thr)) | (roi_grad >= float(grad_thr))) & (roi_mask > 0)
        cand_u8 = (cand.astype(np.uint8) * 255)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
        keep = np.zeros_like(cand_u8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area <= max_cc_area:
                keep[labels == i] = 255

        residual[y0p:y1p, x0p:x1p] = cv2.bitwise_or(residual[y0p:y1p, x0p:x1p], keep)

    residual = cv2.bitwise_and(residual, mask)
    return sanitize_mask(residual, (h, w))

def two_pass_erase(
    bgr: np.ndarray,
    base_mask: np.ndarray,
    regions: List[TextRegion],
    primary_inpainter: Optional[BaseInpaint],
    refine_radius: int = 7,
    refine_method: str = "ns",
    residual_deltaE_thr: float = 12.0,
    residual_grad_thr: float = 40.0,
    residual_max_cc_area: int = 2000,
    max_refine_iters: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = fill_mask_holes(base_mask)
    m = sanitize_mask(m, bgr.shape[:2])

    if primary_inpainter is not None:
        out = primary_inpainter.inpaint(bgr.copy(), m)
    else:
        out = cv2.inpaint(bgr.copy(), m, 3, cv2.INPAINT_TELEA)

    if out.shape[:2] != bgr.shape[:2]:
        out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    last_resid = np.zeros_like(m)

    for _ in range(max_refine_iters):
        resid = detect_residual_ink_mask(
            out, m, regions,
            ring_dilate=4,
            deltaE_thr=residual_deltaE_thr,
            grad_thr=residual_grad_thr,
            max_cc_area=residual_max_cc_area
        )
        resid = sanitize_mask(resid, out.shape[:2])
        last_resid = resid.copy()

        if int(resid.sum()) == 0:
            break

        if refine_method.lower() == "telea":
            out = cv2.inpaint(out, resid, int(refine_radius), cv2.INPAINT_TELEA)
        else:
            out = cv2.inpaint(out, resid, int(refine_radius), cv2.INPAINT_NS)

    return out, m, last_resid

# -----------------------------
# Fonts + layout (stroke-aware wrapping; shrink-on-cutoff placement)
# -----------------------------
def discover_default_fonts() -> List[str]:
    candidates = []
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    candidates += [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    return [p for p in candidates if os.path.exists(p)]

class FontManager:
    def __init__(self, font_paths: List[str]) -> None:
        self.font_paths = [p for p in font_paths if p and os.path.exists(p)]
        if not self.font_paths:
            self.font_paths = discover_default_fonts()
        if not self.font_paths:
            raise RuntimeError("No usable fonts found. Provide a system font or install DejaVu/Arial.")
        self._cache: Dict[Tuple[str,int], ImageFont.FreeTypeFont] = {}

    def get(self, size: int) -> ImageFont.FreeTypeFont:
        size = int(max(6, size))
        last_err = None
        for fp in self.font_paths:
            key = (fp, size)
            if key in self._cache:
                return self._cache[key]
            try:
                f = ImageFont.truetype(fp, size=size)
                self._cache[key] = f
                return f
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Failed to load any font at size {size}: {last_err}")

def measure_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_w: int = 0) -> Tuple[int,int]:
    bb = draw.textbbox((0,0), text, font=font, stroke_width=int(stroke_w))
    return int(bb[2]-bb[0]), int(bb[3]-bb[1])

def tokenize_for_wrap(text: str) -> List[str]:
    text = (text or "")
    if not text:
        return []
    # Keep separators AND whitespace
    parts = re.split(r"(\s+|/|,|;|:|\||\(|\)|\-|—)", text)
    toks: List[str] = []
    for p in parts:
        if p is None or p == "":
            continue
        if p.isspace():
            toks.append(" ")          # normalize any whitespace run to a single space
        else:
            toks.append(p)
    # Optional: collapse multiple spaces (already normalized above) and avoid leading/trailing spaces later
    return toks

def wrap_tokens_to_k_lines(
    draw: ImageDraw.ImageDraw,
    font,
    toks: List[str],
    k: int,
    max_w: int,
    stroke_w: int = 0
) -> List[str]:
    if k <= 1:
        return ["".join(toks).strip()]

    def clean_line(s: str) -> str:
        # remove leading/trailing space, keep internal spaces
        return s.strip()

    lines: List[str] = []
    cur = ""
    idx = 0

    while idx < len(toks) and len(lines) < k:
        t = toks[idx]

        # Avoid starting a line with a space
        if not cur and t == " ":
            idx += 1
            continue

        trial = cur + t

        w, _ = measure_line(draw, trial, font, stroke_w=stroke_w)
        if w <= max_w or not cur:
            cur = trial
            idx += 1
        else:
            lines.append(clean_line(cur))
            cur = ""
            if len(lines) == k - 1:
                rest = clean_line("".join(toks[idx:]))
                lines.append(rest)
                idx = len(toks)
                break

    if cur and len(lines) < k:
        lines.append(clean_line(cur))

    while len(lines) < k:
        lines.append("")
    return lines[:k]

def _line_bbox_metrics(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_w: int):
    l, t, r, b = draw.textbbox((0, 0), text, font=font, stroke_width=int(stroke_w))
    return int(l), int(t), int(r), int(b)

def _try_place_lines_in_mask_bands(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    stroke_w: int,
    mx0: int, my0: int, mx1: int, my1: int,   # mask bbox in patch coords
    pad_x: int, pad_y: int,
    bands_patch: List[Tuple[int,int]],         # len == len(lines), in patch coords
) -> Optional[Dict[str, Any]]:
    left_bound  = mx0 + pad_x
    right_bound = mx1 - pad_x
    top_bound   = my0 + pad_y
    bot_bound   = my1 - pad_y
    if right_bound <= left_bound or bot_bound <= top_bound:
        return None

    metrics = []
    min_l = 10**9
    max_r = -10**9
    for ln in lines:
        l, t, r, b = _line_bbox_metrics(draw, ln, font, stroke_w)
        metrics.append((l, t, r, b))
        min_l = min(min_l, l)
        max_r = max(max_r, r)

    x_lo = left_bound - min_l
    x_hi = right_bound - max_r
    if x_hi < x_lo:
        return None

    avail_w = right_bound - left_bound
    text_w = max_r - min_l
    desired_x = left_bound + max(0, (avail_w - text_w)//2) - min_l
    x = int(clamp(int(desired_x), int(x_lo), int(x_hi)))

    ys: List[int] = []
    union_L = 10**9
    union_T = 10**9
    union_R = -10**9
    union_B = -10**9

    if len(bands_patch) != len(lines):
        return None

    for i, ((l, t, r, b), (by0, by1)) in enumerate(zip(metrics, bands_patch)):
        band_h = max(1, by1 - by0)
        margin = max(1, int(band_h * 0.08))

        band_top = max(top_bound, by0 + margin)
        band_bot = min(bot_bound, by1 - margin)
        if band_bot <= band_top:
            return None

        y_lo = band_top - t
        y_hi = band_bot - b
        if y_hi < y_lo:
            return None

        center = 0.5 * (band_top + band_bot)
        desired_y = center - 0.5 * (t + b)
        y = int(clamp(int(round(desired_y)), int(y_lo), int(y_hi)))
        ys.append(y)

        L = x + l
        T = y + t
        R = x + r
        B = y + b
        union_L = min(union_L, L)
        union_T = min(union_T, T)
        union_R = max(union_R, R)
        union_B = max(union_B, B)

    if union_L < left_bound or union_R > right_bound or union_T < top_bound or union_B > bot_bound:
        return None

    return {"x": x, "ys": ys, "metrics": metrics, "union": (union_L, union_T, union_R, union_B)}

def render_text_patch(
    bgr_erased: np.ndarray,
    region: TextRegion,
    fm: FontManager,
    min_font_px: int,
    max_font_px: int,
    allow_extra_line: bool,
    max_extra_lines: int,
    min_good_font_px: int,
) -> Tuple[np.ndarray, Dict[str,Any]]:
    x0,y0,x1,y1 = region.bbox
    box_w = max(1, x1-x0)
    box_h = max(1, y1-y0)

    patch_bgr = bgr_erased[y0:y1, x0:x1].copy()
    bg_rgb = median_rgb(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))

    black = (0,0,0)
    white = (255,255,255)
    fg = black if contrast_ratio(black, bg_rgb) >= contrast_ratio(white, bg_rgb) else white
    stroke = white if fg == black else black

    base_text = (region.en_text or "").strip()
    if not base_text and region.en_lines:
        base_text = " ".join([l.strip() for l in region.en_lines if l and l.strip()]).strip()

    target_lines = max(1, region.line_count)

    mx0,my0,mx1,my1 = region.mask_bbox_local
    mask_w = max(1, mx1 - mx0)
    mask_h = max(1, my1 - my0)

    raw_bands = region.mask_bands_raw or []
    toks = tokenize_for_wrap(base_text) or [""]

    pad_candidates = [0.10, 0.07, 0.05, 0.03]
    extra_max = max_extra_lines if allow_extra_line else 0

    tmp = Image.new("RGBA", (box_w, box_h), (0,0,0,0))
    draw = ImageDraw.Draw(tmp)

    best: Optional[Dict[str,Any]] = None

    for pr in pad_candidates:
        pad_x = int(mask_w * pr)
        pad_y = int(mask_h * pr)
        inner_w = max(1, mask_w - 2*pad_x)

        for extra in range(0, extra_max + 1):
            k = max(1, target_lines + extra)

            # bands in MASK-LOCAL coords [0..mask_h), shift into patch coords by +my0
            bands_local = select_bands_for_k(raw_bands, k, mask_h)
            bands_patch = [(my0 + y0b, my0 + y1b) for (y0b, y1b) in bands_local]

            denom = (k * 1.15 + max(0, k-1) * 0.18)
            base_sz = int(mask_h / max(1e-6, denom))
            base_sz = int(min(max_font_px, max(min_font_px, base_sz)))

            for sz in range(base_sz, min_font_px-1, -1):
                font = fm.get(sz)
                stroke_w = int(max(1, round(sz * 0.08)))

                lines = wrap_tokens_to_k_lines(draw, font, toks, k, inner_w, stroke_w=stroke_w)

                placed = _try_place_lines_in_mask_bands(
                    draw=draw,
                    lines=lines,
                    font=font,
                    stroke_w=stroke_w,
                    mx0=mx0, my0=my0, mx1=mx1, my1=my1,
                    pad_x=pad_x, pad_y=pad_y,
                    bands_patch=bands_patch
                )
                if placed is None:
                    continue

                cand = {
                    "sz": sz,
                    "k": k,
                    "pr": pr,
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                    "lines": lines,
                    "font": font,
                    "stroke_w": stroke_w,
                    "placed": placed,
                    "bands_patch": bands_patch
                }

                if best is None:
                    best = cand
                else:
                    if cand["sz"] > best["sz"]:
                        best = cand
                    elif cand["sz"] == best["sz"]:
                        if cand["sz"] < min_good_font_px and cand["k"] > best["k"]:
                            best = cand
                        elif cand["sz"] >= min_good_font_px and cand["k"] < best["k"]:
                            best = cand
                        elif cand["k"] == best["k"] and cand["pr"] < best["pr"]:
                            best = cand

                break  # max sz for this (pr,k)

    if best is None:
        sz = min_font_px
        font = fm.get(sz)
        stroke_w = int(max(1, round(sz * 0.08)))
        lines = wrap_tokens_to_k_lines(draw, font, toks, 1, max(1, mask_w - 2), stroke_w=stroke_w)
        bands_patch = [(my0, my1)]
        placed = _try_place_lines_in_mask_bands(
            draw, lines, font, stroke_w,
            mx0,my0,mx1,my1,
            pad_x=1, pad_y=1,
            bands_patch=bands_patch
        )
        if placed is None:
            placed = {"x": mx0+1, "ys":[my0+1], "union": (mx0+1,my0+1,mx1-1,my1-1)}

        best = {
            "sz": sz, "k": 1, "pr": 0.03, "pad_x": 1, "pad_y": 1,
            "lines": lines, "font": font, "stroke_w": stroke_w,
            "placed": placed, "bands_patch": bands_patch
        }

    patch = Image.new("RGBA", (box_w, box_h), (0,0,0,0))
    d = ImageDraw.Draw(patch)

    x = int(best["placed"]["x"])
    ys = best["placed"]["ys"]
    font = best["font"]
    stroke_w = int(best["stroke_w"])
    lines = best["lines"]

    for y, ln in zip(ys, lines):
        d.text(
            (x, int(y)),
            ln,
            font=font,
            fill=(fg[0], fg[1], fg[2], 255),
            stroke_width=stroke_w,
            stroke_fill=(stroke[0], stroke[1], stroke[2], 255),
        )

    dbg = {
        "fg": fg,
        "bg": bg_rgb,
        "stroke": stroke,
        "stroke_w": stroke_w,
        "font_size": int(getattr(font, "size", 12)),
        "pad_ratio": float(best["pr"]),
        "pad_x": int(best["pad_x"]),
        "pad_y": int(best["pad_y"]),
        "box": [x0,y0,x1,y1],
        "mask_bbox_local": [mx0,my0,mx1,my1],
        "bands_patch": best["bands_patch"],
        "lines": lines,
        "union_bbox": list(best["placed"].get("union", (0,0,0,0))),
        "k_lines": int(best["k"]),
    }
    return np.array(patch), dbg

def overlay_patch(base_bgr: np.ndarray, region: TextRegion, patch_rgba: np.ndarray, use_mask_soft: bool = True) -> np.ndarray:
    x0,y0,x1,y1 = region.bbox
    H,W = base_bgr.shape[:2]
    x0 = clamp(x0,0,W); x1 = clamp(x1,0,W)
    y0 = clamp(y0,0,H); y1 = clamp(y1,0,H)

    patch = patch_rgba[:(y1-y0), :(x1-x0), :]
    if patch.shape[0] <= 0 or patch.shape[1] <= 0:
        return base_bgr

    alpha = patch[:,:,3].astype(np.float32) / 255.0

    if use_mask_soft:
        m = region.mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        m_soft = cv2.GaussianBlur(m, (0,0), sigmaX=0.8)
        outside = alpha * (1.0 - (m > 0.5).astype(np.float32))
        outside_ratio = float(outside.sum() / max(1e-6, alpha.sum()))
        if outside_ratio <= 0.08:
            alpha = alpha * m_soft

    bgr_roi = base_bgr[y0:y1, x0:x1].astype(np.float32)
    rgb = patch[:,:,:3].astype(np.float32)
    rgb_bgr = rgb[:,:,::-1]
    out = bgr_roi*(1.0-alpha[:,:,None]) + rgb_bgr*alpha[:,:,None]
    base_bgr[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return base_bgr

# -----------------------------
# Debug render overlay
# -----------------------------
def save_render_debug_overlay(
    final_bgr: np.ndarray,
    full_mask: np.ndarray,
    regions: List[TextRegion],
    debug_items: List[Dict[str,Any]],
    out_path: str
) -> None:
    img = final_bgr.copy()
    H,W = img.shape[:2]
    m = sanitize_mask(full_mask, (H,W))
    m_bin = (m > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,0,255), 2)  # red outline

    dbg_by_id = {d.get("id",""): d for d in debug_items if isinstance(d, dict)}
    for r in regions:
        d = dbg_by_id.get(r.id, {})
        x0,y0,x1,y1 = r.bbox

        cv2.rectangle(img, (x0,y0), (x1-1,y1-1), (0,255,255), 1)  # bbox yellow

        mx0,my0,mx1,my1 = r.mask_bbox_local
        cv2.rectangle(img, (x0+mx0, y0+my0), (x0+mx1-1, y0+my1-1), (0,255,0), 2)  # mask bbox green

        render_dbg = d.get("render_dbg", {}) if isinstance(d, dict) else {}
        bands = render_dbg.get("bands_patch", [])
        if isinstance(bands, list):
            for i, b in enumerate(bands):
                try:
                    by0, by1 = int(b[0]), int(b[1])
                    cv2.rectangle(img, (x0+mx0, y0+by0), (x0+mx1-1, y0+by1-1), (255,255,0), 1)  # bands cyan
                    cv2.putText(img, str(i+1), (x0+mx0+2, y0+by0+12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)
                except Exception:
                    pass

        union = render_dbg.get("union_bbox", None)
        if isinstance(union, list) and len(union) == 4:
            try:
                ux0,uy0,ux1,uy1 = map(int, union)
                cv2.rectangle(img, (x0+ux0, y0+uy0), (x0+ux1-1, y0+uy1-1), (255,0,255), 1)  # union magenta
            except Exception:
                pass

    cv2.imwrite(out_path, img)

# -----------------------------
# Core processing
# -----------------------------
def process_one(
    path: str,
    ocr: BaseOCR,
    schema: Dict[str,Any],
    translator: Optional[GeminiTranslator],
    inpainter: Optional[BaseInpaint],
    outdir: str,
    min_font_px: int,
    max_font_px: int,
    gate_mode: str,
    ocr_augment: int,
    debug_dir: Optional[str],
    lama_device_used: str,
    allow_extra_line: bool,
    max_extra_lines: int,
    min_good_font_px: int,
    refine_radius: int,
    refine_method: str,
    residual_deltaE_thr: float,
    residual_grad_thr: float,
    residual_max_cc_area: int,
    refine_iters: int,
    max_mask_ratio: float,
    max_single_poly_area_ratio: float,
) -> FileResult:
    t0 = now_ms()
    bgr = safe_imread(path)
    h, w = bgr.shape[:2]

    base = os.path.basename(path)
    name, ext = os.path.splitext(base)

    # OCR on variants (no downscale)
    tokens_passes: List[List[OCRToken]] = []
    for v in preprocess_variants(bgr, augment=ocr_augment):
        tokens_passes.append(ocr.run(v, img_path=path))
    tokens = merge_tokens_by_iou(tokens_passes, w, h, iou_thr=0.75) if len(tokens_passes) > 1 else (tokens_passes[0] if tokens_passes else [])

    full_mask, zh_tokens, _ = build_chinese_mask_and_tokens(tokens, (h,w), max_single_poly_area_ratio=max_single_poly_area_ratio)

    fr = FileResult(
        path=path,
        status="",
        chinese_tokens=len(zh_tokens),
        total_tokens=len(tokens),
        ocr_backend=ocr.name(),
        inpaint_backend=inpainter.name() if inpainter else "none",
        lama_device=lama_device_used
    )

    # Only regenerate when Chinese exists
    if len(zh_tokens) == 0:
        fr.status = "skipped"
        fr.reason = "no_chinese"
        fr.time_ms = now_ms() - t0
        return fr

    if debug_dir:
        ensure_dir(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, f"{name}.mask.png"), full_mask)
        save_mask_overlay_debug(bgr, full_mask, os.path.join(debug_dir, f"{name}.mask_overlay.png"), fill_alpha=0.22, outline_thickness=2)

    ratio = mask_coverage_ratio(full_mask)
    if ratio > float(max_mask_ratio):
        if debug_dir:
            save_mask_overlay_debug(bgr, full_mask, os.path.join(debug_dir, f"{name}.MASK_TOO_BIG_overlay.png"), fill_alpha=0.12)
        raise RuntimeError(f"Mask too large ({ratio:.1%}). Refusing to inpaint entire image.")

    # Chart gating
    gate_pass, gate_info = ocr_gate_is_chart(tokens, schema)
    fr.chart_gate_pass = bool(gate_pass)

    if gate_mode == "off":
        gate_pass = True
    elif gate_mode == "strict":
        gate_pass = bool(gate_info.get("table_like")) or (gate_info.get("strong_kw_hits", 0) >= 2 and gate_info.get("total_texts", 0) >= 10)

    if not gate_pass:
        fr.status = "skipped"
        fr.reason = f"gate_fail:{gate_info}"
        fr.time_ms = now_ms() - t0
        return fr

    # Regions
    regions = build_regions_from_chinese_tokens(zh_tokens, (h,w), gap_px=8, iou_thr=0.03)
    build_group_keys(regions)
    compute_region_mask_geometry(regions, (h,w))
    fr.regions = len(regions)

    if not regions:
        fr.status = "skipped"
        fr.reason = "no_regions"
        fr.time_ms = now_ms() - t0
        return fr

    if translator is None:
        fr.status = "error"
        fr.reason = "translator_not_configured"
        fr.time_ms = now_ms() - t0
        return fr

    extra_for_translate = (max_extra_lines if allow_extra_line else 0)
    tr_map = translator.translate_regions(
        regions,
        min_font_px=min_font_px,
        max_font_px=max_font_px,
        max_extra_lines=extra_for_translate,
        max_retries=3
    )

    for r in regions:
        target = int(max(1, r.line_count))
        max_lines = int(target + extra_for_translate)
        lines = tr_map.get(r.id, [])
        lines = normalize_lines(lines, target_lines=target, max_lines=max_lines)
        r.en_lines = lines
        r.en_text = " ".join([l.strip() for l in lines if l and l.strip()]).strip()

    # Erase Chinese (two-pass)
    bgr_erased, mask_filled, residual_mask = two_pass_erase(
        bgr=bgr,
        base_mask=full_mask,
        regions=regions,
        primary_inpainter=inpainter,
        refine_radius=refine_radius,
        refine_method=refine_method,
        residual_deltaE_thr=residual_deltaE_thr,
        residual_grad_thr=residual_grad_thr,
        residual_max_cc_area=residual_max_cc_area,
        max_refine_iters=refine_iters,
    )

    # ---------------------------------------------------------
    # EXTRA cleanup: residual strokes inside each region's GREEN bbox
    # (Fix: Chinese strokes visible inside green bbox but outside red bbox)
    # ---------------------------------------------------------
    extra_residual = detect_residual_in_green_boxes(
        bgr=bgr_erased,
        regions=regions,
        residual_max_cc_area=residual_max_cc_area,
    )

    # Safety: never inpaint if mask is empty or too large
    extra_ratio = mask_coverage_ratio(extra_residual)
    if extra_ratio > 0.35:  # conservative safety guard
        if debug_dir:
            save_mask_overlay_debug(
                bgr_erased, extra_residual,
                os.path.join(debug_dir, f"{name}.EXTRA_RESID_TOO_BIG_overlay.png"),
                fill_alpha=0.12, outline_thickness=2
            )
        raise RuntimeError(f"Extra residual mask too large ({extra_ratio:.1%}). Refusing extra inpaint.")

    if int(extra_residual.sum()) > 0:
        # IMPORTANT: cv2.inpaint requires mask == (H,W) exactly.
        if refine_method.lower() == "telea":
            bgr_erased = cv2.inpaint(bgr_erased, extra_residual, int(refine_radius), cv2.INPAINT_TELEA)
        else:
            bgr_erased = cv2.inpaint(bgr_erased, extra_residual, int(refine_radius), cv2.INPAINT_NS)

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"{name}.extra_residual_mask.png"), extra_residual)
            save_mask_overlay_debug(
                bgr_erased, extra_residual,
                os.path.join(debug_dir, f"{name}.extra_residual_overlay.png"),
                fill_alpha=0.18, outline_thickness=2
            )
            
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{name}.mask_filled_holes.png"), mask_filled)
        cv2.imwrite(os.path.join(debug_dir, f"{name}.residual_mask.png"), residual_mask)
        save_mask_overlay_debug(bgr_erased, residual_mask, os.path.join(debug_dir, f"{name}.residual_overlay.png"),
                                fill_alpha=0.18, outline_thickness=2)

    fm = FontManager(discover_default_fonts())

    # Render per-region (no group constraint)
    out = bgr_erased.copy()
    debug_items: List[Dict[str,Any]] = []
    for r in regions:
        patch_rgba, dbg = render_text_patch(
            bgr_erased=bgr_erased,
            region=r,
            fm=fm,
            min_font_px=min_font_px,
            max_font_px=max_font_px,
            allow_extra_line=allow_extra_line,
            max_extra_lines=max_extra_lines,
            min_good_font_px=min_good_font_px,
        )
        out = overlay_patch(out, r, patch_rgba, use_mask_soft=True)
        debug_items.append({
            "id": r.id,
            "group": r.group_key,
            "bbox": list(r.bbox),
            "mask_bbox_local": list(r.mask_bbox_local),
            "line_count_zh": r.line_count,
            "zh_text": r.zh_text,
            "en_lines": r.en_lines,
            "en_text_joined": r.en_text,
            "render_dbg": dbg
        })

    ensure_dir(outdir)
    out_path = os.path.join(outdir, f"{name}.en{ext if ext else '.png'}")
    cv2.imwrite(out_path, out)
    fr.out_path = out_path

    if debug_dir:
        ensure_dir(debug_dir)
        with open(os.path.join(debug_dir, f"{name}.regions.json"), "w", encoding="utf-8") as f:
            json.dump(debug_items, f, ensure_ascii=False, indent=2)

        save_mask_overlay_debug(out, full_mask, os.path.join(debug_dir, f"{name}.final_mask_overlay.png"), fill_alpha=0.10, outline_thickness=2)

        save_render_debug_overlay(
            final_bgr=out,
            full_mask=full_mask,
            regions=regions,
            debug_items=debug_items,
            out_path=os.path.join(debug_dir, f"{name}.render_debug_overlay.png")
        )

    fr.status = "processed"
    fr.reason = "ok"
    fr.time_ms = now_ms() - t0
    return fr

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", nargs="+", required=True, help="Input files/dirs/globs, e.g. images/**/*.jpg")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--summary", required=True, help="Summary JSON path")
    ap.add_argument("--schema", required=True, help="Path to chart_translate_schema.json")

    ap.add_argument("--ocr-backend", default="paddle", choices=["paddle","rapid","auto"], help="OCR backend")
    ap.add_argument("--paddle-lang", default="ch", help="Paddle OCR lang (e.g. ch, en, chinese_cht)")
    ap.add_argument("--paddle-device", default=None, help="Paddle device hint (cpu/gpu). Optional.")
    ap.add_argument("--suppress-ocr-init-logs", action="store_true", help="Suppress PaddleOCR init logs")
    ap.add_argument("--paddle-enable-hpi", action="store_true", 
                    help="Enable PaddleOCR High-Performance Inference (3.x)")
    ap.add_argument("--paddlex-config", default=None, 
                    help="Path to PaddleOCR.yaml (paddlex_config). If omitted and --paddle-rec-batch set, script exports default YAML once.")
    ap.add_argument("--paddle-rec-batch", type=int, default=None,
                    help="Override TextRecognition batch_size in paddlex YAML (written to temp file)")
    ap.add_argument("--ocr-augment", type=int, default=2, help="OCR variants: 1=none,2=+CLAHE,3=+sharpen")
    ap.add_argument("--gate-mode", default="auto", choices=["auto","off","strict"], help="Chart gating mode")

    ap.add_argument("--translate-backend", default="gemini", choices=["gemini"], help="Translation backend")
    ap.add_argument("--gcp-project", default=os.environ.get("GOOGLE_CLOUD_PROJECT",""), help="GCP project ID")
    ap.add_argument("--gcp-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION","us-central1"), help="GCP location")
    ap.add_argument("--gemini-model", default="gemini-2.0-flash-001", help="Gemini model")

    ap.add_argument("--inpaint-backend", default="lama", choices=["lama","opencv","none"], help="Inpainting backend (pass1)")
    ap.add_argument("--lama-device", default="auto", choices=["auto","cpu","cuda","mps"], help="LaMa torch device")

    ap.add_argument("--min-font-px", type=int, default=10, help="Min font px")
    ap.add_argument("--max-font-px", type=int, default=64, help="Max font px")

    ap.add_argument("--allow-extra-line", action="store_true", help="Allow N+1 lines to avoid too small/cut text")
    ap.add_argument("--max-extra-lines", type=int, default=1, help="Max extra lines to add (usually 1)")
    ap.add_argument("--min-good-font-px", type=int, default=14, help="Try keep font >= this (encourages extra line)")

    ap.add_argument("--erase-refine-iters", type=int, default=2, help="How many residual-erase refine iterations")
    ap.add_argument("--erase-refine-radius", type=int, default=7, help="OpenCV inpaint radius for residual erase")
    ap.add_argument("--erase-refine-method", default="ns", choices=["ns","telea"], help="Residual erase method")
    ap.add_argument("--erase-residual-deltaE", type=float, default=12.0, help="Residual tone diff threshold")
    ap.add_argument("--erase-residual-grad", type=float, default=40.0, help="Residual gradient threshold")
    ap.add_argument("--erase-residual-max-area", type=int, default=2000, help="Max CC area treated as residual")

    ap.add_argument("--max-mask-ratio", type=float, default=0.75, help="Refuse inpaint if mask covers > this fraction of image")
    ap.add_argument("--max-single-poly-area-ratio", type=float, default=0.70, help="Skip any single OCR bbox > this fraction of image")

    ap.add_argument("--debug-dir", default=None, help="Debug output dir (mask, overlays, regions json)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output already exists")

    args = ap.parse_args()

    schema = load_schema(args.schema)
    files = expand_inputs(args.input)
    if not files:
        print("No input images found.", file=sys.stderr)
        sys.exit(2)

    ocr = build_ocr(
        backend=args.ocr_backend,
        paddle_lang=args.paddle_lang,
        paddle_device=args.paddle_device,
        suppress_init_logs=bool(args.suppress_ocr_init_logs)
    )    

    translator = None
    if args.translate_backend == "gemini":
        if not args.gcp_project:
            raise RuntimeError("Missing --gcp-project (or set GOOGLE_CLOUD_PROJECT).")
        translator = GeminiTranslator(
            project=args.gcp_project,
            location=args.gcp_location,
            model=args.gemini_model,
            schema=schema,
            temperature=0.0
        )

    inpainter, lama_device_used = build_inpainter(args.inpaint_backend, args.lama_device)
    if args.inpaint_backend == "lama":
        print(f"[INFO] LaMa device: {lama_device_used}")

    ensure_dir(args.outdir)
    if args.debug_dir:
        ensure_dir(args.debug_dir)

    results: List[FileResult] = []
    t_all = now_ms()

    total = len(files)
    for i, p in enumerate(files, start=1):
        base = os.path.basename(p)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(args.outdir, f"{name}.en{ext if ext else '.png'}")

        print(f"[{i}/{total}] Processing: {p}")
        t0 = now_ms()

        if args.skip_existing and os.path.exists(out_path):
            fr = FileResult(
                path=p, status="skipped", reason="skip_existing",
                out_path=out_path, time_ms=now_ms()-t0,
                ocr_backend=ocr.name(),
                inpaint_backend=inpainter.name() if inpainter else "none",
                lama_device=lama_device_used
            )
            results.append(fr)
            print(f"  -> SKIP existing: {out_path}   time={fr.time_ms}ms")
            continue

        try:
            fr = process_one(
                path=p,
                ocr=ocr,
                schema=schema,
                translator=translator,
                inpainter=inpainter,
                outdir=args.outdir,
                min_font_px=args.min_font_px,
                max_font_px=args.max_font_px,
                gate_mode=args.gate_mode,
                ocr_augment=args.ocr_augment,
                debug_dir=args.debug_dir,
                lama_device_used=lama_device_used,
                allow_extra_line=bool(args.allow_extra_line),
                max_extra_lines=int(args.max_extra_lines),
                min_good_font_px=int(args.min_good_font_px),
                refine_radius=int(args.erase_refine_radius),
                refine_method=str(args.erase_refine_method),
                residual_deltaE_thr=float(args.erase_residual_deltaE),
                residual_grad_thr=float(args.erase_residual_grad),
                residual_max_cc_area=int(args.erase_residual_max_area),
                refine_iters=int(args.erase_refine_iters),
                max_mask_ratio=float(args.max_mask_ratio),
                max_single_poly_area_ratio=float(args.max_single_poly_area_ratio),
            )
            results.append(fr)
            print(f"  -> {fr.status.upper()}: {fr.reason}   time={fr.time_ms}ms")
            if fr.out_path:
                print(f"     out={fr.out_path}")
        except Exception as e:
            fr = FileResult(
                path=p, status="error", reason=str(e),
                time_ms=now_ms()-t0,
                ocr_backend=ocr.name(),
                inpaint_backend=inpainter.name() if inpainter else "none",
                lama_device=lama_device_used
            )
            results.append(fr)
            print(f"  -> ERROR: {e}\n     time={fr.time_ms}ms")

    summary = {
        "total": len(results),
        "processed": sum(1 for r in results if r.status == "processed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "errors": sum(1 for r in results if r.status == "error"),
        "ocr_backend": ocr.name(),
        "inpaint_backend": inpainter.name() if inpainter else "none",
        "lama_device": lama_device_used,
        "elapsed_ms": now_ms() - t_all,
        "files": [asdict(r) for r in results]
    }

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "total": summary["total"],
        "processed": summary["processed"],
        "skipped": summary["skipped"],
        "errors": summary["errors"],
        "lama_device": lama_device_used
    }, indent=2))
    print(f"Outdir: {os.path.abspath(args.outdir)}")
    print(f"Summary: {os.path.abspath(args.summary)}")

if __name__ == "__main__":
    main()
