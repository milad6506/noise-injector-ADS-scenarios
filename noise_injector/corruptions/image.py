"""Layer A: image corruption families. Each fn(img, params, rng) -> img (uint8 RGB)."""

from __future__ import annotations

import cv2
import numpy as np


def _clip(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def awgn(img, params, rng):
    sigma = params["sigma"]
    noise = rng.normal(0, sigma, img.shape)
    return _clip(img.astype(np.float32) + noise)


def salt_pepper(img, params, rng):
    p = params["p"]
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < p / 2] = 0
    out[(r >= p / 2) & (r < p)] = 255
    return out


def dead_pixels(img, params, rng):
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(params["n_clusters"]):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        dx = int(rng.integers(0, 4))
        dy = int(rng.integers(0, 4))
        val = 0 if rng.random() < 0.5 else 255
        y0, y1 = max(0, cy - dy), min(h, cy + dy + 1)
        x0, x1 = max(0, cx - dx), min(w, cx + dx + 1)
        out[y0:y1, x0:x1] = val
    return out


def low_light(img, params, rng):
    gamma = params["gamma"]
    sigma = params["sigma"]
    out = img.astype(np.float32) / 255.0
    out = np.power(out, gamma) * 255.0
    noise = rng.normal(0, sigma, img.shape)
    return _clip(out + noise)


def overexposure(img, params, rng):
    strength = params["strength"]
    out = img.astype(np.float32)
    lum = out.mean(axis=2, keepdims=True)
    thr = 170.0
    blow = np.clip((lum - thr) / (255.0 - thr), 0.0, 1.0)
    out = out + strength * blow * (255.0 - out)
    return _clip(out)


def motion_blur(img, params, rng):
    k = params["kernel"]
    if k % 2 == 0:
        k += 1
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    kern /= k
    return _clip(cv2.filter2D(img, -1, kern))


def defocus(img, params, rng):
    return _clip(cv2.GaussianBlur(img, (0, 0), params["sigma"]))


def fog(img, params, rng):
    amount = params["amount"]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    depth = np.linspace(0.75, 1.25, w, dtype=np.float32)[None, :, None]
    veil = np.full((h, w, 3), 205.0, np.float32) * depth
    out = out * (1 - amount) + veil * amount
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * (1 - 0.4 * amount) + mean
    return _clip(out)


def rain(img, params, rng):
    n = params["n_streaks"]
    darken = params["darken"]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    layer = np.zeros((h, w), np.float32)
    angle = np.deg2rad(15.0)
    for _ in range(n):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(rng.integers(8, 22))
        dx = int(length * np.sin(angle))
        dy = int(length * np.cos(angle))
        cv2.line(layer, (x0, y0), (x0 + dx, y0 + dy), float(rng.uniform(0.35, 0.8)), 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - layer)
    out *= 1.0 - darken
    return _clip(out)


def snow(img, params, rng):
    density = params["density"]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    out *= 0.92
    layer = np.zeros((h, w), np.float32)
    n = int(density * h * w)
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    vals = rng.uniform(0.6, 1.0, n)
    layer[ys, xs] = vals
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    out = out * (1 - layer[:, :, None]) + 255.0 * layer[:, :, None]
    return _clip(out)


def glare(img, params, rng):
    intensity = params["intensity"]
    radius_frac = params["radius_frac"]
    h, w = img.shape[:2]
    cx = int(rng.integers(int(0.15 * w), int(0.85 * w)))
    cy = int(rng.integers(int(0.05 * h), int(0.4 * h)))
    radius = int(radius_frac * max(w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1.0 - dist / max(1, radius), 0.0, 1.0) ** 2
    glow = glow * intensity
    out = img.astype(np.float32)
    out = out + glow[:, :, None] * (255.0 - out)
    return _clip(out)


def bloom(img, params, rng):
    strength = params["strength"]
    out = img.astype(np.float32)
    lum = out.mean(axis=2)
    mask = np.clip((lum - 200.0) / 55.0, 0.0, 1.0)
    glow = cv2.GaussianBlur(mask, (0, 0), 12)
    glow = glow * strength
    out = out + glow[:, :, None] * (255.0 - out)
    return _clip(out)


def jpeg(img, params, rng):
    quality = max(2, int(params["quality"]))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def banding(img, params, rng):
    amplitude = params["amplitude"]
    h, w = img.shape[:2]
    freq = rng.uniform(6.0, 24.0)
    phase = rng.uniform(0, 2 * np.pi)
    ys = np.arange(h, dtype=np.float32)
    ripple = (amplitude * np.sin(2 * np.pi * ys * freq / h + phase))[:, None]
    out = img.astype(np.float32) + ripple[:, :, None]
    return _clip(out)


def chromatic_aberration(img, params, rng):
    shift = params["shift"]
    out = img.copy()
    out[:, :, 0] = np.roll(img[:, :, 0], shift, axis=1)
    out[:, :, 2] = np.roll(img[:, :, 2], -shift, axis=1)
    return out


def lens_droplets(img, params, rng):
    n = params["n"]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    layer = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(int(0.1 * h), h))
        r = int(rng.integers(2, 9))
        cv2.circle(layer, (cx, cy), r, float(rng.uniform(0.35, 0.65)), -1)
    layer = cv2.GaussianBlur(layer, (5, 5), 0)
    blur = cv2.GaussianBlur(out, (7, 7), 0)
    out = out * (1 - layer[:, :, None]) + blur * layer[:, :, None] + 25.0 * layer[:, :, None]
    return _clip(out)


def lens_dirt(img, params, rng):
    coverage = params["coverage"]
    h, w = img.shape[:2]
    rng2 = np.random.default_rng(int(rng.integers(0, 2**31)))
    blob = rng2.random((max(2, h // 4), max(2, w // 4))).astype(np.float32)
    blob = cv2.resize(blob, (w, h), interpolation=cv2.INTER_CUBIC)
    blob = cv2.GaussianBlur(blob, (31, 31), 0)
    thr = np.quantile(blob, max(0.0, 1.0 - coverage))
    mask = np.clip((blob - thr) / 0.25, 0, 1) * 0.85
    out = img.astype(np.float32)
    smudge = cv2.GaussianBlur(out, (15, 15), 0)
    tint = np.array([70.0, 65.0, 55.0])
    out = out * (1 - mask[:, :, None]) + (smudge * 0.6 + tint) * mask[:, :, None]
    return _clip(out)


def water_splash(img, params, rng):
    alpha = params["alpha"]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    rng2 = np.random.default_rng(int(rng.integers(0, 2**31)))
    blobs = rng2.random((max(2, h // 6), max(2, w // 3))).astype(np.float32)
    blobs = cv2.resize(blobs, (w, h), interpolation=cv2.INTER_CUBIC)
    blobs = cv2.GaussianBlur(blobs, (25, 25), 0)
    blobs = np.clip(blobs - 0.55, 0, 1) / 0.45
    region = np.ones((h, w), np.float32)
    for y in range(h):
        region[y, :] = min(1.0, (y - h * 0.55) / max(1.0, h * 0.45))
    mask = np.clip(blobs * region, 0, 1) * alpha
    blur = cv2.GaussianBlur(out, (9, 9), 0) + 60.0
    out = out * (1 - mask[:, :, None]) + blur * mask[:, :, None]
    return _clip(out)


def rolling_shutter(img, params, rng):
    amplitude = params["amplitude"]
    h, w = img.shape[:2]
    band = max(4, h // 24)
    phase = rng.uniform(0, 2 * np.pi)
    out = img.copy()
    for y0 in range(0, h, band):
        frac = (y0 / h) * np.pi * 2 + phase
        dx = int(amplitude * np.sin(frac))
        out[y0 : y0 + band] = np.roll(img[y0 : y0 + band], dx, axis=1)
    return out


def color_cast(img, params, rng):
    gain = params["gain"]
    tint = int(rng.integers(0, 3))
    out = img.astype(np.float32)
    for c in range(3):
        if c == tint:
            out[:, :, c] *= 1.0 + gain
        else:
            out[:, :, c] *= 1.0 - gain * 0.5
    return _clip(out)


def quantization(img, params, rng):
    levels = int(params["levels"])
    step = 255.0 / (levels - 1)
    out = np.round(img.astype(np.float32) / step) * step
    return _clip(out)


IMAGE_CORRUPTIONS = {
    "awgn": awgn,
    "salt_pepper": salt_pepper,
    "dead_pixels": dead_pixels,
    "low_light": low_light,
    "overexposure": overexposure,
    "motion_blur": motion_blur,
    "defocus": defocus,
    "fog": fog,
    "rain": rain,
    "snow": snow,
    "glare": glare,
    "bloom": bloom,
    "jpeg": jpeg,
    "banding": banding,
    "chromatic_aberration": chromatic_aberration,
    "lens_droplets": lens_droplets,
    "lens_dirt": lens_dirt,
    "water_splash": water_splash,
    "rolling_shutter": rolling_shutter,
    "color_cast": color_cast,
    "quantization": quantization,
}
