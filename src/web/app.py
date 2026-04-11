"""
Gradio-based control center for reverse-SynthID.

Provides a web UI for:
  1. Watermark detection (upload image → confidence gauge)
  2. Watermark bypass (upload image → side-by-side before/after)
  3. Codebook inspector (view loaded profiles)

Launch:
    python -m src.web.app                         # defaults
    python -m src.web.app --codebook path/to.npz  # custom codebook
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.extraction.synthid_bypass import SynthIDBypass, SpectralCodebook
from src.utils.metrics import compute_psnr, compute_ssim

# ---------------------------------------------------------------------------
# Globals (loaded once at startup)
# ---------------------------------------------------------------------------
_codebook: SpectralCodebook | None = None
_bypass: SynthIDBypass | None = None


def _ensure_loaded(codebook_path: str | None = None):
    global _codebook, _bypass
    if _codebook is not None:
        return
    _codebook = SpectralCodebook()
    default = os.path.join("artifacts", "spectral_codebook_v3.npz")
    path = codebook_path or default
    if os.path.exists(path):
        _codebook.load(path)
    else:
        print(f"[web] codebook not found at {path} — bypass will be unavailable")
        _codebook = None
    _bypass = SynthIDBypass()


# ---------------------------------------------------------------------------
# Tab 1 — Detect
# ---------------------------------------------------------------------------

def _detect(image: np.ndarray | None) -> str:
    if image is None:
        return "Upload an image first."
    _ensure_loaded()
    if _codebook is None or not _codebook.profiles:
        return "No codebook loaded — detection unavailable."

    img = image if image.dtype == np.uint8 else (image * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w = img.shape[:2]
    profile, (ph, pw), exact = _codebook.get_profile(h, w)

    work = img.astype(np.float64)
    coherences = []
    for ch in range(3):
        fft_ch = np.fft.fft2(work[:, :, ch])
        phase = np.angle(fft_ch)
        ref_phase = profile["phase_template"][:, :, ch]
        if exact:
            diff = np.abs(np.cos(phase - ref_phase))
        else:
            ref_resized = cv2.resize(ref_phase, (w, h), interpolation=cv2.INTER_NEAREST)
            diff = np.abs(np.cos(phase - ref_resized))
        top_k = int(0.001 * diff.size)
        coherences.append(float(np.sort(diff.ravel())[-top_k:].mean()))

    avg_coh = float(np.mean(coherences))
    g_coh = coherences[1]

    verdict = "LIKELY WATERMARKED" if g_coh > 0.55 else "UNLIKELY WATERMARKED"

    lines = [
        f"## {verdict}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Image resolution | {h} x {w} |",
        f"| Codebook profile | {ph} x {pw} ({'exact' if exact else 'fallback'}) |",
        f"| Green channel coherence | {g_coh:.4f} |",
        f"| Red channel coherence | {coherences[0]:.4f} |",
        f"| Blue channel coherence | {coherences[2]:.4f} |",
        f"| Average coherence | {avg_coh:.4f} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2 — Bypass
# ---------------------------------------------------------------------------

def _bypass_image(
    image: np.ndarray | None,
    version: str,
    strength: str,
) -> tuple[np.ndarray | None, str]:
    if image is None:
        return None, "Upload an image first."
    _ensure_loaded()

    img = image if image.dtype == np.uint8 else (image * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if _bypass is None:
        return None, "Bypass engine not initialized."

    if version == "V1 (JPEG)":
        result = _bypass.bypass_simple(img, jpeg_quality=50, verify=False)
    elif version == "V2 (Multi-transform)":
        result = _bypass.bypass_v2(img, strength=strength.lower(), verify=False)
    else:
        if _codebook is None or not _codebook.profiles:
            return None, "V3 requires a loaded codebook."
        result = _bypass.bypass_v3(img, _codebook, strength=strength.lower(), verify=False)

    psnr = compute_psnr(img, result.cleaned_image)
    ssim = compute_ssim(img, result.cleaned_image)

    lines = [
        f"## Bypass Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Version | {version} |",
        f"| Strength | {strength} |",
        f"| PSNR | {psnr:.2f} dB |",
        f"| SSIM | {ssim:.4f} |",
        f"| Stages | {', '.join(result.stages_applied)} |",
    ]
    return result.cleaned_image, "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3 — Codebook Inspector
# ---------------------------------------------------------------------------

def _inspect_codebook() -> str:
    _ensure_loaded()
    if _codebook is None or not _codebook.profiles:
        return "No codebook loaded."

    lines = ["## Loaded Codebook Profiles", ""]
    for (h, w), prof in _codebook.profiles.items():
        nb = prof["n_black_refs"]
        nw = prof["n_white_refs"]
        nr = prof["n_random_refs"]
        cg = prof["phase_consistency"][:, :, 1].copy()
        cg[0, 0] = 0
        top5 = np.sort(cg.ravel())[-5:][::-1]
        lines.append(f"### {h} x {w}")
        lines.append(f"- References: {nb} black, {nw} white, {nr} random")
        lines.append(f"- Top-5 G-channel coherence: {', '.join(f'{v:.4f}' for v in top5)}")
        has_agree = prof.get("black_white_agreement") is not None
        lines.append(f"- Cross-validation data: {'yes' if has_agree else 'no'}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app(codebook_path: str | None = None) -> gr.Blocks:
    _ensure_loaded(codebook_path)

    with gr.Blocks(
        title="Reverse-SynthID Control Center",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Reverse-SynthID Control Center\n"
            "Detect, bypass, and inspect Google SynthID watermarks."
        )

        with gr.Tab("Detect"):
            with gr.Row():
                detect_img = gr.Image(label="Upload Image", type="numpy")
                detect_out = gr.Markdown(label="Detection Result")
            detect_btn = gr.Button("Detect Watermark", variant="primary")
            detect_btn.click(_detect, inputs=detect_img, outputs=detect_out)

        with gr.Tab("Bypass"):
            with gr.Row():
                with gr.Column(scale=1):
                    bypass_img = gr.Image(label="Input Image", type="numpy")
                    bypass_ver = gr.Radio(
                        ["V1 (JPEG)", "V2 (Multi-transform)", "V3 (Spectral)"],
                        value="V3 (Spectral)",
                        label="Bypass Version",
                    )
                    bypass_str = gr.Radio(
                        ["Gentle", "Moderate", "Aggressive", "Maximum"],
                        value="Aggressive",
                        label="Strength",
                    )
                    bypass_btn = gr.Button("Run Bypass", variant="primary")
                with gr.Column(scale=1):
                    bypass_out_img = gr.Image(label="Cleaned Image", type="numpy")
                    bypass_out_md = gr.Markdown(label="Metrics")
            bypass_btn.click(
                _bypass_image,
                inputs=[bypass_img, bypass_ver, bypass_str],
                outputs=[bypass_out_img, bypass_out_md],
            )

        with gr.Tab("Codebook"):
            cb_out = gr.Markdown()
            cb_btn = gr.Button("Inspect Codebook", variant="secondary")
            cb_btn.click(_inspect_codebook, outputs=cb_out)

    return app


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reverse-SynthID Web UI")
    parser.add_argument("--codebook", default=None, help="Path to .npz codebook")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    app = build_app(args.codebook)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
