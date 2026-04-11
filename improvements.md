# Reverse-SynthID — Improvements Roadmap

## Current State Summary

**~14,600 lines of Python** across 12 source files. Pure signal-processing pipeline (no DL) for detecting, analyzing, and removing Google SynthID watermarks from AI-generated images. All interactions are CLI-only — there is **no web interface or control center**.

### What Exists Today

| Area | Files | Status |
|------|-------|--------|
| V1/V2/V3 Bypass + SpectralCodebook | `src/extraction/synthid_bypass.py` (2304 LOC) | Production-ready |
| Multi-scale Detector (90% acc) | `src/extraction/robust_extractor.py` (806 LOC) | Production-ready |
| Signature-based Remover | `src/extraction/watermark_remover.py` (631 LOC) | Working |
| Benchmarking Suite | `src/extraction/benchmark_extraction.py` (465 LOC) | Partial (API mismatch) |
| Legacy Codebook Extractor | `src/extraction/synthid_codebook_extractor.py` (345 LOC) | Legacy |
| Deep FFT/Phase Analysis | `src/analysis/deep_synthid_analysis.py` (645 LOC) | Working |
| Carrier Frequency Finder | `src/analysis/synthid_codebook_finder.py` (706 LOC) | Working |
| Reference Image Generator | `generate_references.py` (184 LOC) | Working |
| HF Dataset Downloader | `scripts/download_images.py` (51 LOC) | Working |
| Early Investigation (archived) | `watermark_investigation/` (7 scripts) | Archived |
| Pre-built Codebooks | `artifacts/` (.npz, .pkl) | Available |

---

## 1. Web Control Center (High Priority)

The project has **zero** web UI. A dashboard would dramatically improve usability.

### Proposed: Flask/FastAPI Dashboard

- **Image Upload & Process** — drag-and-drop an image, select bypass version (V1/V2/V3) and strength, see before/after side-by-side
- **Watermark Detection** — upload image, get instant detection result with confidence gauge
- **Batch Processing** — upload folder or ZIP, process all images, download results
- **Codebook Manager** — view loaded codebook profiles, their resolutions, carrier counts; upload new reference images to extend the codebook
- **Live Metrics Dashboard** — PSNR, SSIM, phase coherence drop, carrier energy drop shown as charts
- **Comparison View** — side-by-side V1 vs V2 vs V3 output with quality metrics
- **API Endpoints** — REST API for programmatic access (`/detect`, `/bypass`, `/codebook/build`)

### Tech Stack Suggestion

- Backend: **FastAPI** (async, OpenAPI docs out of the box)
- Frontend: **React** or **Svelte** with Tailwind CSS (or a simpler approach: **Gradio** for rapid prototyping)
- Containerization: Docker + docker-compose

---

## 2. Code Architecture & Quality

### 2.1 Duplicated Code

The following logic is copy-pasted across multiple files:

| Duplicated Function | Files |
|---|---|
| `wavelet_denoise()` | `synthid_bypass.py`, `robust_extractor.py`, `deep_synthid_analysis.py`, `synthid_codebook_finder.py`, `synthid_codebook_extractor.py` |
| SSIM computation | `synthid_bypass.py` (appears 3 times inline), `watermark_remover.py` |
| PSNR computation | `synthid_bypass.py`, `watermark_remover.py` |
| Image loading + extension filtering | At least 5 files |
| JPEG compression helper | `synthid_bypass.py`, `watermark_remover.py` |

**Fix:** Extract shared utilities into `src/utils/` (e.g., `image_io.py`, `metrics.py`, `denoise.py`).

### 2.2 No Package Structure

There is no `pyproject.toml`, `setup.py`, or `__init__.py` files. The code uses `sys.path.insert(0, ...)` hacks for imports.

**Fix:**
```
reverse-SynthID/
├── pyproject.toml
├── src/
│   └── reverse_synthid/
│       ├── __init__.py
│       ├── bypass/
│       ├── detection/
│       ├── analysis/
│       ├── utils/
│       └── web/          # new control center
```

### 2.3 No Type Checking

No `mypy` config, no `py.typed` marker. Many functions accept `np.ndarray` but don't specify shapes or dtypes.

**Fix:** Add `mypy.ini` or `[tool.mypy]` in pyproject.toml, add type stubs, use `NDArray[np.float64]` annotations.

### 2.4 Inconsistent Float/Uint8 Conventions

Some functions expect `[0, 1]` float images, others expect `[0, 255]` uint8. The conversions are scattered and error-prone.

**Fix:** Standardize on one internal representation (e.g., always float64 [0, 255] internally, convert at boundaries).

---

## 3. Testing

### Current: Zero Tests

There are no unit tests, integration tests, or property-based tests anywhere in the project.

### Proposed

- **Unit tests** for every utility function (PSNR, SSIM, wavelet_denoise, JPEG compression)
- **Integration tests** for the full V3 pipeline (synthetic test image → bypass → verify PSNR threshold)
- **Regression tests** using the validation images in `validation_images/`
- **Codebook round-trip tests** (save → load → verify profiles are identical)
- **Property-based tests** (hypothesis) for image processing invariants

Use `pytest` + `pytest-cov` with a target of 80%+ coverage on core modules.

---

## 4. Performance & Scalability

### 4.1 No GPU Acceleration

All FFT operations use `scipy.fft` (CPU). For large images (1536x2816), the V3 bypass is slow.

**Fix:** Add optional CuPy or torch.fft backend with automatic fallback to CPU.

### 4.2 No Parallel Processing

Batch operations are sequential. Codebook building processes images one at a time.

**Fix:** Use `concurrent.futures.ProcessPoolExecutor` for image processing, `threading` for I/O.

### 4.3 Memory Management

`SpectralCodebook.extract_from_references()` loads all images sequentially but accumulates large FFT arrays without explicit cleanup.

**Fix:** Add `gc.collect()` at strategic points, use memory-mapped arrays for large codebooks, add memory profiling hooks.

### 4.4 Large Codebook Files

The V3 codebook (`.npz`) can be large. V2 format already uses rfft + float16 + sparse storage — good. But the legacy `.pkl` codebooks don't compress at all.

**Fix:** Migrate all codebook formats to the V2 compact format, add LZ4 compression option.

---

## 5. DevOps & Infrastructure

### 5.1 No Docker Support

**Fix:** Add `Dockerfile` + `docker-compose.yml` for reproducible environments.

### 5.2 No CI/CD

**Fix:** Add GitHub Actions for:
- Linting (`ruff`)
- Type checking (`mypy`)
- Tests (`pytest`)
- Codebook integrity check (verify stored codebooks load correctly)

### 5.3 No Logging Framework

All output uses `print()`. No structured logging, no log levels.

**Fix:** Replace with `logging` module. Use `structlog` for JSON-formatted output when running as a service.

### 5.4 No Configuration System

Parameters are hardcoded (carrier frequencies, strength levels, channel weights). No config file.

**Fix:** Add YAML/TOML config file support. Example:
```yaml
bypass:
  default_version: v3
  default_strength: aggressive
  channel_weights: [0.85, 1.0, 0.70]

detection:
  scales: [256, 512, 1024]
  n_carriers: 100

codebook:
  cons_threshold: 0.15
  compression: lz4
```

---

## 6. Documentation & Developer Experience

### 6.1 API Documentation

No docstring standards enforced. No auto-generated API docs.

**Fix:** Add `sphinx` or `mkdocs` with `mkdocstrings` for auto-generated API documentation.

### 6.2 Architecture Decision Records

Major decisions (why V3 over V2, why phase coherence > magnitude, why channel weights are [0.85, 1.0, 0.70]) are only in code comments.

**Fix:** Add `docs/adr/` with formal Architecture Decision Records.

### 6.3 Contributing Guide

The README mentions contributions (black/white images) but there's no `CONTRIBUTING.md`.

---

## 7. Research & Algorithm Improvements

### 7.1 Cross-Resolution Generalization

V3 currently requires per-resolution codebook profiles. When no exact match exists, it falls back to spatial-domain resize, which is lossy.

**Improvement:** Investigate resolution-invariant carrier detection (normalized frequency space).

### 7.2 Adaptive Strength Selection

Users must manually choose strength (`gentle`/`moderate`/`aggressive`/`maximum`). 

**Improvement:** Auto-select strength based on detected watermark energy level — if carrier energy is already low, use gentle; if strong, use aggressive.

### 7.3 Multi-Model Support

Currently tuned only for Google Gemini's SynthID. Other AI models (DALL-E, Midjourney, Stable Diffusion) may use different watermarking.

**Improvement:** Modular watermark profile system — one codebook per model family.

### 7.4 Perceptual Quality Metrics

Only PSNR and SSIM are used. Neither correlates well with perceived quality.

**Improvement:** Add LPIPS (requires torch), FID for batch evaluation, DISTS metric.

### 7.5 Content-Aware Bypass

V3 applies uniform subtraction regardless of image content. Flat regions are more sensitive to artifacts than textured regions.

**Improvement:** Weight subtraction by local texture complexity (Laplacian variance mask).

---

## 8. Security & Ethical Considerations

### 8.1 Access Control

If a web UI is built, there should be rate limiting, authentication, and audit logging.

### 8.2 Watermark Verification Logging

Track all bypass operations with cryptographic hashes of input/output for accountability.

### 8.3 Responsible Use Policy

The existing disclaimer is good. Add a machine-readable `SECURITY.md` and consider requiring user acknowledgment before processing.

---

## Priority Matrix

| # | Improvement | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | Web Control Center (Gradio MVP) | Very High | Medium | P0 |
| 2 | Extract shared utilities (dedup) | High | Low | P0 |
| 3 | Add proper package structure | High | Low | P0 |
| 4 | Unit + integration tests | High | Medium | P1 |
| 5 | Docker + CI/CD | Medium | Low | P1 |
| 6 | Logging framework | Medium | Low | P1 |
| 7 | Configuration system | Medium | Low | P1 |
| 8 | Auto-strength selection | High | Medium | P2 |
| 9 | GPU acceleration (optional) | Medium | Medium | P2 |
| 10 | Parallel batch processing | Medium | Low | P2 |
| 11 | Content-aware bypass weighting | High | High | P2 |
| 12 | Perceptual quality metrics | Medium | Medium | P3 |
| 13 | Multi-model support | High | High | P3 |
| 14 | API documentation (mkdocs) | Medium | Low | P3 |
| 15 | Resolution-invariant carriers | Very High | Very High | P3 |

---

## Quick Wins (< 1 day each)

1. Create `src/utils/metrics.py` with shared PSNR/SSIM/wavelet_denoise
2. Add `__init__.py` files and remove `sys.path` hacks
3. Add `Dockerfile` for reproducible setup
4. Add `pytest` skeleton with 5 basic tests
5. Replace `print()` with `logging` in all modules
6. Fix `benchmark_extraction.py` API mismatch (references non-existent `RemovalResult` fields)
7. Add a Gradio single-page demo (detect + bypass in ~100 lines)
