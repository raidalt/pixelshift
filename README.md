# Pixelshift

A local macOS app for combining Sony α7R V pixel-shift RAW captures into a 16-bit TIFF, with a zoomable before-and-after comparison.

Pixelshift reconstructs color from the original sensor samples. It is an independent implementation, not Sony software or a certified replacement for Imaging Edge.

## Features

- Find complete pixel-shift sequences anywhere within a selected SD card or drive.
- Verify capture groups and frame order using Sony metadata rather than filenames.
- Align frames using detailed regions of the image and reconstruct directly from RAW color samples.
- Inspect the result with a draggable comparison slider, automatic detail selection, and zoom controls.
- Save to `~/Pictures/Pixel Shift` with unique filenames, preserving the original ARWs.

| Capture mode | Output dimensions | Pixel count |
| --- | --- | --- |
| 4 frames | 9,504 × 6,336 | 60.2 MP |
| 16 frames | 19,008 × 12,672 | 240.8 MP |

Pixel count describes the output dimensions; resolved detail also depends on focus, optics, exposure, and scene stability.

## Requirements

- macOS and Python 3.11 or newer
- [ExifTool](https://exiftool.org/) available on your `PATH`
- A complete 4- or 16-shot sequence of full-size Sony α7R V ARW files
- Several GB of available RAM and approximately 10 GB of free disk space for a 16-frame job

If you use Homebrew, you can install the prerequisites with:

```sh
brew install python exiftool
```

## Get started

1. Download or clone this repository.
2. Open `Start.command`. The first launch creates a local Python environment, installs dependencies, and opens the app in your browser.
3. Select **4 frames** or **16 frames**, then choose your SD card or photo directory. Nested folders are scanned automatically.
4. Select a verified capture set and choose an output directory.
5. Click **Build composite**. Keep the launcher open and the source drive connected until processing finishes.

To launch from Terminal instead:

```sh
zsh Start.command
```

The app exports an uncompressed, 16-bit RGB BigTIFF using sRGB encoding, with an embedded sRGB profile when available. A JSON report beside the TIFF records alignment measurements, warnings, and interpolation statistics.

## Compare the result

After a merge, drag across **Original / Composite** or use the slider. Choose **Best detail**, the center, or another region, and adjust the zoom from 1× to 8×.

Both sides use the same scene coordinates, white balance, exposure, and color transform. No sharpening is added. For 16-frame composites, the original is enlarged 2× with nearest-neighbor sampling to match the composite's field of view. Previews are 8-bit; the saved TIFF remains 16-bit.

## How reconstruction works

Pixelshift reads the original CFA samples through RawPy/LibRaw, applies the recorded crop and black levels, and estimates global translations from texture-selected regions. It combines the measured color samples on the output grid using normalized convolution. On ideal integer- or half-pixel sampling positions, this reduces to direct color interleaving with two-green averaging.

Sampling gaps use a wider interpolation kernel; the app and report disclose their frequency. Camera white balance, a camera-to-sRGB matrix, and the sRGB transfer curve are applied after reconstruction.

## Limitations

- Supports full-size α7R V RAWs and complete, matching capture groups.
- Corrects global translation, not rotation, local subject motion, or optical distortion. Moving subjects can produce artifacts.
- Conflicting alignment estimates produce a warning and continue using the median translation. Unusable alignment still stops processing. A warning does not prove the camera moved.
- Does not export ARQ or apply lens corrections, highlight reconstruction, denoising, or sharpening.
- Has been exercised with real 4- and 16-frame captures and synthetic tests. Equivalence to Sony's output and quantitative resolution gains remain unverified.

## Privacy

Photo processing happens on your Mac. The server listens only on `127.0.0.1`, and processing requests require a per-launch token. There are no photo uploads, accounts, or telemetry.

This repository contains no sample photos, capture reports, credentials, or personal settings. Generated photos, reports, caches, and local environments are excluded by `.gitignore`. Reports generated during use can contain local filenames and capture metadata; review them before sharing.

## Development

After the launcher has installed the dependencies:

```sh
.venv/bin/python test_engine.py
```

Tests cover exact synthetic 4- and 16-frame reconstruction, color calibration, sequence validation, nested-folder discovery, alignment, output naming, and local request authentication. No camera files are required.

## References

- [Sony α7R V pixel-shift shooting and processing guide](https://support.d-imaging.sony.co.jp/support/ilc/psms/ilce7rm5/en/index.html)
- [ExifTool Sony metadata definitions](https://exiftool.org/TagNames/Sony.html)
- [Dietz, “Leveraging Pixel Value Certainty in Pixel-Shift and Other Multi-Shot Super-Resolution Processing”](https://doi.org/10.2352/EI.2024.36.15.COIMG-142)
