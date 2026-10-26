# Pixelshift

A local macOS workspace for Sony α7R V pixel-shift reconstruction and focus stacking, with 16-bit TIFF export and a full-resolution before-and-after inspector.

Pixelshift reconstructs color from the original sensor samples. It is an independent implementation, not Sony software or a certified replacement for Imaging Edge.

## Features

- Find complete pixel-shift sequences anywhere within a selected SD card or drive.
- Verify capture groups and frame order using Sony metadata rather than filenames.
- Align frames using detailed regions of the image and reconstruct directly from RAW color samples.
- Inspect the full image with a comparison slider, pan, true 100%/200% pixel views, blink, amplified differences, and registration overlays.
- Tune alignment thresholds, strict or warning mode, include/exclude regions, reconstruction kernels, and sample support.
- Inspect RAW support, alignment confidence, interpolation maps, and per-frame measurements.
- Adjust exposure, camera or custom white balance, optional sharpening and denoising, with a live histogram and clipping overlays.
- Save local presets and export sRGB or linear RGB TIFFs with embedded ICC profiles, custom names, and border cropping.
- Track processing stages, elapsed time, estimated reconstruction time remaining, and cancel a job.
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
- Several GB of available RAM and approximately 12 GB of free disk space for a 16-frame job

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

The app exports uncompressed 16-bit RGB BigTIFFs with an embedded sRGB or linear-sRGB ICC profile. A JSON report records settings, alignment measurements, warnings, and interpolation statistics. Editable camera data and the first reference image are kept in a hidden `.pixelshift-data` folder beside the output. Keep this folder to revisit adjustments and support maps. Additional original frames require the source drive.

## Focus stacking

Choose **Focus stack** in **Capture workflow**, select the SD card or source folder, and review the newest candidate sequence. Detection follows capture times, sequence counters, matching exposure settings, and image dimensions; it does **not** certify that focus changed or that the sweep is complete. Pixel-shift captures are excluded. Matching RAW/rendered pairs are deduplicated with RAW preferred. Use the frame checkboxes to exclude unwanted photos, then click **Build focus stack**.

Focus mode supports 2–100 matching ARW, SDR sRGB HIF/HEIF, JPEG, or TIFF frames. For Sony HIF decoding install [libheif](https://github.com/strukturag/libheif):

```sh
brew install libheif
```

HIF data is decoded at 16-bit precision from its original 10-bit content. HDR/HLG/PQ and wide-gamut rendered inputs must first be converted to display-encoded sRGB. JPEG inputs retain their original 8-bit source precision despite 16-bit output. Rendered inputs retain their recorded white balance; calibrated RAW white-balance controls are unavailable for focus stacks.

The engine registers scale, rotation, and translation to the first frame to accommodate focus breathing, measures local sharpness with a noise-smoothed modified Laplacian, and blends a three-level Laplacian pyramid. Tune **Focus measurement radius** and **Sharpness selectivity**, or turn alignment off for already aligned sources. The output keeps the source dimensions—approximately 60 MP for full-size α7R V images. It does not multiply resolution.

The **Focus source map** identifies the locally preferred frame; **Focus selection margin** shows the separation between the best two sharpness scores. Neither is a physical depth measurement. Every aligned source reference is cached locally for offline comparisons. Allow roughly 6 GB for a 13-frame, 60 MP stack plus additional temporary space; the app checks available disk space before starting.

A stack cannot restore regions blurred in every frame. Identical recorded focus positions trigger a warning, but some cameras do not record a useful focus distance. Motion, parallax, reflections, defocus halos, and insufficient overlap can still cause artifacts. Inspect boundaries at 100% and use the comparison and maps before relying on the output.

## Compare the result

After a build, use **Swipe comparison**, **Blink**, **Difference**, or **Registration overlay**. Drag to pan, scroll to zoom, and choose **100%** for one output pixel per device pixel. Hold Space to blink. Select any available original frame as the reference.

Both sides use the same scene coordinates, white balance, exposure, and finishing. Sharpening and denoising default to zero. For 16-frame composites, the original is enlarged 2× with nearest-neighbor sampling. Previews are 8-bit; the saved TIFF is 16-bit. Histograms and clipping percentages describe the visible region.

RAW support shows normalized sample coverage; alignment confidence shows registration correlation, not a probability of correctness. The interpolation map identifies pixels requiring wider sample support. Legacy TIFFs without an editable cache support exposure and finishing using their existing rendered color; RAW white balance, support maps, and offline original comparisons require a new build.

Alignment, region, and reconstruction changes require **Build composite**. Color, finishing, encoding, naming, and border changes can use **Export current adjustments**. Existing files are never overwritten.

## How reconstruction works

Pixelshift reads the original CFA samples through RawPy/LibRaw, applies the recorded crop and black levels, and estimates global translations from texture-selected regions. It combines the measured color samples on the output grid using normalized convolution. On ideal integer- or half-pixel sampling positions, this reduces to direct color interleaving with two-green averaging.

Sampling gaps use a wider interpolation kernel; the app and report disclose their frequency. Camera white balance, a camera-to-sRGB matrix, and the selected transfer curve are applied after reconstruction. Custom temperature/tint uses an approximate Planckian white point and the camera calibration matrix.

## Limitations

- Supports full-size α7R V RAWs and complete, matching capture groups.
- Corrects global translation, not rotation, local subject motion, or optical distortion. Moving subjects can produce artifacts.
- In warning mode, conflicting alignment estimates continue using the median translation; strict mode stops at the configured thresholds. Unusable alignment still stops processing. A warning does not prove the camera moved.
- Does not export ARQ or apply lens corrections, local motion correction, or highlight reconstruction.
- Has been exercised with real 4- and 16-frame captures and synthetic tests. Equivalence to Sony's output and quantitative resolution gains remain unverified.

## Privacy

Photo processing happens on your Mac. The server listens only on `127.0.0.1`, and processing requests require a per-launch token. There are no photo uploads, accounts, or telemetry.

This repository contains no sample photos, capture reports, credentials, or personal settings. Generated photos, reports, caches, and local environments are excluded by `.gitignore`. Reports generated during use contain local source paths and capture metadata, and editable caches contain image data. Keep these private unless you intend to share them.

## Development

After the launcher has installed the dependencies:

```sh
.venv/bin/python -m unittest discover -p 'test*.py'
```

Tests cover exact synthetic 4- and 16-frame reconstruction, color calibration, sequence validation, nested-folder discovery, alignment, output naming, local request authentication, tuning validation, masks, strict thresholds, presets, ICC profiles, offline inspection, and preview/export parity. Focus tests additionally cover sequence detection, duplicate selection, HDR rejection, focus breathing, complementary sharp-region recovery, offline maps, and the full stacking workflow. No camera files are required.

## References

- [Sony α7R V pixel-shift shooting and processing guide](https://support.d-imaging.sony.co.jp/support/ilc/psms/ilce7rm5/en/index.html)
- [ExifTool Sony metadata definitions](https://exiftool.org/TagNames/Sony.html)
- [Dietz, “Leveraging Pixel Value Certainty in Pixel-Shift and Other Multi-Shot Super-Resolution Processing”](https://doi.org/10.2352/EI.2024.36.15.COIMG-142)
