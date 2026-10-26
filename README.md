# PixelShift Lab — RAW reconstruction v2

Independent local Mac app for Sony a7R V pixel-shift RAWs. This version replaces the original RGB-averaging prototype. It does not call Imaging Edge or PixelShift2DNG.

## Start and use

Double-click `Start.command` (Python 3.11+ required). First launch installs NumPy, OpenCV, RawPy and tifffile into a private environment. ExifTool must also be installed and in PATH; it is used read-only for Sony capture metadata. On macOS, install ExifTool from its official website or with `brew install exiftool`.

Choose a mounted SD card/drive, or paste `/Volumes/CAMERA_CARD`. Scanning includes arbitrarily nested and hidden folders, follows directory links inside the chosen root with cycle detection, and reports inaccessible folders. Choose 4 or 16 frames. Only complete groups confirmed by Sony PixelShiftInfo, frame indices, model and matching exposure metadata are listed; captures can span directories. Filename order is irrelevant. Sony metadata determines the available 4- and 16-shot groups; ordinary bursts are excluded.

Output defaults to `~/Pictures/Pixel Shift`, created automatically. A folder receives a unique filename; an explicit TIFF path is also accepted. Original ARWs are read-only. Existing outputs are never overwritten. Keep the launcher and card connected until processing completes.

## RAW reconstruction

1. Validate and order the complete capture group using ExifTool's decoded Sony MakerNote `PixelShiftInfo`.
2. Extract original CFA sensor values with LibRaw through RawPy. Respect the RAW's recorded active crop, CFA pattern, channel black levels and white level. Store mosaics on disk, rather than developed RGB frames.
3. Rank spatially separated reference regions by high-frequency green-channel texture, exclude weak background regions, and estimate global translations from up to five selected patches. Reuse the exact region coordinates on every frame. Small-displacement ECC starts at zero to avoid phase-correlation jumps between repeating print patterns. Only displacement estimation uses demosaicing. Reject insufficient detail, low correlation, or excessive translation. Large patch disagreement continues using the median translation and is shown as an uncertainty warning in the UI and JSON report; it does not prove camera movement.
4. Reconstruct each output color channel directly from original black-subtracted RAW samples on the target grid using compact tent weights centered on the measured sensel positions. At ideal integer/half-pixel lattice positions, this reduces to exact per-channel interleaving, with the two measured green samples averaged. Synthetic high-frequency scenes verify exact 4- and 16-shot recovery without demosaiced RGB entering the merge.
5. Positions with no compact-kernel RAW support use a wider RAW-sample kernel. The JSON report and UI disclose the fraction of channel positions needing this interpolation. This is an independent normalized-convolution reconstruction, not Sony's proprietary implementation.
6. Apply one consistent camera white balance, a camera-to-sRGB matrix derived from LibRaw's camera/XYZ matrix, and the standard sRGB transfer curve. Export 16-bit RGB BigTIFF with the system sRGB ICC profile when available. No automatic exposure, denoising, sharpening or tone enhancement is added.

Four shots produce 9504 × 6336 (~60.2 MP); sixteen produce 19008 × 12672 (~240.8 MP). The old four-shot 240MP interpolation option has been removed. Higher pixel count does not guarantee equivalent resolved detail: lens, diffraction, focus, sampling, motion and exposure all matter.

## In-app before/after scrubber

After a successful merge, **Detail Inspector** compares the first original frame against the new composite. Both sides use identical scene coordinates, white balance, exposure and color transform. The original alone receives conventional CFA demosaicing, while the composite uses the RAW reconstruction described above. No sharpening is added to either side.

Drag on the image or use the keyboard-accessible slider to scrub. Pick the automatically selected Best detail region, center, or one of four other detail regions and magnifications of 1×, 2×, 4× or 8×. Crops cover 512 × 512 native-pixel scene coordinates; 16-shot crops have twice the output sampling density. For that comparison the original is enlarged with nearest-neighbor sampling. Browser preview PNGs are 8-bit sRGB; the saved TIFF remains 16-bit. Screen magnification depends on the viewport; the zoom control scales the displayed crop, not a guaranteed physical 1:1 display scale.

## Limits and validation

This is not Sony-certified or bit-for-bit equivalent to Imaging Edge. Camera-specific rendering, lens corrections, local motion masks, rotation/optical-flow correction, highlight reconstruction, and ARQ export are not implemented. Translation alignment compensates global camera movement; moving subjects can still create artifacts, and alignment failures stop the merge. Larger RAW sampling gaps require interpolation and can reduce detail. No AI detail generation is used.

Tested on synthetic exact 4/16-frame high-frequency scenes, color ordering, black-level correction, metadata ordering/rejection, deep cross-directory grouping, output naming and HTTP token protection. Full-size decoding and reconstruction have been exercised with real four- and sixteen-shot a7R V capture sets. No camera files or capture metadata are included in this repository. Parity with Sony and quantitative resolution gains remain unverified.

Run `.venv/bin/python test_engine.py` for automated tests. Allow several GB RAM and around 10 GB disk headroom for a 16-shot job; processing uses banded reconstruction and disk-backed sensor arrays. Temporary `.pixelshift-*` folders and `.partial` files are removed on handled completion/error/cancellation; a force quit may leave them for manual cleanup when no merge is running.

The HTTP server binds only to loopback on a random port by default, and processing requests require a per-launch token.

## Technical references

- Sony shooting and supported processing workflow: https://support.d-imaging.sony.co.jp/support/ilc/psms/ilce7rm5/en/index.html
- ExifTool Sony MakerNote tag definitions: https://exiftool.org/TagNames/Sony.html
- Dietz, “Leveraging Pixel Value Certainty in Pixel-Shift and Other Multi-Shot Super-Resolution Processing,” Electronic Imaging 2024: https://doi.org/10.2352/EI.2024.36.15.COIMG-142

The reconstruction implementation is original Python/NumPy code; dependencies retain their own licenses.

Alignment-region disagreements above 1.5 pixels no longer abort a merge. They remain visible while processing and after saving. A median translation is used; possible artifacts are disclosed. Jobs still stop if fewer than two usable high-correlation patch estimates exist.

Registration update v2.1: fixed regions could select blurred background and dominate the median displacement. Texture-based selection and fixed reference coordinates replace those regions. JSON reports record selected region coordinates, texture scores, and individual patch estimates for audit. Eleven automated checks include texture selection and small-shift recovery.
