# Processing performance

Pixelshift keeps native imaging libraries and removes unnecessary conversions, copies, and repeated work. Image dimensions, source precision, reconstruction methods, and default finishing settings are unchanged.

## Measurements

Development measurements on a 14-core Mac with 36 GiB RAM, using thirteen 9,504 × 6,336, 10-bit HIF frames:

| Work measured | Previous pipeline | Optimized pipeline |
| --- | ---: | ---: |
| Decode all 13 originals, generate exact registration previews, and write linear references | 59.1 s | 15.0 s |
| Fuse and export all 60.2 MP pixels from the same references and alignment transforms | 46.1 s | 12.2 s |
| Complete repeat build through the app, with verified references and previews already cached | — | 11.5 s |
| Scan the same source directory | 2.44 s initially | 0.009 s on repeat |

The decode and fusion/export measurements are separate stage benchmarks, not one combined end-to-end stopwatch run. Their sum fell from 105.2 s to 27.2 s, excluding alignment and setup. “Decode originals” means no decoded-reference reuse; it does not mean that operating-system file caches were cleared. Timings vary with hardware, storage, settings, background activity, and focus-frame count.

The full fusion/export comparison had zero differing TIFF channel values and zero differing focus-map values. After preserving exact float registration previews, a complete app rebuild also matched the previous TIFF and alignment transforms exactly. A wide synthetic RAW-band benchmark improved from approximately 0.73 s to 0.54 s with one worker; complete pixel-shift job speedups depend on alignment and interpolation requirements.

## Changes

- **Direct HIF decoding.** The optional libheif C API exposes the decoded RGB buffer directly. Conversion uses lookup tables at the original bit depth and writes linear references in bands, avoiding temporary PNG encoding, decoding, and disk traffic. Systems without a loadable native library retain the `heif-convert` fallback.
- **Exact registration previews.** Registration uses the same float previews as the original implementation, rather than resizing quantized references. Small grayscale registration previews are retained for repeat builds.
- **Less redundant fusion.** Larger pyramid-aligned cores use a conservatively sized halo, checked against whole-image fusion across the supported focus radii. Streaming top-two sharpness selection avoids sorting the entire stack at every pixel. Scratch buffers are reused.
- **Bounded concurrency.** Up to two independent decode or processing tasks run concurrently on machines with adequate RAM and CPU cores. Only a small queue is kept in flight. Conservative machines use one worker.
- **Reference reuse.** A private, versioned index checks the source file and decoded cache. Unchanged references and exact previews are shared with hard links on supported filesystems; changes invalidate reuse. Rebuilds therefore avoid decoding and duplicating the same references. Unsupported filesystems continue with normal decoding.
- **RAW reconstruction.** Planar accumulators and reusable calibration buffers reduce memory traffic. Confidence work is omitted when diagnostics are not requested. Independent bands use the same bounded worker queue.
- **Export.** Display encoding is computed once per band and reused for histograms. Finished TIFFs are published with an exclusive hard link where supported, avoiding an extra full-file copy. An exclusive-copy fallback preserves compatibility and prevents overwriting existing outputs.
- **Scanning and inspection.** File-versioned metadata is cached in memory, shared between workflows, and read in larger ExifTool batches. Full-image references are sampled directly from read-only mappings. Repeated tiles use a bounded encoded-preview cache.

The private reference index, previews, image caches, and performance reports remain beside generated images, outside the repository. Reports retain per-stage timings and the decoder/worker choices for that job.

## Check performance locally

Run a small synthetic fusion benchmark:

```sh
.venv/bin/python benchmark.py --runs 3 --workers 1
.venv/bin/python benchmark.py --runs 3 --workers 2
```

Use an existing focus-stack report to benchmark a representative native-width band. References are opened read-only; this does not build or export another image:

```sh
.venv/bin/python benchmark.py --report '/path/to/stack.tif.json' --workers 2
```

For troubleshooting, `PIXELSHIFT_WORKERS=1` forces serial processing. `PIXELSHIFT_LIBHEIF` can point to a particular installed libheif library. Neither variable is needed for normal use.

```sh
PIXELSHIFT_WORKERS=1 .venv/bin/python app.py
```

Tests cover source precision, exact registration previews, whole-image versus banded fusion, reference invalidation, reuse without duplicate storage, cancellation, metadata caching, preview caching, and export publication/fallback behavior, alongside the numerical reconstruction tests.
