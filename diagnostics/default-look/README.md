# Default-look SSAO evidence

From the repository root:

```sh
bun diagnostics/default-look/run.ts --ssao
bun diagnostics/default-look/run.ts --cost --resolution=3024x1964
# Exhaustive qualification only:
bun diagnostics/default-look/run.ts --cost --qualify --resolution=3024x1964
```

`--ssao` captures every fixed scene in aperture-carbon with SSAO Off, Low, Medium, High, Ultra and Custom. Custom uses the component's defaults (3 slices and 3 samples per side); thickness and radius are unchanged for every quality. Captures use 1280×720 device pixels and fixed noise. Open `test-results/default-look/index.html` to compare them. PNGs live in `test-results/default-look/aperture-carbon/<quality>/`. Without a flag the existing palette comparison still runs.

Add `--ao` to `--ssao` to save the denoised AO alongside each enabled PNG as `<scene>.f32`: 1280×720 tightly packed float32 visibility values in row order (0 occluded, 1 visible). Compare two saved quality directories with `bun diagnostics/default-look/compare-ao.ts <before-aperture-carbon> <after-aperture-carbon>`. It reports maximum and mean absolute difference over every pixel, including unchanged pixels, without display encoding.

For repeatability, hash the 24 PNGs and comparison page, repeat the same command on the same adapter and browser, then check the saved hashes:

```sh
find test-results/default-look/aperture-carbon -mindepth 2 -name '*.png' -print | sort | xargs shasum -a 256 > /tmp/ssao.sha256
shasum -a 256 test-results/default-look/index.html >> /tmp/ssao.sha256
bun diagnostics/default-look/run.ts --ssao
shasum -a 256 -c /tmp/ssao.sha256
```

`--cost` runs the cost page unattended in headless Chromium on a hardware adapter and writes `test-results/default-look/cost.json` and a static `test-results/default-look/cost.html` report. Open the latter to read the completed table without rerunning the measurement. Resolution defaults to the M4 Max seat's 3024×1964 native panel size; pass the measured native resolution for another host. The capture resolution is independent of this setting. Preserve each JSON file before repeating the run. To regenerate the report without measuring again, run `bun diagnostics/default-look/render-cost.ts <saved-cost.json>`. Compare every frame and GPU-span median, not just an aggregate. The first run is not discarded automatically, and no unstable row is filtered out.

The default bench uses primitives-greybox, Off and every quality level, on seven fresh pages per level. Each page warms for 30 frames, then takes medians over the last 60 of 120 queue-awaited frames, as in the fog bench. The extra queued frames allow asynchronous GPU diagnostics to fill before selecting the final 60 samples. `--scene=<id>` selects another scene. `--quality=<level>` narrows the bench to Off and that level; `--quality=all` selects all levels and is the default. `--pages=<count>` changes the page count; seven is the default. Every page is reported separately, not pooled or automatically retried.

`--qualify` explicitly selects the exhaustive protocol: all four scenes and six quality levels, with a 120-second Ultra device warm-up and then 10 seconds warm-up and 20 seconds measurement per row. Run it for a proposed default, preserving each run's JSON before the next. It is not the default bench.

The cost page steps one frame at a time without rAF pacing. Frame time covers `world.step` through its submission fence, including rendering, profiling and host/GPU synchronization; it is not a refresh interval or GPU-only total. SSAO's seven GPU spans include preprocess, four separately dispatched depth mips, evaluation and denoise. They exclude the prepasses and PBR draw that consume the result; the frame time includes those. GPU timing uses ProfilePlugin's asynchronous readback. When `timestamp-query` is unavailable the row says so and reports frame time alone. Browser, adapter, backing resolution and sample counts accompany each result. Keep the page foreground and the viewport unchanged during measurement.
