# Basis Universal level records

`crates/basisu/tests/records` holds the length and SHA-256 digest of every level in
StainedGlassLamp and AnisotropyBarnLamp's `glTF-KTX-BasisU` images, plus the five
committed fixtures in `crates/basisu/tests/fixtures`. The generator hashes
BasisLZ levels unchanged and UASTC levels after optional decompression with the
bundled Zstandard decoder. Decoded bytes are temporary generation inputs, not
committed records. `etc1s.tsv` additionally hashes every ETC1S level in ETC1,
ETC2 RGBA (EAC A8), BC7 mode 5 and RGBA8, through the pinned reference's default
transcode path, including its BC7 chroma filter. The Rust entry is
`Ktx2::transcode_etc1s(level, Target)`; output is tightly packed blocks or pixel
rows. ETC1 drops alpha; the other three targets preserve it.
`uastc.tsv` hashes every UASTC level in ASTC 4×4, BC7 and RGBA8. The Rust
entry is `Ktx2::transcode_uastc(level, UastcTarget)`; RGBA8 uses the reference's
linear ASTC interpolation, independently of the image's transfer declaration.

Regenerate with clean checkouts:

```sh
BASIS_UNIVERSAL=/path/to/basis-universal \
GLTF_SAMPLE_MODELS=/path/to/glTF-Sample-Assets/Models \
bun diagnostics/basisu-records/generate.ts
```

The generator checks Basis Universal `073eb6a0a34a23d77e75e54f41dcd9653a52da83`
and glTF-Sample-Assets `2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf`.
It requires CMake, clang and clang++; builds the encoder and reference decoder;
and encodes deterministic 8×8 TGA inputs. The reference uses
`ktx2_transcoder::init`, `start_transcoding`, `transcode_image_level` and
`zstd/zstddeclib.c`. The C++ build disables floating-point contraction to match
scalar WebAssembly's separate multiply/add operations in the BC7 chroma filter
and encoder; otherwise native ARM FMA changes a few endpoint rounding decisions. The pinned encoder does not assign
red/red-green DFD channel IDs from its swizzles: the generator changes only those
IDs after encoding the corresponding channels. `uastc-none` uses
`-ktx2_no_zstandard`; the constant `uastc-red` fixture exercises a compressed
Zstandard block, not just a raw frame.

`cargo test -p shallot-basisu` syncs the pinned corpus with SHA-256 verification.
`GLTF_SAMPLE_MODELS` selects local inputs; without it, the test downloads the same
files from their pinned revision. Corpus images are cached, not committed.
Neither the encoder nor C++ decoder is needed to run the tests. Records and
fixtures are read beside the tests. A mismatch names the asset and level with
expected and actual lengths and SHA-256 digests.

```sh
bun diagnostics/basisu-records/mutations.ts
```

This perturbs one parser/decoder rule at a time, requires an assertion failure
(not a compiler failure), and restores the source files. Primaries and the
extension's SHOULD for a complete mip pyramid are intentionally not rejection
rules. Arbitrary payload bit flips need not be detectable without a checksum;
tests reject malformed headers, frame structure, truncations and bad checksums.
The ETC1S transcoder refuses malformed BasisLZ ranges, codebooks, Huffman tables,
indices, predictors, runs and truncated entropy. Like the reference, it does not
support global/hybrid selector codebooks. Video image flags are refused.

```sh
bun diagnostics/basisu-records/etc1s-mutations.ts
```

This perturbs each ETC1S target's output in turn and requires the corresponding
reference-digest assertion to fail, restoring the source after every run.
Malformed BasisLZ tests explicitly catch panics: structural and entropy failures
must be errors; arbitrary entropy bit flips may still encode valid images.

## WebAssembly entry

`src/extras/gltf/basisu.ts` owns the JavaScript boundary. `openBasisImage` copies
its KTX2 input into WebAssembly; `transcode` returns independently owned target
bytes, including Zstandard decompression where required; `dispose` releases the
input and last output. Importing this module instantiates nothing until an image
is opened. It is not re-exported by the glTF barrel.

```sh
bun crates/basisu/scripts/build.ts
bun test scripts/check-basisu.test.ts src/extras/gltf/basisu.test.ts
bun diagnostics/basisu-records/uastc-mutations.ts
BASIS_UNIVERSAL=/path/to/basis-universal bun diagnostics/basisu-records/timing.oracle.ts
```

The build stamps its Rust sources and build inputs, as the other crate artifacts
do. The JavaScript test checks all 264 ETC1S and 438 UASTC target records through
the built artifact. The mutation check requires each target's reference-digest assertion
to fail on perturbed output, then restores the source.

The timing oracle runs three fresh Bun processes. It compares complete models,
with input/output copies, image setup, every image level and optional Zstandard,
in the M4 Max's selected targets: ETC1S to ETC1 or ETC2 RGBA, UASTC to ASTC 4×4.
It alternates order during warmup, calibration and 256 sampled batches and
reports minimum milliseconds per model, following meshoptimizer's
`tools/codecbench.cpp` maximum-throughput statistic. It also reports the first
128 batches' minimum so later improvement remains visible. The reference is Binomial's committed
`webgl/transcoder/build/basis_transcoder.js` and `.wasm`, through `KTX2File`.
Commit `32c7b74` built that pair from `3784d52`; the transcoder, Zstandard and
WebGL transcoder inputs are unchanged between `3784d52` and `073eb6a0`.
The oracle checks the pin, clean checkout, unchanged inputs and matching output
before timing. Timings are measurements, not test assertions.

```sh
BASIS_UNIVERSAL=/path/to/basis-universal bun diagnostics/basisu-records/attribution.oracle.ts
```

The attribution oracle requires the native extractor built by `generate.ts`.
It verifies each extracted level against its record and compares both JavaScript
entries on an equivalent KTX2 without supercompression. It reports setup,
disposal, observed JavaScript boundary copies, ETC1, ETC2 RGBA, ASTC without
Zstandard, and the compressed-minus-control ASTC time. That last value is a net
Zstandard increment, not an independently timed decoder. Each entry/control
and component keeps its independent minimum over 256 interleaved observations
after four warmups; component minima do not sum to a measured model run.
Each invocation is one fresh process; run it three times alongside the
full-model timing oracle. Repeat each three-process or three-page sweep
independently. Compare each entry's best per-model minimum across the three
replicas for sweep repeatability, retaining all per-replica observations;
those can still reflect contention.

```sh
BASIS_UNIVERSAL=/path/to/basis-universal bun diagnostics/basisu-records/browser.oracle.ts
BASIS_UNIVERSAL=/path/to/basis-universal bun diagnostics/basisu-records/browser.oracle.ts --attribution
```

The browser oracle requires installed Google Chrome and launches it headlessly
through Playwright. It runs the same comparison or attribution loop as Bun in
three fresh pages, with both entries in each page. Inputs are verified before
serving; output equality is checked in the page before measuring. The attribution
mode requires the same native extractor. `--profile` additionally writes Chrome
CPU profiles under `test-results`, outside the repository's records.

## Attribution

The StainedGlassLamp records derive from **Stained Glass Lamp**, Eric Chadwick,
Wayfair, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode):
<https://github.com/KhronosGroup/glTF-Sample-Assets/tree/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/StainedGlassLamp>.

The AnisotropyBarnLamp records derive from **Anisotropy Barn Lamp**, Eric Chadwick,
Copyright 2023 Wayfair, LLC,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode):
<https://github.com/KhronosGroup/glTF-Sample-Assets/tree/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/AnisotropyBarnLamp>.

The records hash extracted, Zstandard-decompressed and ETC1S-transcoded
and UASTC-transcoded texture data. Model license notices are in `crates/basisu/licenses`.
