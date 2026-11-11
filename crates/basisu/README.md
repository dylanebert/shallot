# Basis Universal level records

`crates/basisu/tests/records` holds the length and SHA-256 digest of every level in
StainedGlassLamp and AnisotropyBarnLamp's `glTF-KTX-BasisU` images, plus the five
committed fixtures in `crates/basisu/tests/fixtures`. The generator hashes
BasisLZ levels unchanged and UASTC levels after optional decompression with the
bundled Zstandard decoder. Decoded bytes are temporary generation inputs, not
committed records. No transcoding is performed in these records.

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
`ktx2_transcoder::init` and `zstd/zstddeclib.c`. The pinned encoder does not assign
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
BasisLZ entropy validation belongs to the ETC1S transcoder.

## Attribution

The StainedGlassLamp records derive from **Stained Glass Lamp**, Eric Chadwick,
Wayfair, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode):
<https://github.com/KhronosGroup/glTF-Sample-Assets/tree/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/StainedGlassLamp>.

The AnisotropyBarnLamp records derive from **Anisotropy Barn Lamp**, Eric Chadwick,
Copyright 2023 Wayfair, LLC,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode):
<https://github.com/KhronosGroup/glTF-Sample-Assets/tree/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/AnisotropyBarnLamp>.

The records hash extracted and, where applicable, Zstandard-decompressed
texture data. Model license notices are in `crates/basisu/licenses`.
