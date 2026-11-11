# Draco header and connectivity records

`shallot_draco::decode_faces(bytes: &[u8], expected_faces: usize, expected_points: usize)
-> Result<Vec<[u32; 3]>>` decodes Draco **2.2 triangle-mesh connectivity** into
the pinned decoder's face and
point-index order. It supports sequential raw and compressed indices, standard
and valence Edgebreaker traversal, rANS raw/tagged symbols, topology splits,
connected components, open boundaries and attribute seams. This stage does not
decode attribute values. It stops at the end of connectivity; valid attribute
data is not required to obtain faces. It is not yet a complete glTF Draco
primitive decoder.

The caller supplies trusted counts: for a glTF primitive, use the indices
accessor's `count / 3` for `expected_faces` and the `POSITION` accessor's `count`
for `expected_points`. Do not derive these bounds from the untrusted Draco
header or from the largest decoded index; unused points still count. These
counts bound stage 1's first geometry allocations. Face counts must match
exactly. Sequential point counts must also match exactly; Edgebreaker's encoded
vertex count cannot exceed the expected point count, and its final point count
after seam expansion and vertex compaction must match exactly. Declared counts
that exceed the caller's bounds are refused by name before geometry allocation
or entropy decoding. All allocations driven by untrusted data are fallible and
return an error on allocation failure.

`parse_header` returns the method, attribute metadata keyed by unique attribute
ID, file metadata, and consumed byte count. Metadata keys and values are bytes;
values are not assumed to be UTF-8. Other bitstream versions, point clouds and
deprecated predictive Edgebreaker are refused by name. Unknown mesh and traversal
methods are errors. Truncated or structurally corrupt connectivity returns an
error, not a panic. Draco has no payload checksum: arbitrary bit changes can
encode another valid mesh and are not universally detectable.

## Pinned records

`tests/records/faces.tsv` records face count, flattened output byte length and
SHA-256 of **little-endian u32 indices**, three indices per face, for all 370
unique Draco primitive buffer views in the pinned corpus (377 primitive
references), plus 16 committed encoder fixtures. No face sorting, winding
normalization or vertex renumbering is applied before hashing. Each row also
pins the input digest, the connectivity end offset, observed decoder paths and
the reference mesh's actual `num_points()`. Tests pass those pinned face and
point counts to the decoder, including for truncated and corrupted streams.
`inputs.json` names each source file, its SHA-256, each extracted range and its
mesh/primitive references. Corpus `.bin` and GLB files are cached, not committed;
there are no standalone `.drc` files to discover in these models.

The corpus has **18** models with actual Draco primitive streams at this pin,
not 17: ABeautifulGame (GLB), Avocado, BarramundiFish, BoomBox, Box, BrainStem,
CarConcept, CesiumMan, CesiumMilkTruck, Corset, Duck, Lantern,
MorphPrimitivesTest, RiggedFigure, RiggedSimple, SunglassesKhronos, VirtualCity
and WaterBottle. SunglassesKhronos participates even though the importer refuses
its required WebP extension. ABeautifulGame reuses eight buffer views across
15 primitive references; each unique stream is decoded once.

```sh
DRACO=/path/to/draco \
GLTF_SAMPLE_MODELS=/path/to/glTF-Sample-Assets/Models \
bun diagnostics/draco-records/generate.ts

GLTF_SAMPLE_MODELS=/path/to/glTF-Sample-Assets/Models cargo test -p shallot-draco
```

The generator verifies clean tracked checkouts at Draco
`554628821c3448319d7d4ae48db170b8c63e1aaf` (1.5.9) and glTF-Sample-Assets
`2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf`. It requires CMake, a C++ compiler
and Bun. It archives the pinned Draco revision into
`test-results/draco-reference/source`, instruments that disposable copy, builds
its native library, and calls `Decoder::DecodeMeshFromBuffer`. The reference
checkout is never modified. The C++ encoder and decoder are generation tools,
not runtime or Cargo-test dependencies.

Cargo tests use Bun only to synchronize hash-verified corpus source files into
`test-results/draco-assets` and extract the recorded ranges into
`test-results/draco-streams`. With no `GLTF_SAMPLE_MODELS`, they download those
same files from the pinned public
revision. Missing premises fail rather than skip. Tests check each input hash,
face count, byte length and output digest; mismatches name the stream. Fixtures
are small, committed `.drc` files, generated through the pinned C++ encoder API.
`licenses` contains the upstream port's Apache 2.0 license and authors, and each
corpus model's license notice.

## Allocation bounds

`tests/memory.rs` meters decoder allocations on the calling thread. The
reviewer's 40-byte valence stream (10 million symbols) and its 43-byte variant
(600 million symbols) both return `Draco face count mismatch` with **zero heap
allocation**, using trusted counts of two faces and four points.

Malformed attribute-metadata, entry, value, child, face, point, encoded-vertex,
symbol, split, event, context and entropy-alphabet counts are also checked under
a 64 KiB test allocation budget. A truncated 255-seam-decoder case exercises the
attribute count; a tiny raw stream requesting a 4 MiB lookup exercises a denied
allocation. The tests assert bounded allocation calls and bytes, not a timing
threshold. Every allocation request in every accepted committed fixture is
then denied in turn: each denial must return an error rather than abort the
process. This covers metadata copies/stacks, entropy tables/output, topology
maps/stacks, valences, seam tables, point maps and output faces. The test budget
is allocator fault injection, not an additional runtime format restriction.

## Measured paths and fixtures

`tests/records/coverage.json` maps each of 49 instrumented paths to the streams
that actually take it. The trace is active only while decoding connectivity,
so attribute entropy does not incorrectly count toward this stage. Variant
names and encoder speed settings are not evidence of a path.

- `sequential-u8`, `-u16`, `-varint`, `-u32` select all four raw index widths via
  `Mesh::set_num_points`, including indices near each width's upper bound,
  without allocating millions of attribute values.
- `sequential-compressed-raw` and `-tagged` select both entropy methods by their
  actual index distributions. `sequential-compressed-wide` selects raw entropy
  explicitly via the native `EncodeSymbols` option in a protected
  `MeshSequentialEncoder` connectivity hook; the default sequential encoder
  passes no entropy options. Its wide alphabet covers the three-byte frequency
  token and four-byte final rANS state that neither corpus nor small fixtures
  reaches. The rest of its header/attribute encoding is the unmodified native
  encoder. Every frequency token (0/1/2 and zero run 3), every symbol rANS final
  state width (1–4 bytes), renormalization, and binary state width (1–3 bytes)
  is observed in the resulting records.
- `standard-*` and `valence-*` include a tetrahedron, an open grid with a
  disconnected component, a torus with topology split events, and a UV-seamed
  pair of triangles. They cover both start-face configurations and all CLERS
  symbols. Corpus traces additionally demonstrate left and right split edges,
  invalid-vertex removal/retention, seam/no-seam branches, boundary/interior
  point assignment, and all six valence contexts (2 through 7).
- The fixtures set `EncoderOptions::SetGlobalInt("edgebreaker_method", ...)`.
  At the pin, `SetEncodingSubmethod` sets `encoding_submethod`, but the mesh
  encoder reads `edgebreaker_method`; relying on that setter's name silently
  produced standard streams under both fixture labels. The generator verifies
  the actual decoder branch instead.
- `metadata-long` contains attribute metadata, a 130-byte binary value and
  nested metadata. This is an encoder-produced regression for the metadata
  value-length erratum below.

Header tests exercise signature, exact version, geometry type, encoding method,
named point-cloud/predictive refusals, trusted face/point count mismatches,
metadata boundaries and malformed varints. Connectivity tests catch panics explicitly while rejecting truncated
prefixes, impossible mesh counts, indices outside the point count, unsupported
entropy schemes/precisions, invalid tables/data sizes and a topology symbol
that requires an absent active edge. They also decode the exact
connectivity-only prefix of every successful record, so stage 1 cannot silently
acquire an attribute-payload dependency.

```sh
bun diagnostics/draco-records/mutations.ts --verify
bun diagnostics/draco-records/mutations.ts
# Or a focused family:
bun diagnostics/draco-records/mutations.ts --only=header.
```

Run mutation checks without concurrent decoder edits. The script first
requires a passing baseline, perturbs header/metadata rules and each connectivity
family in turn, requires a test assertion failure (not a compiler failure), and
restores the original source after every case. `--list` and `--verify` do not
modify source. Branch-removal mutants cover the recorded entropy state widths,
frequency-token forms, traversal methods, CLERS symbols, split events, start-face
configurations, seams, vertex assignment and valence contexts. Count-mismatch
mutants also check that caller bounds are enforced before connectivity payload
processing and that seam-expanded point counts match. A source edit made
concurrently is never overwritten by restoration.

## Decoder/text errata

The bit-level bar is Draco at the pin, including its refusals and arithmetic
on accepted streams. These behaviors take precedence over the 2.2 pseudocode;
undefined or host-dependent C++ behavior requires a separate decision.

1. **Metadata value length is varUI32, not UI8.**
   `docs/spec/metadata.decoder.md` labels `metadata.value_size[i]` UI8.
   `MetadataEncoder::EncodeMetadata` and `MetadataDecoder::DecodeEntry`
   write/read it with `EncodeVarint`/`DecodeVarint`. Keys and child names still
   have UI8 lengths. `metadata_values_use_varui32_not_the_texts_ui8` checks the
   encoder's 130-byte value and the following nested metadata, so treating its
   `0x82 0x01` size as one byte cannot accidentally pass.
2. **Zero-byte metadata values are refused.** The pseudocode would consume zero
   bytes, but `MetadataDecoder::DecodeEntry` returns false when the
   decoded value length is zero. The generator independently checks that
   refusal with the pinned decoder;
   `zero_length_metadata_value_is_refused_like_the_pinned_decoder` is the Rust
   regression. This is not a refusal of empty keys or empty metadata objects.
3. **The sequential face-count sanity bound also applies to compressed data.**
   The pinned `MeshSequentialDecoder::DecodeConnectivity` requires
   `num_faces <= remaining_size / 3` before reading the connectivity selector,
   even though rANS can encode more faces in fewer bytes. The text contains no
   such rejection. The pinned encoder's attribute-free 64-face repeated mesh
   is consequently refused by its own decoder. `reject-too-compressed.drc` is
   that unmodified encoder output, pinned in `rejections.tsv`, independently
   refused during generation and checked by the Rust regression
   `encoder_output_refused_by_the_pinned_decoders_face_count_bound_is_refused`.

4. **Binary rANS complements narrow to uint8.** `RabsDescRead` in
   `rans.decoding.md` uses `p = 256 - p0`. The decoder's `AnsP8` narrows that
   complement to eight bits. For `p0 = 0`, it is zero, not 256: binary data
   `00 01 00` yields zero bits. The encoder clamps probabilities to `[1,255]`,
   but the decoder accepts zero. `zero_binary_probability_uses_pinned_uint8_complement`
   checks the primitive; `zero_binary_probability_matches_pinned_faces` checks
   a full stream accepted by the pinned decoder, with faces `[[0,1,2],[1,3,2]]`.
5. **Every nonzero sequential connectivity selector means raw indices.** The
   text dispatches only selectors zero and one. The decoder uses `else` for
   all nonzero values, though the encoder writes one. The
   `every_nonzero_sequential_connectivity_method_is_raw_like_the_pin` regression
   checks all 255 nonzero selectors against the pinned encoder's raw fixture.
6. **Direct bits are least-significant first, without per-group padding.**
   The conventions text describes `f[n]` as high-to-low; `BitEncoder::PutBits`
   and `BitDecoder::GetBits` iterate from bit zero upward. In addition,
   `DecodeTaggedSymbols` in `rans.decoding.md` resets the bit reader after
   each component group; the implementation keeps one continuous value-bit
   stream across all groups and aligns only at its end. Draco 1.0.0
   (`81d7390`) already encoded both this bit order and continuous packing.
   `bit_fields_cross_bytes_without_alignment`,
   `tagged_groups_are_packed_continuously_by_pinned_encoder` and
   `split_event_deltas_share_one_byte_packed_edge_stream` protect these rules,
   alongside the recorded standard-traversal faces.
7. **Corner state and point numbering follow the implementation.** The text's
   `MapCornerToVertex` updates vertex-corner state implicitly; the library
   changes only the corner-to-vertex map and explicitly maintains leftmost
   corners. It also compacts vertices isolated by S symbols, only without
   attribute connectivity, by filling each hole from the last valid vertex,
   rather than stably renumbering. With attributes, it retains isolated base
   IDs and numbers surviving seam sectors during point assignment. The
   `invalid_vertices_are_filled_from_last_valid_not_stably_renumbered` and
   `interior_seam_rotation_uses_first_attribute_with_a_discontinuity` tests
   protect the numbering and seam-rotation order.
8. **Decoder validation omitted by the pseudocode remains enforced.**
   Metadata refuses duplicate child names and nesting above the pinned limit
   (the first child is level zero; 1,001 nested children are accepted, 1,002
   refused). The conventions text requires unique entry keys and attribute
   metadata IDs, and matching attribute IDs. The decoder instead overwrites
   duplicate entry keys and retains duplicate or unmatched attribute metadata
   IDs; empty names remain valid. The regressions are
   `metadata_duplicate_entries_overwrite_and_duplicate_children_are_refused`,
   `attribute_metadata_ids_can_repeat_and_need_not_name_an_attribute` (also
   verified as a full stream with the pinned decoder), and
   `metadata_nesting_limit_matches_pinned_levels`. Unsigned varints retain the
   destination type's bits on overflow, as `DecodeVarintUnsigned` does;
   `ranges_and_varints_are_bounded` covers that narrowing and the byte limit.
   Entropy tables require an in-range alphabet, bounded zero runs, a complete
   probability sum, valid final-state widths/ranges and available encoded
   data. Sequential indices must fit their point count and signed difference
   reconstruction. Edgebreaker checks possible face/vertex/edge counts,
   symbol/split/context counts, split deltas/order, active edges, degeneracy,
   vertex fans, symbol IDs and final face counts. `tests/malformed.rs`,
   `malformed_probability_tables_and_states_are_refused` and the Edgebreaker
   unit tests exercise those refusals; record-prefix tests exercise byte
   bounds. Metadata uses fallible vectors and in-place key sorting, not scans
   of preceding siblings; `metadata_handles_many_distinct_entries_and_children`
   covers 40,000 of each.

Header flags not used for metadata are ignored, as by the reference; they are
not a reserved-bit rejection rule in these tests.

`bun diagnostics/draco-records/entropy-mutations.ts` additionally perturbs
entropy output, continuous tagged packing, probability validation and binary
complement narrowing. Run it serially with the main mutation script, with no
concurrent decoder edits. It restores source and requires test failures rather
than compiler failures; it never changes records to make a mutation pass.
