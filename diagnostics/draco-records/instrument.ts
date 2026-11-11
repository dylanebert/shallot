import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Instrument only a disposable archive of the clean, pinned reference. */
export function instrument(source: string) {
    writeFileSync(join(source, "src/records_trace.h"), `#ifndef RECORDS_TRACE_H
#define RECORDS_TRACE_H
#include <set>
#include <string>
inline bool &record_active() { static bool value = false; return value; }
inline long long &record_end() { static long long value = -1; return value; }
inline long long &record_size() { static long long value = 0; return value; }
inline std::set<std::string> &record_paths() { static std::set<std::string> value; return value; }
inline void record_path(const char *path) { if (record_active()) record_paths().insert(path); }
#endif
`);
    function patch(file: string, edits: [string, string][]) {
        const path = join(source, "src/draco", file);
        let text = readFileSync(path, "utf8");
        for (const [before, after] of edits) {
            if (text.split(before).length !== 2) throw new Error(`${file}: instrumentation anchor not unique: ${before}`);
            text = text.replace(before, after);
        }
        writeFileSync(path, `#include "records_trace.h"\n${text}`);
    }
    patch("compression/mesh/mesh_decoder.cc", [
        ["  if (!DecodeConnectivity()) {", "  record_active() = true;\n  if (!DecodeConnectivity()) {"],
        ["  return PointCloudDecoder::DecodeGeometryData();", "  record_end() = record_size() - buffer()->remaining_size();\n  record_active() = false;\n  return PointCloudDecoder::DecodeGeometryData();"],
    ]);
    patch("compression/mesh/mesh_sequential_decoder.cc", [
        ["  uint32_t num_faces;", "  record_path(\"sequential\");\n  uint32_t num_faces;"],
        ["  if (connectivity_method == 0) {", "  if (connectivity_method == 0) {\n    record_path(\"sequential.compressed\");"],
        ["      // Decode indices as uint8_t.", "      record_path(\"sequential.u8\");\n      // Decode indices as uint8_t."],
        ["      // Decode indices as uint16_t.", "      record_path(\"sequential.u16\");\n      // Decode indices as uint16_t."],
        ["      // Decode indices as uint32_t.", "      record_path(\"sequential.varint\");\n      // Decode indices as uint32_t."],
        ["      // Decode faces as uint32_t (default).", "      record_path(\"sequential.u32\");\n      // Decode faces as uint32_t (default)."],
    ]);
    patch("compression/mesh/mesh_edgebreaker_decoder.cc", [
        ["if (traversal_decoder_type == MESH_EDGEBREAKER_STANDARD_ENCODING) {", "if (traversal_decoder_type == MESH_EDGEBREAKER_STANDARD_ENCODING) {\n    record_paths().insert(\"edgebreaker.standard\");"],
        ["} else if (traversal_decoder_type == MESH_EDGEBREAKER_VALENCE_ENCODING) {", "} else if (traversal_decoder_type == MESH_EDGEBREAKER_VALENCE_ENCODING) {\n    record_paths().insert(\"edgebreaker.valence\");"],
    ]);
    patch("compression/mesh/mesh_edgebreaker_decoder_impl.cc", [
        ["    if (symbol == TOPOLOGY_C) {", "    if (symbol == TOPOLOGY_C) {\n      record_path(\"topology.C\");"],
        ["    } else if (symbol == TOPOLOGY_R || symbol == TOPOLOGY_L) {", "    } else if (symbol == TOPOLOGY_R || symbol == TOPOLOGY_L) {\n      record_path(symbol == TOPOLOGY_R ? \"topology.R\" : \"topology.L\");"],
        ["    } else if (symbol == TOPOLOGY_S) {", "    } else if (symbol == TOPOLOGY_S) {\n      record_path(\"topology.S\");"],
        ["    } else if (symbol == TOPOLOGY_E) {", "    } else if (symbol == TOPOLOGY_E) {\n      record_path(\"topology.E\");"],
        ["      if (it != topology_split_active_corners.end()) {", "      if (it != topology_split_active_corners.end()) {\n        record_path(\"topology.split-merge\");"],
        ["      while (IsTopologySplit(encoder_symbol_id, &split_edge,\n                             &encoder_split_symbol_id)) {", "      while (IsTopologySplit(encoder_symbol_id, &split_edge,\n                             &encoder_split_symbol_id)) {\n        record_path(\"topology.split-event\");"],
        ["        if (split_edge == RIGHT_FACE_EDGE) {", "        record_path(split_edge == RIGHT_FACE_EDGE ? \"topology.split-right\" : \"topology.split-left\");\n        if (split_edge == RIGHT_FACE_EDGE) {"],
        ["    if (interior_face) {", "    record_path(interior_face ? \"start.interior\" : \"start.boundary\");\n    if (interior_face) {"],
        ["      if (remove_invalid_vertices) {", "      record_path(remove_invalid_vertices ? \"vertices.remove-invalid\" : \"vertices.retain-invalid\");\n      if (remove_invalid_vertices) {"],
        ["  if (attribute_data_.empty()) {", "  record_path(attribute_data_.empty() ? \"attributes.none\" : \"attributes.connectivity\");\n  if (attribute_data_.empty()) {"],
        ["    if (is_vert_hole_[v]) {", "    record_path(is_vert_hole_[v] ? \"vertices.boundary\" : \"vertices.interior\");\n    if (is_vert_hole_[v]) {"],
    ]);
    patch("compression/mesh/mesh_edgebreaker_traversal_decoder.h", [
        ["    return attribute_connectivity_decoders_[attribute].DecodeNextBit();", "    const bool seam = attribute_connectivity_decoders_[attribute].DecodeNextBit();\n    record_path(seam ? \"seams.yes\" : \"seams.no\");\n    return seam;"],
    ]);
    patch("compression/mesh/mesh_edgebreaker_traversal_valence_decoder.h", [
        ["    if (active_context_ != -1) {", "    if (active_context_ != -1) {\n      const std::string path = \"valence.context.\" + std::to_string(active_context_ + min_valence_);\n      record_path(path.c_str());"],
    ]);
    patch("compression/entropy/rans_symbol_decoder.h", [
        ["    const int token = prob_data & 3;", "    const int token = prob_data & 3;\n    const std::string path = \"rans.frequency.token.\" + std::to_string(token);\n    record_path(path.c_str());"],
        ["  // Advance the buffer past the rANS data.", "  if (bytes_encoded) {\n    const std::string path = \"rans.symbol.init.\" + std::to_string(data_head[bytes_encoded - 1] >> 6);\n    record_path(path.c_str());\n  }\n  // Advance the buffer past the rANS data."],
    ]);
    patch("compression/bit_coders/rans_bit_decoder.cc", [
        ["  source_buffer->Advance(size_in_bytes);", "  if (size_in_bytes) {\n    const auto tail = static_cast<uint8_t>(source_buffer->data_head()[size_in_bytes - 1]);\n    const std::string path = \"rans.binary.init.\" + std::to_string(tail >> 6);\n    record_path(path.c_str());\n  }\n  source_buffer->Advance(size_in_bytes);"],
        ["  return bit > 0;", "  record_path(bit ? \"rans.binary.one\" : \"rans.binary.zero\");\n  return bit > 0;"],
    ]);
    patch("compression/entropy/ans.h", [
        ["    while (ans_.state < l_rans_base && ans_.buf_offset > 0) {", "    while (ans_.state < l_rans_base && ans_.buf_offset > 0) {\n      record_path(\"rans.symbol.renormalize\");"],
    ]);
    patch("compression/entropy/symbol_decoding.cc", [
        ["  if (scheme == SYMBOL_CODING_TAGGED) {", "  if (scheme == SYMBOL_CODING_TAGGED) {\n    record_path(\"symbols.tagged\");"],
        ["  } else if (scheme == SYMBOL_CODING_RAW) {", "  } else if (scheme == SYMBOL_CODING_RAW) {\n    record_path(\"symbols.raw\");"],
    ]);
}
