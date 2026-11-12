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
inline bool &record_attribute_active() { static bool value = false; return value; }
inline void record_attribute(const char *path) { if (record_attribute_active()) record_paths().insert(path); }
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
    const attribute = (path: string) => `record_attribute("attribute.${path}");`;
    patch("compression/point_cloud/point_cloud_decoder.cc", [
        ["  if (!DecodePointAttributes()) {", "  record_attribute_active() = true;\n  if (!DecodePointAttributes()) {"],
    ]);
    patch("compression/mesh/mesh_sequential_decoder.cc", [
        ["  // Always create the basic attribute decoder.", `  ${attribute("sequencer.linear")}`],
    ]);
    patch("compression/mesh/mesh_edgebreaker_decoder_impl.cc", [
        ["    if (traversal_method == MESH_TRAVERSAL_PREDICTION_DEGREE) {", `    if (traversal_method == MESH_TRAVERSAL_PREDICTION_DEGREE) {\n      ${attribute("traversal.prediction-degree")}`],
        ["    } else if (traversal_method == MESH_TRAVERSAL_DEPTH_FIRST) {", `    } else if (traversal_method == MESH_TRAVERSAL_DEPTH_FIRST) {\n      ${attribute("traversal.depth-first")}`],
        ["    // Per-corner attribute decoder.", `    ${attribute("traversal.corner")}`],
    ]);
    patch("compression/attributes/sequential_attribute_decoders_controller.cc", [
        ["    case SEQUENTIAL_ATTRIBUTE_ENCODER_GENERIC:", `    case SEQUENTIAL_ATTRIBUTE_ENCODER_GENERIC:\n      ${attribute("decoder.generic")}`],
        ["    case SEQUENTIAL_ATTRIBUTE_ENCODER_INTEGER:", `    case SEQUENTIAL_ATTRIBUTE_ENCODER_INTEGER:\n      ${attribute("decoder.integer")}`],
        ["    case SEQUENTIAL_ATTRIBUTE_ENCODER_QUANTIZATION:", `    case SEQUENTIAL_ATTRIBUTE_ENCODER_QUANTIZATION:\n      ${attribute("decoder.quantization")}`],
        ["    case SEQUENTIAL_ATTRIBUTE_ENCODER_NORMALS:", `    case SEQUENTIAL_ATTRIBUTE_ENCODER_NORMALS:\n      ${attribute("decoder.normals")}`],
    ]);
    patch("compression/attributes/sequential_integer_attribute_decoder.cc", [
        ["  if (prediction_scheme_method != PREDICTION_NONE) {", `  if (prediction_scheme_method == PREDICTION_NONE) ${attribute("prediction.none")}\n  if (prediction_scheme_method != PREDICTION_NONE) {`],
        ["        static_cast<PredictionSchemeTransformType>(prediction_transform_type));", `        static_cast<PredictionSchemeTransformType>(prediction_transform_type));\n    if (!prediction_scheme_) ${attribute("prediction.dropped")}`],
        ["  if (compressed > 0) {", `  if (compressed > 0) {\n    ${attribute("values.compressed")}`],
        ["    if (num_bytes == DataTypeLength(DT_INT32)) {", `    if (num_bytes == DataTypeLength(DT_INT32)) {\n      ${attribute("values.raw.4")}`],
        ["      if (in_buffer->remaining_size() <", `      ${attribute("values.raw.narrow")}\n      if (in_buffer->remaining_size() <`],
        ...([["UINT8", "uint8_t", "u8"], ["INT8", "int8_t", "i8"], ["UINT16", "uint16_t", "u16"], ["INT16", "int16_t", "i16"], ["UINT32", "uint32_t", "u32"], ["INT32", "int32_t", "i32"]] as const).map(([dt, type, name]): [string, string] => [`    case DT_${dt}:\n      StoreTypedValues<${type}>(num_values);`, `    case DT_${dt}:\n      ${attribute(`integer.${name}`)}\n      StoreTypedValues<${type}>(num_values);`]),
    ]);
    const schemes = "compression/attributes/prediction_schemes/";
    for (const [file, name] of [["prediction_scheme_delta_decoder.h", "delta"], ["mesh_prediction_scheme_parallelogram_decoder.h", "parallelogram"], ["mesh_prediction_scheme_multi_parallelogram_decoder.h", "multi-parallelogram"], ["mesh_prediction_scheme_constrained_multi_parallelogram_decoder.h", "constrained-multi-parallelogram"], ["mesh_prediction_scheme_tex_coords_decoder.h", "tex-coords-deprecated"], ["mesh_prediction_scheme_tex_coords_portable_decoder.h", "tex-coords-portable"]]) {
        const edits: [string, string][] = [["  this->transform().Init(num_components);", `  ${attribute(`prediction.${name}`)}\n  this->transform().Init(num_components);`]];
        if (name.includes("parallelogram")) edits.push(["      const int src_offset = (p - 1) * num_components;", `      ${attribute(`prediction.${name}.fallback`)}\n      const int src_offset = (p - 1) * num_components;`]);
        if (name.startsWith("constrained")) edits.push(["        const bool is_crease = is_crease_edge_[context][pos];", `        const bool is_crease = is_crease_edge_[context][pos];\n        if (is_crease) ${attribute(`prediction.${name}.crease`)}`]);
        patch(schemes + file, edits);
    }
    patch(schemes + "mesh_prediction_scheme_geometric_normal_decoder.h", [
        ["  this->SetQuantizationBits(this->transform().quantization_bits());", `  ${attribute("prediction.geometric-normal")}\n  this->SetQuantizationBits(this->transform().quantization_bits());`],
        ["    if (flip_normal_bit_decoder_.DecodeNextBit()) {", `    if (flip_normal_bit_decoder_.DecodeNextBit()) {\n      ${attribute("prediction.geometric-normal.flip")}`],
    ]);
    patch(schemes + "mesh_prediction_scheme_tex_coords_portable_predictor.h", [
        ["    if (p_uv == n_uv) {", `    if (p_uv == n_uv) {\n      ${attribute("prediction.tex-coords-portable.equal")}`],
        ["        const bool orientation = orientations_.back();", `        const bool orientation = orientations_.back();\n        ${attribute("prediction.tex-coords-portable.orientation")}`],
        ["  if (next_data_id < data_id) {", `  ${attribute("prediction.tex-coords-portable.fallback")}\n  if (next_data_id < data_id) {`],
    ]);
    patch(schemes + "prediction_scheme_wrap_decoding_transform.h", [
        ["    if (min_value > max_value) {", `    ${attribute("transform.wrap")}\n    if (min_value > max_value) {`],
        ["      if (value > this->max_value()) {", `      if (value > this->max_value()) {\n        ${attribute("transform.wrap.high")}`],
        ["      } else if (value < this->min_value()) {", `      } else if (value < this->min_value()) {\n        ${attribute("transform.wrap.low")}`],
    ]);
    patch(schemes + "prediction_scheme_wrap_transform_base.h", [
        ["      if (predicted_val[i] > max_value_) {", `      if (predicted_val[i] > max_value_ || predicted_val[i] < min_value_) ${attribute("transform.wrap.clamp")}\n      if (predicted_val[i] > max_value_) {`],
    ]);
    patch(schemes + "prediction_scheme_normal_octahedron_decoding_transform.h", [
        ["    return this->set_max_quantized_value(max_quantized_value);", `    ${attribute("transform.octahedron")}\n    return this->set_max_quantized_value(max_quantized_value);`],
    ]);
    patch(schemes + "prediction_scheme_normal_octahedron_canonicalized_decoding_transform.h", [
        ["    if (this->quantization_bits() < 2) {", `    ${attribute("transform.octahedron-canonicalized")}\n    if (this->quantization_bits() < 2) {`],
    ]);
    patch("compression/entropy/symbol_decoding.cc", [
        ["  if (scheme == SYMBOL_CODING_TAGGED) {", "  if (scheme == SYMBOL_CODING_TAGGED) {\n    record_path(\"symbols.tagged\");"],
        ["  } else if (scheme == SYMBOL_CODING_RAW) {", "  } else if (scheme == SYMBOL_CODING_RAW) {\n    record_path(\"symbols.raw\");"],
    ]);
}
