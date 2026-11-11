// Deterministic encoder-API inputs; no CLI defaults select the missing paths.
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "draco/compression/expert_encode.h"
#include "draco/compression/mesh/mesh_sequential_encoder.h"
#include "draco/compression/entropy/symbol_encoding.h"
#include "draco/core/varint_encoding.h"
#include "draco/mesh/mesh.h"
#include "draco/metadata/geometry_metadata.h"

using Face = std::array<uint32_t, 3>;
using Position = std::array<float, 3>;

// MeshSequentialEncoder passes nullptr to EncodeSymbols. This protected hook
// exposes the pinned entropy encoder's method option without patching Draco or
// post-editing the stream. Delta coding is its CompressAndEncodeIndices mapping.
class RawSequentialEncoder : public draco::MeshSequentialEncoder {
 protected:
  draco::Status EncodeConnectivity() override {
    draco::EncodeVarint(uint32_t(mesh()->num_faces()), buffer());
    draco::EncodeVarint(uint32_t(mesh()->num_points()), buffer());
    buffer()->Encode(uint8_t(0));
    std::vector<uint32_t> symbols;
    int32_t previous = 0;
    for (draco::FaceIndex i(0); i < mesh()->num_faces(); ++i) {
      for (int j = 0; j < 3; ++j) {
        const int32_t index = mesh()->face(i)[j].value();
        const int32_t delta = index - previous;
        symbols.push_back((uint32_t(std::abs(delta)) << 1) | uint32_t(delta < 0));
        previous = index;
      }
    }
    draco::Options options;
    options.SetInt("symbol_encoding_method", draco::SYMBOL_CODING_RAW);
    if (!draco::EncodeSymbols(symbols.data(), symbols.size(), 1, &options, buffer()))
      return draco::Status(draco::Status::DRACO_ERROR, "forced raw entropy encoding failed");
    return draco::OkStatus();
  }
};

void encode(const std::string &directory, const std::string &name,
            uint32_t points, const std::vector<Face> &faces,
            const std::vector<Position> &positions, int submethod = -1,
            bool compressed = false, bool seams = false, bool metadata = false,
            bool attribute_compression = true, bool force_raw = false) {
  draco::Mesh mesh;
  mesh.set_num_points(points);
  for (const auto &f : faces) {
    draco::Mesh::Face face;
    for (int j = 0; j < 3; ++j) face[j] = draco::PointIndex(f[j]);
    mesh.AddFace(face);
  }
  if (!positions.empty()) {
    draco::GeometryAttribute attribute;
    attribute.Init(draco::GeometryAttribute::POSITION, nullptr, 3,
                   draco::DT_FLOAT32, false, sizeof(Position), 0);
    const int id = mesh.AddAttribute(attribute, true, points);
    for (uint32_t i = 0; i < points; ++i)
      mesh.attribute(id)->SetAttributeValue(draco::AttributeValueIndex(i), positions[i].data());
    if (seams) {
      attribute.Init(draco::GeometryAttribute::TEX_COORD, nullptr, 2,
                     draco::DT_FLOAT32, false, 2 * sizeof(float), 0);
      const int uv = mesh.AddAttribute(attribute, true, points);
      for (uint32_t i = 0; i < points; ++i) {
        const float value[2] = {float(i % 7), float(i / 7)};
        mesh.attribute(uv)->SetAttributeValue(draco::AttributeValueIndex(i), value);
      }
    }
  }
  if (metadata) {
    auto meta = std::unique_ptr<draco::GeometryMetadata>(new draco::GeometryMetadata());
    std::vector<uint8_t> long_value(130);
    for (int i = 0; i < 130; ++i) long_value[i] = uint8_t(i);
    meta->AddEntryBinary("long", long_value);
    auto child = std::unique_ptr<draco::Metadata>(new draco::Metadata());
    child->AddEntryString("note", "child");
    meta->AddSubMetadata("nested", std::move(child));
    mesh.AddMetadata(std::move(meta));
    auto attr = std::unique_ptr<draco::AttributeMetadata>(new draco::AttributeMetadata());
    attr->AddEntryString("semantic", "position");
    mesh.AddAttributeMetadata(0, std::move(attr));
  }
  if (seams) mesh.DeduplicateAttributeValues();
  draco::ExpertEncoder encoder(mesh);
  auto options = draco::EncoderOptions::CreateDefaultOptions();
  options.SetGlobalBool("compress_connectivity", compressed);
  if (submethod >= 0) options.SetGlobalInt("edgebreaker_method", submethod);
  encoder.Reset(options);
  encoder.SetEncodingMethod(submethod < 0 ? draco::MESH_SEQUENTIAL_ENCODING
                                         : draco::MESH_EDGEBREAKER_ENCODING);
  encoder.SetSpeedOptions(5, 5);
  if (submethod >= 0) encoder.SetEncodingSubmethod(submethod);
  if (!positions.empty()) encoder.SetAttributeQuantization(0, 12);
  if (seams) encoder.SetAttributeQuantization(1, 10);
  encoder.SetUseBuiltInAttributeCompression(attribute_compression);
  draco::EncoderBuffer buffer;
  RawSequentialEncoder raw_encoder;
  raw_encoder.SetMesh(mesh);
  const auto status = force_raw ? raw_encoder.Encode(encoder.options(), &buffer)
                                : encoder.EncodeToBuffer(&buffer);
  if (!status.ok()) {
    std::cerr << name << ": " << status.error_msg() << '\n';
    std::exit(2);
  }
  std::ofstream output(directory + "/" + name + ".drc", std::ios::binary);
  output.write(buffer.data(), buffer.size());
  if (!output.good()) std::exit(3);
  std::cout << name << '\n';
}

int main(int argc, char **argv) {
  if (argc != 2) return 1;
  const std::string directory(argv[1]);
  for (const auto &spec : std::vector<std::pair<std::string, uint32_t>>{
           {"sequential-u8", 255}, {"sequential-u16", 65535},
           {"sequential-varint", 2097151}, {"sequential-u32", 2097152}}) {
    const uint32_t n = spec.second;
    encode(directory, spec.first, n,
           {{0, n / 2, n - 1}, {n - 1, 1, 0}, {n / 2, 0, 1}}, {});
  }
  std::vector<Face> raw, tagged;
  uint32_t state = 42;
  for (int i = 0; i < 64; ++i) {
    if (i < 3) raw.push_back({0, 1, 2});
    Face f;
    for (auto &v : f) {
      state = state * 1664525u + 1013904223u;
      v = (state >> 8) & 131071;
    }
    tagged.push_back(f);
  }
  encode(directory, "sequential-compressed-raw", 3, raw, {}, -1, true);
  encode(directory, "reject-too-compressed", 3, std::vector<Face>(64, {0, 1, 2}), {}, -1, true);
  encode(directory, "sequential-compressed-tagged", 131072, tagged, {}, -1, true);
  // More than 2^11 unique deltas plus a dominant small delta require wider
  // rANS precision and three-byte frequency tokens. Uncompressed attributes
  // keep this encoder stream above the decoder's remaining-size face bound.
  std::vector<Face> wide;
  std::vector<Position> wide_positions;
  for (uint32_t i = 0; i < 2050; ++i) {
    wide_positions.push_back({float(i), float(i % 31), float(i % 127)});
    if (i < 2048) wide.push_back({0, i + 1, i + 2});
  }
  wide.insert(wide.end(), 1024, {0, 1, 2});
  encode(directory, "sequential-compressed-wide", 2050, wide, wide_positions,
         -1, true, false, false, false, true);
  const std::vector<Position> tetra = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const std::vector<Face> closed = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
  encode(directory, "metadata-long", 4, closed, tetra, -1, false, false, true);
  for (int method : {0, 2}) {
    const std::string prefix = method == 0 ? "standard-" : "valence-";
    encode(directory, prefix + "closed", 4, closed, tetra, method);
    std::vector<Face> grid;
    std::vector<Position> positions;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) positions.push_back({float(x), float(y), 0});
    for (uint32_t y = 0; y < 7; ++y)
      for (uint32_t x = 0; x < 7; ++x) {
        const uint32_t a = y * 8 + x;
        grid.push_back({a, a + 1, a + 8});
        grid.push_back({a + 1, a + 9, a + 8});
      }
    // A disconnected component exercises multiple start faces.
    positions.insert(positions.end(), {{20, 0, 0}, {21, 0, 0}, {20, 1, 0}});
    grid.push_back({64, 65, 66});
    encode(directory, prefix + "boundary", positions.size(), grid, positions, method);
    std::vector<Face> torus;
    positions.clear();
    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 8; ++x) {
        const float u = x * 6.283185307179586f / 8;
        const float v = y * 6.283185307179586f / 6;
        positions.push_back({(3 + std::cos(v)) * std::cos(u),
                             (3 + std::cos(v)) * std::sin(u), std::sin(v)});
      }
    for (uint32_t y = 0; y < 6; ++y)
      for (uint32_t x = 0; x < 8; ++x) {
        const uint32_t a = y * 8 + x, b = y * 8 + (x + 1) % 8;
        const uint32_t c = ((y + 1) % 6) * 8 + x, d = ((y + 1) % 6) * 8 + (x + 1) % 8;
        torus.push_back({a, b, c});
        torus.push_back({b, d, c});
      }
    encode(directory, prefix + "topology", positions.size(), torus, positions, method);
    // Duplicate the corners of two faces at identical positions with distinct UVs.
    // Edgebreaker joins position vertices but retains the UV seam connectivity.
    const std::vector<Position> seam_positions = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    encode(directory, prefix + "seams", 6, {{0, 1, 2}, {3, 4, 5}}, seam_positions,
           method, false, true);
  }
}
