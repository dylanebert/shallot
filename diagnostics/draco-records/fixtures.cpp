// Deterministic encoder-API inputs; no CLI defaults select the missing paths.
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_multi_parallelogram_encoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_encoder.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_delta_encoder.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_encoder_factory.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_encoding_transform.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_wrap_encoding_transform.h"
#include "draco/compression/attributes/sequential_attribute_encoders_controller.h"
#include "draco/compression/attributes/sequential_normal_attribute_encoder.h"
#include "draco/compression/attributes/sequential_quantization_attribute_encoder.h"
#include "draco/compression/expert_encode.h"
#include "draco/compression/mesh/mesh_edgebreaker_encoder.h"
#include "draco/compression/mesh/traverser/depth_first_traverser.h"
#include "draco/compression/mesh/traverser/mesh_attribute_indices_encoding_observer.h"
#include "draco/compression/mesh/traverser/mesh_traversal_sequencer.h"
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

// The pinned encoder still contains the multi-parallelogram and deprecated
// texture-coordinate predictors and the non-canonicalized octahedron
// transform, which the pinned default-build decoder takes for bitstream 2.2,
// but its factories never select them. These protected hooks construct those
// pinned encoder classes; everything else is the unmodified encoder.
// The default transform's type is PREDICTION_TRANSFORM_DELTA, which the
// pinned integer decoder reads but builds no prediction scheme for.
enum class Hook { MultiParallelogram, TexCoordsDeprecated, OctahedronGeometric, OctahedronDelta, DeltaTransform };

using CornerData = draco::MeshPredictionSchemeData<draco::CornerTable>;

CornerData corner_data(const draco::PointCloudEncoder *encoder, int att_id) {
  const auto *mesh_encoder = static_cast<const draco::MeshEncoder *>(encoder);
  const auto *data = mesh_encoder->GetAttributeEncodingData(att_id);
  CornerData md;
  md.Set(mesh_encoder->mesh(), mesh_encoder->GetCornerTable(),
         &data->encoded_attribute_value_index_to_corner_map,
         &data->vertex_to_encoded_attribute_value_index_map);
  return md;
}

class HookedQuantization : public draco::SequentialQuantizationAttributeEncoder {
 public:
  explicit HookedQuantization(Hook hook) : hook_(hook) {}

 protected:
  std::unique_ptr<draco::PredictionSchemeTypedEncoderInterface<int32_t>>
  CreateIntPredictionScheme(draco::PredictionSchemeMethod) override {
    using Transform = draco::PredictionSchemeWrapEncodingTransform<int32_t>;
    const auto *att = encoder()->point_cloud()->attribute(attribute_id());
    const CornerData md = corner_data(encoder(), attribute_id());
    if (hook_ == Hook::DeltaTransform)
      return std::unique_ptr<draco::PredictionSchemeTypedEncoderInterface<int32_t>>(
          new draco::PredictionSchemeDeltaEncoder<
              int32_t, draco::PredictionSchemeEncodingTransform<int32_t, int32_t>>(att));
    if (hook_ == Hook::MultiParallelogram)
      return std::unique_ptr<draco::PredictionSchemeTypedEncoderInterface<int32_t>>(
          new draco::MeshPredictionSchemeMultiParallelogramEncoder<int32_t, Transform, CornerData>(
              att, Transform(), md));
    return std::unique_ptr<draco::PredictionSchemeTypedEncoderInterface<int32_t>>(
        new draco::MeshPredictionSchemeTexCoordsEncoder<int32_t, Transform, CornerData>(
            att, Transform(), md));
  }

 private:
  Hook hook_;
};

class HookedNormals : public draco::SequentialNormalAttributeEncoder {
 public:
  explicit HookedNormals(Hook hook) : hook_(hook) {}

 protected:
  std::unique_ptr<draco::PredictionSchemeTypedEncoderInterface<int32_t>>
  CreateIntPredictionScheme(draco::PredictionSchemeMethod) override {
    using Transform = draco::PredictionSchemeNormalOctahedronEncodingTransform<int32_t>;
    const int32_t bits = encoder()->options()->GetAttributeInt(attribute_id(), "quantization_bits", -1);
    return draco::CreatePredictionSchemeForEncoder<int32_t, Transform>(
        hook_ == Hook::OctahedronGeometric ? draco::MESH_PREDICTION_GEOMETRIC_NORMAL
                                           : draco::PREDICTION_DIFFERENCE,
        attribute_id(), encoder(), Transform((1 << bits) - 1));
  }

 private:
  Hook hook_;
};

class HookedController : public draco::SequentialAttributeEncodersController {
 public:
  HookedController(std::unique_ptr<draco::PointsSequencer> sequencer, int att_id, Hook hook)
      : SequentialAttributeEncodersController(std::move(sequencer), att_id), hook_(hook) {}

 protected:
  std::unique_ptr<draco::SequentialAttributeEncoder> CreateSequentialEncoder(int i) override {
    const auto type = encoder()->point_cloud()->attribute(GetAttributeId(i))->attribute_type();
    if (((hook_ == Hook::MultiParallelogram || hook_ == Hook::DeltaTransform) &&
         type == draco::GeometryAttribute::POSITION) ||
        (hook_ == Hook::TexCoordsDeprecated && type == draco::GeometryAttribute::TEX_COORD))
      return std::unique_ptr<draco::SequentialAttributeEncoder>(new HookedQuantization(hook_));
    if ((hook_ == Hook::OctahedronGeometric || hook_ == Hook::OctahedronDelta) &&
        type == draco::GeometryAttribute::NORMAL)
      return std::unique_ptr<draco::SequentialAttributeEncoder>(new HookedNormals(hook_));
    return SequentialAttributeEncodersController::CreateSequentialEncoder(i);
  }

 private:
  Hook hook_;
};

// Edgebreaker with one depth-first, per-vertex attribute decoder over the
// position connectivity ("split_mesh_on_seams"), whose controller is hooked.
class HookedEdgebreaker : public draco::MeshEdgebreakerEncoder {
 public:
  explicit HookedEdgebreaker(Hook hook) : hook_(hook) {}

 protected:
  bool GenerateAttributesEncoder(int32_t att_id) override {
    if (num_attributes_encoders() > 0) {
      attributes_encoder(0)->AddAttributeId(att_id);
      return true;
    }
    using Observer = draco::MeshAttributeIndicesEncodingObserver<draco::CornerTable>;
    using Traverser = draco::DepthFirstTraverser<draco::CornerTable, Observer>;
    auto *data = const_cast<draco::MeshAttributeIndicesEncodingData *>(GetAttributeEncodingData(att_id));
    std::unique_ptr<draco::MeshTraversalSequencer<Traverser>> sequencer(
        new draco::MeshTraversalSequencer<Traverser>(mesh(), data));
    Observer observer(GetCornerTable(), mesh(), sequencer.get(), data);
    Traverser traverser;
    traverser.Init(GetCornerTable(), observer);
    sequencer->SetTraverser(traverser);
    AddAttributesEncoder(std::unique_ptr<draco::AttributesEncoder>(
        new HookedController(std::move(sequencer), att_id, hook_)));
    return true;
  }
  bool EncodeAttributesEncoderIdentifier(int32_t) override {
    buffer()->Encode(int8_t(-1));
    buffer()->Encode(uint8_t(draco::MESH_VERTEX_ATTRIBUTE));
    buffer()->Encode(uint8_t(draco::MESH_TRAVERSAL_DEPTH_FIRST));
    return true;
  }

 private:
  Hook hook_;
};

void write(const std::string &directory, const std::string &name,
           const draco::EncoderBuffer &buffer) {
  std::ofstream output(directory + "/" + name + ".drc", std::ios::binary);
  output.write(buffer.data(), buffer.size());
  if (!output.good()) std::exit(3);
  std::cout << name << '\n';
}

template <class T>
int add(draco::Mesh &mesh, draco::GeometryAttribute::Type type, draco::DataType data_type,
        int components, bool normalized, const std::vector<T> &values) {
  draco::GeometryAttribute attribute;
  attribute.Init(type, nullptr, components, data_type, normalized, sizeof(T) * components, 0);
  const uint32_t points = mesh.num_points();
  const int id = mesh.AddAttribute(attribute, true, points);
  for (uint32_t i = 0; i < points; ++i)
    mesh.attribute(id)->SetAttributeValue(draco::AttributeValueIndex(i),
                                          &values[(i * components) % values.size()]);
  return id;
}

// A lat/long sphere band: a manifold grid with distinct positions.
void sphere(draco::Mesh &mesh, int columns, int rows) {
  mesh.set_num_points(columns * rows);
  for (int y = 0; y + 1 < rows; ++y)
    for (int x = 0; x < columns; ++x) {
      const uint32_t a = y * columns + x, b = y * columns + (x + 1) % columns;
      draco::Mesh::Face f0, f1;
      f0[0] = draco::PointIndex(a), f0[1] = draco::PointIndex(b), f0[2] = draco::PointIndex(a + columns);
      f1[0] = draco::PointIndex(b), f1[1] = draco::PointIndex(b + columns), f1[2] = draco::PointIndex(a + columns);
      mesh.AddFace(f0);
      mesh.AddFace(f1);
    }
  std::vector<float> positions, normals, uvs;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < columns; ++x) {
      const float u = x * 6.283185307179586f / columns, v = 0.3f + y * 2.5f / rows;
      const float n[3] = {std::sin(v) * std::cos(u), std::sin(v) * std::sin(u), std::cos(v)};
      for (int c = 0; c < 3; ++c) {
        positions.push_back(n[c] * (2 + 0.1f * ((x * 7 + y * 3) % 5)));
        normals.push_back(n[c]);
      }
      uvs.push_back(u * 0.3f + 0.01f * (y % 3));
      uvs.push_back(v * 0.7f);
    }
  add(mesh, draco::GeometryAttribute::POSITION, draco::DT_FLOAT32, 3, false, positions);
  add(mesh, draco::GeometryAttribute::NORMAL, draco::DT_FLOAT32, 3, false, normals);
  add(mesh, draco::GeometryAttribute::TEX_COORD, draco::DT_FLOAT32, 2, false, uvs);
}

void hooked(const std::string &directory, const std::string &name, Hook hook) {
  draco::Mesh mesh;
  sphere(mesh, 12, 7);
  auto options = draco::EncoderOptions::CreateDefaultOptions();
  options.SetGlobalBool("split_mesh_on_seams", true);
  options.SetAttributeInt(0, "quantization_bits", 11);
  options.SetAttributeInt(1, "quantization_bits", 8);
  options.SetAttributeInt(2, "quantization_bits", 10);
  options.SetSpeed(5, 5);
  HookedEdgebreaker encoder(hook);
  encoder.SetMesh(mesh);
  draco::EncoderBuffer buffer;
  const auto status = encoder.Encode(options, &buffer);
  if (!status.ok()) {
    std::cerr << name << ": " << status.error_msg() << '\n';
    std::exit(2);
  }
  write(directory, name, buffer);
}

// Every integer width, normalized and unnormalized, plus the generic raw types.
void typed(const std::string &directory, const std::string &name, bool edgebreaker,
           bool compression) {
  draco::Mesh mesh;
  mesh.set_num_points(4);
  for (const Face &f : std::vector<Face>{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}}) {
    draco::Mesh::Face face;
    for (int j = 0; j < 3; ++j) face[j] = draco::PointIndex(f[j]);
    mesh.AddFace(face);
  }
  using G = draco::GeometryAttribute;
  add<float>(mesh, G::POSITION, draco::DT_FLOAT32, 3, false, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1});
  add<uint8_t>(mesh, G::COLOR, draco::DT_UINT8, 3, true, {0, 255, 128, 7, 200, 1, 64, 0, 255, 254, 3, 90});
  add<int8_t>(mesh, G::GENERIC, draco::DT_INT8, 2, false, {-128, 127, -1, 0, 5, -77, 100, -3});
  add<uint16_t>(mesh, G::GENERIC, draco::DT_UINT16, 1, true, {0, 65535, 32768, 255});
  add<int16_t>(mesh, G::GENERIC, draco::DT_INT16, 4, false,
               {-32768, 32767, -1, 0, 128, -129, 255, 256, 12, -300, 1000, -1000, 7, 8, -9, 10});
  const int u32 = add<uint32_t>(mesh, G::GENERIC, draco::DT_UINT32, 1, false,
                                {0, 16777215u, 300, 65536});
  add<int32_t>(mesh, G::GENERIC, draco::DT_INT32, 2, true,
               {-1000000, 1000000, -1, 1, -65536, 65535, 0, -300});
  add<float>(mesh, G::GENERIC, draco::DT_FLOAT32, 2, true,
             {0.f, 1.f, 0.5f, -0.25f, 1.5f, 0.999f, 0.0001f, 0.25f});
  add<float>(mesh, G::TEX_COORD, draco::DT_FLOAT32, 2, false,
             {-128.5f, 127.f, 255.f, -0.5f, 32767.f, 65535.f, 2147483648.f, 3e9f});
  add<double>(mesh, G::GENERIC, draco::DT_FLOAT64, 1, false, {-1.5, 4294967294.5, 1e300, 0.25});
  add<int64_t>(mesh, G::GENERIC, draco::DT_INT64, 1, false,
               {-9223372036854775807ll - 1, -129, 4294967296ll, 70000});
  add<uint64_t>(mesh, G::GENERIC, draco::DT_UINT64, 1, true, {0, 18446744073709551615ull, 255, 1});
  add<uint8_t>(mesh, G::GENERIC, draco::DT_BOOL, 1, false, {0, 1, 1, 0});
  // Values every narrower output type accepts, so conversions are not all refused.
  add<float>(mesh, G::GENERIC, draco::DT_FLOAT32, 1, true, {0.f, 1.f, 0.5f, 0.0001f});
  add<float>(mesh, G::GENERIC, draco::DT_FLOAT32, 3, false,
             {-3.5f, 100.f, 0.75f, 126.99f, -0.f, 12.f, 1.f, -100.f, 64.5f, 7.f, 0.f, 99.f});
  add<double>(mesh, G::GENERIC, draco::DT_FLOAT64, 2, false, {0.5, 126.9, 3.0, -2.25, 1e-9, 77, 0, 5});
  add<int64_t>(mesh, G::GENERIC, draco::DT_INT64, 1, true, {-5, 0, 100, 127});
  draco::ExpertEncoder encoder(mesh);
  auto options = draco::EncoderOptions::CreateDefaultOptions();
  encoder.Reset(options);
  encoder.SetEncodingMethod(edgebreaker ? draco::MESH_EDGEBREAKER_ENCODING
                                        : draco::MESH_SEQUENTIAL_ENCODING);
  encoder.SetSpeedOptions(5, 5);
  encoder.SetAttributeQuantization(0, 10);
  encoder.SetAttributePredictionScheme(u32, draco::PREDICTION_NONE);
  encoder.SetUseBuiltInAttributeCompression(compression);
  draco::EncoderBuffer buffer;
  const auto status = encoder.EncodeToBuffer(&buffer);
  if (!status.ok()) {
    std::cerr << name << ": " << status.error_msg() << '\n';
    std::exit(2);
  }
  write(directory, name, buffer);
}

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
  typed(directory, "attributes-sequential", false, true);
  typed(directory, "attributes-edgebreaker", true, true);
  typed(directory, "attributes-raw", false, false);
  hooked(directory, "prediction-multi-parallelogram", Hook::MultiParallelogram);
  hooked(directory, "prediction-tex-coords-deprecated", Hook::TexCoordsDeprecated);
  hooked(directory, "normals-octahedron-geometric", Hook::OctahedronGeometric);
  hooked(directory, "normals-octahedron-delta", Hook::OctahedronDelta);
  hooked(directory, "prediction-dropped", Hook::DeltaTransform);
}
