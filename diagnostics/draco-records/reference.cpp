// Generation-only oracle. The pinned Draco library supplies outputs and paths;
// attribute conversions go through the pinned JavaScript wrapper's
// Decoder::GetAttributeDataArrayForAllPoints, which DRACOLoader calls.
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include "draco/compression/decode.h"
#include "draco/javascript/emscripten/decoder_webidl_wrapper.h"
#include "draco/mesh/mesh.h"
#include "records_trace.h"

static bool write(const std::string &path, const void *data, size_t size) {
  std::ofstream output(path, std::ios::binary);
  output.write(static_cast<const char *>(data), size);
  return output.good();
}

int main(int argc, char **argv) {
  if (argc != 4) return 1;
  std::ifstream input(argv[1], std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(input)), {});
  draco::DecoderBuffer buffer;
  buffer.Init(bytes.data(), bytes.size());
  record_size() = bytes.size();
  draco::Decoder decoder;
  auto result = decoder.DecodeMeshFromBuffer(&buffer);
  if (!result.ok()) {
    std::cerr << result.status().error_msg() << '\n';
    return 2;
  }
  const auto &mesh = *result.value();
  std::ofstream output(argv[2], std::ios::binary);
  for (draco::FaceIndex i(0); i < mesh.num_faces(); ++i) {
    for (int j = 0; j < 3; ++j) {
      const uint32_t v = mesh.face(i)[j].value();
      for (int b = 0; b < 4; ++b) output.put((v >> (8 * b)) & 255);
    }
  }
  if (!output.good()) return 3;
  std::cout << mesh.num_faces() << '\t' << record_end() << '\t';
  bool comma = false;
  for (const auto &path : record_paths()) {
    if (comma) std::cout << ',';
    std::cout << path;
    comma = true;
  }
  std::cout << '\t' << mesh.num_points() << '\n';
  // One line per attribute: index, unique id, type, data type, components,
  // normalized, then each output data type whose conversion the pinned
  // wrapper accepts. "<dir>/<index>-<data type>.bin" holds those values.
  const std::string directory(argv[3]);
  for (int i = 0; i < mesh.num_attributes(); ++i) {
    const draco::PointAttribute &pa = *mesh.attribute(i);
    std::cout << i << '\t' << pa.unique_id() << '\t' << pa.attribute_type()
              << '\t' << pa.data_type() << '\t' << int(pa.num_components())
              << '\t' << int(pa.normalized()) << '\t';
    // The decoded type goes through the wrapper too, which refuses the
    // 64-bit and bool types, so it is listed only when accepted.
    for (const draco::DataType type :
         {draco::DT_INT8, draco::DT_UINT8, draco::DT_INT16, draco::DT_UINT16,
          draco::DT_INT32, draco::DT_UINT32, draco::DT_FLOAT32}) {
      const int size = mesh.num_points() * pa.num_components() *
                       draco::DataTypeLength(type);
      std::vector<uint8_t> values(size);
      if (!Decoder::GetAttributeDataArrayForAllPoints(mesh, pa, type, size,
                                                      values.data()))
        continue;
      if (!write(directory + "/" + std::to_string(i) + "-" +
                     std::to_string(type) + ".bin",
                 values.data(), values.size()))
        return 3;
      std::cout << type << ',';
    }
    std::cout << '\n';
  }
}
