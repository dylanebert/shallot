// Generation-only oracle. The pinned Draco library supplies both outputs and paths.
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "draco/compression/decode.h"
#include "draco/mesh/mesh.h"
#include "records_trace.h"

int main(int argc, char **argv) {
  if (argc != 3) return 1;
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
}
