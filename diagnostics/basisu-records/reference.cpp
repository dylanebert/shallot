// Basis Universal 073eb6a0's KTX2 level decompression boundary.
#include "basisu_transcoder.h"
#include "zstd.h"
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) return 1;
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    basist::basisu_transcoder_init();
    basist::ktx2_transcoder decoder;
    if (!decoder.init(bytes.data(), static_cast<uint32_t>(bytes.size()))) return 2;
    auto level = static_cast<uint32_t>(std::stoul(argv[2]));
    if (level >= decoder.get_levels()) return 5;
    const auto &index = decoder.get_level_index()[level];
    const uint8_t *source = bytes.data() + index.m_byte_offset.get_uint64();
    size_t length = index.m_byte_length.get_uint64();
    std::vector<uint8_t> unpacked;
    if (decoder.get_header().m_supercompression_scheme == 2) {
        unpacked.resize(index.m_uncompressed_byte_length.get_uint64());
        size_t actual = ZSTD_decompress(unpacked.data(), unpacked.size(), source, length);
        if (ZSTD_isError(actual) || actual != unpacked.size()) return 3;
        source = unpacked.data(); length = unpacked.size();
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char *>(source), length);
    return output ? 0 : 4;
}
