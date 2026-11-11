// Basis Universal 073eb6a0's KTX2 level decompression boundary.
#include "basisu_transcoder.h"
#include "zstd.h"
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) return 1;
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
    if (argc == 5) {
        basist::transcoder_texture_format format;
        const std::string target(argv[4]);
        if (target == "etc1") format = basist::transcoder_texture_format::cTFETC1_RGB;
        else if (target == "etc2-rgba") format = basist::transcoder_texture_format::cTFETC2_RGBA;
        else if (target == "astc") format = basist::transcoder_texture_format::cTFASTC_4x4_RGBA;
        else if (target == "bc7") format = basist::transcoder_texture_format::cTFBC7_RGBA;
        else if (target == "rgba8") format = basist::transcoder_texture_format::cTFRGBA32;
        else return 6;
        if (!decoder.start_transcoding()) return 7;
        basist::ktx2_image_level_info info;
        if (!decoder.get_image_level_info(info, level, 0, 0)) return 8;
        const uint32_t count = target == "rgba8" ? info.m_orig_width * info.m_orig_height : info.m_total_blocks;
        unpacked.resize(count * basist::basis_get_bytes_per_block_or_pixel(format));
        if (!decoder.transcode_image_level(level, 0, 0, unpacked.data(), count, format)) return 9;
        source = unpacked.data(); length = unpacked.size();
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char *>(source), length);
    return output ? 0 : 4;
}
