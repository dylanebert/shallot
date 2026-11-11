import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
const parser = "crates/basisu/src/lib.rs";
const decoder = "crates/basisu/src/zstd.rs";
const mutations: [string, string, string, string, string][] = [];
function header(name: string, before: string, after: string) {
    mutations.push([name, parser, before, after, "--test headers"]);
}
for (const [name, before, after] of [
    ["identifier", 'range(bytes, 0, 12)? != b"\\xabKTX 20\\xbb\\r\\n\\x1a\\n"', "false"],
    ["vkFormat", "word(bytes, 12)? != 0", "false"],
    ["typeSize", "word(bytes, 16)? != 1", "false"],
    ["width multiple of four", "width % 4 != 0", "false"],
    ["height multiple of four", "height % 4 != 0", "false"],
    ["2D depth", "word(bytes, 28)? != 0", "false"],
    ["2D layers", "word(bytes, 32)? != 0", "false"],
    ["2D faces", "word(bytes, 36)? != 1", "false"],
    ["ETC1S BasisLZ scheme", "(163, 1)", "(163, 2)"],
    ["ETC1S DFD model", "(163, 1)", "(162, 1)"],
    ["UASTC None scheme", "(166, 0 | 2)", "(166, 2)"],
    ["UASTC Zstandard scheme", "(166, 0 | 2)", "(166, 0)"],
    ["UASTC DFD model", "(166, 0 | 2)", "(165, 0 | 2)"],
    ["ETC1S RGB channel", "(Model::Etc1s, [0])", "(Model::Etc1s, [1])"],
    ["ETC1S RGBA channels", "(Model::Etc1s, [0, 15])", "(Model::Etc1s, [0, 14])"],
    ["ETC1S red channel", "(Model::Etc1s, [3])", "(Model::Etc1s, [2])"],
    ["ETC1S red-green channels", "(Model::Etc1s, [3, 4])", "(Model::Etc1s, [3, 5])"],
    ["UASTC RGB channel", "(Model::Uastc, [0])", "(Model::Uastc, [1])"],
    ["UASTC RGBA channel", "(Model::Uastc, [3])", "(Model::Uastc, [2])"],
    ["UASTC red channel", "(Model::Uastc, [4])", "(Model::Uastc, [5])"],
    ["UASTC red-green channel", "(Model::Uastc, [6])", "(Model::Uastc, [7])"],
    ["no premultiplied alpha", "dfd[15] != 0", "false"],
    ["KTXswizzle", 'value != b"rgba"', "false"],
    ["KTXorientation", 'value != b"rd"', "false"],
    ["linear transfer", "1 => Transfer::Linear", "1 => Transfer::Srgb"],
    ["sRGB transfer", "2 => Transfer::Srgb", "2 => Transfer::Linear"],
    ["red and red-green linear", "&& transfer != Transfer::Linear", "&& false"],
    ["texture-role transfer", "self.transfer != transfer", "false"],
] as const) header(name, before, after);
for (const [name, before] of [
    ["raw", "output.extend_from_slice(r.take(size)?);"],
    ["RLE", "output.resize(start + size, r.number(1)? as u8);"],
    ["compressed", "2 => context.block(r.take(size)?, &mut output, limit, window)?,"],
] as const) {
    const after = name === "compressed" ? '2 => return Err("perturbed compressed block"),' : `${before}\noutput.bytes[start] ^= 1;`;
    mutations.push([`Zstandard ${name} block`, decoder, before, after, `--lib zstd::tests::${name === "RLE" ? "rle" : name}_block`]);
}
mutations.push(["Zstandard reserved block refusal", decoder, '_ => return Err("reserved Zstandard block type"),', "_ => output.extend_from_slice(r.take(size)?),", "--lib zstd::tests::reserved_block"]);
mutations.push(["Zstandard unused frame bit ignored", decoder, "flags & 0x08 != 0", "flags & 0x18 != 0", "--lib zstd::tests::unused_frame_bit_matches_reference"]);
mutations.push(["Zstandard reserved frame bit refused", decoder, "flags & 0x08 != 0", "false", "--lib zstd::tests::reserved_frame_bit_is_refused"]);
mutations.push(["Zstandard literal copy bounds", decoder, "length > lit.0.len() - 8", "false", "--lib zstd::tests::literal_copy_uses_only_initialized_padding_and_refuses_overconsumption"]);
mutations.push(["Zstandard literal copy bytes", decoder, "let chunk: [u8; 8] = lit.0[src..src + 8].try_into().unwrap();", "let chunk: [u8; 8] = [0; 8];", "--lib zstd::tests::literal_copy_uses_only_initialized_padding_and_refuses_overconsumption"]);
mutations.push(["Zstandard reverse bit budget", decoder, "self.end - self.consumed", "self.end", "--lib zstd::tests::reverse_lookahead_matches_bits_at_every_word_and_tail_boundary"]);
const originals = new Map([parser, decoder].map((path) => [path, readFileSync(resolve(root, path), "utf8")]));
try {
    for (const [name, path, before, after, selection] of mutations) {
        const original = originals.get(path)!;
        if (original.split(before).length !== 2) throw new Error(`${name}: mutation must match once`);
        writeFileSync(resolve(root, path), original.replace(before, after));
        const result = spawnSync("cargo", ["test", "-p", "shallot-basisu", ...selection.split(" ")], { cwd: root, encoding: "utf8" });
        writeFileSync(resolve(root, path), original);
        if (result.status === 0 || !result.stdout.includes("test result: FAILED")) throw new Error(`${name}: not a test red\n${result.stdout}\n${result.stderr}`);
        console.log(`RED ${name}`);
    }
} finally {
    for (const [path, original] of originals) writeFileSync(resolve(root, path), original);
}
console.log(`${mutations.length} mutants caught; sources restored`);
