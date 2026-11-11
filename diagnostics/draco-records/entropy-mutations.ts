import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
const path = resolve(root, "crates/draco/src/entropy.rs");
const original = readFileSync(path, "utf8");
const mutations = [
    ["binary uint8 complement", "u32::from((256 - self.zero) as u8)", "256 - self.zero", "zero_binary_probability_uses_pinned_uint8_complement"],
    ["raw symbols", "values.push(decoder.read());", "values.push(0);", "raw_symbols_match_pinned_encoder"],
    ["tagged symbols", "values.push(bits.read(width)?);", "values.push(bits.read(width)? ^ 1);", "tagged_groups_are_packed_continuously_by_pinned_encoder"],
    ["continuous tagged packing", "values.push(bits.read(width)?);", "values.push(bits.read(width)?);\n                    bits.read((8 - width % 8) % 8)?;", "tagged_groups_are_packed_continuously_by_pinned_encoder"],
    ["rANS symbol lookup", "let symbol = self.lookup[remainder as usize];", "let symbol = 0;", "raw_symbols_match_pinned_encoder"],
    ["rANS probability sum", "cumulative != precision", "false", "malformed_probability_tables_and_states_are_refused"],
    ["rANS state range", "u64::from(value) >= u64::from(base) * 256", "false", "malformed_probability_tables_and_states_are_refused"],
] as const;
try {
    for (const [name, before, after, test] of mutations) {
        if (original.split(before).length !== 2) throw new Error(`${name}: mutation must match once`);
        writeFileSync(path, original.replace(before, after));
        const result = spawnSync("cargo", ["test", "--locked", "-p", "shallot-draco", "--lib", test], { cwd: root, encoding: "utf8" });
        writeFileSync(path, original);
        if (result.status === 0 || !result.stdout.includes("test result: FAILED")) throw new Error(`${name}: expected assertion failure\n${result.stdout}\n${result.stderr}`);
        console.log(`RED ${name}`);
    }
} finally {
    writeFileSync(path, original);
}
console.log(`${mutations.length} entropy mutations failed their tests; source restored`);
