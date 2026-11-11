import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import coverage from "../../crates/draco/tests/records/coverage.json";

// Run without concurrent decoder edits. Each mutant must
// produce a test assertion failure, not a compiler failure, and is restored.
interface Mutation {
    name: string;
    file: string;
    before: string;
    after: string;
    test: string;
    filter: string;
}
const mutations: Mutation[] = [];
function add(name: string, file: string, before: string, after: string, test = "records", filter = "every_recorded_face_list_is_bit_equal_to_the_pinned_decoder") {
    mutations.push({ name, file, before, after, test, filter });
}
function header(name: string, file: string, before: string, after: string, filter: string) {
    add(name, file, before, after, "headers", filter);
}
const refusal = 'if core::hint::black_box(true) { return Err("mutated Draco rule"); }';
header("header.signature", "lib.rs", 'if r.take(5)? != b"DRACO"', 'if { r.take(5)?; false }', "signature_is_draco");
header("header.version", "lib.rs", "if r.byte()? != 2 || r.byte()? != 2", "if { r.byte()?; r.byte()?; false }", "only_bitstream_version_2_2_is_claimed");
header("header.point-cloud", "lib.rs", '0 => return Err("Draco point clouds are not supported"),', "0 => (),", "point_cloud_is_refused_by_name");
header("header.geometry", "lib.rs", '_ => return Err("invalid Draco geometry type"),', "_ => (),", "geometry_type_must_be_known");
header("header.method", "lib.rs", '_ => return Err("unsupported Draco mesh encoding method"),', "_ => Method::Sequential,", "mesh_method_must_be_known");
header("header.predictive", "edgebreaker.rs", '1 => return Err("deprecated Draco predictive Edgebreaker is not supported"),', "1 => false,", "predictive_edgebreaker_is_refused_by_name");
header("header.metadata-varint", "lib.rs", "let size = r.count()?;\n        if size == 0", "let size = usize::from(r.byte()?);\n        if size == 0", "metadata_values_use_varui32_not_the_texts_ui8");
header("header.metadata-zero-value", "lib.rs", "if size == 0 {", "if false {", "zero_length_metadata_value_is_refused_like_the_pinned_decoder");
header("header.unknown-traversal", "edgebreaker.rs", '_ => return Err("unsupported Draco Edgebreaker traversal decoder"),', "_ => false,", "unknown_edgebreaker_traversal_is_refused");
header("header.reserved-flags", "lib.rs", "if flags & 0x8000 != 0", "if flags != 0", "reserved_header_flags_are_ignored_like_the_pinned_decoder");
header("counts.sequential.faces", "lib.rs", "if count != expected_faces {", "if false {", "declared_counts_are_checked_before_connectivity_payload");
header("counts.sequential.points", "lib.rs", "if points as usize != expected_points {", "if false {", "declared_counts_are_checked_before_connectivity_payload");
header("counts.edgebreaker.faces", "edgebreaker.rs", "if faces != expected_faces {", "if false {", "declared_counts_are_checked_before_connectivity_payload");
header("counts.edgebreaker.vertices", "edgebreaker.rs", "if vertices > expected_points {", "if false {", "declared_counts_are_checked_before_connectivity_payload");
header("counts.edgebreaker.seam-points", "edgebreaker.rs", "if count as usize != expected_points {", "if false {", "trusted_point_counts_must_match");

for (const [name, expression] of [
    ["sequential.u8", "u32::from(r.byte()?)"],
    ["sequential.u16", "u32::from(u16::from_le_bytes(r.take(2)?.try_into().unwrap()))"],
    ["sequential.varint", "r.varint(32)? as u32"],
    ["sequential.u32", "u32::from_le_bytes(r.take(4)?.try_into().unwrap())"],
] as const) {
    // Varint also appears in the point count; anchor it to its width branch.
    if (name === "sequential.varint") add(name, "lib.rs", `} else if points < 1 << 21 {\n                    ${expression}`, `} else if points < 1 << 21 {\n                    (${expression}) ^ 1`);
    else add(name, "lib.rs", `                    ${expression}`, `                    (${expression}) ^ 1`);
}
add("sequential", "lib.rs", "Method::Sequential => sequential(&mut r, expected_faces, expected_points),", 'Method::Sequential => Err("mutated Draco sequential decoder"),');
add("sequential.compressed", "lib.rs", "let values = entropy::symbols(r, count * 3, 1)?;", `${refusal}\n        let values = entropy::symbols(r, count * 3, 1)?;`);
add("symbols.tagged", "entropy.rs", "let mut decoder = Symbols::new(r, 5)?;", `${refusal}\n            let mut decoder = Symbols::new(r, 5)?;`);
add("symbols.raw", "entropy.rs", "let mut decoder = Symbols::new(r, width)?;", `${refusal}\n            let mut decoder = Symbols::new(r, width)?;`);
for (let token = 0; token < 4; token++) add(`rans.frequency.token.${token}`, "entropy.rs", "let token = first & 3;", `let token = first & 3;\n            if token == ${token} { ${refusal} }`);
for (const [kind, max, sizes] of [["symbol", 4, 4], ["binary", 3, 3]] as const) {
    for (let size = 1; size <= sizes; size++) add(`rans.${kind}.init.${size - 1}`, "entropy.rs", "let offset = bytes.len() - size;", `if max_bytes == ${max} && size == ${size} { ${refusal} }\n        let offset = bytes.len() - size;`);
}
add("rans.symbol.renormalize", "entropy.rs", "self.value = self.value * 256 + u32::from(self.bytes[self.offset]);", "self.value = self.value * 255 + u32::from(self.bytes[self.offset]);");
for (const value of [true, false]) add(`rans.binary.${value ? "one" : "zero"}`, "entropy.rs", "let value = remainder < p;", `let value = remainder < p;\n        if value == ${value} { ${refusal} }`);
add("edgebreaker.standard", "edgebreaker.rs", "Symbols::Standard(bits) => {", `Symbols::Standard(bits) => { ${refusal}`);
add("edgebreaker.valence", "edgebreaker.rs", "let Some(context) = *active else {", `${refusal}\n                let Some(context) = *active else {`);
for (const symbol of ["C", "L", "E", "R", "S"]) {
    if (symbol === "L" || symbol === "R") add(`topology.${symbol}`, "edgebreaker.rs", "R | L => {", `R | L => { if symbol == ${symbol} { ${refusal} }`);
    else add(`topology.${symbol}`, "edgebreaker.rs", `            ${symbol} => {`, `            ${symbol} => { ${refusal}`);
}
add("topology.split-event", "edgebreaker.rs", "if event.source != encoder_id {\n                    break;\n                }", `if event.source != encoder_id {\n                    break;\n                }\n                ${refusal}`);
add("topology.split-merge", "edgebreaker.rs", "if let Some(&a) = split_active.get(&symbol_id) {", `if let Some(&a) = split_active.get(&symbol_id) { ${refusal}`);
for (const right of [true, false]) add(`topology.split-${right ? "right" : "left"}`, "edgebreaker.rs", "let split_corner = if event.right {", `if event.right == ${right} { ${refusal} }\n                let split_corner = if event.right {`);
add("start.boundary", "edgebreaker.rs", "if !traversal.starts.read()? {\n            continue;\n        }", `if !traversal.starts.read()? {\n            ${refusal}\n            continue;\n        }`);
add("start.interior", "edgebreaker.rs", "if !traversal.starts.read()? {\n            continue;\n        }", `if !traversal.starts.read()? {\n            continue;\n        }\n        ${refusal}`);
for (const seam of [true, false]) add(`seams.${seam ? "yes" : "no"}`, "edgebreaker.rs", "if opp == INVALID || decoder.read()? {", `let seam = opp == INVALID || decoder.read()?;\n            if opp != INVALID && seam == ${seam} { ${refusal} }\n            if seam {`);
for (const boundary of [true, false]) add(`vertices.${boundary ? "boundary" : "interior"}`, "edgebreaker.rs", "if !table.holes[v] {", `if table.holes[v] == ${boundary} { ${refusal} }\n            if !table.holes[v] {`);
add("vertices.remove-invalid", "edgebreaker.rs", "if compact {\n                    push(&mut invalid, n)?;", `if compact {\n                    ${refusal}\n                    push(&mut invalid, n)?;`);
add("vertices.retain-invalid", "edgebreaker.rs", "if compact {\n                    push(&mut invalid, n)?;\n                }", `if compact {\n                    push(&mut invalid, n)?;\n                } else { ${refusal} }`);
for (let context = 0; context < 6; context++) add(`valence.context.${context + 2}`, "edgebreaker.rs", "let id = contexts[context]", `if context == ${context} { ${refusal} }\n                let id = contexts[context]`);

add("attributes.none", "edgebreaker.rs", "if attributes.is_empty() {", `if attributes.is_empty() { ${refusal}`);
add("attributes.connectivity", "edgebreaker.rs", "let mut count = 0u32;", `${refusal}\n        let mut count = 0u32;`);
const missingPaths = Object.keys(coverage).filter((path) => !mutations.some((mutation) => mutation.name === path));
if (missingPaths.length) throw new Error(`recorded decoder paths without mutations: ${missingPaths.join(", ")}`);

const root = resolve(import.meta.dir, "../..");
const only = process.argv.find((arg) => arg.startsWith("--only="))?.slice(7);
const selected = mutations.filter((m) => !only || m.name.startsWith(only));
if (!selected.length) throw new Error(`no mutations match ${only}`);
if (process.argv.includes("--list")) {
    console.log(selected.map((m) => `${m.name}\t${m.file}\t${m.test}:${m.filter}`).join("\n"));
} else if (process.argv.includes("--verify")) {
    const failures = selected.filter((mutation) => readFileSync(resolve(root, "crates/draco/src", mutation.file), "utf8").split(mutation.before).length !== 2);
    if (failures.length) throw new Error(`non-unique source anchors: ${failures.map((m) => m.name).join(", ")}`);
    console.log(`${selected.length} mutation anchors verified`);
} else {
    const baseline = spawnSync("cargo", ["test", "--locked", "-p", "shallot-draco", "--tests"], { cwd: root, encoding: "utf8" });
    if (baseline.status !== 0) throw new Error(`baseline tests must pass before mutations\n${baseline.stdout}${baseline.stderr}`);
    for (const mutation of selected) {
        const path = resolve(root, "crates/draco/src", mutation.file);
        const original = readFileSync(path, "utf8");
        if (original.split(mutation.before).length !== 2) throw new Error(`${mutation.name}: source anchor is not unique`);
        const changed = original.replace(mutation.before, mutation.after);
        writeFileSync(path, changed);
        try {
            const result = spawnSync("cargo", ["test", "--locked", "-p", "shallot-draco", "--test", mutation.test, mutation.filter, "--", "--exact"], { cwd: root, encoding: "utf8" });
            const output = result.stdout + result.stderr;
            if (result.status === 0 || !output.includes("test result: FAILED") || !output.includes("panicked at") || /error\[E\d+\]|could not compile/.test(output)) throw new Error(`${mutation.name}: expected assertion failure, not compile failure/pass\n${output}`);
            console.log(`RED ${mutation.name}`);
        } finally {
            // Never overwrite concurrent source changes.
            if (readFileSync(path, "utf8") !== changed) throw new Error(`${mutation.name}: source changed concurrently; restore the mutation manually`);
            writeFileSync(path, original);
        }
    }
    console.log(`${selected.length} decoder mutations shown red and restored`);
}
