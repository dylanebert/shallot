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
// Attribute paths: each must turn a recorded attribute output red.
function attribute(name: string, file: string, before: string, after: string) {
    add(`attribute.${name}`, file, before, after, "records", "every_recorded_attribute_is_bit_equal_to_the_pinned_decoder");
}
attribute("decoder.generic", "attributes.rs", "values.copy_from_slice(bytes);", "values.copy_from_slice(bytes);\n                    values[0] ^= 1;");
attribute("decoder.integer", "attributes.rs", "out.copy_from_slice(&value.to_le_bytes()[..size]);", "out.copy_from_slice(&(value ^ 1).to_le_bytes()[..size]);");
attribute("decoder.quantization", "attributes.rs", "quantization.min[i % components]", "quantization.min[(i + 1) % components]");
attribute("decoder.normals", "attributes.rs", "tool.unit_vector(st[0], st[1])", "tool.unit_vector(st[1], st[0])");
for (const [name, type] of [["u8", "Uint8"], ["i8", "Int8"], ["u16", "Uint16"], ["i16", "Int16"], ["u32", "Uint32"], ["i32", "Int32"]]) attribute(`integer.${name}`, "attributes.rs", "out.copy_from_slice(&value.to_le_bytes()[..size]);", `if attribute.data_type == DataType::${type} { ${refusal} }\n                        out.copy_from_slice(&value.to_le_bytes()[..size]);`);
attribute("sequencer.linear", "attributes.rs", "ids.extend(0..points as u32);", "ids.extend(0..points.saturating_sub(1) as u32);");
attribute("traversal.depth-first", "attributes.rs", "0 => Order::DepthFirst,", "0 => Order::PredictionDegree,");
attribute("traversal.prediction-degree", "attributes.rs", "1 => Order::PredictionDegree,", "1 => Order::DepthFirst,");
attribute("traversal.corner", "attributes.rs", "let corner = element != 0;", "let corner = element == 7;");
attribute("prediction.none", "attributes.rs", "if method != -2 {", "if method != -3 {");
attribute("prediction.dropped", "attributes.rs", "        (_, 1) => Transform::wrap(),\n        _ => return None,", "        _ => Transform::wrap(),");
attribute("values.compressed", "attributes.rs", "*value = symbol as i32;", "*value = (symbol ^ 2) as i32;");
attribute("values.raw.4", "attributes.rs", "bytes[..width].copy_from_slice(r.take(width)?);", `if width == 4 { ${refusal} }\n            bytes[..width].copy_from_slice(r.take(width)?);`);
attribute("values.raw.narrow", "attributes.rs", "bytes[..width].copy_from_slice(r.take(width)?);", `if width < 4 { ${refusal} }\n            bytes[..width].copy_from_slice(r.take(width)?);`);
attribute("prediction.delta", "prediction.rs", "let mut at = components;", "let mut at = components + components;");
attribute("prediction.parallelogram", "prediction.rs", "if parallelogram(p, corner, mesh, data, components, &mut pred)? {", "if parallelogram(p, corner, mesh, data, components, &mut pred)? && p % 2 == 0 {");
attribute("prediction.parallelogram.fallback", "prediction.rs", "                    } else {\n                        let previous = prior(data, at, components);", `                    } else {\n                        ${refusal}\n                        let previous = prior(data, at, components);`);
attribute("prediction.multi-parallelogram", "prediction.rs", "*pred = pred.wrapping_add(*single);", "*pred = pred.wrapping_sub(*single);");
attribute("prediction.multi-parallelogram.fallback", "prediction.rs", "self.average_or_previous(&mut pred, count, data, p, components)?;", `if count == 0 { ${refusal} }\n                    self.average_or_previous(&mut pred, count, data, p, components)?;`);
attribute("prediction.constrained-multi-parallelogram", "prediction.rs", "if count == 4 {", "if count == 3 {");
attribute("prediction.constrained-multi-parallelogram.crease", "prediction.rs", "if !*self.creases[context]", "if *self.creases[context]");
attribute("prediction.constrained-multi-parallelogram.fallback", "prediction.rs", "self.average_or_previous(&mut pred, used, data, p, components)?;", `if used == 0 { ${refusal} }\n                    self.average_or_previous(&mut pred, used, data, p, components)?;`);
attribute("prediction.tex-coords-deprecated", "prediction.rs", "let pnus = pn_uv[0] * s + n_uv[0];", "let pnus = pn_uv[0] * s - n_uv[0];");
attribute("prediction.tex-coords-portable", "prediction.rs", "let root = int_sqrt(cx_norm.wrapping_mul(pn_norm)) as i64;", "let root = int_sqrt(cx_norm.wrapping_mul(pn_norm)) as i64 + 1;");
attribute("prediction.tex-coords-portable.equal", "prediction.rs", "return Ok([p_uv[0] as i32, p_uv[1] as i32]);", "return Ok([p_uv[1] as i32, p_uv[0] as i32]);");
attribute("prediction.tex-coords-portable.orientation", "prediction.rs", "let orientation = self.pop_orientation()?;", "let orientation = !self.pop_orientation()?;");
attribute("prediction.tex-coords-portable.fallback", "prediction.rs", "let offset = if next < p {", "let offset = if next < p && p % 2 == 0 {");
attribute("prediction.geometric-normal", "prediction.rs", "tool.canonicalize_vector(&mut normal);", "tool.canonicalize_vector(&mut normal);\n                    normal.swap(1, 2);");
attribute("prediction.geometric-normal.flip", "prediction.rs", "normal = normal.map(i32::wrapping_neg);", "normal = normal.map(|v| v);");
attribute("transform.wrap", "prediction.rs", "let (low, high) = (r.i32()?, r.i32()?);", "let (low, high) = (r.i32()?.wrapping_add(1), r.i32()?);");
attribute("transform.wrap.clamp", "prediction.rs", "i64::from(pred.clamp(min, max))", "i64::from(pred)");
attribute("transform.wrap.high", "prediction.rs", "value -= i64::from(max_dif);", "value -= i64::from(max_dif) - 1;");
attribute("transform.wrap.low", "prediction.rs", "value += i64::from(max_dif);", "value += i64::from(max_dif) - 1;");
attribute("transform.octahedron", "prediction.rs", "(pred[0] as u32).wrapping_sub(center) as i32,", "(pred[0] as u32).wrapping_sub(center + 1) as i32,");
attribute("transform.octahedron-canonicalized", "prediction.rs", "pred[0].wrapping_sub(tool.center),", "pred[0].wrapping_sub(tool.center + 1),");
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
