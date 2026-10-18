import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary } from "./native";

/** Compile the selected upstream classes verbatim, without the samples app's graphics dependencies. */
export function sampleBinary(samples = join(process.env.BOX3D ?? "", "samples")): string {
    const stacking = readFileSync(join(samples, "sample_stacking.cpp"), "utf8");
    const joints = readFileSync(join(samples, "sample_joint.cpp"), "utf8");
    const sample = readFileSync(join(samples, "sample.cpp"), "utf8");
    function section(source: string, start: string, end: string): string {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        if (first < 0 || last < 0) throw new Error(`missing sample source boundary: ${start} / ${end}`);
        return source.slice(first, last);
    }
    const sources = [
        section(sample, "b3BodyId Sample::AddGroundBox(", "struct RowDef"),
        section(stacking, "class BoxStack :", "static int sampleBoxStack"),
        section(joints, "class Bridge :", "static int sampleBridgeIndex"),
    ].join("\n");
    const source = readFileSync(join(import.meta.dir, "sample.cpp"), "utf8").replace(
        "// SAMPLE_SOURCES",
        sources,
    );
    const path = join(tmpdir(), `box3d-sample-${Bun.hash(source).toString(16)}.cpp`);
    writeFileSync(path, source);
    return nativeBinary(path);
}
