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
    const grab = section(sample, "\t\t// Ctrl click grabs", "\n\telse if ( modifiers & MOD_SHIFT )")
        .replace(/\t\tPickRay pickRay = m_camera->BuildPickRay\( p.x, p.y \);/, "")
        .replace(/\n\t}\s*$/, "");
    const up = section(sample, "void Sample::MouseUp(", "void Sample::MouseMove(");
    const source = readFileSync(join(import.meta.dir, "sample.cpp"), "utf8")
        .replace("// SAMPLE_GRAB", grab)
        .replace("// SAMPLE_UP", up.slice(up.indexOf("{")))
        .replace("// SAMPLE_SOURCES", sources);
    const path = join(tmpdir(), `box3d-sample-${Bun.hash(source).toString(16)}.cpp`);
    writeFileSync(path, source);
    return nativeBinary(path);
}
