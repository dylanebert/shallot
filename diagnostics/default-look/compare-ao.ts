import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SCENES } from "./scenes";
import { AO_LEVELS } from "./ssao";

export function aoDifference(before: Float32Array, after: Float32Array) {
    if (!before.length || before.length !== after.length)
        throw new Error("AO capture sizes differ or are empty");
    let max = 0;
    let sum = 0;
    for (let i = 0; i < before.length; i++) {
        const difference = Math.abs(before[i]! - after[i]!);
        if (!Number.isFinite(difference)) throw new Error("Non-finite AO capture");
        max = Math.max(max, difference);
        sum += difference;
    }
    return { pixels: before.length, max, mean: sum / before.length };
}

if (import.meta.main) {
    const [before, after] = process.argv.slice(2);
    if (!before || !after)
        throw new Error(
            "Usage: bun diagnostics/default-look/compare-ao.ts <before-directory> <after-directory>",
        );
    const rows = [];
    for (const quality of AO_LEVELS) {
        if (quality === "Off") continue;
        for (const scene of SCENES) {
            const arrays = await Promise.all(
                [before, after].map(async (directory) => {
                    const bytes = await readFile(resolve(directory, quality, `${scene.id}.f32`));
                    return new Float32Array(
                        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
                    );
                }),
            );
            rows.push({ scene: scene.id, quality, ...aoDifference(arrays[0]!, arrays[1]!) });
        }
    }
    console.log(JSON.stringify(rows, null, 2));
}
