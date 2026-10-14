import { expect, test } from "bun:test";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import gold from "../../crates/physics/tests/hull_build.gold.json";
import { nativeBinary, run } from "./native";

test("hull construction and every generator match native byte images", async () => {
    const native = JSON.parse(run([nativeBinary("hull-build.c")])) as typeof gold;
    expect(native).toEqual(gold);
    await init(undefined, { threads: 0 });
    const k = kernel(undefined);
    let compared = 0;
    for (const item of native) {
        const count = item.points.length;
        const inputPtr = k.hullBuildInputPtr(count);
        const input = new Float32Array(k.memory.buffer, inputPtr, 10 + 3 * count);
        const words = new Uint32Array(input.buffer, input.byteOffset, input.length);
        for (let i = 0; i < item.args.length; ++i) words[i] = item.args[i];
        for (let i = 0; i < count; ++i) words.set(item.points[i], 10 + 3 * i);
        const ptr = k.hullBuild(item.operation, count, item.budget);
        if (item.words === null) {
            expect(ptr, item.name).toBe(0);
        } else {
            expect(ptr, item.name).not.toBe(0);
            const output = new Uint32Array(k.memory.buffer, ptr, item.words.length);
            expect(Array.from(output), item.name).toEqual(item.words);
            ++compared;
        }
    }
    console.log(JSON.stringify({ cases: native.length, images: compared }));
});
