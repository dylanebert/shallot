import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/geometry_read.gold.json";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import { nativeBinary, run } from "./native";

test("shape AABBs, safe scale and triangle decoding equal native Box3D bits", async () => {
    const native = JSON.parse(run([nativeBinary("geometry-read.c")])) as typeof gold;
    expect(native).toEqual(gold);
    await init(undefined, { threads: 0 });
    const k = kernel(undefined);
    for (const [i, c] of native.entries()) {
        const ptr = k.geometryReadInputPtr(c.input.length);
        new Uint32Array(k.memory.buffer, ptr, c.input.length).set(c.input);
        const output = k.geometryRead(c.operation);
        expect(Array.from(new Uint32Array(k.memory.buffer, output, 16)), `case ${i}`).toEqual(c.output);
    }
    console.log(JSON.stringify({ readers: 7, cases: native.length }));
});
