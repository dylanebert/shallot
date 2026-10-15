import { expect, test } from "bun:test";
import gold from "../../crates/physics/tests/math_validity.gold.json";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import { nativeBinary, run } from "./native";

test("joint quaternion validity equals native Box3D at normalization boundaries and nonfinite inputs", async () => {
    const native = JSON.parse(run([nativeBinary("math-validity.c")])) as typeof gold;
    expect(native).toEqual(gold);
    await init(undefined, { threads: 0 });
    const k = kernel(undefined);
    for (const row of native) {
        const q = new Float32Array(new Uint32Array(row.slice(0, 4)).buffer);
        expect(k.mathQuatIsValid(q[0], q[1], q[2], q[3]), row.join(",")).toBe(row[4]);
    }
});
