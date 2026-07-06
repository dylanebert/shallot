import { beforeEach, describe, expect, test } from "bun:test";
import { Camera, f32, Orbit, State, sparse, vec4 } from "@dylanebert/shallot";
import { clear, register } from "@dylanebert/shallot/ecs/core";
import { copyField, syncCameraEffects } from "./camera";

describe("copyField", () => {
    test("copies a typed-array lane", () => {
        const arr = new Float32Array(4);
        arr[0] = 5;
        copyField(arr, 0, 1);
        expect(arr[1]).toBe(5);
    });

    test("copies a plain-array lane", () => {
        const arr = [3, 0];
        copyField(arr, 0, 1);
        expect(arr[1]).toBe(3);
    });

    test("copies through a Single/Pair/Quad {get,set} surface", () => {
        let stored = -1;
        copyField(
            {
                get: (e: number) => (e === 0 ? 9 : stored),
                set: (_e: number, v: number) => {
                    stored = v;
                },
            },
            0,
            1,
        );
        expect(stored).toBe(9);
    });

    test("copies a Quad through all four lanes (a vec4 effect field, e.g. Glaze.slope)", () => {
        const q = sparse(vec4);
        q.set(0, 1, 2, 3, 4);
        copyField(q, 0, 1);
        expect(q.x.get(1)).toBe(1);
        expect(q.y.get(1)).toBe(2);
        expect(q.z.get(1)).toBe(3);
        expect(q.w.get(1)).toBe(4);
    });

    test("no-ops on a null field", () => {
        expect(() => copyField(null, 0, 1)).not.toThrow();
    });
});

// a non-excluded effect component, the kind syncCameraEffects should mirror from scene camera to editor.
// `tint` is a vec4 lane (Glaze.slope/offset/power shape) — the sync must copy every lane, not just a scalar.
const Fx = { value: sparse(f32), tint: sparse(vec4) };

describe("syncCameraEffects", () => {
    beforeEach(() => {
        clear();
        register("camera", Camera);
        register("orbit", Orbit);
        register("fx", Fx);
    });

    test("copies camera fields + effects, preserving the editor camera's own components", () => {
        const s = new State();
        const from = s.create();
        const to = s.create();
        s.add(from, Camera as never);
        s.add(to, Camera as never);
        Camera.fov.set(from, 1.23);

        s.add(from, Fx as never);
        Fx.value.set(from, 7);
        Fx.tint.set(from, 0.1, 0.2, 0.3, 0.4);

        // Orbit is the editor camera's own pose (excluded); present on `to`, absent on `from` — it must
        // survive the sync, or editing a scene camera would strip the editor's orbit controls.
        s.add(to, Orbit as never);

        // clearColor mirrors the scene camera so the edit viewport reads WYSIWYG against play mode —
        // the scene camera's background must overwrite the editor camera's.
        Camera.clearColor.set(from, 0x111111);
        Camera.clearColor.set(to, 0x242019);

        syncCameraEffects(s, from, to);

        expect(Camera.fov.get(to)).toBeCloseTo(1.23);
        expect(s.has(to, Fx as never)).toBe(true);
        expect(Fx.value.get(to)).toBeCloseTo(7);
        expect(Fx.tint.z.get(to)).toBeCloseTo(0.3);
        expect(s.has(to, Orbit as never)).toBe(true);
        expect(Camera.clearColor.get(to)).toBe(0x111111);
    });

    test("removes a non-excluded component the source lacks", () => {
        const s = new State();
        const from = s.create();
        const to = s.create();
        s.add(from, Camera as never);
        s.add(to, Camera as never);
        s.add(to, Fx as never);

        syncCameraEffects(s, from, to);

        expect(s.has(to, Fx as never)).toBe(false);
    });
});
