import { beforeEach, describe, expect, test } from "bun:test";
import { State } from "../..";
import { clear, register } from "../../engine/ecs/core";
import { Camera } from "../render";
import { Slab } from "../slab";
import { Transform, TransformsPlugin } from "../transforms";
import type { BodyState, PhysicsBackend } from "./index";
import { forwardRay, worldToLocal } from "./pick";

// The live-state pick layer over the pose-agnostic raycast: forwardRay reads the camera Transform, and
// worldToLocal converts a world hit into a held body's local anchor off the backend's live pose. Both must
// honor their contracts — a normalized ray dir, and a null (not a silent snap-to-origin) on a vanished body.

function backend(read: (eid: number) => BodyState | null): PhysicsBackend {
    return {
        step() {},
        readBody: read,
        setKinematic() {},
        setVelocity() {},
        setSprings() {},
        setJoints() {},
        get gravity() {
            return -10;
        },
        get dt() {
            return 1 / 60;
        },
        compose() {},
    };
}

describe("worldToLocal", () => {
    test("returns null when the backend has no live pose for the body", () => {
        // a body that despawned between the cast and the grab — null lets the caller drop the grab rather
        // than pinning the joint to a plausible-looking [0,0,0] local anchor.
        const b = backend(() => null);
        expect(worldToLocal(b, 7, [1, 2, 3])).toBeNull();
    });

    test("expresses the world point in the body's local frame at an identity pose", () => {
        const b = backend(() => ({ pos: [0, 0, 0], quat: [0, 0, 0, 1], vel: [0, 0, 0] }));
        const local = worldToLocal(b, 7, [1, 2, 3])!;
        expect(local).not.toBeNull();
        expect(local[0]).toBeCloseTo(1, 9);
        expect(local[1]).toBeCloseTo(2, 9);
        expect(local[2]).toBeCloseTo(3, 9);
    });
});

describe("forwardRay", () => {
    let state: State;

    beforeEach(() => {
        clear();
        state = new State();
        register("Camera", Camera);
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        Slab.collect();
    });

    test("normalizes the direction even from a non-unit camera quaternion, origin at the camera", () => {
        const cam = state.create();
        state.add(cam, Camera);
        state.add(cam, Transform);
        Transform.pos.set(cam, 1, 2, 3, 1);
        // 2× the 90°-about-Y unit quat: the raw qRotate of −Z is (−4, 0, 3), length 5 — the contract needs
        // it normalized to (−0.8, 0, 0.6).
        const s = Math.SQRT1_2;
        Transform.rot.set(cam, 0, 2 * s, 0, 2 * s);

        const ray = forwardRay(state, cam)!;
        expect(ray).not.toBeNull();
        expect(Math.hypot(ray.dir[0], ray.dir[1], ray.dir[2])).toBeCloseTo(1, 6);
        expect(ray.dir[0]).toBeCloseTo(-0.8, 6);
        expect(ray.dir[1]).toBeCloseTo(0, 6);
        expect(ray.dir[2]).toBeCloseTo(0.6, 6);
        // the origin stays AT the camera (no near-plane offset, unlike cursorRay)
        expect(ray.origin).toEqual([1, 2, 3]);
    });

    test("null for a non-camera or non-transform eid", () => {
        expect(forwardRay(state, -1)).toBeNull();
        const bare = state.create();
        expect(forwardRay(state, bare)).toBeNull();
    });
});
