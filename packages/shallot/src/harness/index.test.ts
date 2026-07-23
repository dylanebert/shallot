import { afterEach, describe, expect, test } from "bun:test";
import type { State } from "../engine";
import { Physics } from "../standard/physics";
import { installHarness } from "./index";

// a stub State exposing only what installHarness reads: the elapsed clock + membership check.
const stubState = (elapsed: number, has: (eid: number) => boolean): State =>
    ({ time: { elapsed }, has: (eid: number) => has(eid) }) as unknown as State;

afterEach(() => {
    Physics.backend = null;
});

describe("installHarness", () => {
    test("ready follows the elapsed clock (built + drawn once a frame has stepped)", () => {
        expect(installHarness(stubState(0, () => false)).ready).toBe(false);
        expect(installHarness(stubState(0.016, () => false)).ready).toBe(true);
    });

    test("default run reports a booted pass", async () => {
        const v = await installHarness(stubState(1, () => false)).run!();
        expect(v.ok).toBe(true);
        expect(v.checks).toEqual([{ name: "booted", ok: true }]);
    });

    test("read returns the physics pose for a body, velocity included", () => {
        Physics.backend = {
            readBody: () => ({ pos: [1, 2, 3], quat: [0, 0, 0, 1], vel: [4, 5, 6] }),
        } as unknown as NonNullable<typeof Physics.backend>;
        const pose = installHarness(stubState(1, () => false)).read!(0);
        expect(pose).toEqual({ pos: [1, 2, 3], quat: [0, 0, 0, 1], vel: [4, 5, 6] });
    });

    test("read is null when the eid carries neither a body nor a transform", () => {
        expect(installHarness(stubState(1, () => false)).read!(7)).toBeNull();
    });
});
