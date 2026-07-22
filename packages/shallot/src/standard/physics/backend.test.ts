import { afterAll, beforeEach, describe, expect, test } from "bun:test";
import { installBackend, Physics, type PhysicsBackend, uninstallBackend } from "./index";

// The substrate's single-backend guard (specs/tumble-shallot.md "Locked decision"): a scene runs exactly
// one physics backend at a time, so a second install must fail loud rather than silently clobbering the
// first. No device or state needed — installBackend/uninstallBackend are pure over the module singleton.

function fakeBackend(): PhysicsBackend {
    return {
        step() {},
        readBody: () => null,
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

describe("physics backend install guard", () => {
    beforeEach(() => {
        uninstallBackend();
    });

    // `Physics.backend` is a module singleton, not scoped to this file — the last test here would
    // otherwise leave a fake handle installed, which fails the next test FILE's own installBackend
    // call (alphabetically, standard/tumble's).
    afterAll(() => {
        uninstallBackend();
    });

    test("installBackend sets Physics.backend to the handle", () => {
        const handle = fakeBackend();
        installBackend(handle);
        expect(Physics.backend).toBe(handle);
    });

    test("installBackend throws when a backend is already installed", () => {
        installBackend(fakeBackend());
        expect(() => installBackend(fakeBackend())).toThrow();
    });

    test("uninstallBackend clears the handle, allowing a fresh install", () => {
        installBackend(fakeBackend());
        uninstallBackend();
        expect(Physics.backend).toBeNull();
        const handle = fakeBackend();
        expect(() => installBackend(handle)).not.toThrow();
        expect(Physics.backend).toBe(handle);
    });
});
