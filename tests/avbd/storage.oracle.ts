import { describe, expect, test } from "bun:test";
import { UnsupportedError } from "../../src/engine/runtime/gpu";
import {
    CONTACT_VEC4,
    CONTACTS_PER_PAIR,
    checkContactStore,
    PAIRS_PER_BODY,
} from "../../src/standard/avbd/step";

// the Phase-4.9 device-limit guard: the per-eid contact store is the step's largest single storage binding,
// and at a high capacity it exceeds the WebGPU spec defaults. The guard (a pure size-vs-limit check, so it
// runs with no device) fails loud + clear before an opaque bind-group validation error. These tie the byte
// sizing (the PAIRS_PER_BODY · CONTACTS_PER_PAIR · CONTACT_VEC4 constants) to the limit.
const MB = 1 << 20;
const DEFAULT_BINDING_LIMIT = 128 * MB; // the WebGPU maxStorageBufferBindingSize spec default
const storeBytes = (eidCount: number): number =>
    eidCount * PAIRS_PER_BODY * CONTACTS_PER_PAIR * CONTACT_VEC4 * 16;

describe("physics contact-store device-limit guard", () => {
    test("a full-capacity store exceeds the 128 MB default and fails loud + clear", () => {
        // the case the guard exists for: at 65536 eids the per-eid store is ~235 MB, over the default
        // binding limit — it binds only because acquireDevice now requests the adapter's full size.
        expect(storeBytes(65536)).toBeGreaterThan(DEFAULT_BINDING_LIMIT);
        let caught: unknown;
        try {
            checkContactStore(65536, DEFAULT_BINDING_LIMIT);
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(UnsupportedError);
        // the message names the buffer + the needed-vs-available + a remedy (loud + clear, not a bare throw).
        // The remedy points at `capacity` — the engine sizes the pool to it (Phase 4.9 size-to-capacity).
        const msg = (caught as Error).message;
        expect(msg).toContain("contact store");
        expect(msg).toContain("maxStorageBufferBindingSize");
        expect(msg).toContain("capacity");
    });

    test("a small-capacity store fits under the default", () => {
        // an 8192-eid store → ~28 MB, well under the 128 MB default — a small-capacity scene needs no raised
        // limit (the store sizes to capacity, so this is a capacity ≈ 8192 scene).
        expect(storeBytes(8192)).toBeLessThanOrEqual(DEFAULT_BINDING_LIMIT);
        expect(() => checkContactStore(8192, DEFAULT_BINDING_LIMIT)).not.toThrow();
    });

    test("the boundary is exclusive: exactly at the limit fits, one byte over throws", () => {
        const exact = storeBytes(8192);
        expect(() => checkContactStore(8192, exact)).not.toThrow();
        expect(() => checkContactStore(8192, exact - 1)).toThrow(UnsupportedError);
    });
});
