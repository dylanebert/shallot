import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { FRAME_UNIFORM_SIZE, FrameGpu, frameWgsl } from "./frame";

// The Frame schema is the one source of truth on both sides: `writeFrame`'s CPU staging writer stages
// against these exact byte offsets, and the emitted WGSL struct resolves from the same schema —
// reordering a field must move the CPU write, not silently shift the WGSL struct out from under it (the
// layout-mismatch class the port exists to kill, the `Step`/`View` precedent).
test("the FrameGpu schema pins the uniform layout the CPU writer stages against", () => {
    expect(d.sizeOf(FrameGpu)).toBe(12);
    const at = (f: keyof (typeof FrameGpu)["propTypes"]) =>
        d.memoryLayoutOf(FrameGpu, (s) => s[f]).offset;
    expect(at("time")).toBe(0);
    expect(at("dt")).toBe(4);
    expect(at("frame")).toBe(8);
});

// Frame's natural size (12 — three 4-byte scalars) is the first schema this port has hit that isn't
// already a multiple of 16: `View`/`Step`/`Lighting` all land on one because their largest member forces
// it. WGSL requires a uniform-address-space struct's alignment to round up to 16, so the real minimum
// binding size is 16, not `d.sizeOf`'s 12 — `FRAME_UNIFORM_SIZE` must apply that rounding, not just
// mirror `d.sizeOf` (the bug this pins: a naive `d.sizeOf(FrameGpu)` under-allocates the GPU buffer).
test("FRAME_UNIFORM_SIZE rounds the schema's natural size up to the 16-byte uniform alignment", () => {
    expect(FRAME_UNIFORM_SIZE).toBe(16);
});

describe("frameWgsl", () => {
    test("emits the Frame struct from the schema", () => {
        const wgsl = frameWgsl();
        expect(wgsl).toContain("struct Frame {");
        expect(wgsl).toContain("time: f32");
        expect(wgsl).toContain("dt: f32");
        expect(wgsl).toContain("frame: u32");
    });

    test("is memoized — repeat calls return the same string", () => {
        expect(frameWgsl()).toBe(frameWgsl());
    });
});
