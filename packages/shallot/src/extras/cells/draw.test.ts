import { describe, expect, test } from "bun:test";
import { drawWgsl } from "./draw";

// The draw pipeline's device-free structural seam: `tgpu.resolve` runs no device, so this proves the vs/fs
// pair type-checks and resolves to valid WGSL text with no adapter — the same shape `grid.ts`'s `gridWgsl`
// and `extras/outline`'s `wgslArtifacts` use. Real-device correctness (the instanced draw's actual pixels)
// is `bun bench`/Playwright tier truth (`testing.md`), not this file's.
describe("drawWgsl", () => {
    test("resolves the vertex + fragment pair with no device", () => {
        const wgsl = drawWgsl();
        expect(wgsl).toContain("@vertex");
        expect(wgsl).toContain("@fragment");
    });

    test("the vertex stage reads @builtin(vertex_index) and @builtin(instance_index) — no vertex buffer", () => {
        const wgsl = drawWgsl();
        expect(wgsl).toContain("vertex_index");
        expect(wgsl).toContain("instance_index");
    });
});
