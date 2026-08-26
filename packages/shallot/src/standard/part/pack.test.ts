import { describe, expect, spyOn, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../tests/wgsl";
import { capacity } from "../../engine";
import { Meshes, Surfaces } from "../render/core";
import { packWgsl } from "./pack";
import { PartTraits } from "./part";

// The pack kernels' device-free structural seam. Real-GPU truth — survivor counts, survivor identity,
// per-view slot offsets — is the gym `render` scenario (`bun bench --scenario render`); what's assertable
// without a device is the emitted WGSL's shape, and that's where the TGSL port's two silent-wrong classes
// live: a literal seeded as `i32` (which flips the arithmetic signed and litters conversions), and a bare
// `/` on integers (which transpiles to *float* division).

const MEMBERSHIP_BASE = 65536;
const MEMBERSHIP_MASK = 4;
const SURFACES = 3;
const wgsl = packWgsl(MEMBERSHIP_BASE, MEMBERSHIP_MASK, SURFACES);
const all = [wgsl.count, wgsl.scan, wgsl.scatter];

describe("integer discipline", () => {
    test("the pack divides nothing — it has no arithmetic TGSL's float `/` could corrupt", () => {
        // the pair index is a multiply-add and the scan walks by addition, so a `/` appearing here at
        // all is the signal, whether it went through `idiv` or (worse) didn't
        for (const src of all) expect(flat(src)).not.toMatch(/[)\w] ?\/ ?/);
    });

    test("every integer local stays u32 — no signed literals or conversions leak in", () => {
        for (const src of all) integerDiscipline(src);
    });
});

describe("shared cull group", () => {
    test("count and scatter emit byte-identical cull declarations and `visible`", () => {
        // the shared-layout convention: one `cullLayout` both kernels reference, so the prefix is not
        // re-declared per kernel and the two gates cannot drift
        expect(body(wgsl.scatter, "fn visible(")).toBe(body(wgsl.count, "fn visible("));
        expect(body(wgsl.scatter, "fn partPair(")).toBe(body(wgsl.count, "fn partPair("));
        for (const name of ["surfaceField", "meshField", "membership", "transforms"]) {
            const decl = new RegExp(`@group\\(0\\) @binding\\(\\d+\\) var<storage, read> ${name}:`);
            expect(wgsl.count).toMatch(decl);
            expect(wgsl.scatter).toMatch(decl);
        }
    });

    test("each kernel's own I/O lands outside the shared cull group", () => {
        expect(wgsl.count).toMatch(/@group\(1\) @binding\(\d+\) var<storage, read_write> counts:/);
        expect(wgsl.scatter).toMatch(
            /@group\(1\) @binding\(\d+\) var<storage, read_write> packedEids:/,
        );
        // the scan shares no cull input, so it declares exactly one group
        expect(wgsl.scan).not.toContain("@group(1)");
        expect(wgsl.scan).not.toContain("cullVolumes");
    });
});

describe("folded constants", () => {
    test("the membership gate, surface count and capacity resolve to literals — no uniform for them", () => {
        for (const src of [wgsl.count, wgsl.scatter]) {
            expect(src).toContain(`membership[(${MEMBERSHIP_BASE}u + eid)] & ${MEMBERSHIP_MASK}u`);
            expect(src).toContain(`eid >= ${capacity}u`);
            expect(src).toContain(`sid >= ${SURFACES}u`);
            expect(src).toContain(`(mid * ${SURFACES}u)`);
        }
        // the pair count is the one late-arriving dimension, so it stays a uniform read
        expect(wgsl.count).toContain("params.pairCount");
    });

    test("the scan writes instanceCount and firstInstance, never the static lanes", () => {
        const main = body(wgsl.scan, "@compute");
        expect(main).toContain("drawArgs[idx].instanceCount");
        expect(main).toContain("drawArgs[idx].firstInstance");
        expect(main).not.toContain("drawArgs[idx].indexCount");
        expect(main).not.toContain("drawArgs[idx].firstIndex");
        expect(main).not.toContain("drawArgs[idx].baseVertex");
        expect(main).toContain(`slot * ${capacity}u`); // the slot's packedEids region base
    });
});

// PartTraits.defaults resolves Surfaces.id("default") and Meshes.id("cube") at entity-creation time.
// A miss (no SearPlugin → "default" unregistered, or PartPlugin not initialized → "cube" unregistered)
// must be loud — warn, matching the sear module's warn+skip idiom (the batch-drop warn in atlas.ts).
// The pre-fix `?? 0` silently bound whatever surface/mesh held registry id 0 — a plausible-wrong resource
// invisible to every green gate. The fix warns at the call site, naming the wiring bug, while still
// returning 0 (the same fallback value) so a valid Part-without-SearPlugin build (the conformance roster)
// doesn't abort.
describe("PartTraits defaults — miss is loud", () => {
    test("defaults() warns when the default surface is unregistered (no SearPlugin)", () => {
        const savedSurfaces = [...Surfaces.values()];
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            // "cube" must be registered (PartPlugin.initialize → initMeshes) so only the surface miss fires
            if (Meshes.id("cube") === undefined) Meshes.register({ name: "cube" } as any);
            Surfaces.clear();
            const result = PartTraits.defaults();
            // the miss is loud: a warning names the wiring bug
            expect(warn).toHaveBeenCalled();
            expect(warn.mock.calls.some((c) => /default/.test(c[0] as string))).toBe(true);
            // the fallback is still 0 (same value as pre-fix, but now warned)
            expect(result.surface).toBe(0);
        } finally {
            Surfaces.clear();
            for (const s of savedSurfaces) Surfaces.register(s);
            warn.mockRestore();
        }
    });

    test("defaults() warns when the cube mesh is unregistered (PartPlugin not initialized)", () => {
        const savedMeshes = [...Meshes.values()];
        const savedSurfaces = [...Surfaces.values()];
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            // "default" must be registered so only the mesh miss fires
            if (Surfaces.id("default") === undefined) Surfaces.register({ name: "default" } as any);
            Meshes.clear();
            const result = PartTraits.defaults();
            expect(warn).toHaveBeenCalled();
            expect(warn.mock.calls.some((c) => /cube/.test(c[0] as string))).toBe(true);
            expect(result.mesh).toBe(0);
        } finally {
            Meshes.clear();
            for (const m of savedMeshes) Meshes.register(m);
            Surfaces.clear();
            for (const s of savedSurfaces) Surfaces.register(s);
            warn.mockRestore();
        }
    });

    test("defaults() does not warn when both are registered", () => {
        const savedSurfaces = [...Surfaces.values()];
        const savedMeshes = [...Meshes.values()];
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            if (Surfaces.id("default") === undefined) Surfaces.register({ name: "default" } as any);
            if (Meshes.id("cube") === undefined) Meshes.register({ name: "cube" } as any);
            PartTraits.defaults();
            expect(warn).not.toHaveBeenCalled();
        } finally {
            Surfaces.clear();
            for (const s of savedSurfaces) Surfaces.register(s);
            Meshes.clear();
            for (const m of savedMeshes) Meshes.register(m);
            warn.mockRestore();
        }
    });
});
