import { beforeEach, describe, expect, spyOn, test } from "bun:test";
import {
    build,
    Compute,
    capacity,
    entity,
    f16x4,
    f32,
    type Plugin,
    State,
    slab,
    srgb8x4,
    u8,
    u16,
    u32,
    vec2,
    vec4,
} from "../..";
import { clear, lanes, register } from "../../engine/ecs/core";
import { Slab, SlabPlugin } from "./";
import { elementBytes, scatterWgsl } from "./scatter";

// Pure-CPU slab logic (no device): CPU-storage alloc, set/get + dirty bits, defaults-through-.set,
// and the sub-32-bit warn. The real-GPU scatter flush (dirty slots → the canonical buffer, per type,
// dirty-clear, edit-mode) is the gym `render` scenario's transport round-trip (`bun bench --scenario
// render`) — the single source of truth for anything that binds a device. `Slab.collect()` is the
// device-free CPU-alloc pass `build()` runs over the registry; the `.gpu` mirror is `prepare()` at warm.
// One exception binds a device here: the reset/in-flight-stager lifecycle race, unreachable from the gym.

beforeEach(() => {
    clear();
});

describe("Slab allocation", () => {
    test("a registered slab field is allocated to capacity", () => {
        const Marker: { x: Slab } = { x: new Slab() };
        register("Marker", Marker);
        Slab.collect();
        expect(Marker.x).toBeInstanceOf(Slab);
        expect(Marker.x.array.length).toBe(65536);
        expect(Marker.x.dirty.length).toBe(65536 / 32);
    });

    test("an unregistered slab is not allocated (no component carries it)", () => {
        const orphan = new Slab();
        Slab.collect(); // allocs only slab fields of registered components; a loose handle stays empty
        expect(orphan.array).toBeUndefined();
    });
});

describe("lanes classification", () => {
    // the Slab class declares x/y/z/w fields, so a scalar slab carries them as `undefined`. lanes() must
    // classify by an actual lane handle, not key presence — else a scalar slab reads as a Quad and a
    // reflection walk (readFields / schema) crashes or mis-renders it (Part.surface / Part.mesh).
    test("scalar slab reports 1 lane despite declared x/y/z/w fields", () => {
        expect(lanes(slab(u32))).toBe(1);
        expect(lanes(slab(f32))).toBe(1);
    });

    test("multi-lane slabs report their lane count", () => {
        expect(lanes(slab(vec2))).toBe(2);
        expect(lanes(slab(vec4))).toBe(4);
    });
});

describe("Slab.set / .get", () => {
    test("set writes the slot and marks dirty; get reads back", () => {
        const Marker: { x: Slab } = { x: new Slab() };
        register("Marker", Marker);
        Slab.collect();

        Marker.x.set(7, 1.5);
        expect(Marker.x.get(7)).toBeCloseTo(1.5, 6);
        expect((Marker.x.dirty[0] >>> 7) & 1).toBe(1);

        expect(Marker.x.get(8)).toBe(0);
        expect((Marker.x.dirty[0] >>> 8) & 1).toBe(0);
    });

    test("sparse writes across word boundaries land their bits", () => {
        const Marker: { v: Slab } = { v: new Slab() };
        register("Marker", Marker);
        Slab.collect();

        const eids = [0, 1, 31, 32, 63, 64, 1000, 65000];
        for (const eid of eids) Marker.v.set(eid, eid + 0.25);

        for (const eid of eids) {
            expect(Marker.v.get(eid)).toBeCloseTo(eid + 0.25, 6);
            const word = eid >>> 5;
            const bit = (Marker.v.dirty[word] >>> (eid & 31)) & 1;
            expect(bit).toBe(1);
        }
    });
});

describe("state.add routes defaults through .set", () => {
    test("Traits.defaults flows into slab via .set and marks dirty", () => {
        const Health: { current: Slab; max: Slab } = { current: new Slab(), max: new Slab() };
        register("Health", Health, { defaults: () => ({ current: 100, max: 250 }) });
        Slab.collect();
        const state = new State();

        const eid = state.create();
        state.add(eid, Health);

        expect(Health.current.get(eid)).toBe(100);
        expect(Health.max.get(eid)).toBe(250);
        expect((Health.current.dirty[eid >>> 5] >>> (eid & 31)) & 1).toBe(1);
        expect((Health.max.dirty[eid >>> 5] >>> (eid & 31)) & 1).toBe(1);
    });

    test("absent defaults leave the slot clean", () => {
        const Body: { mass: Slab } = { mass: new Slab() };
        register("Body", Body);
        Slab.collect();
        const state = new State();

        const eid = state.create();
        state.add(eid, Body);

        // no Traits.defaults declared → applyDefaults is a no-op, no .set call
        expect(Body.mass.get(eid)).toBe(0);
        expect((Body.mass.dirty[eid >>> 5] >>> (eid & 31)) & 1).toBe(0);
    });
});

describe("sub-32-bit slabs warn and stay CPU-only", () => {
    // WGSL has no native storage for sub-32-bit integers, so slab(u8)/slab(u16) warn at construction
    // and never allocate a `.gpu` mirror (the scatter flush no-ops them — a device concern, gym-side).
    test("u8 slab warns and has no GPU buffer", () => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            const Tag: { flag: Slab } = { flag: new Slab(u8) }; // warns at construction
            expect(warn).toHaveBeenCalled();
            const msg = String(warn.mock.calls[0][0]);
            expect(msg).toContain("u8");
            expect(msg).toContain("u32");

            register("Tag", Tag);
            Slab.collect();
            Tag.flag.set(5, 200);
            expect(Tag.flag.get(5)).toBe(200);
            expect(Tag.flag.gpuSupported).toBe(false);
            expect(Tag.flag.gpu).toBeNull();
        } finally {
            warn.mockRestore();
        }
    });

    test("u16 slab warns and has no GPU buffer", () => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            const Tag: { code: Slab } = { code: new Slab(u16) };
            expect(warn).toHaveBeenCalled();
            const msg = String(warn.mock.calls[0][0]);
            expect(msg).toContain("u16");

            register("Tag", Tag);
            Slab.collect();
            Tag.code.set(5, 50000);
            expect(Tag.code.get(5)).toBe(50000);
            expect(Tag.code.gpuSupported).toBe(false);
            expect(Tag.code.gpu).toBeNull();
        } finally {
            warn.mockRestore();
        }
    });
});

describe("reset during an in-flight flush", () => {
    // flush re-pools its stager in a mapAsync continuation, so a dispose (Slab.reset) can land
    // while the stager is in flight — it isn't in the pool yet, escapes release(), and a stale
    // re-pool would hand the NEXT build a prior build's buffer (prior size, possibly prior
    // device). The epoch guard in flush's continuation is what this pins.
    test("a stager in flight during reset is destroyed, never re-pooled", async () => {
        clear();
        const Thing = { v: slab(f32) };
        const P: Plugin = { name: "thing", components: { Thing } };
        const app = await build({ plugins: [SlabPlugin, P], defaults: false });
        const eid = app.state.create();
        app.state.add(eid, Thing);
        const pool = (Thing.v as unknown as { _stagingPool: GPUBuffer[] })._stagingPool;
        const settle = async (want: number) => {
            for (let i = 0; i < 20 && pool.length !== want; i++) {
                await new Promise((r) => setTimeout(r, 5));
            }
        };

        // live build: the continuation re-pools the stager — proves the mapAsync path runs here,
        // so the absence assertion below can't pass vacuously
        Thing.v.set(eid, 1);
        app.state.step();
        await settle(1);
        expect(pool.length).toBe(1);

        Thing.v.set(eid, 2);
        app.state.step(); // pops the stager; it's mapAsync-pending again
        app.dispose(); // Slab.reset() — the in-flight stager escapes the pool teardown
        await settle(1);
        expect(pool.length).toBe(0); // destroyed by the epoch guard, not re-pooled stale
    });
});

describe("staging pool allocation", () => {
    // `createStager` (module-private in index.ts) is the site that carries `lazy: true` (index.ts:49) —
    // it fires whenever `Slab.flush()` finds an empty `_stagingPool` for a dirty slab, i.e. real GPU
    // backpressure (no prior stager has mapped back yet). This drives `flush()`'s real branch and reads
    // the descriptor `device.createBuffer` actually received, rather than restating the literal against
    // itself. The pipeline/bind-group fields `prepare()` would normally set are device-work this file
    // doesn't do (see the file header) — stand them in with the minimal fakes `flush()` actually calls,
    // the same bypass-private-fields pattern the "in-flight flush" test above uses for `_stagingPool`.
    test("a stager grown under real backpressure carries the lazy mark", () => {
        const Marker: { v: Slab } = { v: new Slab() };
        register("Marker", Marker);
        Slab.collect();
        Marker.v.set(7, 1.5); // dirties the slot flush() checks

        const s = Marker.v as unknown as {
            gpu: unknown;
            _rawSlots: unknown;
            _rawValues: unknown;
            _bound: { with: (pass: unknown) => { dispatchWorkgroups: (n: number) => void } };
        };
        s.gpu = {};
        s._rawSlots = {};
        s._rawValues = {};
        s._bound = { with: () => ({ dispatchWorkgroups: () => {} }) };

        const created: (GPUBufferDescriptor & { lazy?: boolean })[] = [];
        const stagerBytes = (capacity + 1) * 4 + capacity * 4; // f32 element, matches flush()'s own math
        const fakeStager = {
            getMappedRange: () => new ArrayBuffer(stagerBytes),
            unmap: () => {},
            mapAsync: () => new Promise<void>(() => {}), // never resolves — the test only needs the descriptor
        } as unknown as GPUBuffer;
        const device = {
            createCommandEncoder: () => ({
                copyBufferToBuffer: () => {},
                beginComputePass: () => ({ end: () => {} }),
                finish: () => ({}),
            }),
            createBuffer: (desc: GPUBufferDescriptor) => {
                created.push(desc as GPUBufferDescriptor & { lazy?: boolean });
                return fakeStager;
            },
            queue: { submit: () => {} },
        } as unknown as GPUDevice;
        const prevDevice = Compute.device;
        Object.assign(Compute, { device });

        try {
            Slab.flush();
        } finally {
            Object.assign(Compute, { device: prevDevice });
        }

        expect(created.length).toBe(1);
        expect(created[0].label).toBe("slab-staging");
        expect(created[0].lazy).toBe(true);
    });
});

describe("packed-mirror slabs (srgb8x4 / f16x4)", () => {
    // Two 4-lane formats whose CPU storage stays lossless f32 (set/read/serialize unchanged) while the
    // GPU mirror packs at flush via `type.gpu`: srgb8x4 → one sRGB u32 (4 B), f16x4 → two u32 words of
    // two f16 each, bound as `vec2<u32>` (8 B). The real-GPU flush is the gym `render` transport assert —
    // these pin the device-free CPU-side losslessness + the per-slot word count + the scatter dedup.
    test("srgb8x4 mirrors as one u32 (4 B); f16x4 as a vec2<u32> f16 pair (8 B)", () => {
        const c = slab(srgb8x4);
        const m = slab(f16x4);
        expect(lanes(c)).toBe(4);
        expect(lanes(m)).toBe(4);
        expect((c as unknown as Slab).type.gpu).toEqual({
            wgsl: "u32",
            bytes: 4,
            pack: expect.any(Function),
        });
        expect((m as unknown as Slab).type.gpu).toEqual({
            wgsl: "vec2<u32>",
            bytes: 8,
            pack: expect.any(Function),
        });
    });

    test("CPU storage stays lossless f32 for both — set/read round-trips exactly", () => {
        const C = { rgba: new Slab(srgb8x4) };
        const M = { params: new Slab(f16x4) };
        register("C", C);
        register("M", M);
        Slab.collect();
        C.rgba.set(3, 0.1, 0.2, 0.3, 1); // arbitrary linear color
        M.params.set(5, 0.137, 0.42, 256, 1); // 256 is HDR-range — only the f16 mirror would round it
        const out = new Float32Array(4);
        C.rgba.read(3, out);
        expect([...out]).toEqual([0.1, 0.2, 0.3, 1].map((v) => Math.fround(v)));
        M.params.read(5, out);
        expect([...out]).toEqual([0.137, 0.42, 256, 1].map((v) => Math.fround(v)));
        expect((M.params.dirty[0] >>> 5) & 1).toBe(1); // a write dirties the slot, as for any slab
    });

    test("gpu.pack writes the right word count (srgb8x4 → 1 u32, f16x4 → 2)", () => {
        // the packer the flush calls per dirty slot: srgb8x4 emits one sRGB u32, f16x4 two f16-pair words.
        // These exact words are what the shader's `unpack2x16float` consumes — the packer is the wire format
        const out = new Uint32Array(4);
        srgb8x4.gpu!.pack(out, 0, 1, 1, 1, 1); // white → 0xffffffff
        expect(out[0]).toBe(0xffffffff);
        expect(out[1]).toBe(0); // single word — second slot untouched
        f16x4.gpu!.pack(out, 2, 1, 0, 0, 0); // 1.0 → f16 0x3c00 in the low half of the first word
        expect(out[2] & 0xffff).toBe(0x3c00);
        expect(out[3]).toBe(0); // f16(0) = 0 for the z/w pair
    });

    test("scatter pipelines dedup by GPU element (srgb8x4 + u32 share one; f16x4 its own)", () => {
        const Comp = { col: new Slab(srgb8x4), mat: new Slab(f16x4), id: new Slab(u32) };
        register("Comp", Comp);
        Slab.collect();
        const keys = Slab.gpuTypes()
            .map((t) => t.gpu?.wgsl ?? t.wgsl)
            .sort();
        expect(keys).toEqual(["u32", "vec2<u32>"]); // srgb8x4 + u32 collapse to "u32"; f16x4 separate
    });
});

describe("scatter kernel", () => {
    // the emitted WGSL, resolved with no device (testing.md: structural validation is the `bun test`
    // tier; the kernel actually running is the gym `render` transport round-trip).
    test("the element schema drives every array binding", () => {
        const wgsl = scatterWgsl(vec4);
        expect(wgsl).toContain("var<storage, read> slots: array<u32>");
        expect(wgsl).toContain("var<storage, read> values: array<vec4f>");
        expect(wgsl).toContain("var<storage, read_write> canonical: array<vec4f>");
        expect(wgsl).toContain("canonical[slots[(i + 1u)]] = values[i]");
        // the count guard is what keeps a dispatch's padding threads off the buffer
        expect(wgsl).toContain("if ((i >= count))");
    });

    test("a packed type emits its mirror's element, not its CPU lanes", () => {
        // srgb8x4 is four lossless CPU floats mirrored as one u32 — the kernel copies the mirror
        expect(scatterWgsl(srgb8x4)).toBe(scatterWgsl(u32));
        expect(scatterWgsl(f16x4)).toContain("array<vec2u>");
    });

    test("element bytes match the packer's word count", () => {
        // `Type.gpu.bytes` drives `pack`'s stride into the stager and the schema drives the buffer size;
        // a disagreement would scatter a slot's words into the wrong slot
        expect(elementBytes(srgb8x4)).toBe(srgb8x4.gpu!.bytes);
        expect(elementBytes(f16x4)).toBe(f16x4.gpu!.bytes);
        expect(elementBytes(vec4)).toBe(16);
        expect(elementBytes(f32)).toBe(4);
        // keyed by the WGSL element, not the descriptor's name: `entity` is a u32 that a name-keyed
        // lookup misses, and a slab of one would size its stager off a missing element width
        expect(elementBytes(entity)).toBe(4);
        expect(elementBytes(u8)).toBeNull();
    });
});
