import { describe, expect, test } from "bun:test";
import { body, flat, IDIV_LEAF, integerDiscipline, noDivision } from "../../../tests/wgsl";
import { Compute, precompile, precompileAll, requestGPU } from "../../engine/runtime";
import { createSceneBounds } from "./bounds";
import { createBuild } from "./build";
import { createBvh } from "./core";
import { createMorton } from "./morton";
import { createRadixSort, radixWgsl } from "./sort";
import { createRadixSortLds } from "./sort-lds";

// The Onesweep arm's real gate is the `accel` gym scenario's `subgroup sort` rows on the device (sorted
// + stable over eleven distributions). Device-free, what's checkable is that the port kept the four
// properties the algorithm's correctness rests on, each of which fails silently if it drifts:
//
//   1. every integer division goes through `idiv` — this kernel divides by the *subgroup size*, so a
//      fractional f32 quotient would corrupt every wave index at every width;
//   2. the tile descriptor is ONE atomic word packing `value<<2 | flag` (gpu.md "the decoupled-scan
//      exception") — per-location coherence is the only guarantee WGSL gives, so a two-word descriptor
//      would be the unsound cross-workgroup handoff this design exists to avoid;
//   3. the lookback's early exit is gated by `workgroupUniformLoad`, which is what makes the in-loop
//      barriers legal, with the subgroup-uniformity diagnostic present at module scope;
//   4. the fallback rescans the *source* keys (prior-dispatch, stable), never the destination.

const wgsl = radixWgsl();
const all = [wgsl.init, wgsl.globalHist, wgsl.scan, wgsl.binning, wgsl.prepare];
const WG = 256;
const RADIX = 256;
const PART_SIZE = 3584;
const FLAG_REDUCTION = 1;
const FLAG_INCLUSIVE = 2;
const FLAG_MASK = 3;

describe("integer discipline", () => {
    test("no kernel divides anything but through `idiv`, and every local stays u32", () => {
        for (const src of all) {
            noDivision(src);
            integerDiscipline(src);
        }
    });

    test("every division is an idiv call — the wave indices and the block counts", () => {
        expect(flat(wgsl.binning)).toContain(IDIV_LEAF);
        // the subgroup-width divides: wave index, wave count, ballot-word count, digit-lane count
        for (const call of ["idiv(tid, sgsize)", "idiv(256u, sgsize)", "idiv((sgsize + 31u), 32u)"])
            expect(flat(wgsl.binning)).toContain(call);
        // and the block counts the prepare derives from the GPU count
        expect(flat(wgsl.prepare)).toContain("idiv((count[0i] + 3583u), 3584u)");
        expect(flat(wgsl.prepare)).toContain("idiv((padded + 32767u), 32768u)");
        // the only bare `/` anywhere is inside the escape leaf itself
        for (const src of all) {
            const withoutLeaf = src.replace(
                "fn idivWgsl(a: u32, b: u32) -> u32 { return a / b; }",
                "",
            );
            expect(flat(withoutLeaf)).not.toMatch(/[)\w] ?\/ ?/);
        }
    });
});

describe("tile descriptors", () => {
    test("the descriptor is one atomic word packing value<<2 | flag", () => {
        // the seed scan writes partition 0's INCLUSIVE descriptor
        expect(flat(wgsl.scan)).toContain(`= ((base << 2u) | ${FLAG_INCLUSIVE}u)`);
        // the binning pass CAS-publishes its own count as a REDUCTION
        expect(flat(wgsl.binning)).toContain(
            `compareExchange((&passHist[succ]), 0u, (${FLAG_REDUCTION}u | (histReduction << 2u)))`,
        );
        // and bumps a successor's REDUCTION to INCLUSIVE with a plain add of the flag delta
        expect(flat(wgsl.binning)).toContain(
            `atomicAdd(&passHist[succ], (${FLAG_REDUCTION}u | (lookbackReduction << 2u)))`,
        );
        // reads mask the flag out of the same word — never a second location
        expect(flat(wgsl.binning)).toContain(`(flagPayload & ${FLAG_MASK}u)`);
        expect(flat(wgsl.binning)).toContain("(flagPayload >> 2u)");
    });

    test("passHist is bound atomic in the binning pass and plain where only one writer exists", () => {
        expect(wgsl.binning).toMatch(/var<storage, read_write> passHist: array<atomic<u32>>/);
        expect(wgsl.scan).toMatch(/var<storage, read_write> passHist: array<u32>/);
        expect(wgsl.init).toMatch(/var<storage, read_write> passHist: array<u32>/);
    });

    test("the CAS goes through the escape leaf, which reads old_value", () => {
        expect(flat(wgsl.binning)).toContain(
            "return atomicCompareExchangeWeak(p, cmp, val).old_value;",
        );
    });
});

describe("lookback with fallback", () => {
    test("the early exit is a workgroupUniformLoad gate, which is what legalizes the in-loop barriers", () => {
        expect(flat(wgsl.binning)).toContain(
            "fn uniformLoad(p: ptr<workgroup,u32>) -> u32 { return workgroupUniformLoad(p); }",
        );
        const main = flat(body(wgsl.binning, "@compute"));
        expect(main).toContain("while (true) { if ((uniformLoad((&wgDone)) != 0u)) { break; }");
        // the gate is a plain (non-atomic) workgroup var — an atomicLoad here is what Tint rejects
        expect(wgsl.binning).toContain("var<workgroup> wgDone: u32;");
        expect(main).toContain("wgDone = 1u;");
    });

    test("the subgroup-uniformity diagnostic is the first thing in the module", () => {
        // WGSL requires every directive ahead of every global declaration, and typegpu emits
        // declarations in first-use order — so this pins the ordering the port depends on, which
        // otherwise fails at pipeline creation with nothing pointing at the cause. The seam is not
        // byte-identical to the shipped module: the pipeline path also prepends `enable subgroups;`
        // from the root's features, so the real module's ordering is `bun bench --scenario accel`
        expect(wgsl.binning.indexOf("diagnostic(off, subgroup_uniformity);")).toBe(0);
        expect(flat(wgsl.binning)).toContain("uniformityOptOut();");
        // it rides a WGSL-bodied fn because `$uses` is unavailable on a transform-produced body, and
        // the two ops it covers are the in-loop reductions
        expect(flat(wgsl.binning)).toContain(
            "fn lookbackAny(x: bool) -> bool { return subgroupAny(x); }",
        );
        expect(wgsl.binning).toContain(
            "fn lookbackAll(x: bool) -> bool { return subgroupAll(x); }",
        );
        // no other kernel carries the opt-out
        for (const src of [wgsl.init, wgsl.globalHist, wgsl.scan, wgsl.prepare])
            expect(src).not.toContain("diagnostic(");
    });

    test("the spin is capped, then the workgroup rescans cooperatively", () => {
        const main = flat(body(wgsl.binning, "@compute"));
        expect(main).toContain("if ((spinCount >= 4u)) { break; }");
        // the fallback reads the SOURCE keys — the prior dispatch's stable input, never the
        // destination this dispatch is still writing
        expect(main).toContain(
            `while ((f < ${PART_SIZE}u)) { atomicAdd(&gD[(${RADIX}u + ((srcKeys[`,
        );
        expect(main).not.toContain("dstKeys[(fbBase");
        // the predecessor it recomputes is the partition one below the one it stalled on
        expect(main).toContain("let fbBase = ((lookbackPart - 1u) * 3584u);");
    });

    test("the loop walks strictly downward, so it terminates at the seeded partition 0", () => {
        const main = flat(body(wgsl.binning, "@compute"));
        expect(main).toContain("var lookbackPart = part;");
        expect(main).toContain("lookbackPart = (lookbackPart - 1u);");
    });

    test("a resolved subgroup bumps the completion counter exactly once, from lane 0", () => {
        const main = flat(body(wgsl.binning, "@compute"));
        expect(main).toContain("warpComplete = lookbackAll(lookbackComplete);");
        expect(main).toContain("if ((warpComplete && (sid == 0u))) { atomicAdd(&gD[0i], 1u); }");
        expect(main).toContain("atomicLoad(&gD[0i]) >= numSub");
    });
});

describe("shared workgroup arena", () => {
    test("gD spans the sorted partition plus the per-digit device base", () => {
        expect(wgsl.binning).toContain(
            `var<workgroup> gD: array<atomic<u32>, ${PART_SIZE + RADIX}>`,
        );
        // the device base for digit `tid` lands past the partition
        expect(flat(wgsl.binning)).toContain(
            `atomicStore(&gD[(tid + ${PART_SIZE}u)], (lookbackReduction - exclusiveHistReduction))`,
        );
    });

    test("the per-thread key, offset, digit and payload arrays are 14 slots each", () => {
        const main = wgsl.binning;
        expect(main.match(/array<u32, 14>\(\)/g)).toHaveLength(4);
        // the ballot needs one word per 32 lanes of the widest subgroup
        expect(main).toContain("array<u32, 4>(4294967295u, 4294967295u, 4294967295u, 4294967295u)");
    });

    test("the multisplit mask seeds u32, not a signed literal", () => {
        // `1 << (sid & 31)` on a bare literal transpiles i32 and overflows at lane 31
        expect(flat(wgsl.binning)).toContain("((1u << (sid & 31u)) - 1u)");
        expect(flat(wgsl.binning)).not.toContain("(1i <<");
    });
});

describe("global histogram", () => {
    test("all four byte positions are counted in one pass over the keys", () => {
        const main = flat(body(wgsl.globalHist, "@compute"));
        for (const sh of [
            "(k & 255u)",
            "((k >> 8u) & 255u)",
            "((k >> 16u) & 255u)",
            "((k >> 24u) & 255u)",
        ])
            expect(main).toContain(sh);
    });

    test("two sub-histograms halve the shared-atomic contention, then reduce to the device", () => {
        expect(wgsl.globalHist).toContain("var<workgroup> gHist: array<atomic<u32>, 2048>");
        expect(flat(wgsl.globalHist)).toContain("let subOff = (idiv(tid, 64u) * 1024u);");
        expect(flat(body(wgsl.globalHist, "@compute"))).toContain("atomicAdd(&globalHist[");
    });

    test("the last tile clamps to numKeys instead of running past the padded end", () => {
        expect(flat(wgsl.globalHist)).toContain(
            "select(((wid.x + 1u) * 32768u), P.numKeys, (wid.x == (P.histBlocks - 1u)))",
        );
    });
});

describe("params", () => {
    test("all four passes' params are written by one prepare dispatch, shift 0/8/16/24", () => {
        const main = flat(body(wgsl.prepare, "@compute"));
        for (const shift of [0, 8, 16, 24])
            expect(main).toContain(`Params(padded, binBlocks, histBlocks, ${shift}u)`);
    });

    test("params ride their own bind group, so the four binning dispatches swap only it", () => {
        expect(wgsl.binning).toMatch(/@group\(0\) @binding\(0\) var<storage, read> P: Params/);
        expect(wgsl.binning).toMatch(/@group\(1\) @binding\(0\) var<storage, read> srcKeys/);
        // the init pass needs no params at all
        expect(wgsl.init).not.toContain("Params");
    });

    test("the params buffer stays storage, not uniform — the prepare writes it on the GPU", () => {
        for (const src of [wgsl.globalHist, wgsl.scan, wgsl.binning])
            expect(src).not.toContain("var<uniform>");
        expect(wgsl.prepare).toMatch(/var<storage, read_write> p0: Params/);
    });

    test("the workgroup size is RADIX so digit-indexed work needs no guard", () => {
        for (const src of [wgsl.init, wgsl.scan, wgsl.binning])
            expect(src).toContain(`@workgroup_size(${WG})`);
        expect(WG).toBe(RADIX);
    });
});

// One app stands up more than one sorter — the `accel` gym scenario runs two arms, each a BVH whose
// builder sorts internally plus a standalone sort beside it — and the precompile queue rejects a
// duplicate label. So the labels are per-instance: the first keeps the bare names (stable profiler
// rows for the single-sorter apps), each later one takes a numbered scope.
describe("per-instance precompile labels", () => {
    // no adapter (testing.md): the sorters are built against a recording stub, which is enough to
    // reach the precompile registrations — nothing here dispatches.
    const stub = (): GPUDevice =>
        ({
            features: new Set(["subgroups"]),
            limits: {},
            queue: { writeBuffer() {} },
            createBuffer: (d: GPUBufferDescriptor) => ({ ...d, destroy() {} }),
        }) as unknown as GPUDevice;

    test("a public factory created after the build drain awaits its scoped fence", async () => {
        const saved = { ...Compute };
        const events: string[] = [];
        let release!: () => void;
        const fence = new Promise<void>((resolve) => {
            release = resolve;
        });
        const device = {
            features: new Set(["subgroups"]),
            limits: {},
            queue: {
                writeBuffer() {},
                onSubmittedWorkDone() {
                    events.push("fence");
                    return fence;
                },
            },
            createBuffer: (d: GPUBufferDescriptor) => ({ ...d, destroy() {} }),
            pushErrorScope: () => events.push("push"),
            popErrorScope: async () => {
                events.push("pop");
                return null;
            },
        } as unknown as GPUDevice;
        const bound = {
            $name() {
                return this;
            },
            with() {
                return this;
            },
            dispatchWorkgroups() {
                events.push("force");
                return this;
            },
        };
        try {
            await requestGPU(device);
            await precompileAll();
            Object.assign(Compute, {
                root: {
                    createComputePipeline: () => bound,
                    createBindGroup: () => ({}),
                },
            });

            let resolved = false;
            const creating = createMorton(device, 8).then((value) => {
                resolved = true;
                return value;
            });
            await Promise.resolve();
            await Promise.resolve();
            expect(resolved).toBe(false);
            expect(events).toEqual(["push", "force", "fence"]);
            release();
            await creating;
            expect(events).toEqual(["push", "force", "fence", "pop"]);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("a second sorter on one device takes a scoped label instead of colliding", async () => {
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);
            const count = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });

            await createRadixSort(device, 1 << 14, { count }, true);
            expect(() => precompile("radix-init", () => true)).toThrow(/duplicate/);

            await createRadixSort(device, 1 << 14, { count }, true);
            expect(() => precompile("radix-2-init", () => true)).toThrow(/duplicate/);
            expect(() => precompile("radix-2-binning", () => true)).toThrow(/duplicate/);

            // the subgroup-free sibling is the same class of factory, on its own scope
            await createRadixSortLds(device, 1 << 14, { count });
            expect(() => precompile("radix-lds-hist", () => true)).toThrow(/duplicate/);
            await createRadixSortLds(device, 1 << 14, { count });
            expect(() => precompile("radix-lds-2-hist", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("a BVH's internal sort keeps the bare label; a sibling sort beside it takes the scoped one", async () => {
        // the reported shape: the `accel` gym scenario's makeArm builds one createBvh (whose builder
        // sorts internally) plus a standalone createRadixSort beside it, on the same device
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);

            await createBvh(device, 1 << 10);
            expect(() => precompile("radix-init", () => true)).toThrow(/duplicate/);
            expect(() => precompile("bounds-reduce", () => true)).toThrow(/duplicate/);
            expect(() => precompile("morton", () => true)).toThrow(/duplicate/);
            expect(() => precompile("build-prepare", () => true)).toThrow(/duplicate/);

            const count = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
            await createRadixSort(device, 1 << 14, { count }, true);
            expect(() => precompile("radix-2-init", () => true)).toThrow(/duplicate/);
            expect(() => precompile("radix-2-binning", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("a second bounds/build/morton instance on one device takes its scoped label", async () => {
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);
            const shared = {
                prims: device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE }),
                bounds: device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE }),
                keys: device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
                payload: device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
                nodes: device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE }),
                count: device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE }),
            };

            await createSceneBounds(device, 1 << 10, shared, true);
            expect(() => precompile("bounds-reduce", () => true)).toThrow(/duplicate/);
            await createSceneBounds(device, 1 << 10, shared, true);
            expect(() => precompile("bounds-2-reduce", () => true)).toThrow(/duplicate/);
            expect(() => precompile("bounds-2-finalize", () => true)).toThrow(/duplicate/);

            await createBuild(device, 1 << 10, shared);
            expect(() => precompile("build-prepare", () => true)).toThrow(/duplicate/);
            await createBuild(device, 1 << 10, shared);
            expect(() => precompile("build-2-prepare", () => true)).toThrow(/duplicate/);
            expect(() => precompile("build-2-sweep", () => true)).toThrow(/duplicate/);

            await createMorton(device, 1 << 10, shared);
            expect(() => precompile("morton", () => true)).toThrow(/duplicate/);
            await createMorton(device, 1 << 10, shared);
            expect(() => precompile("morton-2", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    });
});
