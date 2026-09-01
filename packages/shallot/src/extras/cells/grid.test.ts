import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { Compute } from "../../engine";
import { CELL_BYTES } from "./cell";
import { createCellGrid, fillCellGrid, gridWgsl, resetPipeline } from "./grid";

afterEach(() => {
    Object.assign(Compute, { root: undefined, device: undefined });
});

describe("cell grid compute-pass contract (device-free structural)", () => {
    test("resolves to a compute entry point bound to the dims uniform + the mutable cell array", () => {
        const wgsl = gridWgsl();
        expect(wgsl).toContain("@compute");
        expect(wgsl).toContain("struct GridParams");
        expect(wgsl).toContain("cols: u32");
        expect(wgsl).toContain("rows: u32");
        expect(wgsl).toContain("glyphCount: u32");
        expect(wgsl).toMatch(/var<uniform>/);
        expect(wgsl).toMatch(/var<storage, read_write>/);
        // the fill kernel calls the same packCell body cell.test.ts pins on the CPU — one source, both
        // sides, gpu.md rule 6's lattice-drift property
        expect(wgsl).toContain("fn packCell(");
    });
});

describe("createCellGrid", () => {
    test("allocates cols*rows cells, threads dims through untouched, and requests storage usage", () => {
        const created: { elementCount: number }[] = [];
        const fakeBuffer = {
            $usage(usage: string) {
                (this as { usage?: string }).usage = usage;
                return this;
            },
            $name() {
                return this;
            },
        };
        Object.assign(Compute, {
            root: {
                createBuffer: (arraySchema: { elementCount: number }) => {
                    created.push({ elementCount: arraySchema.elementCount });
                    return fakeBuffer;
                },
            },
        });

        const grid = createCellGrid(80, 24, 95);

        expect(grid.cols).toBe(80);
        expect(grid.rows).toBe(24);
        expect(grid.glyphCount).toBe(95);
        expect(grid.buffer).toBe(fakeBuffer as unknown as typeof grid.buffer);
        expect(created).toEqual([{ elementCount: 80 * 24 }]);
        expect((fakeBuffer as { usage?: string }).usage).toBe("storage");
    });

    test("rejects a glyphCount below 1 — the fill kernel's modulo would be undefined behavior at 0", () => {
        expect(() => createCellGrid(80, 24, 0)).toThrow(/glyphCount/);
    });

    test("CELL_BYTES-derived stride arithmetic matches a hand-computed literal for a fixed 10x5 grid (does not call createCellGrid — see the test above for the buffer actually allocated)", () => {
        // structural pin on CELL_BYTES itself, not on createCellGrid's own buffer sizing (the test above
        // this one already exercises createCellGrid and asserts its elementCount/dims wiring): this only
        // proves cols * rows * CELL_BYTES equals the schema-derived byte count a caller would compute by
        // hand, so a change to CELL_BYTES's own size shows up here even though this arm never allocates.
        expect(CELL_BYTES).toBeGreaterThan(0);
        const cols = 10;
        const rows = 5;
        expect(cols * rows * CELL_BYTES).toBe(600);
    });
});

// fillCellGrid + resetPipeline (device-free structural): a full fake of the WebGPU surface fillCellGrid
// calls (buffer/bind-group/pipeline/pass/submit), so this proves *wiring* — call counts, not readback
// values — with no device. Real dispatch correctness (the actual GPU intrinsic vs. the CPU packCell
// reference) is the `cells` gym scenario's job (`bun bench --scenario cells`), per testing.md's tier
// split; this tier can't and doesn't try to prove that.
function fakeBuffer() {
    const buf = {
        $usage() {
            return buf;
        },
        $name() {
            return buf;
        },
        destroy() {},
    };
    return buf;
}

function fakeRoot(onCreatePipeline: () => void) {
    const bindGroup = {};
    return {
        createBuffer: () => fakeBuffer(),
        createBindGroup: () => bindGroup,
        createComputePipeline: () => {
            onCreatePipeline();
            const p: {
                with: () => typeof p;
                dispatchWorkgroups: () => void;
                $name(): typeof p;
            } = {
                with: () => p,
                dispatchWorkgroups: () => {},
                $name() {
                    return p;
                },
            };
            return p;
        },
    };
}

function fakeDevice() {
    return {
        createCommandEncoder: () => ({
            beginComputePass: () => ({ end: () => {} }),
            finish: () => ({}),
        }),
        queue: { submit: () => {} },
    };
}

describe("fillCellGrid + resetPipeline (device-free structural)", () => {
    // `_pipeline` is grid.ts's own module-singleton memo, shared across every test in the process — each
    // test starts from a known-clean slate rather than leaning on file execution order.
    beforeEach(() => {
        resetPipeline();
    });

    test("memoizes the fill pipeline across dispatches against the same root", () => {
        let calls = 0;
        Object.assign(Compute, { root: fakeRoot(() => calls++), device: fakeDevice() });
        const grid = createCellGrid(4, 4, 3);

        fillCellGrid(grid);
        fillCellGrid(grid);

        expect(calls).toBe(1);
    });

    // The hazard `resetPipeline`'s own docblock names, armed: swapping the adopted root WITHOUT calling
    // `resetPipeline` leaves the memoized pipeline bound to the destroyed root — `pipeline()` returns the
    // stale object instead of building one against the new root, silently.
    test("without resetPipeline, a re-adopted root keeps dispatching against the stale pipeline", () => {
        let calls = 0;
        Object.assign(Compute, { root: fakeRoot(() => calls++), device: fakeDevice() });
        const gridA = createCellGrid(4, 4, 3);
        fillCellGrid(gridA);
        expect(calls).toBe(1);

        // re-adopt: a fresh root + device, no resetPipeline
        Object.assign(Compute, { root: fakeRoot(() => calls++), device: fakeDevice() });
        const gridB = createCellGrid(4, 4, 3);
        fillCellGrid(gridB);

        expect(calls).toBe(1); // stale — the new root's createComputePipeline was never called
    });

    test("resetPipeline forces a fresh pipeline for the re-adopted root", () => {
        let calls = 0;
        Object.assign(Compute, { root: fakeRoot(() => calls++), device: fakeDevice() });
        const gridA = createCellGrid(4, 4, 3);
        fillCellGrid(gridA);
        expect(calls).toBe(1);

        resetPipeline();
        Object.assign(Compute, { root: fakeRoot(() => calls++), device: fakeDevice() });
        const gridB = createCellGrid(4, 4, 3);
        fillCellGrid(gridB);

        expect(calls).toBe(2); // fresh — the new root built its own pipeline
    });
});
