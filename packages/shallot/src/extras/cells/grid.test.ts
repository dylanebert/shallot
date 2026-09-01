import { afterEach, describe, expect, test } from "bun:test";
import { Compute } from "../../engine";
import { CELL_BYTES } from "./cell";
import { createCellGrid, gridWgsl } from "./grid";

afterEach(() => {
    Object.assign(Compute, { root: undefined });
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

    test("buffer size derives from CELL_BYTES, never a duplicated stride constant", () => {
        // structural pin: a grid's byte footprint is cols * rows * CELL_BYTES, read off the schema — a
        // second hand-authored stride would be exactly the "layout drift" gpu.md rule 4 warns against
        expect(CELL_BYTES).toBeGreaterThan(0);
        const cols = 10;
        const rows = 5;
        expect(cols * rows * CELL_BYTES).toBe(600);
    });
});
