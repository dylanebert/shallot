import { expect, test } from "bun:test";
import { COLORS } from "../../examples/city/src/city";
import { layout, POPULATION, SEED } from "../../examples/city/src/layout";
import { refuseReport, WORKLOAD } from "./workload";

test("the record doubles from 4096 through a million parts and declares the example kit", () => {
    expect(WORKLOAD.seed).toBe(SEED);
    expect(WORKLOAD.populations).toEqual(Array.from({ length: 9 }, (_, i) => 4096 * 2 ** i));
    expect(WORKLOAD.populations).toContain(POPULATION);
    const placements = layout(WORKLOAD.seed, 4096);
    expect(placements.length).toBe(4096);
    expect(new Set(placements.map((p) => p.part)).size).toBe(WORKLOAD.kit.meshes);
    expect(WORKLOAD.kit.materials).toBe(COLORS.length);
    expect([WORKLOAD.width, WORKLOAD.height]).toEqual([1280, 720]);
    expect(WORKLOAD.configurations.map((c) => c.population)).toEqual(WORKLOAD.populations);
    expect(WORKLOAD.shadowMapsEnabled).toBe(false);
});

test("every street camera looks along a gap between building columns, not through a footprint", () => {
    for (const { population, poses } of WORKLOAD.configurations) {
        expect(poses.map((p) => p.name)).toEqual(["whole-city", "street"]);
        const street = poses[1]!;
        const [x, y, z] = street.translation;
        const side = Math.ceil(Math.sqrt((population - 1) / 5));
        const columns = Array.from({ length: side }, (_, i) => (i - (side - 1) / 2) * 12);
        // Layout blocks are at most eight units wide; roof caps add half a unit.
        expect(Math.min(...columns.map((column) => Math.abs(column - x!)))).toBe(6);
        expect(y).toBe(2);
        expect(z).toBe(0);
        expect(street.target).toEqual([x, y, -100]);
    }
});

test("reporting rejects enabled shadows, any shadow pass and either population mismatch", () => {
    expect(() => refuseReport(false, 10, 10, ["main", "cull"])).not.toThrow();
    expect(() => refuseReport(true, 10, 10, [])).toThrow("shadow-enabled");
    for (const label of ["standard-shadow", "standard-pointshadow", "standard-cascadeshadow"])
        expect(() => refuseReport(false, 10, 10, ["main", label])).toThrow("shadow pass");
    for (const instances of [9, 11])
        expect(() => refuseReport(false, 10, instances, [])).toThrow("spawned");
});
