import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { emptyQueryHit } from "@dylanebert/shallot/standard/physics";
import { BenchmarkVillageState, benchmarkVillageDrawing } from "../src/benchmark-village";
import { changeControl, controlValues } from "../src/scenes";

test("Village Benchmark draw controls retain casts while hiding their presentation", () => {
    const world = new World();
    const s = world.resource(BenchmarkVillageState);
    s.translation = { x: 0, y: -12, z: 0 };
    const result = emptyQueryHit();
    result.fraction = 0.5;
    result.point = { x: 1, y: 4, z: 2 };
    result.normal = { x: 0, y: 1, z: 0 };
    s.casts.push({ origin: { x: 1, y: 10, z: 2 }, hit: true, result });
    expect(benchmarkVillageDrawing(world)).toEqual([
        [1, 1, 10, 2, 1, 4, 2],
        [0, 1, 4, 2, 4],
        [1, 1, 4, 2, 1, 4.5, 2],
        [4, 1, 4, 2, 0, 0, 0, 1, 0, 0, 0, 0.05],
    ]);
    const values = controlValues("benchmarkVillage");
    changeControl(world, "benchmarkVillage", [], values, "Draw Spheres", 0);
    expect(benchmarkVillageDrawing(world)).toHaveLength(3);
    changeControl(world, "benchmarkVillage", [], values, "Draw Casts", 0);
    expect(benchmarkVillageDrawing(world)).toEqual([]);
    expect(s.casts).toHaveLength(1);
    changeControl(world, "benchmarkVillage", [], values, "Draw Spheres", 1);
    expect(benchmarkVillageDrawing(world)).toEqual([]);
    changeControl(world, "benchmarkVillage", [], values, "Draw Casts", 1);
    expect(benchmarkVillageDrawing(world)).toHaveLength(4);
});
