import type { World } from "@dylanebert/shallot";
import { BenchmarkVillageState } from "../src/benchmark-village";
import { drawingDigest } from "./digest";
export function benchmarkVillageRecord(world: World): string {
    const s = world.resource(BenchmarkVillageState);
    return drawingDigest([
        [Number(s.drawCasts), Number(s.drawSpheres)],
        [s.translation.x, s.translation.y, s.translation.z],
        ...s.casts.map((c) => [
            c.origin.x,
            c.origin.y,
            c.origin.z,
            Number(c.hit),
            c.hit ? c.result.fraction : 1,
            ...(c.hit
                ? [
                      c.result.point.x,
                      c.result.point.y,
                      c.result.point.z,
                      c.result.normal.x,
                      c.result.normal.y,
                      c.result.normal.z,
                  ]
                : [0, 0, 0, 0, 0, 0]),
        ]),
    ]);
}
