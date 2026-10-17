import { expect, test } from "bun:test";
import { columnSlope, DetailKind, detailCount, detailKind } from "./detail";
import { height } from "./noise";
import { CONFIG } from "./terrain-data";

test("seeded detail pins named columns and places all five kinds within nine chunks", () => {
    for (const [x, z, kind] of [
        [-16, -16, DetailKind.none],
        [-15, -16, DetailKind.grass],
        [8, -13, DetailKind.amber],
        [16, -16, DetailKind.cyan],
        [21, 18, DetailKind.rock],
        [2, -14, DetailKind.tree],
    ] as const)
        expect(detailKind(x, z, CONFIG)).toBe(kind);
    const kinds = new Set<number>();
    for (let cz = -1; cz <= 1; cz++) {
        for (let cx = -1; cx <= 1; cx++) {
            let count = 0;
            for (let z = cz * 16; z < (cz + 1) * 16; z++) {
                for (let x = cx * 16; x < (cx + 1) * 16; x++) {
                    const kind = detailKind(x, z, CONFIG);
                    kinds.add(kind);
                    if (kind !== DetailKind.none) count++;
                    if (kind === DetailKind.tree) {
                        expect(columnSlope(x, z, CONFIG)).toBe(0);
                        // Eight-voxel lattice spacing also holds across chunk boundaries.
                        for (let dz = -7; dz <= 7; dz++)
                            for (let dx = -7; dx <= 7; dx++)
                                if (dx || dz)
                                    expect(detailKind(x + dx, z + dz, CONFIG)).not.toBe(
                                        DetailKind.tree,
                                    );
                    }
                    if (kind === DetailKind.rock)
                        expect(columnSlope(x, z, CONFIG) >= 2 || height(x, z, CONFIG) >= 20).toBe(
                            true,
                        );
                }
            }
            expect(detailCount(cx, cz, CONFIG)).toBe(count);
        }
    }
    expect(kinds).toEqual(new Set(Object.values(DetailKind)));
});
