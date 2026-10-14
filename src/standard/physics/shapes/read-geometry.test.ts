import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/geometry_read.gold.json";
import { computeCapsuleAABB, computeSphereAABB } from "./geometry";
import { getHeightFieldTriangle, type HeightFieldData } from "./heightfield";
import { computeHullAABB, type HullData } from "./hull";
import { computeMeshAABB, getMeshTriangle, type MeshData, safeScale } from "./mesh";

test("safe scale rounds input fields before choosing their signs, like native float fields", () => {
    const expected = new Float32Array(
        new Uint32Array(gold.find((c) => c.operation === 4)!.output).buffer,
    )[0];
    expect(safeScale({ x: -1e-50, y: 1, z: 1 }).x).toBe(expected);
});

for (let operation = 0; operation < 7; ++operation) {
    test(`geometry reader ${operation} materializes native bits`, () => {
        for (const c of gold.filter((c) => c.operation === operation)) {
            const f = new Float32Array(new Uint32Array(c.input).buffer);
            const v = (i: number) => ({ x: f[i], y: f[i + 1], z: f[i + 2] });
            const pose = { p: v(0), q: { v: v(3), s: f[6] } };
            const aabb = { lowerBound: v(8), upperBound: v(11) };
            let actual: number[];
            if (operation < 4) {
                const box =
                    operation === 0
                        ? computeSphereAABB({ center: v(8), radius: f[11] }, pose)
                        : operation === 1
                          ? computeCapsuleAABB(
                                { center1: v(8), center2: v(11), radius: f[14] },
                                pose,
                            )
                          : operation === 2
                            ? computeHullAABB({ aabb } as HullData, pose)
                            : computeMeshAABB({ bounds: aabb } as MeshData, pose, v(14));
                actual = [
                    box.lowerBound.x,
                    box.lowerBound.y,
                    box.lowerBound.z,
                    box.upperBound.x,
                    box.upperBound.y,
                    box.upperBound.z,
                ];
                actual = Array.from(new Uint32Array(new Float32Array(actual).buffer));
            } else if (operation === 4) {
                const scale = safeScale(v(8));
                actual = Array.from(
                    new Uint32Array(new Float32Array([scale.x, scale.y, scale.z]).buffer),
                );
            } else {
                const t =
                    operation === 5
                        ? getMeshTriangle(
                              {
                                  scale: v(21),
                                  data: {
                                      vertices: [v(8), v(11), v(14)],
                                      triangles: [
                                          {
                                              index1: c.input[17],
                                              index2: c.input[18],
                                              index3: c.input[19],
                                          },
                                      ],
                                      flags: [c.input[20]],
                                  } as MeshData,
                              },
                              0,
                          )
                        : getHeightFieldTriangle(
                              {
                                  columnCount: c.input[8],
                                  minHeight: f[10],
                                  heightScale: f[11],
                                  scale: v(12),
                                  clockwise: !!c.input[15],
                                  flags: Array.from({ length: 12 }, () => c.input[16]),
                                  compressedHeights: c.input.slice(32),
                              } as HeightFieldData,
                              c.input[9],
                          );
                actual = Array.from(
                    new Uint32Array(
                        new Float32Array(t.vertices.flatMap((v) => [v.x, v.y, v.z])).buffer,
                    ),
                );
                actual.push(t.i1, t.i2, t.i3, t.flags);
            }
            expect(actual).toEqual(c.output.slice(0, actual.length));
        }
    });
}
