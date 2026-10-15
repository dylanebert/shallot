import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/reader_regressions.gold.json";
import { BodyType, defaultSurfaceMaterial, type SurfaceMaterial } from "../common/types";
import { kernel } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/build-hull";
import { createCompound } from "../shapes/compound";
import { createMesh } from "../shapes/mesh";
import {
    createCompoundShape,
    createHullShape,
    createMeshShape,
    destroyShape,
} from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { hashWorldState } from "../world/hash";
import { ContactField, contactField, contactIds } from "./contact";
import { readContactMaterial } from "./manifoldstore";

test("numeric default material mixing matches native Box3D callback bits", () => {
    const k = kernel(undefined);
    for (const [aBits, bBits, friction, restitution] of gold.materialDefaults) {
        const [a, b] = new Float32Array(new Uint32Array([aBits, bBits]).buffer);
        const result = new Float32Array([
            k.materialMixDefault(0, a, b),
            k.materialMixDefault(1, a, b),
        ]);
        expect(Array.from(new Uint32Array(result.buffer))).toEqual([friction, restitution]);
    }
});

const identity = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const transform = { p: { x: 0, y: 0, z: 0 }, q: identity };
const unitScale = { x: 1, y: 1, z: 1 };

function mesh() {
    return createMesh({
        vertices: [
            { x: -2, y: 0, z: -2 },
            { x: 2, y: 0, z: -2 },
            { x: 2, y: 0, z: 2 },
            { x: -2, y: 0, z: 2 },
        ],
        indices: [0, 2, 1, 0, 3, 2],
        identifyEdges: true,
    });
}

test("custom mixing reads only the active geometry after a mesh slot is recycled as a compound", () => {
    const k = kernel(undefined);
    let mixes = 0;
    const world = createTestWorld({
        enableSleep: false,
        frictionCallback: (a, _ia, b, _ib) => {
            mixes++;
            return k.materialMixDefault(0, a, b);
        },
    });
    try {
        const ground = createTestBody(world, { type: BodyType.Static });
        const meshData = mesh();
        if (!meshData) throw new Error("mesh construction failed");
        const old = createMeshShape(world, ground, testShapeDef(), meshData, unitScale)!;
        const slot = old + 1;
        destroyShape(world, old, true);
        const hull = makeBoxHull(0.5, 0.5, 0.5);
        const compound = createCompound({
            hulls: [{ hull, material: defaultSurfaceMaterial(), transform }],
        })!;
        expect(createCompoundShape(world, ground, testShapeDef(), compound)! + 1).toBe(slot);
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 0.9, z: 0 },
        });
        createHullShape(world, body, testShapeDef(), hull);
        stepTestWorld(world, 1 / 60, 1);
        expect(mixes).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("kernel default mixing equals callback mixing for convex, flipped compound and mesh materials", () => {
    const k = kernel(undefined);
    const frictionCallback = (a: number, _ia: bigint, b: number) => k.materialMixDefault(0, a, b);
    const restitutionCallback = (a: number, _ia: bigint, b: number) =>
        k.materialMixDefault(1, a, b);
    for (const kind of ["hull", "sphere", "capsule", "mesh"] as const) {
        const worlds = [
            {},
            { frictionCallback },
            { restitutionCallback },
            { frictionCallback, restitutionCallback },
        ].map((callbacks) =>
            createTestWorld({
                gravity: { x: 0, y: 0, z: 0 },
                enableSleep: false,
                enableContinuous: false,
                ...callbacks,
            }),
        );
        try {
            for (const world of worlds) {
                world.contactRecycleDistance = 0;
                const material: SurfaceMaterial = {
                    ...defaultSurfaceMaterial(),
                    friction: 0.7,
                    restitution: 0.2,
                    rollingResistance: 0.3,
                    tangentVelocity: [0.4, 0.2, 0.3],
                };
                const hull = makeBoxHull(0.5, 0.5, 0.5);
                const meshData = mesh();
                if (!meshData) throw new Error("mesh construction failed");
                const compound = createCompound(
                    kind === "mesh"
                        ? {
                              meshes: [
                                  {
                                      meshData,
                                      scale: unitScale,
                                      materials: [material],
                                      materialCount: 1,
                                      transform,
                                  },
                              ],
                          }
                        : kind === "hull"
                          ? { hulls: [{ hull, material, transform }] }
                          : kind === "sphere"
                            ? {
                                  spheres: [
                                      {
                                          material,
                                          sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                                      },
                                  ],
                              }
                            : {
                                  capsules: [
                                      {
                                          material,
                                          capsule: {
                                              center1: { x: 0, y: -0.25, z: 0 },
                                              center2: { x: 0, y: 0.25, z: 0 },
                                              radius: 0.5,
                                          },
                                      },
                                  ],
                              },
                );
                if (!compound) throw new Error("compound construction failed");
                const ground = createTestBody(world, { type: BodyType.Static });
                createCompoundShape(world, ground, testShapeDef(), compound);
                const body = createTestBody(world, {
                    type: BodyType.Dynamic,
                    position: { x: 0, y: kind === "mesh" ? 0.4 : 0.8, z: 0 },
                });
                createHullShape(
                    world,
                    body,
                    testShapeDef({
                        baseMaterial: {
                            ...material,
                            friction: 0.4,
                            restitution: 0.5,
                            rollingResistance: 0.6,
                            tangentVelocity: [-0.1, 0.3, 0.2],
                        },
                    }),
                    hull,
                );
            }
            for (let step = 0; step < 4; ++step) {
                for (const world of worlds) stepTestWorld(world, 1 / 60, 1);
                for (const world of worlds.slice(1))
                    expect(hashWorldState(worlds[0]!)).toBe(hashWorldState(world));
                const rows = worlds.map((world) =>
                    contactIds(world)
                        .filter((c) => contactField(world, c, ContactField.manifoldCount) > 0)
                        .map((c) => readContactMaterial(world.manifoldStore.dirF, c)),
                );
                expect(rows[0]!.length).toBeGreaterThan(0);
                for (const row of rows.slice(1)) expect(rows[0]).toEqual(row);
            }
        } finally {
            for (const world of worlds) destroyTestWorld(world);
        }
    }
});
