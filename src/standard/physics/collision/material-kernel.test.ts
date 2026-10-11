import { expect, test } from "bun:test";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { createCompound } from "../shapes/compound";
import { makeBoxHull } from "../shapes/hull";
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
import { defaultFrictionCallback, defaultRestitutionCallback } from "../world/world";
import { ContactField, contactField, contactIds } from "./contact";
import { readContactMaterial } from "./manifoldstore";

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
    let mixes = 0;
    const world = createTestWorld({
        enableSleep: false,
        frictionCallback: (a, ia, b, ib) => {
            mixes++;
            return defaultFrictionCallback(a, ia, b, ib);
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
    for (const kind of ["hull", "sphere", "capsule", "mesh"] as const) {
        const worlds = [false, true].map((callbacks) =>
            createTestWorld({
                gravity: { x: 0, y: 0, z: 0 },
                enableSleep: false,
                enableContinuous: false,
                ...(callbacks
                    ? {
                          frictionCallback: defaultFrictionCallback,
                          restitutionCallback: defaultRestitutionCallback,
                      }
                    : {}),
            }),
        );
        try {
            for (const world of worlds) {
                world.contactRecycleDistance = 0;
                const material = {
                    ...defaultSurfaceMaterial(),
                    friction: 0.7,
                    restitution: 0.2,
                    rollingResistance: 0.3,
                    tangentVelocity: { x: 0.4, y: 0.2, z: 0.3 },
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
                            tangentVelocity: { x: -0.1, y: 0.3, z: 0.2 },
                        },
                    }),
                    hull,
                );
            }
            for (let step = 0; step < 4; ++step) {
                for (const world of worlds) stepTestWorld(world, 1 / 60, 1);
                expect(hashWorldState(worlds[0]!)).toBe(hashWorldState(worlds[1]!));
                const rows = worlds.map((world) =>
                    contactIds(world)
                        .filter((c) => contactField(world, c, ContactField.manifoldCount) > 0)
                        .map((c) => readContactMaterial(world.manifoldStore.dirF, c)),
                );
                expect(rows[0]!.length).toBeGreaterThan(0);
                expect(rows[0]).toEqual(rows[1]);
            }
        } finally {
            for (const world of worlds) destroyTestWorld(world);
        }
    }
});
