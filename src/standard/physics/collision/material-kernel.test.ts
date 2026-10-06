import { expect, test } from "bun:test";
import {
    BodyType,
    createCompound,
    createMesh,
    defaultSurfaceMaterial,
    hash,
    makeBoxHull,
    PhysicsWorld,
} from "../api";
import { defaultFrictionCallback, defaultRestitutionCallback } from "../world/world";
import { ContactField, contactField, contactIds } from "./contact";
import { readContactMaterial } from "./manifoldstore";

const identity = { v: { x: 0, y: 0, z: 0 }, s: 1 };

test("custom mixing reads only the active geometry after a mesh slot is recycled as a compound", () => {
    let mixes = 0;
    const world = new PhysicsWorld({
        enableSleep: false,
        frictionCallback: (a, ia, b, ib) => {
            mixes++;
            return defaultFrictionCallback(a, ia, b, ib);
        },
    });
    try {
        const ground = world.createBody({ type: BodyType.Static });
        const mesh = createMesh({
            vertices: [
                { x: -2, y: 0, z: -2 },
                { x: 2, y: 0, z: -2 },
                { x: 2, y: 0, z: 2 },
                { x: -2, y: 0, z: 2 },
            ],
            indices: [0, 2, 1, 0, 3, 2],
            identifyEdges: true,
        })!;
        const old = ground.createMesh({}, mesh, { x: 1, y: 1, z: 1 });
        const slot = old.id.index1;
        old.destroy();
        const hull = makeBoxHull(0.5, 0.5, 0.5);
        const compound = createCompound({
            hulls: [
                {
                    hull,
                    material: defaultSurfaceMaterial(),
                    transform: { p: { x: 0, y: 0, z: 0 }, q: identity },
                },
            ],
        })!;
        expect(ground.createCompound({}, compound)!.id.index1).toBe(slot);
        world
            .createBody({ type: BodyType.Dynamic, position: { x: 0, y: 0.9, z: 0 } })
            .createHull({}, hull);
        world.step(1 / 60, 1);
        expect(mixes).toBeGreaterThan(0);
    } finally {
        world.destroy();
    }
});

test("kernel default mixing equals callback mixing for convex, flipped compound and mesh materials", () => {
    for (const kind of ["hull", "sphere", "capsule", "mesh"] as const) {
        const worlds = [false, true].map(
            (callbacks) =>
                new PhysicsWorld({
                    gravity: { x: 0, y: 0, z: 0 },
                    enableSleep: false,
                    enableContinuous: false,
                    ...(callbacks
                        ? {
                              frictionCallback: (a, ia, b, ib) =>
                                  defaultFrictionCallback(a, ia, b, ib),
                              restitutionCallback: (a, ia, b, ib) =>
                                  defaultRestitutionCallback(a, ia, b, ib),
                          }
                        : {}),
                }),
        );
        try {
            for (const world of worlds) {
                world.state.contactRecycleDistance = 0;
                const material = {
                    ...defaultSurfaceMaterial(),
                    friction: 0.7,
                    restitution: 0.2,
                    rollingResistance: 0.3,
                    tangentVelocity: { x: 0.4, y: 0.2, z: 0.3 },
                };
                const hull = makeBoxHull(0.5, 0.5, 0.5);
                const mesh = createMesh({
                    vertices: [
                        { x: -2, y: 0, z: -2 },
                        { x: 2, y: 0, z: -2 },
                        { x: 2, y: 0, z: 2 },
                        { x: -2, y: 0, z: 2 },
                    ],
                    indices: [0, 2, 1, 0, 3, 2],
                    identifyEdges: true,
                });
                if (!mesh) throw new Error("mesh construction failed");
                const compound = createCompound(
                    kind === "mesh"
                        ? {
                              meshes: [
                                  {
                                      meshData: mesh,
                                      scale: { x: 1, y: 1, z: 1 },
                                      materials: [material],
                                      materialCount: 1,
                                      transform: { p: { x: 0, y: 0, z: 0 }, q: identity },
                                  },
                              ],
                          }
                        : kind === "hull"
                          ? {
                                hulls: [
                                    {
                                        hull,
                                        material,
                                        transform: { p: { x: 0, y: 0, z: 0 }, q: identity },
                                    },
                                ],
                            }
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
                world.createBody({ type: BodyType.Static }).createCompound({}, compound);
                world
                    .createBody({
                        type: BodyType.Dynamic,
                        position: { x: 0, y: kind === "mesh" ? 0.4 : 0.8, z: 0 },
                    })
                    .createHull(
                        {
                            baseMaterial: {
                                ...material,
                                friction: 0.4,
                                restitution: 0.5,
                                rollingResistance: 0.6,
                                tangentVelocity: { x: -0.1, y: 0.3, z: 0.2 },
                            },
                        },
                        hull,
                    );
            }
            for (let step = 0; step < 4; ++step) {
                for (const world of worlds) world.step(1 / 60, 1);
                expect(hash(worlds[0])).toBe(hash(worlds[1]));
                const rows = worlds.map((world) =>
                    contactIds(world.state)
                        .filter((c) => contactField(world.state, c, ContactField.manifoldCount) > 0)
                        .map((c) => readContactMaterial(world.state.manifoldStore.dirF, c)),
                );
                expect(rows[0].length).toBeGreaterThan(0);
                expect(rows[0]).toEqual(rows[1]);
            }
        } finally {
            for (const world of worlds) world.destroy();
        }
    }
});
