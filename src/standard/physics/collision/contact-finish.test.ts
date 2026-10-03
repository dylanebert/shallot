import { expect, test } from "bun:test";
import {
    BodyType,
    createCompound,
    createMesh,
    defaultSurfaceMaterial,
    makeBoxHull,
    PhysicsWorld,
} from "../api";
import { ContactFlags } from "./contact";

const identity = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const transform = { p: { x: 0, y: 0, z: 0 }, q: identity };
const scale = { x: 1, y: 1, z: 1 };

test("compound convex children mix their mapped material for callbacks, rolling resistance and tangent velocity, including flipped pairs", () => {
    for (const kind of ["sphere", "capsule", "hull"] as const) {
        for (const partner of ["sphere", "hull"] as const) {
            const friction: [number, bigint, number, bigint][] = [];
            const restitution: [number, bigint, number, bigint][] = [];
            const world = new PhysicsWorld({
                gravity: { x: 0, y: 0, z: 0 },
                enableSleep: false,
                enableContinuous: false,
                frictionCallback: (...args) => {
                    friction.push(args);
                    return args[1] === 9n ? args[0] : args[2];
                },
                restitutionCallback: (...args) => {
                    restitution.push(args);
                    return args[1] === 9n ? args[0] : args[2];
                },
            });
            try {
                const wrong = {
                    ...defaultSurfaceMaterial(),
                    userMaterialId: 1n,
                    friction: 0.1,
                    restitution: 0.2,
                    rollingResistance: 0.3,
                    tangentVelocity: { x: 0.1, y: 0, z: 0 },
                };
                const selected = {
                    ...defaultSurfaceMaterial(),
                    userMaterialId: 9n,
                    friction: 0.9,
                    restitution: 0.8,
                    rollingResistance: 0.7,
                    tangentVelocity: { x: 0.4, y: 0.2, z: 0.3 },
                };
                const hull = makeBoxHull(0.5, 0.5, 0.5);
                const materials = [wrong, selected];
                const compound = createCompound(
                    kind === "sphere"
                        ? {
                              spheres: materials.map((material, i) => ({
                                  material,
                                  sphere: {
                                      center: { x: i === 0 ? -4 : 0, y: 0, z: 0 },
                                      radius: 0.5,
                                  },
                              })),
                          }
                        : kind === "capsule"
                          ? {
                                capsules: materials.map((material, i) => ({
                                    material,
                                    capsule: {
                                        center1: { x: i === 0 ? -4 : 0, y: -0.25, z: 0 },
                                        center2: { x: i === 0 ? -4 : 0, y: 0.25, z: 0 },
                                        radius: 0.5,
                                    },
                                })),
                            }
                          : {
                                hulls: materials.map((material, i) => ({
                                    material,
                                    hull,
                                    transform: {
                                        p: { x: i === 0 ? -4 : 0, y: 0, z: 0 },
                                        q: identity,
                                    },
                                })),
                            },
                );
                if (!compound) throw new Error("compound construction failed");
                const ground = world.createBody({ type: BodyType.Static });
                ground.createCompound({ enableContactEvents: true }, compound);
                const body = world.createBody({
                    type: BodyType.Dynamic,
                    position: { x: 0, y: 0.8, z: 0 },
                });
                const def = { baseMaterial: { ...defaultSurfaceMaterial(), userMaterialId: 20n } };
                if (partner === "sphere")
                    body.createSphere(def, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
                else body.createHull(def, hull);
                world.step(1 / 60, 1);
                const begin = world.getContactEvents().beginEvents;
                expect(begin.length).toBe(1);
                expect(begin[0].contact.getData().manifolds.length).toBe(1);
                const contact = world.state.contacts[begin[0].contact.id.index1 - 1];
                expect(contact.childIndex).toBe(1);
                const flipped = partner === "hull" && kind !== "hull";
                const side = flipped ? 2 : 0;
                const idSide = flipped ? 3 : 1;
                expect(friction.at(-1)![idSide]).toBe(9n);
                expect(friction.at(-1)![side]).toBe(Math.fround(0.9));
                expect(restitution.at(-1)![idSide]).toBe(9n);
                expect(restitution.at(-1)![side]).toBe(Math.fround(0.8));
                expect(contact.friction).toBe(Math.fround(0.9));
                expect(contact.restitution).toBe(Math.fround(0.8));
                const radius = kind === "hull" && partner === "hull" ? 0.125 : 0.5;
                expect(contact.rollingResistance).toBe(Math.fround(Math.fround(0.7) * radius));
                const sign = flipped ? -1 : 1;
                expect(contact.tangentVelocity).toEqual({
                    x: sign * Math.fround(0.4),
                    y: sign * Math.fround(0.2),
                    z: sign * Math.fround(0.3),
                });
            } finally {
                world.destroy();
            }
        }
    }
});

test("mesh and compound-mesh contacts clear hit-event eligibility when their public contact ends with no manifolds", () => {
    for (const compoundChild of [false, true]) {
        const world = new PhysicsWorld({
            gravity: { x: 0, y: 0, z: 0 },
            enableSleep: false,
            enableContinuous: false,
        });
        try {
            world.state.contactRecycleDistance = 0;
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
            const ground = world.createBody({ type: BodyType.Static });
            const def = { enableContactEvents: true, enableHitEvents: true };
            if (compoundChild) {
                const compound = createCompound({
                    meshes: [
                        {
                            meshData: mesh,
                            transform,
                            scale,
                            materials: [defaultSurfaceMaterial()],
                            materialCount: 1,
                        },
                    ],
                });
                if (!compound) throw new Error("compound construction failed");
                ground.createCompound(def, compound);
            } else ground.createMesh(def, mesh);
            const sphere = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: 0.4, z: 0 },
            });
            sphere.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            world.step(1 / 60, 1);
            const begin = world.getContactEvents().beginEvents;
            expect(begin.length).toBe(1);
            const handle = begin[0].contact;
            expect(handle.getData().manifolds.length).toBe(1);
            const contact = world.state.contacts[handle.id.index1 - 1];
            expect(contact.flags & ContactFlags.simEnableHitEvent).not.toBe(0);
            sphere.setTransform({ x: 0, y: 0.6, z: 0 }, identity);
            world.step(1 / 60, 1);
            expect(world.getContactEvents().endEvents.length).toBe(1);
            expect(handle.isValid()).toBe(true);
            expect(handle.getData().manifolds.length).toBe(0);
            expect(contact.flags & ContactFlags.simEnableHitEvent).toBe(0);
        } finally {
            world.destroy();
        }
    }
});
