import { expect } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    Hulls,
    PhysicsMeshes,
    Shape,
    ShapeKind,
    Transform,
} from "@dylanebert/shallot";
import {
    hashPhysics,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    timeOfImpactBodyMover,
} from "@dylanebert/shallot/standard/physics";

export async function bodyMoverTOIRows(): Promise<string> {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const rows: string[] = [];
    const bits = (value: number) =>
        new Uint32Array(new Float32Array([value]).buffer)[0]!.toString(16).padStart(8, "0");
    try {
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        for (let kind = 0; kind < 4; ++kind) {
            const body = world.create();
            world.add(body, Transform, { translation: [100, 0, 0] });
            world.add(body, Body, { type: kind === 3 ? BodyType.Static : BodyType.Dynamic });
            const shapes: number[] = [];
            for (let i = 0; i < 3; ++i) {
                const shape = world.create();
                world.add(shape, Shape, {
                    body,
                    kind: [ShapeKind.Sphere, ShapeKind.Capsule, ShapeKind.Hull, ShapeKind.Mesh][
                        kind
                    ]!,
                    radius: 0.5,
                    sphereCenter: [0.5 * i, 0, 0],
                    capsuleCenter1: [0.5 * i, -0.5, 0],
                    capsuleCenter2: [0.5 * i, 0.5, 0],
                    geometry:
                        kind === 2
                            ? world.resource(Hulls).add({
                                  name: `box ${i}`,
                                  halfExtents: [0.5, 0.5, 0.5],
                                  transform: { position: [0.5 * i, 0, 0], rotation: [0, 0, 0, 1] },
                              })
                            : kind === 3
                              ? world.resource(PhysicsMeshes).add({
                                    name: `torus ${i}`,
                                    torus: {
                                        radialResolution: 8,
                                        tubularResolution: 8,
                                        radius: 2,
                                        thickness: 1,
                                    },
                                })
                              : 0,
                    filterCategoryLow: 1 << i,
                    filterCategoryHigh: 1 << (8 + i),
                });
                shapes.push(shape);
            }
            world.tick();
            const hash = hashPhysics(world);
            for (let c = 0; c < 10; ++c) {
                const origin = { x: 10, y: 2, z: -3 };
                const mover = {
                    center1: { x: 0, y: -0.2, z: 0 },
                    center2: { x: 0, y: 0.2, z: 0 },
                    radius: 0.25,
                };
                const start = {
                    p: { x: origin.x + 4, y: 2, z: -3 },
                    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
                };
                const end = structuredClone(start);
                let translation = { x: 8, y: 0, z: 0 };
                const filter = { categoryBits: 0xffffffffffffffffn, maskBits: 0xffffffffffffffffn };
                if (c === 1) translation.x = -8;
                if (c === 2) start.p = end.p = origin;
                if (c === 3) filter.maskBits = 0n;
                if (c === 4) filter.maskBits = 4n;
                if (c === 5) {
                    translation = { x: 0, y: 0, z: 0 };
                    end.p = origin;
                }
                if (c === 6) end.q = { v: { x: 0, y: 0.6, z: 0 }, s: 0.8 };
                if (c === 7) {
                    start.p.x = origin.x - 4;
                    end.p.x = origin.x + 4;
                    translation.x = 0;
                }
                if (c === 8) filter.maskBits = 4n << 40n;
                if (c === 9) filter.categoryBits = 0n;
                const out = timeOfImpactBodyMover(
                    world,
                    body,
                    origin,
                    mover,
                    translation,
                    start,
                    end,
                    filter,
                );
                expect(out.body).toBe(out.shape ? body : 0);
                rows.push(
                    `${kind} ${c} ${out.shape ? shapes.indexOf(out.shape) + 1 : 0} ${[out.fraction, out.point.x, out.point.y, out.point.z, out.normal.x, out.normal.y, out.normal.z].map(bits).join(" ")}`,
                );
                expect(
                    timeOfImpactBodyMover(
                        world,
                        body,
                        origin,
                        mover,
                        translation,
                        start,
                        end,
                        filter,
                    ),
                ).toBe(out);
            }
            expect(hashPhysics(world)).toBe(hash);
            for (const shape of shapes) world.destroy(shape);
            world.destroy(body);
        }
        return `${rows.join("\n")}\n`;
    } finally {
        app.dispose();
    }
}
