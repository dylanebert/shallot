import {
    Body,
    BodyType,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    castBodyRay,
    castBodyShape,
    collideBodyMover,
    emptyQueryHit,
    overlapBodyShape,
    type PhysicsBodyPlanes,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";

export const CastState = {
    create: () => ({
        body: 0,
        placement: { p: { x: -10, y: 2, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        ray: emptyQueryHit(),
        sphere: emptyQueryHit(),
        rayHit: false,
        sphereHit: false,
        overlap: false,
        planes: { records: [], count: 0 } as PhysicsBodyPlanes,
    }),
};
export const rayOrigin = { x: -9.75, y: 3, z: -4 };
export const rayTranslation = { x: 0, y: 0, z: 8 };
export const sphereOrigin = { x: -14.5, y: 2.5, z: 0.5 };
export const sphereTranslation = { x: 8, y: 0, z: 0 };
export const sphereProxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.2 };
export const overlapOrigin = { x: -10, y: 1, z: 0.5 };
export const overlapProxy = {
    points: [
        { x: -0.5, y: 1, z: 0 },
        { x: 0.5, y: 0, z: 0 },
    ],
    count: 2,
    radius: 0.5,
};
export const moverOrigin = { x: -10, y: 2, z: -0.75 };
export const mover = {
    center1: { x: -0.25, y: 0, z: 0 },
    center2: { x: 0.25, y: 1, z: 0 },
    radius: 0.3,
};
export const CastPlugin: Plugin = {
    name: "Cast",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(CastState),
    systems: [
        {
            name: "body-cast-queries",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(CastState);
                if (!s.body) return;
                s.rayHit = castBodyRay(
                    world,
                    s.body,
                    rayOrigin,
                    rayTranslation,
                    s.placement,
                    s.ray,
                );
                s.sphereHit = castBodyShape(
                    world,
                    s.body,
                    sphereOrigin,
                    sphereProxy,
                    sphereTranslation,
                    s.placement,
                    s.sphere,
                    undefined,
                    1,
                    true,
                );
                s.overlap = overlapBodyShape(
                    world,
                    s.body,
                    overlapOrigin,
                    overlapProxy,
                    s.placement,
                );
                s.planes = collideBodyMover(world, s.body, moverOrigin, mover, s.placement);
            },
        },
    ],
};
function construct(world: World): number[] {
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    const body = world.create();
    world.add(body, Transform, { translation: [5, 5, 0] });
    world.add(body, Body, { type: BodyType.Kinematic, angularVelocity: [0.1, -0.1, 0.1] });
    world.add(body, Shape, {
        geometry: world
            .resource(Hulls)
            .add({ name: "cylinder", cylinder: { height: 2, radius: 0.5, yOffset: 0, sides: 16 } }),
    });
    const s = world.resource(CastState);
    s.body = body;
    const f = Math.fround;
    const inv = f(1 / f(Math.sqrt(14)));
    const q = rotationFromAxisAngle(
        world,
        { x: inv, y: f(-2 * inv), z: f(3 * inv) },
        f(f(0.75) * f(Math.PI)),
    );
    s.placement.q = { v: { x: q[0], y: q[1], z: q[2] }, s: q[3] };
    return [body];
}
export const castScene = {
    construct,
    gridExtent: 10,
    gridSpacing: 1,
    camera: { yaw: 15, pitch: 25, distance: 30, phoneDistance: 55, target: [-4, 3, 0] },
    controls: undefined,
};
