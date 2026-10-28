import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    DistanceJoint,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    StandardPhysicsPlugin,
    WheelJoint,
} from "@dylanebert/shallot";
import {
    applyLinearImpulseToCenter,
    isAwake,
    setAwake,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSample, EventSamplePlugin } from "../src/events";
import { GeometricMoverState } from "../src/geometric-mover";
import { scenePresentation, shapeMaterial } from "../src/presentation";
import { RigidBodyState } from "../src/rigid-body";
import { MotorizedJoint } from "../src/roles";
import {
    categories,
    changeControl,
    controlValues,
    createScene,
    enabledControls,
    type Scene,
    scenes,
} from "../src/scenes";
import { SensorSample } from "../src/sensor";

setDefaultTimeout(TIMEOUT.node);

test("Body Type follows its selected type's role and Weeble draws only its explosion rings", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const bodies = createScene(world, "bodyType");
        world.tick();
        const values = controlValues("bodyType");
        for (const type of [BodyType.Static, BodyType.Kinematic, BodyType.Dynamic]) {
            changeControl(world, "bodyType", bodies, values, "Type", type);
            const presentation = scenePresentation(world, "bodyType");
            for (const index of [1, 2, 4, 5, 6])
                expect(
                    presentation.shapes.find((shape) => shape.body === bodies[index])!.role,
                ).toBe(
                    type === BodyType.Static
                        ? "ground"
                        : type === BodyType.Kinematic
                          ? "motorized"
                          : "subject",
                );
            expect(presentation.explosion).toEqual([]);
        }
    } finally {
        app.dispose();
    }
    const weeble = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            createScene(world, "weeble");
        },
    });
    try {
        const rings = scenePresentation(weeble.world, "weeble").explosion;
        expect(rings).toHaveLength(72);
        for (const line of rings) {
            expect(line.color).toBe(0x00ffff);
            for (const point of [line.a, line.b])
                expect(Math.hypot(point[0], point[1] + 0.1, point[2])).toBeCloseTo(8, 10);
        }
    } finally {
        weeble.dispose();
    }
});

// The accepted critical-set baseline, in eid order; counts compress adjacent equal roles.
const acceptedRoles = {
    move: [
        ["ground", 1],
        ["subject", 1],
    ],
    sensorHits: [
        ["ground", 2],
        ["sensor", 3],
        ["subject", 1],
    ],
    persistentContact: [
        ["ground", 1],
        ["subject", 1],
    ],
    driving: [
        ["ground", 1],
        ["subject", 5],
    ],
    gearLift: [
        ["ground", 3],
        ["motorized", 35],
        ["subject", 236],
    ],
    distance: [
        ["ground", 1],
        ["subject", 1],
    ],
    door: [
        ["ground", 1],
        ["subject", 1],
    ],
    spherical: [
        ["ground", 1],
        ["subject", 1],
    ],
    motionLocks: [
        ["ground", 1],
        ["subject", 4],
    ],
    ballAndChain: [["subject", 33]],
    foldOut: [
        ["ground", 1],
        ["subject", 6],
    ],
    topDownFriction: [
        ["ground", 4],
        ["subject", 100],
    ],
    weld: [
        ["ground", 1],
        ["subject", 1],
    ],
    contact: [
        ["ground", 5],
        ["motorized", 1],
    ],
    hit: [
        ["ground", 1],
        ["subject", 22],
    ],
    joint: [
        ["ground", 1],
        ["subject", 4],
    ],
    sensor: [
        ["subject", 1],
        ["sensor", 1],
    ],
    bridge: [
        ["ground", 1],
        ["subject", 150],
    ],
    filter: [
        ["ground", 1],
        ["subject", 2],
    ],
    motor: [
        ["ground", 1],
        ["motorized", 1],
        ["subject", 1],
    ],
    parallel: [
        ["ground", 5],
        ["subject", 1],
    ],
    prismatic: [
        ["ground", 1],
        ["motorized", 1],
    ],
    revolute: [
        ["ground", 1],
        ["motorized", 1],
    ],
    wheel: [
        ["ground", 1],
        ["motorized", 1],
    ],
    boxStack: [
        ["ground", 1],
        ["subject", 40],
    ],
} as const;
const acceptedColors = {
    ground: [0.085, 0.08, 0.072, 1],
    subject: [0.72, 0.4, 0.05, 1],
    motorized: [0.04, 0.43, 0.53, 1],
    sensor: [0.04, 0.43, 0.53, 1],
};

test("the accepted role baseline covers every critical-set scene", () => {
    expect(Object.keys(acceptedRoles).sort()).toEqual(
        categories
            .filter(
                (category) =>
                    ![
                        "Benchmark",
                        "Bodies",
                        "Character",
                        "Compound",
                        "Continuous",
                        "Shapes",
                    ].includes(category.name),
            )
            .flatMap((category) =>
                category.scenes
                    .filter((scene) => category.name !== "Stacking" || scene.id === "boxStack")
                    .map((scene) => scene.id),
            )
            .sort(),
    );
});
for (const [id, runs] of Object.entries(acceptedRoles)) {
    test(`${id}: default roles and colours retain the accepted critical-set baseline`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin, EventSamplePlugin],
            setup(world) {
                createScene(world, id as Scene);
            },
        });
        try {
            const presentation = scenePresentation(app.world, id as Scene);
            const shapes = presentation.shapes.sort((a, b) => a.eid - b.eid);
            const expected = runs.flatMap(([role, count]) =>
                Array.from({ length: count }, () => role),
            );
            expect(shapes.map((shape) => shape.role)).toEqual(expected);
            for (const role of expected)
                expect(Array.from(presentation.materials[role].baseColor)).toEqual(
                    acceptedColors[role].map(Math.fround),
                );
        } finally {
            app.dispose();
        }
    });
}

test("sleep changes a body's colour without changing its role", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const [body] = createScene(app.world, "boxStack");
        app.world.tick();
        expect(isAwake(app.world, body!)).toBe(true);
        const before = Array.from(shapeMaterial(app.world, "subject", body!).baseColor);
        for (let i = 0; i < 1800 && isAwake(app.world, body!); i++) app.world.tick();
        expect(isAwake(app.world, body!)).toBe(false);
        expect(Array.from(shapeMaterial(app.world, "subject", body!).baseColor)).not.toEqual(
            before,
        );
        // Compare chromaticity: brightness changes with state and lighting, affordance does not.
        const chromaticity = (color: ArrayLike<number>) => {
            const sum = color[0]! + color[1]! + color[2]!;
            return [color[0]! / sum, color[1]! / sum, color[2]! / sum];
        };
        const distance = (a: number[], b: number[]) =>
            Math.hypot(...a.map((value, i) => value - b[i]!));
        const ground = chromaticity(shapeMaterial(app.world, "ground", body!).baseColor);
        for (const role of ["subject", "motorized", "sensor", "reaction"] as const) {
            if (role === "reaction")
                app.world.resource(EventSample).reactions.add(app.world.ref(body!));
            const materialRole = role === "reaction" ? "subject" : role;
            setAwake(app.world, body!, true);
            const awake = shapeMaterial(app.world, materialRole, body!);
            setAwake(app.world, body!, false);
            const asleep = shapeMaterial(app.world, materialRole, body!);
            const tint = chromaticity(asleep.baseColor);
            expect(distance(tint, chromaticity(awake.baseColor)), role).toBeLessThan(
                distance(tint, ground),
            );
            expect(asleep.perceptualRoughness).toBeGreaterThan(awake.perceptualRoughness);
            expect(asleep.baseColor[0]! + asleep.baseColor[1]! + asleep.baseColor[2]!).toBeLessThan(
                awake.baseColor[0]! + awake.baseColor[1]! + awake.baseColor[2]!,
            );
        }
    } finally {
        app.dispose();
    }
});

test("a broken joint recolours its body", async () => {
    const app = await createApp({ defaults: false, plugins: [EventSamplePlugin] });
    try {
        const bodies = createScene(app.world, "joint");
        const before = bodies.map((body) =>
            Array.from(shapeMaterial(app.world, "subject", body).baseColor),
        );
        app.world.tick();
        applyLinearImpulseToCenter(app.world, bodies[0]!, { x: 0, y: -1000, z: 0 });
        for (let i = 0; i < 120; i++) app.world.tick();
        expect(
            bodies.some(
                (body, i) =>
                    JSON.stringify(
                        Array.from(shapeMaterial(app.world, "subject", body).baseColor),
                    ) !== JSON.stringify(before[i]),
            ),
        ).toBe(true);
    } finally {
        app.dispose();
    }
});

for (const { id, name } of categories.flatMap((category) => category.scenes)) {
    // Their camera draw sets are held by large-world-view.node.ts and village-view.node.ts.
    if (id === "largeWorld" || id === "village") continue;
    test(`${name} builds every shape mesh, role material and controls panel without a GPU`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                createScene(world, id);
            },
        });
        try {
            const world = app.world;
            const presentation = scenePresentation(world, id);
            expect(presentation.shapes.map((shape) => shape.eid).sort()).toEqual(
                [...world.query([Shape])].sort(),
            );
            expect(presentation.shapes.length).toBeGreaterThan(0);
            const driven = new Set<number>();
            for (const joint of [DistanceJoint, PrismaticJoint, RevoluteJoint, WheelJoint]) {
                const storage = world.storage(joint);
                for (const eid of world.query([joint, MotorizedJoint])) {
                    driven.add(storage.a.get(eid));
                    driven.add(storage.b.get(eid));
                }
            }
            for (const { eid, body, role, mesh } of presentation.shapes) {
                expect(mesh.vertices.length).toBeGreaterThan(0);
                const mover = world.resource(GeometricMoverState);
                const characterRole =
                    mover.active &&
                    (eid === mover.ignore || mover.policies.get(eid)?.clipVelocity === false)
                        ? "motorized"
                        : mover.active &&
                            world.get(body, Body)!.type === BodyType.Static &&
                            mover.policies.has(eid)
                          ? "sensor"
                          : null;
                const expected =
                    characterRole ??
                    (world.resource(SensorSample).sensors.get(world.ref(eid))?.active
                        ? "ground"
                        : world.get(eid, Shape)!.isSensor
                          ? "sensor"
                          : world.get(body, Body)!.type === BodyType.Static
                            ? "ground"
                            : world.get(body, Body)!.type === BodyType.Kinematic ||
                                driven.has(body) ||
                                body === world.resource(EventSample).player ||
                                body === world.resource(RigidBodyState).character.body
                              ? "motorized"
                              : "subject");
                expect(role).toBe(expected);
                expect(presentation.materials[role]).toBeDefined();
                const material = shapeMaterial(world, role, eid);
                expect("cullMode" in material ? material.cullMode : "back").toBe("back");
            }
            expect(Array.from(presentation.materials.sensor.baseColor)).toEqual(
                [0.04, 0.43, 0.53, 1].map(Math.fround),
            );
            if (id === "sensor") {
                const ground = presentation.ground!;
                const bounds = [0, 1, 2].map((axis) => {
                    const values = Array.from(ground.mesh.vertices).filter(
                        (_, index) => index % 8 === axis,
                    );
                    return [
                        Math.min(...values) + ground.translation[axis]!,
                        Math.max(...values) + ground.translation[axis]!,
                    ];
                });
                expect(bounds[0]).toEqual([-10, 10]);
                expect(bounds[2]).toEqual([-10, 10]);
                expect(bounds[1]![0]).toBeCloseTo(-0.2);
                expect(bounds[1]![1]).toBeCloseTo(0);
                expect(ground.mesh.indices.length).toBeGreaterThan(0);
            } else expect(presentation.ground).toBeUndefined();
            const extent = {
                simpleCompound: 8,
                compoundSpheres: 12,
                compoundHulls: 12,
                tileFloor: 208,
                meshTile: 12,
                village: 800,
                bulletVersusStack: 50,
                meshDrop: 20,
                thinWall: 40,
                bounceHouse: 10,
                spinningStick: 10,
                needleMesh: 10,
                humpMesh: 20,
                isFast: 40,
                stall: 500,
                geometricMover: 80,
                rigidBody: 3300,
                capsulePlane: 10,
                moverOverlap: 12,
                cast: 10,
                classRing: 100,
                restitution: 50,
                staticInvoke: 20,
                wind: 20,
                inclinedPlane: 50,
                rollingResistance: 80,
                highResistance: 50,
                isotropicFriction: 100,
                slideTwist: 50,
                conveyorBelt: 20,
                conveyorMesh: 20,
                windDrop: 15,
                windFlap: 50,
                gyroscopicPrecession: 40,
                bodyType: 20,
                weeble: 30,
                disable: 20,
                spinningBook: 10,
                gyroscopicTorque: 20,
                kinematic: 20,
                lockMixing: 20,
                fixedRotation: 20,
                offsetKinematic: 20,
                driving: 200,
                gearLift: 20,
                distance: 20,
                door: 20,
                spherical: 20,
                motionLocks: 20,
                heightField: 30,
                hull: 5,
                ballAndChain: 40,
                foldOut: 20,
                topDownFriction: 20,
                weld: 20,
                boxStack: 40,
                cardHouse: 10,
                sphereStack: 15,
                capsuleStack: 40,
                singleBox: 20,
                cylinder: 10,
                cylinderStack: 10,
                jenga: 20,
                dominoes: 80,
                wedge: 20,
                arch: 40,
                doubleDomino: 20,
                pyramid2D: 40,
                edgeCrossing: 40,
                bridge: 25,
                contact: 80,
                hit: 60,
                joint: 20,
                sensor: 10,
                motor: 10,
                revolute: 10,
                prismatic: 10,
                wheel: 10,
                filter: 10,
                parallel: 20,
                largePyramid: 400,
                widePyramid: 100,
                manyPyramids: 77,
                fallingBoxes: 100,
                jointGrid: 100,
                rain: 75,
                junkyard: 120,
                washer: 60,
                convexPile: 250,
                candyCups: 60,
                fallingTrees: 100,
                chains: 40,
                explosion: 20,
                destruction: 20,
                benchmarkSensor: 130,
                move: 40,
                persistentContact: 20,
                sensorHits: 10,
            }[id];
            const grid = presentation.grid;
            const spacing = grid.lines[2]!.a[2] - grid.lines[0]!.a[2];
            expect(spacing).toBe(scenes[id].gridSpacing);
            // At the 720 CSS-pixel review camera, adjacent cells at the ground focus
            // reserve at least 4 pixels in both screen directions (at most 1 line / 4 px).
            const camera = scenes[id].camera;
            const pitch = (camera.pitch * Math.PI) / 180;
            const depth = camera.distance + camera.target[1]! * Math.sin(pitch);
            const focal = 720 / (2 * Math.tan(Math.PI / 6));
            const across = (focal * spacing) / depth;
            const along =
                (focal * spacing * (camera.distance * Math.sin(pitch) + camera.target[1]!)) /
                (depth * depth - ((Math.cos(pitch) * spacing) / 2) ** 2);
            expect(
                Math.min(across, along),
                `${name} grid cell pixels at the ground focus`,
            ).toBeGreaterThanOrEqual(4);
            expect(grid).toBeDefined();
            expect(grid.width).toBe(1);
            expect(grid.opacity).toBeLessThan(0.45);
            const height = id === "simpleCompound" ? -0.995 : 0.005;
            for (let offset = -extent; offset <= extent; offset += spacing) {
                const color = offset === 0 ? 0x797369 : 0x49453f;
                expect(grid.lines).toContainEqual({
                    a: [-extent, height, offset],
                    b: [extent, height, offset],
                    color,
                });
                expect(grid.lines).toContainEqual({
                    a: [offset, height, -extent],
                    b: [offset, height, extent],
                    color,
                });
            }
            expect(grid.lines).toHaveLength(((2 * extent) / spacing + 1) * 2);
            expect(presentation.controls).toEqual(enabledControls(id, controlValues(id)));
        } finally {
            app.dispose();
        }
    });
}
