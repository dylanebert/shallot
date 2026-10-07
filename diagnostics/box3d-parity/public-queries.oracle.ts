// Stage-4 public differential oracle. Both sides author the same six shapes through their public
// helpers/APIs, receive the same float32 bit patterns, and serialize only public, meaningful fields.
// Callback results are appended as delivered: order is deliberately never sorted.
import { expect, setDefaultTimeout, test } from "bun:test";
import {
    BodyType,
    createBoxMesh,
    createCompound,
    createGrid,
    defaultSurfaceMaterial,
    init,
    makeBoxHull,
    PhysicsWorld,
    type Shape,
} from "../../src/standard/physics/api";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();

const f32 = Math.fround;
const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0];
const hex = (value: number) => (value >>> 0).toString(16).padStart(8, "0");
const zero = { x: 0, y: 0, z: 0 };
const names = ["sphere", "capsule", "hull", "mesh", "heightfield", "compound"] as const;
const operations = [
    "world overlapAABB",
    "world overlapShape",
    "world castRay",
    "world castRayClosest",
    "world castShape",
    "world collideMover",
    "world castMover",
    "body castRay",
    "body castShape",
    "body overlapShape",
    "body getClosestPoint",
    "body collideMover",
] as const;

type Input = {
    operation: number;
    kind: number;
    words: number[];
    label: string;
};

type Pose = {
    position: { x: number; y: number; z: number };
    rotation: { v: { x: number; y: number; z: number }; s: number };
};

type ContactInput = {
    compositeKind: number;
    convexKind: number;
    words: number[];
    label: string;
};

const material = (id: bigint) => ({ ...defaultSurfaceMaterial(), userMaterialId: id });

function fixture(kind: number, pose?: Pose) {
    const world = new PhysicsWorld({ gravity: zero, enableSleep: false });
    const position = pose?.position ?? fixturePosition(kind);
    const rotation = pose?.rotation ?? { v: zero, s: 1 };
    const body = world.createBody({ type: BodyType.Static, position, rotation });
    const def = {
        userData: kind + 1,
        baseMaterial: material(0x100000000n + BigInt(kind + 1)),
    };
    let shape: Shape;
    if (kind === 0) {
        shape = body.createSphere(def, { center: { x: 0, y: -1, z: 0 }, radius: 1 });
    } else if (kind === 1) {
        shape = body.createCapsule(def, {
            center1: { x: 0, y: -1.65, z: 0 },
            center2: { x: 0, y: -0.35, z: 0 },
            radius: 0.35,
        });
    } else if (kind === 2) {
        shape = body.createHull(def, makeBoxHull(1, 0.75, 0.6));
    } else if (kind === 3) {
        shape = body.createMesh(
            { ...def, materials: [material(0x100000004n)] },
            createBoxMesh({ x: 0, y: -0.75, z: 0 }, { x: 1, y: 0.75, z: 0.6 }, true),
        );
    } else if (kind === 4) {
        shape = body.createHeightField(
            { ...def, materials: [material(0x100000005n)] },
            createGrid(4, 4, { x: 1, y: 1, z: 1 }, false),
        );
    } else {
        const compound = createCompound({
            hulls: [
                {
                    hull: makeBoxHull(1, 0.75, 0.6),
                    transform: {
                        p: { x: 0, y: -0.75, z: 0 },
                        q: { v: zero, s: 1 },
                    },
                    material: material(0x100000006n),
                },
            ],
            spheres: [
                {
                    sphere: { center: { x: -0.625, y: -0.2, z: 0 }, radius: 0.2 },
                    material: material(0x200000006n),
                },
                {
                    sphere: { center: { x: 0.625, y: -0.2, z: 0 }, radius: 0.2 },
                    material: material(0x300000006n),
                },
            ],
        });
        if (compound === null) throw new Error("compound fixture creation failed");
        shape = body.createCompound(def, compound);
    }
    return { world, body, shape };
}

function fixturePosition(kind: number): { x: number; y: number; z: number } {
    return kind === 4 ? { x: -1.5, y: 0, z: -1.5 } : kind === 2 ? { x: 0, y: -0.75, z: 0 } : zero;
}

function pushVec(out: number[], value: { x: number; y: number; z: number }): void {
    out.push(bits(value.x), bits(value.y), bits(value.z));
}

function pushU64(out: number[], value: bigint): void {
    out.push(Number(value & 0xffffffffn), Number((value >> 32n) & 0xffffffffn));
}

function pushCast(
    out: number[],
    hit: {
        shape: Shape | null;
        point: { x: number; y: number; z: number };
        normal: { x: number; y: number; z: number };
        fraction: number;
        userMaterialId: bigint;
        triangleIndex: number;
        childIndex?: number;
        hit: boolean;
    },
    includeChild: boolean,
): void {
    out.push(Number(hit.hit));
    if (!hit.hit) return;
    out.push(hit.shape?.getUserData() as number);
    pushVec(out, hit.point);
    pushVec(out, hit.normal);
    out.push(bits(hit.fraction));
    pushU64(out, hit.userMaterialId);
    out.push(hit.triangleIndex);
    if (includeChild) out.push(hit.childIndex ?? 0);
}

function pushPlane(
    out: number[],
    plane: {
        plane: { normal: { x: number; y: number; z: number }; offset: number };
        point: { x: number; y: number; z: number };
        triangleIndex?: number;
        childIndex?: number;
        materialIndex?: number;
    },
): void {
    pushVec(out, plane.plane.normal);
    out.push(bits(plane.plane.offset));
    pushVec(out, plane.point);
}

function runPublic(input: Input): string[] {
    const values = input.words.map((word) => new Float32Array(new Uint32Array([word]).buffer)[0]);
    const pose = {
        position: { x: values[11], y: values[12], z: values[13] },
        rotation: { v: { x: values[14], y: values[15], z: values[16] }, s: values[17] },
    };
    const { world, body } = fixture(input.kind, pose);
    const origin = { x: values[0], y: values[1], z: values[2] };
    const translation = { x: values[3], y: values[4], z: values[5] };
    const proxy = { points: [zero], count: 1, radius: values[6] };
    const mover = {
        center1: { x: 0, y: -values[7], z: 0 },
        center2: { x: 0, y: values[7], z: 0 },
        radius: values[8],
    };
    const extent = values[9];
    const filtered = values[10] < 0;
    const maxFraction = Math.abs(values[10]);
    const filter = filtered ? { categoryBits: 0n, maskBits: 0n } : undefined;
    const out = [input.operation, input.kind];
    try {
        if (input.operation === 0) {
            let count = 0;
            const stats = world.overlapAABB(
                {
                    lowerBound: {
                        x: origin.x - extent,
                        y: origin.y - extent,
                        z: origin.z - extent,
                    },
                    upperBound: {
                        x: origin.x + extent,
                        y: origin.y + extent,
                        z: origin.z + extent,
                    },
                },
                (shape) => {
                    out.push(shape.getUserData() as number);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count, stats.nodeVisits, stats.leafVisits);
        } else if (input.operation === 1) {
            let count = 0;
            const stats = world.overlapShape(
                origin,
                proxy,
                (shape) => {
                    out.push(shape.getUserData() as number);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count, stats.nodeVisits, stats.leafVisits);
        } else if (input.operation === 2 || input.operation === 4) {
            let count = 0;
            const callback = (hit: Parameters<Parameters<PhysicsWorld["castRay"]>[2]>[0]) => {
                out.push(hit.shape.getUserData() as number);
                pushVec(out, hit.point);
                pushVec(out, hit.normal);
                out.push(bits(hit.fraction));
                pushU64(out, hit.userMaterialId);
                out.push(hit.triangleIndex, hit.childIndex);
                count++;
                return hit.fraction;
            };
            const stats =
                input.operation === 2
                    ? world.castRay(origin, translation, callback, filter)
                    : world.castShape(origin, proxy, translation, callback, filter);
            out.push(count, stats.nodeVisits, stats.leafVisits);
        } else if (input.operation === 3) {
            pushCast(out, world.castRayClosest(origin, translation, filter), true);
        } else if (input.operation === 5) {
            let count = 0;
            world.collideMover(
                origin,
                mover,
                (shape, planes) => {
                    out.push(shape.getUserData() as number, planes.length);
                    for (const plane of planes) pushPlane(out, plane);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count);
        } else if (input.operation === 6) {
            let count = 0;
            const fraction = world.castMover(origin, mover, translation, filter, (shape) => {
                out.push(shape.getUserData() as number);
                count++;
                return true;
            });
            out.push(bits(fraction), count);
        } else if (input.operation === 7) {
            pushCast(
                out,
                body.castRay(
                    origin,
                    translation,
                    { p: pose.position, q: pose.rotation },
                    filter,
                    maxFraction,
                ),
                false,
            );
        } else if (input.operation === 8) {
            pushCast(
                out,
                body.castShape(
                    origin,
                    proxy,
                    translation,
                    { p: pose.position, q: pose.rotation },
                    filter,
                    maxFraction,
                ),
                false,
            );
        } else if (input.operation === 9) {
            out.push(
                Number(
                    body.overlapShape(
                        origin,
                        proxy,
                        { p: pose.position, q: pose.rotation },
                        filter,
                    ),
                ),
            );
        } else if (input.operation === 10) {
            const result = body.getClosestPoint(origin);
            pushVec(out, result.point);
            out.push(bits(result.distance));
        } else {
            const planes = body.collideMover(
                origin,
                mover,
                { p: pose.position, q: pose.rotation },
                8,
                filter,
            );
            out.push(planes.length);
            for (const entry of planes) {
                out.push(entry.shape.getUserData() as number);
                pushPlane(out, entry.plane);
            }
        }
        return out.map(hex);
    } finally {
        world.destroy();
    }
}

function contactPublic(input: ContactInput): string[] {
    const values = input.words.map((word) => new Float32Array(new Uint32Array([word]).buffer)[0]);
    const { world } = fixture(input.compositeKind);
    try {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: values[0], y: values[1], z: values[2] },
            rotation: { v: { x: values[3], y: values[4], z: values[5] }, s: values[6] },
        });
        const def = { userData: 99, enableContactEvents: true };
        if (input.convexKind === 0) {
            body.createSphere(def, { center: zero, radius: values[7] });
        } else if (input.convexKind === 1) {
            body.createCapsule(def, {
                center1: { x: 0, y: -values[8], z: 0 },
                center2: { x: 0, y: values[8], z: 0 },
                radius: values[7],
            });
        } else {
            body.createHull(def, makeBoxHull(values[7], values[8], values[9]));
        }
        world.step(f32(1 / 60), 4);
        const events = world.getContactEvents().beginEvents;
        const out = [12, input.compositeKind, input.convexKind, events.length];
        for (const event of events) {
            out.push(event.shapeA.getUserData() as number, event.shapeB.getUserData() as number);
            out.push(Number(event.contact.isValid()));
            const data = event.contact.getData();
            out.push(data.shapeA.getUserData() as number, data.shapeB.getUserData() as number);
            out.push(data.manifolds.length);
            for (const manifold of data.manifolds) {
                out.push(manifold.pointCount);
                pushVec(out, manifold.normal);
                out.push(bits(manifold.twistImpulse));
                pushVec(out, manifold.frictionImpulse);
                pushVec(out, manifold.rollingImpulse);
                for (const point of manifold.points) {
                    pushVec(out, point.anchorA);
                    pushVec(out, point.anchorB);
                    out.push(
                        bits(point.separation),
                        bits(point.baseSeparation),
                        bits(point.normalImpulse),
                    );
                    out.push(bits(point.totalNormalImpulse), bits(point.normalVelocity));
                    out.push(point.featureId, point.triangleIndex, Number(point.persisted));
                }
            }
        }
        return out.map(hex);
    } finally {
        world.destroy();
    }
}

let seed = 0x47d7f7cc;
function random(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 0x100000000;
}

function identityPose(kind: number): Pose {
    return { position: fixturePosition(kind), rotation: { v: zero, s: 1 } };
}

function generatedPose(kind: number): Pose {
    const base = fixturePosition(kind);
    const axis = [random() - 0.5, random() - 0.5, random() - 0.5];
    const axisLength = Math.hypot(...axis);
    const halfAngle = (random() - 0.5) * 0.45;
    const scale = Math.sin(halfAngle) / axisLength;
    return {
        position: {
            x: f32(base.x + (random() - 0.5) * 0.24),
            y: f32(base.y + (random() - 0.5) * 0.12),
            z: f32(base.z + (random() - 0.5) * 0.24),
        },
        rotation: {
            v: { x: f32(axis[0] * scale), y: f32(axis[1] * scale), z: f32(axis[2] * scale) },
            s: f32(Math.cos(halfAngle)),
        },
    };
}

function words(origin: number[], translation: number[], pose: Pose, maxFraction = 1): number[] {
    return [
        ...origin,
        ...translation,
        0.25,
        0.25,
        0.3,
        0.4,
        maxFraction,
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.rotation.v.x,
        pose.rotation.v.y,
        pose.rotation.v.z,
        pose.rotation.s,
    ].map((value) => bits(value));
}

const inputs: Input[] = [];
for (let operation = 0; operation < operations.length; operation++) {
    for (let kind = 0; kind < names.length; kind++) {
        const cast =
            operation === 2 ||
            operation === 3 ||
            operation === 4 ||
            operation === 6 ||
            operation === 7 ||
            operation === 8;
        const closest = operation === 10;
        const controlPose = identityPose(kind);
        inputs.push({
            operation,
            kind,
            words: words(
                cast || closest ? [0, 2, 0] : [0, -0.2, 0],
                cast ? [0, -4, 0] : [0, 0, 0],
                controlPose,
            ),
            label: "deterministic hit",
        });
        const jitterX = f32((random() - 0.5) * 0.35);
        const jitterZ = f32((random() - 0.5) * 0.25);
        const seededPose = generatedPose(kind);
        inputs.push({
            operation,
            kind,
            words: words(
                cast || closest ? [jitterX, 2, jitterZ] : [jitterX, -0.2, jitterZ],
                cast ? [0, -4, 0] : [0, 0, 0],
                seededPose,
            ),
            label: `seeded pose ${hex(bits(jitterX))}/${hex(bits(jitterZ))}`,
        });
        const missPose = generatedPose(kind);
        inputs.push({
            operation,
            kind,
            words: words(
                [3, cast || closest ? 2 : -0.2, 3],
                cast ? [0, -4, 0] : [0, 0, 0],
                missPose,
            ),
            label: "deterministic miss",
        });
        if (operation !== 10) {
            const filteredPose = generatedPose(kind);
            inputs.push({
                operation,
                kind,
                words: words(
                    cast ? [0, 2, 0] : [0, -0.2, 0],
                    cast ? [0, -4, 0] : [0, 0, 0],
                    filteredPose,
                    -1,
                ),
                label: "zero query filter",
            });
        }
    }
}
for (const operation of [2, 3, 4, 7, 8]) {
    inputs.push({
        operation,
        kind: 5,
        words: words([1, 2, 0], [0, -4, 0], identityPose(5)),
        label: "strict compound box clip",
    });
}

function contactWords(pose: Pose, dimensions: number[]): number[] {
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.rotation.v.x,
        pose.rotation.v.y,
        pose.rotation.v.z,
        pose.rotation.s,
        ...dimensions,
    ].map(bits);
}

const contactInputs: ContactInput[] = [];
for (const compositeKind of [3, 4, 5]) {
    const centered: Array<[number, number[]]> = [
        [0, [0.5, 0, 0]],
        [1, [0.3, 0.35, 0]],
        [2, [0.38, 0.45, 0.32]],
    ];
    for (const [convexKind, dimensions] of centered) {
        contactInputs.push({
            compositeKind,
            convexKind,
            words: contactWords(
                { position: { x: 0, y: 0.35, z: 0 }, rotation: { v: zero, s: 1 } },
                dimensions,
            ),
            label: `centered ${names[convexKind]}`,
        });
    }
    for (let sample = 0; sample < 5; sample++) {
        const convexKind = sample % 3;
        const pose = generatedPose(0);
        pose.position.x = f32((random() - 0.5) * 0.3);
        pose.position.y = f32(0.05 + random() * 0.05);
        pose.position.z = f32((random() - 0.5) * 0.3);
        if (compositeKind === 4 && convexKind === 1) pose.position.y = f32(0.35);
        const radius = f32(0.35 + random() * 0.2);
        let dimensions =
            convexKind === 0
                ? [radius, 0, 0]
                : convexKind === 1
                  ? [f32(radius * 0.65), f32(0.2 + random() * 0.35), 0]
                  : [radius, f32(0.28 + random() * 0.25), f32(0.24 + random() * 0.24)];
        if (compositeKind === 4 && convexKind === 1) dimensions = [f32(0.65), f32(0.6), 0];
        contactInputs.push({
            compositeKind,
            convexKind,
            words: contactWords(pose, dimensions),
            label: `seeded ${sample} ${names[convexKind]}`,
        });
    }
}

const nativeInput = [
    ...inputs.map((input) => `${input.operation} ${input.kind} ${input.words.map(hex).join(" ")}`),
    ...contactInputs.map(
        (input) =>
            `12 ${input.compositeKind} ${input.convexKind} ${input.words.map(hex).join(" ")}`,
    ),
].join("\n");
const nativeRows = nativeSseOutput("public-queries.c", nativeInput, [
    ...inputs.map((input) => `${operations[input.operation]}:${names[input.kind]}:${input.label}`),
    ...contactInputs.map((input) => `contact:${names[input.compositeKind]}:${input.label}`),
])
    .trim()
    .split("\n")
    .map((line) =>
        line
            .trim()
            .split(/\s+/)
            .map((v) => v.padStart(8, "0")),
    );
if (nativeRows.length !== inputs.length + contactInputs.length)
    throw new Error(
        `native returned ${nativeRows.length} rows for ${inputs.length + contactInputs.length} cases`,
    );

const actualRows = inputs.map(runPublic);
for (let operation = 0; operation < operations.length; operation++) {
    test(`${operations[operation]}: all six public shape kinds equal Box3D SSE2 bits`, () => {
        for (let i = 0; i < inputs.length; i++) {
            const input = inputs[i];
            if (input.operation !== operation) continue;
            if (JSON.stringify(actualRows[i]) !== JSON.stringify(nativeRows[i])) {
                console.info(
                    JSON.stringify({
                        operation: operations[operation],
                        shape: names[input.kind],
                        case: input.label,
                        input: input.words.map(hex),
                        actual: actualRows[i],
                        native: nativeRows[i],
                    }),
                );
            }
            expect(actualRows[i]).toEqual(nativeRows[i]);
        }
    });
}

for (const kind of [3, 4, 5]) {
    test(`${names[kind]} contacts: generated sphere/capsule/hull public manifolds equal Box3D SSE2 bits`, () => {
        const rows = contactInputs
            .map((input, i) => ({ input, native: nativeRows[inputs.length + i] }))
            .filter(
                (row) =>
                    row.input.compositeKind === kind &&
                    !(kind === 3 && row.input.label === "seeded 0 sphere"),
            );
        const mismatches: string[] = [];
        const untouched: string[] = [];
        for (const { input, native } of rows) {
            const actual = contactPublic(input);
            if (Number.parseInt(actual[3], 16) === 0) untouched.push(input.label);
            if (JSON.stringify(actual) !== JSON.stringify(native)) {
                mismatches.push(input.label);
                console.info(
                    JSON.stringify({
                        shape: names[kind],
                        convex: names[input.convexKind],
                        case: input.label,
                        input: input.words.map(hex),
                        actual,
                        native,
                    }),
                );
            }
        }
        expect(untouched).toEqual([]);
        expect(mismatches).toEqual([]);
    });
}

test.todo("triangle_manifold.rs:578-579 vs math_functions.c:554-555: seeded mesh sphere contact preserves native barycentrics", () => {
    const i = contactInputs.findIndex(
        (input) => input.compositeKind === 3 && input.label === "seeded 0 sphere",
    );
    const actual = contactPublic(contactInputs[i]);
    expect(Number.parseInt(actual[3], 16)).toBeGreaterThan(0);
    expect(actual).toEqual(nativeRows[inputs.length + i]);
});
