import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import {
    cloneAndTransformHull,
    computeCapsuleMass,
    computeHullMass,
    createCylinder,
    createHull,
    type MassData,
    makeBoxHull,
    makeScaledBoxHull,
    StandardPhysicsPlugin,
} from "./index";
import { hullImage } from "./shapes/hullbytes";
import gold from "./standalone-geometry.gold.json";

setDefaultTimeout(TIMEOUT.node);

function massWords(m: MassData): number[] {
    const { center: c, inertia: i } = m;
    return Array.from(
        new Uint32Array(
            new Float32Array([
                m.mass,
                c.x,
                c.y,
                c.z,
                i.cx.x,
                i.cx.y,
                i.cx.z,
                i.cy.x,
                i.cy.y,
                i.cy.z,
                i.cz.x,
                i.cz.y,
                i.cz.z,
            ]).buffer,
        ),
    );
}
const a = gold.transform;
const transform = {
    p: { x: a[0], y: a[1], z: a[2] },
    q: { v: { x: a[3], y: a[4], z: a[5] }, s: a[6] },
};

test("public scaled boxes and standalone masses equal native records, including reflections and degenerate capsules", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    try {
        for (const row of gold.boxes) {
            const hull = makeScaledBoxHull(
                { x: 1, y: 0.5, z: 0.25 },
                transform,
                { x: row.scale[0], y: row.scale[1], z: row.scale[2] },
                world,
            );
            const image = hullImage(hull);
            expect(
                Array.from(new Uint32Array(image.buffer, image.byteOffset, image.byteLength / 4)),
            ).toEqual(row.hull);
            expect(massWords(computeHullMass(hull, 2.3, world))).toEqual(row.mass);
        }
        for (const row of gold.capsules) {
            const r = row.input;
            const capsule = {
                center1: { x: r[0], y: r[1], z: r[2] },
                center2: { x: r[3], y: r[4], z: r[5] },
                radius: r[6],
            };
            expect(massWords(computeCapsuleMass(capsule, r[7], world))).toEqual(row.mass);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});

test("construction results own their storage across later builds and mass calls", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    try {
        const cylinder = createCylinder(1, 0.5, 0, 9, world);
        const source = hullImage(cylinder).slice();
        const copy = cloneAndTransformHull(cylinder, transform, { x: -1, y: 2, z: 0.5 }, world)!;
        const mass = computeHullMass(copy, 1, world);
        const saved = structuredClone(mass);
        const box = makeBoxHull(2, 1, 1, world);
        expect(computeHullMass(box, 0, world).mass).toBe(0);
        expect(
            computeCapsuleMass(
                { center1: { x: -1, y: 0, z: 0 }, center2: { x: 1, y: 0, z: 0 }, radius: 1 },
                1,
                world,
            ).mass,
        ).toBeGreaterThan(0);
        const built = createHull(box.points, 8, world)!;
        expect(built.vertexCount).toBe(8);
        expect(createHull([{ x: 0, y: 0, z: 0 }], 8, world)).toBeNull();
        expect(hullImage(cylinder)).toEqual(source);
        expect(mass).toEqual(saved);
        mass.center.x = 999;
        mass.inertia.cx.x = 999;
        expect(computeHullMass(copy, 1, world)).toEqual(saved);
        for (const density of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(() => computeHullMass(box, density, world)).toThrow(RangeError);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
