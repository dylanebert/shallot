// Continuous impacts against a static mesh, against native Box3D (mesh-impact.c). The world cases
// put a triangle's plane within about 1e-5 m of a fast box's swept bounds corner, where the
// bounds-triangle overlap test's rounding depends on the order it reads the vertices, under a
// reflecting and a plain scale, as a mesh shape and as a compound's mesh child.
import { expect, setDefaultTimeout, test } from "bun:test";
import gold from "../../crates/physics/tests/mesh_impact.gold.json";
import { World } from "../../src/engine";
import { BodyType, defaultShapeDef, defaultSurfaceMaterial } from "../../src/standard/physics/common/types";
import { init } from "../../src/standard/physics/kernel/kernel";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import { createMesh } from "../../src/standard/physics/shapes/build-nonconvex";
import { createCompound } from "../../src/standard/physics/shapes/compound";
import { createCompoundShape, createHullShape, createMeshShape } from "../../src/standard/physics/shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld } from "../../src/standard/physics/testing";
import { bodyGetLinearVelocity, bodyGetTransform } from "../../src/standard/physics/world/body";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(180_000);

const f = Math.fround;
const view = new DataView(new ArrayBuffer(4));
const bits = (v: number) => (view.setFloat32(0, v), view.getUint32(0));
const value = (word: string) => (view.setUint32(0, Number.parseInt(word, 16)), view.getFloat32(0));
const hex = (v: number) => bits(v).toString(16).padStart(8, "0");

test("native Box3D regenerates the mesh impact gold", () => {
    const input = gold.cases
        .map((c) => ["toi", ...c.vertices, ...c.scale, ...c.c1, ...c.c2].join(" "))
        .join("\n");
    const output = gold.cases.map(({ output: o }) =>
        [o.state, o.fraction, o.distance, ...o.point, ...o.normal, o.usedFallback ? 1 : 0].join(" "),
    );
    expect(nativeSseOutput("mesh-impact.c", `${input}\n`).trim().split("\n")).toEqual(output);
});

// Triangles in planes nx x + y + nz z = k through the box's end-pose lower corner (-0.25, 0.25, -0.25),
// offset by up to 1e-5 m and jittered, in either winding; z is stored negated for the reflecting scale.
function cases(count: number, reflected: boolean): number[][] {
    let seed = 12345;
    const random = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
    const out: number[][] = [];
    for (let i = 0; i < count; i++) {
        const nx = f(0.3 + random());
        const nz = f(0.3 + random());
        const k = f(-0.25 * nx + 0.25 - 0.25 * nz + (random() - 0.5) * 2e-5);
        const vertices = [
            [-2.25, -0.25],
            [1.75, -2.25],
            [-0.25, 1.75],
        ].map(([x, z]) => {
            const px = f(x + (random() - 0.5) * 0.1);
            const pz = f(z + (random() - 0.5) * 0.1);
            return [px, f(k - f(nx * px) - f(nz * pz)), reflected ? f(-pz) : pz];
        });
        if (random() < 0.5) [vertices[1], vertices[2]] = [vertices[2], vertices[1]];
        out.push([...vertices.flat(), 1, 1, reflected ? -1 : 1]);
    }
    return out;
}

function step(world: World, kind: number, w: number[]): string {
    const state = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } }, world);
    try {
        const ground = createTestBody(state);
        const mesh = createMesh({
            vertices: [0, 3, 6].map((i) => ({ x: w[i], y: w[i + 1], z: w[i + 2] })),
            indices: [0, 1, 2],
        })!;
        const scale = { x: w[9], y: w[10], z: w[11] };
        if (kind === 0) createMeshShape(state, ground, defaultShapeDef(), mesh, scale);
        else {
            const compound = createCompound({
                meshes: [
                    {
                        meshData: mesh,
                        transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                        scale,
                        materials: [defaultSurfaceMaterial()],
                        materialCount: 1,
                    },
                ],
            })!;
            createCompoundShape(state, ground, defaultShapeDef(), compound);
        }
        const body = createTestBody(state, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 1.5, z: 0 },
            linearVelocity: { x: 0, y: -60, z: 0 },
        });
        createHullShape(state, body, defaultShapeDef(), makeBoxHull(0.25, 0.25, 0.25));
        stepTestWorld(state, f(1 / 60), 4);
        const t = bodyGetTransform(state, body, {
            p: { x: 0, y: 0, z: 0 },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        });
        const v = bodyGetLinearVelocity(state, body, { x: 0, y: 0, z: 0 });
        return [t.p.x, t.p.y, t.p.z, t.q.v.x, t.q.v.y, t.q.v.z, t.q.s, v.x, v.y, v.z].map(hex).join(" ");
    } finally {
        destroyTestWorld(state);
    }
}

const world = new World();
await init(world, { threads: 0 });
const goldWorlds = gold.cases.map((c) => [...c.vertices, ...c.scale].map(value));

for (const kind of [0, 1]) {
    for (const reflected of [true, false]) {
        const child = kind === 0 ? "mesh shape" : "compound mesh child";
        test(`a fast box stepped over a ${reflected ? "reflected" : "plain"} ${child} near its swept corner ends where native Box3D's does`, () => {
            const set = [...(reflected ? goldWorlds : []), ...cases(20_000, reflected)];
            const input = set.map((w) => `world ${kind} ${w.map(hex).join(" ")}`).join("\n");
            const native = nativeSseOutput("mesh-impact.c", `${input}\n`).trim().split("\n");
            const differing = set.filter((w, i) => step(world, kind, w) !== native[i]);
            expect(differing.map((w) => w.map(hex).join(" "))).toEqual([]);
        });
    }
}
