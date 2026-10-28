import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BodySamplePlugin } from "../src/bodies";
import { CastState, moverOrigin, rayOrigin, rayTranslation } from "../src/cast";
import { castDrawing } from "../src/cast-presentation";
import { JOINT_DOT, JOINT_INK, JOINT_LINE } from "../src/joint-presentation";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
const bits = (n: number) =>
    new Uint32Array(new Float32Array([n]).buffer)[0]!.toString(16).padStart(8, "0");
test("Cast holds every native query invocation and draws its unstepped results", async () => {
    const gold = (await Bun.file(new URL("./cast.queries.gold.txt", import.meta.url)).text())
        .trim()
        .split("\n");
    const app = await createApp({
        defaults: false,
        plugins: [BodySamplePlugin],
        setup(world) {
            createScene(world, "cast");
        },
    });
    const lines: string[] = [];
    try {
        let hash: bigint | undefined;
        for (let tick = 0; tick < 600; tick++) {
            app.world.tick();
            hash ??= hashPhysics(app.world);
            expect(hashPhysics(app.world)).toBe(hash);
            const s = app.world.resource(CastState);
            for (const [hit, r] of [
                [s.rayHit, s.ray],
                [s.sphereHit, s.sphere],
            ] as const) {
                lines.push(
                    `cast ${Number(hit)} ${[r.point.x, r.point.y, r.point.z, r.normal.x, r.normal.y, r.normal.z, r.fraction].map(bits).join(" ")} ${r.triangleIndex} ${r.childIndex} ${r.userMaterialId}`,
                );
            }
            lines.push(`overlap ${Number(s.overlap)}`, `mover ${s.planes.count}`);
            for (let i = 0; i < s.planes.count; i++) {
                const { plane: p, shape, body } = s.planes.records[i]!;
                expect(shape).toBe(s.body);
                expect(body).toBe(s.body);
                lines.push(
                    `plane ${[p.plane.normal.x, p.plane.normal.y, p.plane.normal.z, p.plane.offset, p.point.x, p.point.y, p.point.z].map(bits).join(" ")} ${p.triangleIndex} ${p.childIndex} ${p.materialIndex}`,
                );
            }
            const drawing = castDrawing(app.world);
            expect(drawing.segments).toContainEqual({
                a: [rayOrigin.x, rayOrigin.y, rayOrigin.z],
                b: [
                    rayOrigin.x + rayTranslation.x,
                    rayOrigin.y + rayTranslation.y,
                    rayOrigin.z + rayTranslation.z,
                ],
                color: JOINT_INK,
            });
            expect(drawing.marks).toEqual(
                s.planes.records
                    .slice(0, s.planes.count)
                    .map(({ plane }) => [
                        moverOrigin.x + plane.point.x,
                        moverOrigin.y + plane.point.y,
                        moverOrigin.z + plane.point.z,
                    ]),
            );
            expect(drawing.segments.every((s) => s.color === JOINT_INK)).toBe(true);
        }
        expect(lines).toEqual(gold);
        const s = app.world.resource(CastState);
        const before = castDrawing(app.world);
        const hullEdges = shapeEdges(app.world, app.world.get(s.body, Shape)!);
        expect(before.segments).toHaveLength(hullEdges.length + 1 + 3 * 32 * 2 + 2 * 34 * 2 + 1);
        expect(JOINT_LINE).toEqual({ width: 1, depthBias: -1 });
        expect(JOINT_DOT).toEqual({ width: 4, depthBias: -1 });
        s.overlap = false;
        s.rayHit = true;
        Object.assign(s.ray, { point: { x: -9.75, y: 3, z: 0 }, normal: { x: 0, y: 0, z: -1 } });
        s.sphereHit = true;
        s.sphere.fraction = 0.25;
        const after = castDrawing(app.world);
        expect(after.marks[0]).toEqual([-9.75, 3, 0]);
        expect(after.segments).toContainEqual({
            a: [-9.75, 3, 0],
            b: [-9.75, 3, -0.2],
            color: JOINT_INK,
        });
        expect(after.segments.filter((s) => s.color === 0x797369)).toHaveLength(68);
        const sphereEnds = after.segments
            .slice(hullEdges.length + 2 + 96, hullEdges.length + 2 + 192)
            .flatMap((s) => [s.a[0], s.b[0]]);
        expect(Math.min(...sphereEnds)).toBeCloseTo(-12.7);
        expect(Math.max(...sphereEnds)).toBeCloseTo(-12.3);
    } finally {
        app.dispose();
    }
});
