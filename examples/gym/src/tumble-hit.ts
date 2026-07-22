// The tumble.js `HitEvents` sample (`samples/src/samples/events.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Bodies (alternating sphere/box) drop onto the ground with restitution; each fast
// contact reports a hit event. The sample's impact markers are its `render()` overlay ({@link renderHitEvents}):
// each hit event pushes a fading point + approach line. The sample splits this across `update()` (push/age the
// marks) + `render()` (draw them); both are read-only reads of the world's hit events, so the whole layer folds
// into the render path here — the oracle path (build only) stays untouched.
//
// Creation order is load-bearing for the hash: ground, then the sixteen bodies left to right, sphere/hull
// alternating — the sample's exact order.

import {
    BodyType,
    defaultSurfaceMaterial,
    makeBoxHull,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { Overlay, SampleRender } from "./tumble-overlay";

type HitMark = { point: Vec3; normal: Vec3; speed: number; age: number };
let marks: HitMark[] = [];
let total = 0;

/**
 * Author the Hit Events scene into `world`: sixteen bodies (alternating sphere/box) at varied heights,
 * restitution 0.5, dropped onto the ground with hit events enabled.
 */
export function buildHitEvents(world: World): void {
    marks = [];
    total = 0;
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const material = { ...defaultSurfaceMaterial(), restitution: 0.5, userMaterialId: 7n };
    const box = makeBoxHull(0.5, 0.5, 0.5);
    for (let i = 0; i < 16; ++i) {
        const x = -6 + 0.8 * i;
        const y = 6 + 2 * ((i * 7) % 5);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y, z: 0 },
        });
        if (i % 2 === 0) {
            body.createSphere(
                { density: 1, enableHitEvents: true, baseMaterial: material },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.4 },
            );
        } else {
            body.createHull({ density: 1, enableHitEvents: true, baseMaterial: material }, box);
        }
    }
}

/**
 * Drain the step's hit events into fading impact markers, age them, and draw each as a point + an approach
 * line scaled by contact speed (the sample's `update()` + `render()`, folded — a read-only event drain).
 */
export const renderHitEvents: SampleRender = (draw: Overlay, world: World) => {
    for (const e of world.getContactEvents().hitEvents) {
        marks.push({ point: e.point, normal: e.normal, speed: e.approachSpeed, age: 0 });
        total += 1;
    }
    for (const m of marks) m.age += 1;
    marks = marks.filter((m) => m.age < 30);

    for (const m of marks) {
        const len = Math.min(1.5, 0.1 * m.speed);
        const tip = {
            x: m.point.x - len * m.normal.x,
            y: m.point.y - len * m.normal.y,
            z: m.point.z - len * m.normal.z,
        };
        draw.point(m.point, 8, 0xffe000);
        draw.line(m.point, tip, 0xffe000);
    }
    draw.text(`hits ${total}`);
};
