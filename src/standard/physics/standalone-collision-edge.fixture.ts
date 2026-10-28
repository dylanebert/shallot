import type { World } from "../../engine/ecs";
import {
    type DistanceInput,
    getSweepTransform,
    type ShapeCastPairInput,
    type Sweep,
    shapeCast,
    shapeDistance,
    type TOIInput,
    TOIState,
    timeOfImpact,
} from ".";
import {
    bits,
    collisionCases,
    distanceWords,
    emptyCache,
    hex,
    toiWords,
    vf,
} from "./standalone-collision.fixture";
import failed from "./standalone-collision-failed.json";
import history from "./standalone-collision-history.json";

type Case =
    | { operation: 7 | 8; input: DistanceInput; capacity: number; reset: boolean }
    | { operation: 9; input: ShapeCastPairInput }
    | { operation: 5; input: TOIInput }
    | { operation: 20; input: Sweep; fraction: number };

export const edgeCases: Case[] = [];
const moving = collisionCases.find((c) => c.label === "1-3-5")!.input;
for (let i = 0; i < 12; i++) {
    edgeCases.push({
        operation: 8,
        capacity: 32,
        reset: i === 0,
        input: {
            ...moving,
            transform: { ...moving.transform, p: { x: 5 - i * 0.04, y: 0.1 + i * 0.03, z: 0.2 } },
        },
    });
}
for (let capacity = 1; capacity <= 32; capacity++) {
    edgeCases.push({ operation: 7, input: history, capacity, reset: true });
}
for (const { input } of collisionCases) {
    edgeCases.push({
        operation: 9,
        input: {
            ...input,
            translationB: { x: -10, y: 0, z: 0 },
            maxFraction: 1,
            canEncroach: true,
        },
    });
}
edgeCases.push({ operation: 5, input: failed });
for (const input of [failed.sweepA, failed.sweepB]) {
    for (const fraction of [0, 0.125, 0.375, 0.5, 0.875, 1]) {
        edgeCases.push({ operation: 20, input, fraction });
    }
}

export function nativeEdgeInput(): string {
    return `${edgeCases
        .map((c) => {
            const r = Array<number>(834).fill(0);
            const put = (i: number, p: { x: number; y: number; z: number }) =>
                r.splice(i, 3, ...vf(p));
            const sweep = (i: number, s: Sweep) => {
                put(i, s.localCenter);
                put(i + 3, s.c1);
                put(i + 6, s.c2);
                put(i + 9, s.q1.v);
                r[i + 12] = bits(s.q1.s);
                put(i + 13, s.q2.v);
                r[i + 16] = bits(s.q2.s);
            };
            if (c.operation === 20) {
                sweep(800, c.input);
                r[15] = bits(c.fraction);
            } else {
                const input = c.input;
                r[0] = input.proxyA.count;
                r[1] = bits(input.proxyA.radius);
                r[2] = input.proxyB.count;
                r[3] = bits(input.proxyB.radius);
                input.proxyA.points.forEach((p, i) => {
                    put(32 + i * 3, p);
                });
                input.proxyB.points.forEach((p, i) => {
                    put(416 + i * 3, p);
                });
                if (c.operation === 5) {
                    sweep(800, c.input.sweepA);
                    sweep(817, c.input.sweepB);
                    r[15] = bits(c.input.maxFraction);
                } else {
                    put(4, c.input.transform.p);
                    put(7, c.input.transform.q.v);
                    r[10] = bits(c.input.transform.q.s);
                    if (c.operation === 9) {
                        put(12, c.input.translationB);
                        r[15] = bits(c.input.maxFraction);
                        r[11] = Number(c.input.canEncroach);
                    } else {
                        r[11] = Number(c.input.useRadii);
                        r[26] = c.capacity;
                        r[27] = Number(c.reset);
                    }
                }
            }
            return `${c.operation} ${r.length} ${r.map(hex).join(" ")}`;
        })
        .join("\n")}\n21 0\n`;
}

export function publicEdgeOutput(world: World): string {
    let cache = emptyCache();
    return `${edgeCases
        .map((c) => {
            let words: number[];
            switch (c.operation) {
                case 7:
                case 8: {
                    if (c.reset) cache = emptyCache();
                    words = distanceWords(shapeDistance(world, c.input, cache, c.capacity), cache);
                    break;
                }
                case 9: {
                    const cast = shapeCast(world, c.input);
                    words = [
                        Number(cast.hit),
                        cast.iterations,
                        bits(cast.fraction),
                        ...vf(cast.point),
                        ...vf(cast.normal),
                    ];
                    break;
                }
                case 5:
                    words = toiWords(timeOfImpact(world, c.input));
                    break;
                case 20: {
                    const transform = getSweepTransform(world, c.input, c.fraction);
                    words = [...vf(transform.p), ...vf(transform.q.v), bits(transform.q.s)];
                    break;
                }
            }
            return `${c.operation} ${words.map(hex).join(" ")}`;
        })
        .join(
            "\n",
        )}\n21 ${[TOIState.unknown, TOIState.failed, TOIState.overlapped, TOIState.hit, TOIState.separated].map(hex).join(" ")}\n`;
}
