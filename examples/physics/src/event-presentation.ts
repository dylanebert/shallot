import type { World } from "@dylanebert/shallot";
import { EventSample } from "./events";
import { drawInkPoint, type INK_DOT } from "./ink-presentation";

type Point = [number, number, number];
export function drawSensorHitViews(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof INK_DOT) => void,
): void {
    for (const point of world.resource(EventSample).sensorMarks)
        drawInkPoint(world, camera, point, line);
}

export interface EventMark {
    point: [number, number, number];
    end: [number, number, number];
    label: string;
}

export const EVENT_SEGMENT = { width: 2, depthBias: -1 };
export const EVENT_POINT = { width: 3, depthBias: -1 };

export function drawEventMark(
    mark: EventMark,
    line: (
        a: [number, number, number],
        b: [number, number, number],
        color: number,
        options: typeof EVENT_SEGMENT,
    ) => void,
): void {
    const p = mark.point;
    line(mark.end, p, 0xffb62e, EVENT_SEGMENT);
    line([p[0] - 0.2, p[1], p[2]], [p[0] + 0.2, p[1], p[2]], 0xffb62e, EVENT_POINT);
}
