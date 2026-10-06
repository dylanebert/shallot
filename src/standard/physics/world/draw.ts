import { hi32, lo32 } from "../common/bits";
import type { AABB, Pos, WorldTransform } from "../common/math";
import { queryColumns } from "../kernel/querycolumns";
import type { Capsule, Sphere } from "../shapes/geometry";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import type { Mesh } from "../shapes/mesh";
import {
    drawCapsule,
    drawHeightField,
    drawHull,
    drawMesh,
    drawSphere,
    drawTransform,
    drawVector,
} from "../shapes/readgeometry";
import type { WorldState } from "./world";

/** Low-24-bit packed RGB colors (b3HexColor). */
export const DebugColor = {
    red: 0xff0000,
    orange: 0xffa500,
    yellow: 0xffff00,
    lime: 0x00ff00,
    turquoise: 0x40e0d0,
    slateGray: 0x708090,
    lightSlateGray: 0x778899,
    darkGray: 0xa9a9a9,
    tan: 0xd2b48c,
    steelBlue: 0x4682b4,
    lightSteelBlue: 0xb0c4de,
    wheat: 0xf5deb3,
    gold: 0xffd700,
    darkSeaGreen: 0x8fbc8f,
    plum: 0xdda0dd,
    yellowGreen: 0x9acd32,
    azure: 0xf0ffff,
    white: 0xffffff,
} as const;

/** Debug callbacks receive independent geometry and transforms. Colors are packed RGB. */
export type DebugDraw = {
    /** The transform rotation drives the sphere's surface pattern. */
    drawSolidSphere(transform: WorldTransform, sphere: Sphere, color: number): void;
    /** A capsule whose local +x is the long axis. */
    drawSolidCapsule(transform: WorldTransform, capsule: Capsule, color: number): void;
    drawSolidHull(transform: WorldTransform, hull: HullData, color: number): void;
    drawSolidMesh(transform: WorldTransform, mesh: Mesh, color: number): void;
    drawSolidHeightField(
        transform: WorldTransform,
        heightField: HeightFieldData,
        color: number,
    ): void;
    drawSegment(p1: Pos, p2: Pos, color: number): void;
    /** Size in pixels. */
    drawPoint(p: Pos, size: number, color: number): void;
    drawTransform(transform: WorldTransform): void;
    drawAabb(aabb: AABB, color: number): void;
    drawString(p: Pos, s: string, color: number): void;
    /** World bounds culling shapes. */
    drawingBounds: AABB;
    jointScale: number;
    forceScale: number;
    drawShapes: boolean;
    drawJoints: boolean;
    drawJointExtras: boolean;
    drawBounds: boolean;
    drawMass: boolean;
    context: unknown;
};
const noop = (): void => {};
/** No-op callbacks, ±100 m bounds, and every category off (b3DefaultDebugDraw). */
export function defaultDebugDraw(): DebugDraw {
    return {
        drawSolidSphere: noop,
        drawSolidCapsule: noop,
        drawSolidHull: noop,
        drawSolidMesh: noop,
        drawSolidHeightField: noop,
        drawSegment: noop,
        drawPoint: noop,
        drawTransform: noop,
        drawAabb: noop,
        drawString: noop,
        drawingBounds: {
            lowerBound: { x: -100, y: -100, z: -100 },
            upperBound: { x: 100, y: 100, z: 100 },
        },
        jointScale: 1,
        forceScale: 1,
        drawShapes: false,
        drawJoints: false,
        drawJointExtras: false,
        drawBounds: false,
        drawMass: false,
        context: null,
    };
}
const views = new WeakMap<WorldState, { view: DataView; mask: bigint; hi: number; lo: number }>();

/** Dispatch the kernel's read-only snapshot of shapes, bounds, mass and joints. */
export function worldDraw(world: WorldState, draw: DebugDraw, maskBits: bigint): void {
    const flags =
        Number(draw.drawShapes) |
        (Number(draw.drawBounds) << 1) |
        (Number(draw.drawMass) << 2) |
        (Number(draw.drawJoints) << 3) |
        (Number(draw.drawJointExtras) << 4);
    const q = queryColumns(world);
    const k = q.prepare(draw.drawingBounds.lowerBound);
    q.bounds(draw.drawingBounds);
    let state = views.get(world);
    if (!state) {
        state = {
            view: new DataView(k.memory.buffer),
            mask: maskBits,
            hi: hi32(maskBits),
            lo: lo32(maskBits),
        };
        views.set(world, state);
    } else if (state.mask !== maskBits) {
        state.mask = maskBits;
        state.hi = hi32(maskBits);
        state.lo = lo32(maskBits);
    }
    const start = k.worldDraw(world.worldId, flags, state.hi, state.lo, world.invH);
    const end = k.worldDrawLen();
    try {
        for (let position = start; position < end; ) {
            // A nested draw may relocate the buffer or grow linear memory.
            if (state.view.buffer !== k.memory.buffer) state.view = new DataView(k.memory.buffer);
            const v = state.view;
            const o = k.worldDrawPtr() + position * 4;
            position += v.getUint32(o + 4, true);
            const kind = v.getUint32(o, true),
                color = v.getUint32(o + 8, true),
                p = o + 12;
            switch (kind) {
                case 0:
                    draw.drawSolidCapsule(drawTransform(v, p), drawCapsule(v, p + 28), color);
                    break;
                case 2:
                    draw.drawSolidHeightField(
                        drawTransform(v, p),
                        drawHeightField(v, p + 40),
                        color,
                    );
                    break;
                case 3:
                    draw.drawSolidHull(drawTransform(v, p), drawHull(v, p + 40), color);
                    break;
                case 4:
                    draw.drawSolidMesh(drawTransform(v, p), drawMesh(v, p + 28), color);
                    break;
                case 5:
                    draw.drawSolidSphere(drawTransform(v, p), drawSphere(v, p + 28), color);
                    break;
                case 6:
                    draw.drawSegment(drawVector(v, p), drawVector(v, p + 12), color);
                    break;
                case 7:
                    draw.drawPoint(drawVector(v, p), v.getFloat32(p + 12, true), color);
                    break;
                case 8:
                    draw.drawTransform(drawTransform(v, p));
                    break;
                case 9:
                    draw.drawAabb(
                        { lowerBound: drawVector(v, p), upperBound: drawVector(v, p + 12) },
                        color,
                    );
                    break;
                case 10:
                    draw.drawString(
                        drawVector(v, p),
                        `  ${v.getFloat32(p + 12, true).toFixed(2)}`,
                        color,
                    );
                    break;
                case 11:
                    draw.drawString(
                        drawVector(v, p),
                        `f = ${v.getFloat32(p + 12, true).toPrecision(4)}, t = ${v.getFloat32(p + 16, true).toPrecision(4)}`,
                        color,
                    );
                    break;
            }
        }
    } finally {
        k.worldDrawRelease(start);
    }
}
