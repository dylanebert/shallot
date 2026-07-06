import { describe, expect, test } from "bun:test";
import {
    axisDrag,
    closestAxisT,
    cursorRay,
    GIZMO_PX,
    type Glyph,
    gizmoScale,
    glyphs,
    handleSegments,
    localAxes,
    Move,
    manipulatorFor,
    type Pose,
    pickHandles,
    planeDrag,
    pointInQuad,
    project,
    type Ray,
    RING_SCALE,
    Rotate,
    ringAngle,
    SCREEN_RING_SCALE,
    Scale,
    type Vec3,
    WORLD_AXES,
} from "./gizmo";
import { Tool } from "./tool";

// camera-forward for the identity viewProj `I`: its centre pixel shoots straight down +z (see cursorRay
// tests below), so the camera looks along +z.
const EYE: Vec3 = [0, 0, 1];

// a ray dropping straight down -z through (x,y) hits the z=0 plane at (x,y,0)
function down(x: number, y: number): Ray {
    return { origin: [x, y, 1], dir: [0, 0, -1] };
}

// rotate a vector by a quaternion (independent reimplementation, for asserting Rotate's output)
function rot(q: readonly [number, number, number, number], v: Vec3): Vec3 {
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ];
}

// identity viewProj: world (x,y,z) → clip (x,y,z,1), so NDC = (x,y). At 200×200, world origin lands at the
// screen centre (100,100), +x at the right edge, +y at the top.
const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

describe("project", () => {
    test("identity maps the origin to centre, axes to the edges", () => {
        expect(project(I, [0, 0, 0], 200, 200)).toEqual({ x: 100, y: 100, behind: false });
        expect(project(I, [1, 0, 0], 200, 200)).toEqual({ x: 200, y: 100, behind: false });
        // +y is up, so it falls to screen y = 0
        expect(project(I, [0, 1, 0], 200, 200)).toEqual({ x: 100, y: 0, behind: false });
    });

    test("a point past the camera flags behind; a point on the camera plane is null", () => {
        // column-major matrix with W = -z (col2's W row = -1) and W's constant term 0
        const M = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0];
        expect(project(M, [0, 0, 1], 200, 200)?.behind).toBe(true);
        expect(project(M, [0, 0, 0], 200, 200)).toBeNull();
    });
});

describe("cursorRay", () => {
    test("identity: centre pixel shoots straight down +z from the origin", () => {
        const r = cursorRay(I, 100, 100, 200, 200);
        expect(r.origin).toEqual([0, 0, 0]);
        expect(r.dir).toEqual([0, 0, 1]);
    });

    test("identity: an off-centre pixel shifts the ray origin, keeps the +z direction", () => {
        const r = cursorRay(I, 200, 100, 200, 200);
        expect(r.origin[0]).toBeCloseTo(1, 6);
        expect(r.dir).toEqual([0, 0, 1]);
    });
});

describe("closestAxisT", () => {
    test("nearest point parameter along an axis to a crossing ray", () => {
        // X axis; a ray dropping straight down through x=3 is nearest the axis at (3,0,0)
        expect(
            closestAxisT([0, 0, 0], [1, 0, 0], { origin: [3, 5, 0], dir: [0, -1, 0] }),
        ).toBeCloseTo(3, 6);
        // Y axis; a ray along -z through y=2 is nearest at (0,2,0)
        expect(
            closestAxisT([0, 0, 0], [0, 1, 0], { origin: [0, 2, 4], dir: [0, 0, -1] }),
        ).toBeCloseTo(2, 6);
    });

    test("a ray parallel to the axis has no unique nearest point", () => {
        expect(closestAxisT([0, 0, 0], [1, 0, 0], { origin: [0, 3, 0], dir: [1, 0, 0] })).toBeNaN();
    });
});

describe("axisDrag", () => {
    test("translates by the change in nearest-point parameter", () => {
        const start = { origin: [1, 5, 0], dir: [0, -1, 0] } as const;
        const now = { origin: [4, 5, 0], dir: [0, -1, 0] } as const;
        expect(axisDrag([0, 0, 0], [1, 0, 0], start, now)).toBeCloseTo(3, 6);
    });

    test("a parallel (degenerate) ray yields no movement", () => {
        const r = { origin: [0, 0, 0], dir: [1, 0, 0] } as const;
        expect(axisDrag([0, 0, 0], [1, 0, 0], r, r)).toBe(0);
    });
});

describe("gizmoScale", () => {
    test("perspective grows the gizmo with distance to keep a constant on-screen size", () => {
        // fov 90° → half = dist·tan(45°) = dist; scale = GIZMO_PX·2·dist / height
        expect(gizmoScale(true, 90, 10, 1000)).toBeCloseTo((GIZMO_PX * 2 * 10) / 1000, 6);
    });
    test("orthographic uses the view size, independent of distance", () => {
        expect(gizmoScale(false, 5, 999, 1000)).toBeCloseTo((GIZMO_PX * 2 * 5) / 1000, 6);
    });
});

describe("Move manipulator", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };

    test("drag translates pos along the handle axis, leaving rot + scale", () => {
        // X axis; the grabbed point at x=1 tracks to x=4, so +3 along X
        const start = { origin: [1, 5, 0], dir: [0, -1, 0] } as const;
        const now = { origin: [4, 5, 0], dir: [0, -1, 0] } as const;
        const np = Move.drag(0, [0, 0, 0], WORLD_AXES, pose, start, now);
        expect(np.pos[0]).toBeCloseTo(3, 6);
        expect(np.pos[1]).toBe(0);
        expect(np.pos[2]).toBe(0);
        expect(np.rot).toEqual([0, 0, 0, 1]);
        expect(np.scale).toEqual([1, 1, 1]);
    });

    test("a degenerate (axis-parallel) ray doesn't move the entity", () => {
        const r = { origin: [0, 0, 0], dir: [1, 0, 0] } as const;
        const np = Move.drag(0, [0, 0, 0], WORLD_AXES, { ...pose, pos: [2, 0, 0] }, r, r);
        expect(np.pos).toEqual([2, 0, 0]);
    });
});

describe("Scale manipulator", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };

    test("drag scales the axis by the cursor distance ratio, leaving siblings + pos", () => {
        // X axis param: grab at t0 = 2, drag to t1 = 4 → factor 2
        const start = { origin: [2, 5, 0], dir: [0, -1, 0] } as const;
        const now = { origin: [4, 5, 0], dir: [0, -1, 0] } as const;
        const np = Scale.drag(0, [0, 0, 0], WORLD_AXES, pose, start, now);
        expect(np.scale[0]).toBeCloseTo(2, 6);
        expect(np.scale[1]).toBe(1);
        expect(np.scale[2]).toBe(1);
        expect(np.pos).toEqual([0, 0, 0]);
    });

    test("a grab at the gizmo origin (t0 ≈ 0) is a no-op, not an explosion", () => {
        const start = { origin: [0, 5, 0], dir: [0, -1, 0] } as const; // t0 = 0
        const now = { origin: [4, 5, 0], dir: [0, -1, 0] } as const;
        const np = Scale.drag(0, [0, 0, 0], WORLD_AXES, { ...pose, scale: [3, 3, 3] }, start, now);
        expect(np.scale).toEqual([3, 3, 3]);
    });

    test("scale is local-only: the X handle scales lane 0 along the object's local X, even rotated", () => {
        // box turned 90° about Z → local X points along world +Y. The App always passes the local frame to
        // Scale (a per-axis WORLD scale of a rotated object needs shear TRS can't hold — three.js/Unity
        // force local), so dragging the X handle along local X (world Y) grows lane 0, never a world lane.
        const s = Math.SQRT1_2;
        const rotated: Pose = { pos: [0, 0, 0], rot: [0, 0, s, s], scale: [1, 1, 1] };
        const axes = localAxes(rotated.rot);
        const start = { origin: [0, 2, 1], dir: [0, 0, -1] } as const; // along local X (world Y), t0 = 2
        const now = { origin: [0, 4, 1], dir: [0, 0, -1] } as const; // t1 = 4 → ×2
        const np = Scale.drag(0, [0, 0, 0], axes, rotated, start, now);
        expect(np.scale[0]).toBeCloseTo(2, 6); // local X lane
        expect(np.scale[1]).toBe(1);
        expect(np.scale[2]).toBe(1);
    });
});

describe("manipulatorFor", () => {
    test("Move, Rotate, and Scale drive a manipulator; Select picks only", () => {
        expect(manipulatorFor(Tool.Move)).toBe(Move);
        expect(manipulatorFor(Tool.Rotate)).toBe(Rotate);
        expect(manipulatorFor(Tool.Scale)).toBe(Scale);
        expect(manipulatorFor(Tool.Select)).toBeNull();
    });
});

describe("localAxes", () => {
    test("identity gives the world frame", () => {
        const a = localAxes([0, 0, 0, 1]);
        expect(a[0]).toEqual([1, 0, 0]);
        expect(a[1]).toEqual([0, 1, 0]);
        expect(a[2]).toEqual([0, 0, 1]);
    });

    test("a +90° turn about Z sends local X to world +Y", () => {
        const s = Math.SQRT1_2;
        const a = localAxes([0, 0, s, s]); // quat for +90° about Z
        expect(a[0][0]).toBeCloseTo(0, 6);
        expect(a[0][1]).toBeCloseTo(1, 6);
        expect(a[1][0]).toBeCloseTo(-1, 6);
        expect(a[2]).toEqual([0, 0, 1]);
    });
});

describe("planeDrag", () => {
    test("translates by the in-plane delta of the two rays' plane hits", () => {
        // XY plane (normal Z) through the origin; hits move (1,1)→(3,2)
        const d = planeDrag([0, 0, 0], [0, 0, 1], down(1, 1), down(3, 2));
        expect(d[0]).toBeCloseTo(2, 6);
        expect(d[1]).toBeCloseTo(1, 6);
        expect(d[2]).toBeCloseTo(0, 6);
    });

    test("a ray parallel to the plane yields no movement", () => {
        const r: Ray = { origin: [0, 0, 0], dir: [1, 0, 0] }; // in the XY plane
        expect(planeDrag([0, 0, 0], [0, 0, 1], r, r)).toEqual([0, 0, 0]);
    });
});

describe("ringAngle", () => {
    test("the signed angle swept in the ring plane", () => {
        // about Z: hit at (1,0)→(0,1) is +90°
        expect(ringAngle([0, 0, 0], [0, 0, 1], down(1, 0), down(0, 1))).toBeCloseTo(Math.PI / 2, 6);
    });

    test("reversing start and now flips the sign", () => {
        expect(ringAngle([0, 0, 0], [0, 0, 1], down(0, 1), down(1, 0))).toBeCloseTo(
            -Math.PI / 2,
            6,
        );
    });

    test("a ray parallel to the ring plane sweeps nothing", () => {
        const r: Ray = { origin: [0, 0, 0], dir: [1, 0, 0] };
        expect(ringAngle([0, 0, 0], [0, 0, 1], r, r)).toBe(0);
    });
});

describe("Rotate manipulator", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };

    test("the Z ring rotates the orientation about Z, leaving pos + scale", () => {
        // ring Z handle id = 8; a (1,0)→(0,1) sweep is +90° about Z
        const np = Rotate.drag(8, [0, 0, 0], WORLD_AXES, pose, down(1, 0), down(0, 1));
        // the new orientation sends world +X to +Y
        const x = rot(np.rot, [1, 0, 0]);
        expect(x[0]).toBeCloseTo(0, 6);
        expect(x[1]).toBeCloseTo(1, 6);
        expect(np.pos).toEqual([0, 0, 0]);
        expect(np.scale).toEqual([1, 1, 1]);
    });

    test("an entity off the pivot orbits the anchor", () => {
        const off: Pose = { ...pose, pos: [2, 0, 0] };
        const np = Rotate.drag(8, [0, 0, 0], WORLD_AXES, off, down(1, 0), down(0, 1));
        expect(np.pos[0]).toBeCloseTo(0, 6);
        expect(np.pos[1]).toBeCloseTo(2, 6); // (2,0,0) orbited +90° about Z → (0,2,0)
    });

    test("snap quantizes the swept angle to 15°", () => {
        // sweep ~20° (down(1,0)→ a point 20° around) snaps to 15°
        const a = (20 * Math.PI) / 180;
        const np = Rotate.drag(
            8,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            down(1, 0),
            down(Math.cos(a), Math.sin(a)),
            [0, 0, 1],
            true,
        );
        // the resulting half-angle quat z-component is sin(7.5°)
        expect(np.rot[2]).toBeCloseTo(Math.sin(Math.PI / 24), 5);
    });
});

describe("Rotate trackball (screen-facing) ring", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };

    test("the trackball handle (id 11) rolls about the view axis, ignoring the frame", () => {
        // eye looks down +z, so a (1,0)→(0,1) sweep is +90° about the view axis (z). Pass a rotated
        // frame as `axes` to prove the trackball reads `eye`, not the frame — the result still rolls in z.
        const tilted = localAxes([Math.sin(Math.PI / 8), 0, 0, Math.cos(Math.PI / 8)]); // 45° about X
        const np = Rotate.drag(11, [0, 0, 0], tilted, pose, down(1, 0), down(0, 1), EYE);
        const x = rot(np.rot, [1, 0, 0]);
        expect(x[0]).toBeCloseTo(0, 6);
        expect(x[1]).toBeCloseTo(1, 6); // +X → +Y (rotation about z)
        expect(np.pos).toEqual([0, 0, 0]);
        expect(np.scale).toEqual([1, 1, 1]);
    });

    test("the trackball ring projects as a polyline at the screen-ring radius and is pickable", () => {
        const set = Rotate.handles;
        const gs = glyphs(set, [0, 0, 0], WORLD_AXES, I, 200, 200, 1, EYE);
        const ring = gs.find((g): g is Extract<Glyph, { kind: "ring" }> => g?.id === 11) ?? null;
        expect(ring?.kind).toBe("ring");
        // world radius 1·SCREEN_RING_SCALE, projected with I at 200px → centre 100 + radius·100
        const rimX = 100 + SCREEN_RING_SCALE * 100;
        expect(Math.max(...ring!.pts.filter((_, i) => i % 2 === 0))).toBeCloseTo(rimX, 4);
        // a cursor on the screen-ring rim picks the trackball (id 11), not the inner Z ring (RING_SCALE)
        expect(
            pickHandles(set, { x: rimX, y: 100 }, [0, 0, 0], WORLD_AXES, I, 200, 200, EYE, 1),
        ).toBe(11);
    });
});

describe("Rotate free arcball (interior disc)", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };

    test("the free handle (id 12) tumbles about an in-screen axis (arcball)", () => {
        // eye looks down +z over an arcball of radius 1 (scale = 1/SCREEN_RING_SCALE). Dragging from the
        // centre (ball near-point = +z) to halfway out in +x maps +z → (0.5, 0, √0.75): a +30° roll about +y.
        const scale = 1 / SCREEN_RING_SCALE;
        const np = Rotate.drag(
            12,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            down(0, 0),
            down(0.5, 0),
            EYE,
            false,
            scale,
        );
        const z = rot(np.rot, [0, 0, 1]);
        expect(z[0]).toBeCloseTo(0.5, 6);
        expect(z[1]).toBeCloseTo(0, 6);
        expect(z[2]).toBeCloseTo(Math.sqrt(0.75), 6);
        expect(np.pos).toEqual([0, 0, 0]);
        expect(np.scale).toEqual([1, 1, 1]);
    });

    test("the tumble is the same whether the ray points into the scene or back at the camera (reverse-Z)", () => {
        // the engine's reverse-Z cursorRay puts the ray origin on the far side with dir back toward the
        // camera. The arcball maps through the camera plane, so it must give the SAME tumble as a forward
        // ray — not its mirror (the old ray-sphere hit picked the away-from-camera face here, inverting it).
        const scale = 1 / SCREEN_RING_SCALE;
        const revZ = (x: number, y: number): Ray => ({ origin: [x, y, -1], dir: [0, 0, 1] });
        const np = Rotate.drag(
            12,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            revZ(0, 0),
            revZ(0.5, 0),
            EYE,
            false,
            scale,
        );
        const z = rot(np.rot, [0, 0, 1]);
        expect(z[0]).toBeCloseTo(0.5, 6); // +X, not −X (the inverted case)
        expect(z[2]).toBeCloseTo(Math.sqrt(0.75), 6);
    });

    test("grabbing inside the gizmo disc, off every ring, picks the free handle", () => {
        // scale 1 at 200px: axis rings ~78px, the roll ring ~95px; (130,130) is ~42px out — on no ring rim
        const id = pickHandles(
            Rotate.handles,
            { x: 130, y: 130 },
            [0, 0, 0],
            WORLD_AXES,
            I,
            200,
            200,
            EYE,
            1,
        );
        expect(id).toBe(12);
    });

    test("an axis ring still wins over the interior disc on its rim", () => {
        // a 45° point on the Z ring (xy-plane circle, ~78px) — clear of the edge-on X/Y ring lines + the disc
        const rx = 100 + RING_SCALE * 100 * Math.SQRT1_2;
        const ry = 100 - RING_SCALE * 100 * Math.SQRT1_2;
        const id = pickHandles(
            Rotate.handles,
            { x: rx, y: ry },
            [0, 0, 0],
            WORLD_AXES,
            I,
            200,
            200,
            EYE,
            1,
        );
        expect(id).toBe(8); // the Z ring, not the free disc
    });
});

describe("Move plane handles", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };

    test("the XY-plane handle (id 5) translates on two axes; snap grids the delta", () => {
        const np = Move.drag(
            5,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            down(0.2, 0.2),
            down(2.1, 0.9),
            EYE,
            true,
        );
        expect(np.pos[0]).toBeCloseTo(2, 6); // delta 1.9 → grid 2
        expect(np.pos[1]).toBeCloseTo(1, 6); // delta 0.7 → grid 1
    });
});

describe("Scale uniform handle", () => {
    const pose: Pose = { pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] };
    // gizmo on-screen world size — the drag is measured against it, so the feel is zoom-independent (a
    // camera distance read from the reverse-Z cursor ray's far-plane origin would be ~200× too large). The
    // factor is f = 1 + dot(delta, diag)/Gs, diag = the screen up-right unit (PlayCanvas's scale model).
    // EYE = +z (toward camera) → screen up = +Y, right = +X, so diag = (1, 1, 0)/√2.
    const Gs = 0.5;

    test("the uniform handle (id 10) scales all axes by the signed up-right diagonal, finite at the centre", () => {
        // grabbed AT the centre; up-right (0.1, 0.1) → dot with (1,1,0)/√2 = 0.2/√2 = 0.1414; /Gs → +0.2828
        const np = Scale.drag(
            10,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            down(0, 0),
            down(0.1, 0.1),
            EYE,
            false,
            Gs,
        );
        expect(np.scale[0]).toBeCloseTo(1.2828, 4);
        expect(np.scale[1]).toBeCloseTo(1.2828, 4);
        expect(np.scale[2]).toBeCloseTo(1.2828, 4);
    });

    test("dragging down-left shrinks — the signed inverse direction (so scale-down works)", () => {
        // down-left (−0.1, −0.1) → −0.1414 along the diagonal → 1 − 0.2828, a positive factor below 1
        const np = Scale.drag(
            10,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            down(0, 0),
            down(-0.1, -0.1),
            EYE,
            false,
            Gs,
        );
        expect(np.scale[0]).toBeCloseTo(0.7172, 4);
    });

    test("a pure-axis drag projects onto the diagonal (1/√2 of its length), still finite at the centre", () => {
        // straight up (0, 0.1) → 0.1·(1/√2) = 0.0707; /Gs → +0.1414
        const np = Scale.drag(
            10,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            down(0, 0),
            down(0, 0.1),
            EYE,
            false,
            Gs,
        );
        expect(np.scale[0]).toBeCloseTo(1.1414, 4);
    });

    test("scales when the view looks straight down world up (screen-up no longer degenerates)", () => {
        // eye = +Y (top-down). A ray straight down to the xz plane: origin (x,1,z), dir −Y, hits (x,0,z).
        const topDown: Vec3 = [0, 1, 0];
        const ray = (x: number, z: number): Ray => ({ origin: [x, 1, z], dir: [0, -1, 0] });
        // up falls back to +Z, right = −X, so the diagonal is (−1,0,1)/√2; a +Z drag of 0.5 → 0.5/√2; /Gs → 1/√2
        const np = Scale.drag(
            10,
            [0, 0, 0],
            WORLD_AXES,
            pose,
            ray(0, 0),
            ray(0, 0.5),
            topDown,
            false,
            Gs,
        );
        const f = 1 + Math.SQRT1_2;
        expect(np.scale[0]).toBeCloseTo(f, 6);
        expect(np.scale[1]).toBeCloseTo(f, 6);
        expect(np.scale[2]).toBeCloseTo(f, 6);
    });
});

describe("pointInQuad", () => {
    const quad = [0, 0, 10, 0, 10, 10, 0, 10];
    test("inside is true, outside is false", () => {
        expect(pointInQuad(5, 5, quad)).toBe(true);
        expect(pointInQuad(15, 5, quad)).toBe(false);
        expect(pointInQuad(-1, 5, quad)).toBe(false);
    });
});

// the gizmo extent is a world-space size now; 0.5 world units lands the handles on-screen under the
// identity viewProj (origin + axis·0.5 projects to half-way to the edge)
const SCALE = 0.5;

describe("glyphs", () => {
    test("Move yields axis + plane geometry, edge-on planes culled", () => {
        // looking down +z: the XY plane (normal Z, id 5) faces the camera; the YZ/XZ planes are edge-on
        const gs = glyphs(Move.handles, [0, 0, 0], WORLD_AXES, I, 200, 200, SCALE, EYE).filter(
            Boolean,
        ) as Glyph[];
        const byId = new Map(gs.map((g) => [g.id, g]));
        expect(byId.get(5)?.kind).toBe("quad"); // XY plane visible
        expect(byId.has(3)).toBe(false); // YZ plane edge-on → culled
        expect(byId.has(4)).toBe(false); // XZ plane edge-on → culled
        const ax = byId.get(0);
        expect(ax?.kind).toBe("axis");
        expect(ax?.kind === "axis" && ax.cap).toBe("arrow"); // Move axes carry the arrowhead
    });

    test("the X axis projects to a horizontal segment from the centre", () => {
        const gs = glyphs(Move.handles, [0, 0, 0], WORLD_AXES, I, 200, 200, SCALE, EYE);
        const ax = gs.find((g) => g?.id === 0);
        // origin (0,0,0) → centre (100,100); origin + X·0.5 = (0.5,0,0) → (150,100)
        expect(ax?.kind === "axis" && [ax.ox, ax.oy, ax.ex, ax.ey]).toEqual([100, 100, 150, 100]);
    });

    test("Scale axes carry the box cap", () => {
        const gs = glyphs(
            Scale.handles,
            [0, 0, 0],
            WORLD_AXES,
            I,
            200,
            200,
            SCALE,
            EYE,
            "box",
        ).filter(Boolean) as Glyph[];
        const ax = gs.find((g) => g.id === 0);
        expect(ax?.kind === "axis" && ax.cap).toBe("box");
    });

    test("Rotate yields ring polylines", () => {
        const gs = glyphs(Rotate.handles, [0, 0, 0], WORLD_AXES, I, 200, 200, SCALE, EYE).filter(
            Boolean,
        ) as Glyph[];
        const ring = gs.find((g) => g.kind === "ring");
        expect(ring?.kind === "ring" && ring.pts.length).toBeGreaterThan(8);
    });
});

describe("handleSegments", () => {
    test("Move yields an axis shaft per axis + the 2 outer edges of the visible plane (flush)", () => {
        const segs = handleSegments(Move.handles, [0, 0, 0], WORLD_AXES, SCALE, EYE);
        const count = new Map<number, number>();
        for (const s of segs) count.set(s.id, (count.get(s.id) ?? 0) + 1);
        expect(count.get(0)).toBe(1); // axis X shaft
        expect(count.get(1)).toBe(1);
        expect(count.get(2)).toBe(1);
        expect(count.get(5)).toBe(2); // XY plane (id 5): only the 2 outer edges (inner 2 lie on the axes)
        expect(count.has(3)).toBe(false); // YZ / XZ planes edge-on → no edges
        expect(count.has(4)).toBe(false);
        const ax = segs.find((s) => s.id === 0);
        expect(ax?.a).toEqual([0, 0, 0]); // shaft runs origin → origin + X·scale
        expect(ax?.b[0]).toBeCloseTo(SCALE, 6);
    });

    test("a ring is a closed polyline whose every point sits on the (inset) circle", () => {
        const ring = handleSegments(Rotate.handles, [0, 0, 0], WORLD_AXES, SCALE, EYE).filter(
            (s) => s.id === 8, // Z ring
        );
        expect(ring.length).toBe(48);
        const r = SCALE * RING_SCALE; // rings sit inside the axis shafts
        for (const s of ring) {
            expect(Math.hypot(s.a[0], s.a[1], s.a[2])).toBeCloseTo(r, 5);
            expect(Math.hypot(s.b[0], s.b[1], s.b[2])).toBeCloseTo(r, 5);
        }
    });
});

describe("pickHandles", () => {
    test("the cursor on the XY plane picks the plane, not an axis under it", () => {
        // the XY plane quad sits flush in the +x+y corner from the origin (~x 100–120, y 80–100 at SCALE 0.5)
        expect(
            pickHandles(
                Move.handles,
                { x: 110, y: 90 },
                [0, 0, 0],
                WORLD_AXES,
                I,
                200,
                200,
                EYE,
                SCALE,
            ),
        ).toBe(5);
    });

    test("the cursor on the centre picks the camera-facing plane (Move has no centre handle)", () => {
        // the XY plane's inner corner sits at the origin, so a dead-centre grab lands on the view-facing plane
        expect(
            pickHandles(
                Move.handles,
                { x: 100, y: 100 },
                [0, 0, 0],
                WORLD_AXES,
                I,
                200,
                200,
                EYE,
                SCALE,
            ),
        ).toBe(5);
    });

    test("the cursor away from every handle picks nothing", () => {
        expect(
            pickHandles(
                Move.handles,
                { x: 5, y: 5 },
                [0, 0, 0],
                WORLD_AXES,
                I,
                200,
                200,
                EYE,
                SCALE,
            ),
        ).toBe(-1);
    });
});
