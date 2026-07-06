import {
    Body,
    Color,
    Part,
    PointLight,
    Shadow,
    ShapeKind,
    type State,
    Transform,
    Volumetric,
} from "@dylanebert/shallot";
import type { JointDef } from "@dylanebert/shallot/physics/core";
import { Meshes, Surfaces } from "@dylanebert/shallot/render/core";

// Spawn helpers + the prop builders (ropes, bridge, pyramid, brick stack). Joints are authored
// imperatively as JointDefs — the gravity gun appends its grab joint after them on the same
// `setJoints` path, and a scene must use ONE path per constraint type (physics ConstraintSystem).
// Every joint's anchors are coincident at the spawn pose, or jointInit's construction guard
// rejects it (the gym `constraints` rope rig is the pattern).

/** marks a body that clacks — the impact-sound contact filter reads it */
export const Brick = {};

/** hex sRGB (0x9a8068) → linear rgb, the space Color.rgba stores */
export function hex(v: number): [number, number, number] {
    const lin = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    return [lin(((v >> 16) & 0xff) / 255), lin(((v >> 8) & 0xff) / 255), lin((v & 0xff) / 255)];
}

/** a Body+Part box; `mass <= 0` is static; `surface` names a registered surface ("grit" walls). */
export function box(
    state: State,
    pos: readonly [number, number, number],
    half: readonly [number, number, number],
    mass: number,
    color: readonly [number, number, number],
    surface?: string,
    quat: readonly [number, number, number, number] = [0, 0, 0, 1],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.quat.set(eid, quat[0], quat[1], quat[2], quat[3]);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    Body.friction.set(eid, 0.6);
    state.add(eid, Part);
    if (surface !== undefined) Part.surface.set(eid, Surfaces.id(surface) ?? 0);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// a Body+Part sphere (collider radius = render radius); `surface` names a registered surface ("stone")
function sphere(
    state: State,
    pos: readonly [number, number, number],
    radius: number,
    mass: number,
    color: readonly [number, number, number],
    surface?: string,
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Sphere);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, 0, 0, 0, radius);
    Body.mass.set(eid, mass);
    Body.friction.set(eid, 0.6);
    state.add(eid, Part);
    Part.mesh.set(eid, Meshes.id("sphere") ?? 0);
    if (surface !== undefined) Part.surface.set(eid, Surfaces.id(surface) ?? 0);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

const ROPE_LINKS = 18;
const ROPE_SEG = 0.21;
const ROPE_PIN = ROPE_SEG / 2;

/**
 * a hanging spherical chain (the gym `constraints` rope rig, vertical): thin bar links spaced exactly
 * one segment apart, each rigid pin's anchors coincident at the link boundary, rotation free. The
 * mount is an invisible static body at the ceiling point. `weight` > 0 hangs a heavy sphere of that
 * diameter at the tip (it clacks — Brick-tagged).
 */
export function rope(
    state: State,
    x: number,
    topY: number,
    z: number,
    weight: number,
    joints: JointDef[],
): void {
    const mount = state.create();
    state.add(mount, Body);
    Body.shape.set(mount, ShapeKind.Box);
    Body.pos.set(mount, x, topY, z, 0);
    Body.halfExtents.set(mount, 0.05, 0.05, 0.05, 0);
    Body.mass.set(mount, 0);

    const linkColor = hex(0x6a5a45); // light oak — the wooden chain links
    let prev = mount;
    for (let i = 0; i < ROPE_LINKS; i++) {
        const y = topY - ROPE_SEG * (i + 0.5);
        const link = box(state, [x, y, z], [0.05, ROPE_PIN - 0.02, 0.05], 0.5, linkColor, "wood");
        joints.push({
            a: prev,
            b: link,
            rA: i === 0 ? [0, 0, 0] : [0, -ROPE_PIN, 0],
            rB: [0, ROPE_PIN, 0],
        });
        prev = link;
    }

    if (weight > 0) {
        // 12 kg on 0.5 kg links — the iterative solver stretches a chain visibly past ~25× mass ratio
        // (the legacy 50-on-0.2 read as "stretched" even at rest), so the weight reads heavy by SIZE
        const r = weight / 2;
        const tipY = topY - ROPE_SEG * ROPE_LINKS;
        const w = sphere(state, [x, tipY - r, z], r, 12, hex(0x575149), "stone");
        state.add(w, Brick);
        joints.push({ a: prev, b: w, rA: [0, -ROPE_PIN, 0], rB: [0, r, 0] });
    }
}

const PLANKS = 10;
const PLANK_W = 1.5;
const PLANK_T = 0.2;
const PLANK_GAP = 0.15;
const SPAN = 8; // the pit's chord, platform edge to platform edge
const SAG = 0.5; // mid-span drape depth — the slack that lets the bridge move (zero slack = a taut rigid row)
const PIN_X = 0.6;

/**
 * a plank bridge draped over the pit along z, ends static, middle planks dynamic. The planks lie on a
 * pre-sagged arc — a chain of rigid pins authored taut (a straight line between fixed ends) has zero
 * slack and geometrically CANNOT sag or deflect, so it reads as a frozen rigid row. Each adjacent pair
 * is tied by two spherical pins at the side corners (lateral stability), coincident at the shared arc
 * point by construction.
 */
export function bridge(state: State, cx: number, cy: number, cz: number, joints: JointDef[]): void {
    const plankColor = hex(0x4d4030); // aged oak — the wood-grain planks
    const anchorColor = hex(0x332a1d); // darker weathered wood — the fixed end mounts read embedded

    // walk the sag parabola y(t) = -4·SAG·t·(1-t) and place PLANKS+1 pins equally spaced in ARC length —
    // each plank spans two consecutive pins, so its pin-to-pin anchors are coincident on the arc
    const M = 2000;
    const at = (t: number): [number, number] => [SPAN * t, -4 * SAG * t * (1 - t)];
    let arc = 0;
    const lens = new Float64Array(M + 1);
    for (let i = 1; i <= M; i++) {
        const [z0, y0] = at((i - 1) / M);
        const [z1, y1] = at(i / M);
        arc += Math.hypot(z1 - z0, y1 - y0);
        lens[i] = arc;
    }
    const pins: [number, number][] = []; // (z', y) along the arc
    for (let k = 0, i = 0; k <= PLANKS; k++) {
        const target = (arc * k) / PLANKS;
        while (i < M && lens[i + 1] < target) i++;
        const f = lens[i + 1] > lens[i] ? (target - lens[i]) / (lens[i + 1] - lens[i]) : 0;
        const [z0, y0] = at(i / M);
        const [z1, y1] = at((i + 1) / M);
        pins.push([z0 + (z1 - z0) * f, y0 + (y1 - y0) * f]);
    }

    const pitch = arc / PLANKS; // pin-to-pin per plank
    const halfLen = pitch / 2 - PLANK_GAP / 2;
    const half: [number, number, number] = [PLANK_W / 2, PLANK_T / 2, halfLen];
    const zStart = cz - SPAN / 2;
    const planks: number[] = [];
    for (let i = 0; i < PLANKS; i++) {
        const [za, ya] = pins[i];
        const [zb, yb] = pins[i + 1];
        // tilt about x so the plank's local z runs pin to pin: R_x(a)·(0,0,1) = (0, -sin a, cos a)
        const a = -Math.atan2(yb - ya, zb - za);
        const quat: [number, number, number, number] = [Math.sin(a / 2), 0, 0, Math.cos(a / 2)];
        const end = i === 0 || i === PLANKS - 1;
        planks.push(
            box(
                state,
                [cx, cy + (ya + yb) / 2, zStart + (za + zb) / 2],
                half,
                end ? 0 : 8,
                end ? anchorColor : plankColor,
                "wood",
                quat,
            ),
        );
    }
    for (let i = 0; i < PLANKS - 1; i++) {
        for (const sx of [-PIN_X, PIN_X]) {
            joints.push({
                a: planks[i],
                b: planks[i + 1],
                rA: [sx, 0, pitch / 2],
                rB: [sx, 0, -pitch / 2],
            });
        }
    }
}

/** the wood-block pyramid — `levels` rows of 0.7 × 0.4 × 0.4 timbers, every block grabbable + clacking. */
export function pyramid(state: State, cx: number, baseY: number, cz: number, levels: number): void {
    const half: [number, number, number] = [0.35, 0.2, 0.2];
    const gap = 0.01;
    const stepX = half[0] * 2 + gap;
    const stepY = half[1] * 2 + gap;
    const color = hex(0x5a4e3c); // muted oak — the wood-grain base tone
    for (let row = 0; row < levels; row++) {
        const count = levels - row;
        const rowStart = cx - ((count - 1) * stepX) / 2;
        for (let i = 0; i < count; i++) {
            const brick = box(
                state,
                [rowStart + i * stepX, baseY + half[1] + row * stepY, cz],
                half,
                2,
                color,
                "wood",
            );
            state.add(brick, Brick);
        }
    }
}

/** the loose wood-block stack in the pit (it rests on the bridge planks). */
export function brickStack(state: State, x: number, z: number, count: number, baseY: number): void {
    const color = hex(0x5a4e3c); // muted oak — the wood-grain base tone
    for (let i = 0; i < count; i++) {
        const brick = box(
            state,
            [x, baseY + 0.22 + i * 0.42, z],
            [0.35, 0.2, 0.2],
            2,
            color,
            "wood",
        );
        state.add(brick, Brick);
    }
}

// inverse-square beyond the source radius, flat within it. A pinpoint bulb (tiny radius) is a blinding
// hotspot at the source that dies to near-nothing across the room; a wide source radius caps the near
// field to a soft glow (intensity/radius²) and spreads it like an area light, so a brighter intensity
// fills the room evenly without the bulb blowing out. The legacy faked this with bloom + an HDR-emissive
// bulb; the lean engine gets the soft fill from the radius directly. Tune both for the room.
const LAMP_INTENSITY = 78;
const LAMP_RADIUS = 4.0; // a soft area source (metres), far larger than the 0.15 m visible bulb

/**
 * a warm ceiling lamp: an unlit emissive-looking sphere that IS the shadow-casting point light it stands
 * in for. From the light at the sphere's center the shell is entirely back-faces, so sear's depth pass
 * culls it and the lamp never occludes its own light.
 */
export function lamp(state: State, pos: readonly [number, number, number]): void {
    const eid = state.create();
    state.add(eid, Transform);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Transform.scale.set(eid, 0.3, 0.3, 0.3, 0); // built-in sphere radius 0.5 → 0.15, the legacy size
    state.add(eid, Part);
    Part.mesh.set(eid, Meshes.id("sphere") ?? 0);
    Part.surface.set(eid, Surfaces.id("unlit") ?? 0);
    state.add(eid, Color);
    Color.rgba.set(eid, 1.0, 0.85, 0.62, 1);
    state.add(eid, PointLight);
    PointLight.color.set(eid, 0xffd9a8);
    PointLight.intensity.set(eid, LAMP_INTENSITY);
    PointLight.range.set(eid, 24);
    PointLight.radius.set(eid, LAMP_RADIUS);
    state.add(eid, Shadow);
    state.add(eid, Volumetric); // the lamp scatters in the haze — the warm glow near the light
}
