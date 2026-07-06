import {
    AmbientLight,
    Camera,
    CameraMode,
    Color,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    Shadow,
    SlabPlugin,
    type State,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import {
    image,
    Profile,
    ProfilePlugin,
    Sprite,
    SpriteBillboard,
    SpriteBlend,
    SpritePlugin,
} from "@dylanebert/shallot/extras";
import { Draws } from "@dylanebert/shallot/render/core";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// sprite — the SpritePlugin dogfood: procedurally-drawn icons (no asset files) amid cube Parts under a
// sun with shadows. Covers the three billboard modes (screen markers above the cubes, y-locked trees,
// a world-aligned ground decal), the opt-in alpha blend (a ghost marker), clip cutouts casting holed
// shadows, and the perspective↔ortho camera switch (the orrstead top-down consumer's shape — near
// straight-down, exercising the y-locked degeneracy guard's neighborhood). Gated on the per-bucket
// indirect instanceCount read back through a Mirror of the sprite arg buffer; the icon shapes and
// billboarding are visual — read them in the render / screenshot, not here.

// the six (billboard, blend) buckets, billboard-major — mirrors extras/sprite's routing
const BUCKETS = 6;

// one canvas-drawn icon: a filled emblem on a transparent field, with a punched hole so the clip
// cutout (and its holed shadow) is visible. Returns a Blob image() accepts
async function icon(draw: (ctx: OffscreenCanvasRenderingContext2D) => void): Promise<Blob> {
    const canvas = new OffscreenCanvas(128, 128);
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, 128, 128);
    draw(ctx);
    return canvas.convertToBlob();
}

function house(ctx: OffscreenCanvasRenderingContext2D): void {
    ctx.fillStyle = "#e8e4da";
    ctx.fillRect(28, 56, 72, 56);
    ctx.beginPath();
    ctx.moveTo(16, 60);
    ctx.lineTo(64, 16);
    ctx.lineTo(112, 60);
    ctx.closePath();
    ctx.fill();
    ctx.clearRect(52, 76, 24, 36); // the door — a transparent hole the cutout (and shadow) keeps
}

function tree(ctx: OffscreenCanvasRenderingContext2D): void {
    ctx.fillStyle = "#5a8a4a";
    ctx.beginPath();
    ctx.moveTo(64, 8);
    ctx.lineTo(108, 88);
    ctx.lineTo(20, 88);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#7a5a3a";
    ctx.fillRect(56, 88, 16, 32);
}

function star(ctx: OffscreenCanvasRenderingContext2D): void {
    ctx.fillStyle = "#e8c84a";
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
        const r = i % 2 ? 24 : 56;
        const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
        ctx.lineTo(64 + Math.cos(a) * r, 64 + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fill();
}

function box(
    state: State,
    pos: [number, number, number],
    scale: [number, number, number],
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Transform);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Transform.scale.set(eid, scale[0], scale[1], scale[2], 0);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

function sprite(
    state: State,
    img: number,
    pos: [number, number, number],
    fields: Partial<{
        billboard: number;
        blend: number;
        opacity: number;
        anchor: [number, number];
    }> = {},
): number {
    const eid = state.create();
    state.add(eid, Transform);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    state.add(eid, Sprite);
    Sprite.image.set(eid, img);
    Sprite.size.set(eid, 1.4, 1.4);
    if (fields.billboard !== undefined) Sprite.billboard.set(eid, fields.billboard);
    if (fields.blend !== undefined) Sprite.blend.set(eid, fields.blend);
    if (fields.opacity !== undefined) Sprite.opacity.set(eid, fields.opacity);
    if (fields.anchor) Sprite.anchor.set(eid, fields.anchor[0], fields.anchor[1]);
    return eid;
}

let cam = 0;
let hideable = -1; // a screen-clip marker the assert toggles to prove the live rebuild path
let spriteArgs: Mirror | null = null;
let params: Params | null = null;

// per-bucket instanceCount from the sprite indirect args (20-byte DrawIndexedIndirect records, count at word 1)
function bucketCounts(m: Mirror): number[] | null {
    if (!m.snapshot) return null;
    const args = new Uint32Array(m.snapshot.bytes);
    return Array.from({ length: BUCKETS }, (_, b) => args[b * 5 + 1]);
}

// the mode knob is projection only — it flips Camera.mode live without touching the orbit pose, so a
// live drag isn't fought. The ortho top-down pose is a build-time initial pose (see build)
function applyCamera(): void {
    const ortho = params?.mode === "ortho";
    Camera.mode.set(cam, ortho ? CameraMode.Orthographic : CameraMode.Perspective);
    Camera.size.set(cam, 7);
}

const scenario: Scenario = {
    name: "sprite",
    params: [
        {
            key: "mode",
            type: "select",
            default: "perspective",
            options: ["perspective", "ortho"],
            label: "camera",
        },
    ],

    async build(_canvas: HTMLCanvasElement, p: Params) {
        params = p;
        // register images before run() — SpritePlugin.initialize uploads the texture array
        const houseImg = image(await icon(house), "house");
        const treeImg = image(await icon(tree), "tree");
        const starImg = image(await icon(star), "star");

        const { state, dispose } = await run({
            defaults: false,
            capacity: 64,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
                MirrorPlugin,
                SpritePlugin,
            ],
        });

        state.add(state.create(), AmbientLight);
        const sun = state.create();
        state.add(sun, DirectionalLight);
        DirectionalLight.direction.set(sun, -0.4, -0.8, -0.45, 0);
        state.add(sun, Shadow); // clip sprites cast holed shadows through sear's prepass pipeline

        // ground + a few cubes the icons sit amid (depth correctness reads against them)
        box(state, [0, -1.2, 0], [12, 0.4, 12], [0.26, 0.28, 0.3]);
        box(state, [-2.5, -0.5, -1], [1, 1, 1], [0.45, 0.4, 0.5]);
        box(state, [2.5, -0.5, 1], [1, 1, 1], [0.4, 0.5, 0.45]);

        // screen-aligned markers above the cubes (anchor 0.5 0 — the icon's base sits at the pos)
        hideable = sprite(state, houseImg, [-2.5, 0.2, -1], { anchor: [0.5, 0] });
        sprite(state, starImg, [2.5, 0.2, 1], { anchor: [0.5, 0] });
        sprite(state, starImg, [0, 0.4, -3], { anchor: [0.5, 0] });

        // y-locked trees (upright, yawing toward the camera)
        sprite(state, treeImg, [-1, -1, 2.5], {
            billboard: SpriteBillboard.YLocked,
            anchor: [0.5, 0],
        });
        sprite(state, treeImg, [1.5, -1, -2.8], {
            billboard: SpriteBillboard.YLocked,
            anchor: [0.5, 0],
        });

        // a world-aligned ground decal: rotated flat (-90° about X), riding the plain transform
        const decal = sprite(state, starImg, [0, -0.99, 1.8], { billboard: SpriteBillboard.World });
        Transform.rot.set(decal, -Math.SQRT1_2, 0, 0, Math.SQRT1_2);

        // the opt-in translucent mode: a half-faded ghost marker
        sprite(state, houseImg, [0.8, 0.2, 3], { blend: SpriteBlend.Alpha, opacity: 0.5 });

        cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.fov.set(cam, 55);
        Orbit.distance.set(cam, 11);
        Orbit.yaw.set(cam, 0.4);
        // ortho builds at the orrstead top-down framing — near straight-down (just under Orbit's
        // π/2 − 0.01 maxPitch clamp), the y-locked degeneracy guard's neighborhood. Initial pose only;
        // a live drag moves it. Perspective keeps the orbit default pitch
        if (p.mode === "ortho") Orbit.pitch.set(cam, Math.PI / 2 - 0.02);
        applyCamera();

        // the sprite draws register in SpriteSystem.setup (first frame); mirror the arg buffer after it
        await frames(1);
        spriteArgs = mirror(Draws.get("sprite-screen")!.args.indirect);

        return {
            state,
            dispose() {
                cam = 0;
                hideable = -1;
                spriteArgs = null;
                params = null;
                dispose();
            },
        };
    },

    // the gate: per-bucket indirect instanceCounts match the spawned set, the live rebuild path reacts
    // to a visibility edit, the shadow map renders (clip sprites are in it), and the counts hold under
    // the ortho top-down camera. Billboard orientation + cutout shape are visual (screenshot), not here
    async assert(): Promise<Check[]> {
        const checks: Check[] = [];
        if (!spriteArgs) return [{ name: "sprite: args mirror", pass: false, detail: "no mirror" }];

        const expected = [3, 1, 2, 0, 1, 0]; // screen, screen-alpha, y, y-alpha, world, world-alpha
        await settle(spriteArgs);
        let counts = bucketCounts(spriteArgs);
        checks.push({
            name: "sprite: per-bucket instance counts match the spawned set",
            pass: counts !== null && expected.every((n, b) => counts?.[b] === n),
            detail: `counts ${counts?.join(",") ?? "(none)"} (expected ${expected.join(",")})`,
        });

        Sprite.visible.set(hideable, 0);
        await settle(spriteArgs);
        counts = bucketCounts(spriteArgs);
        checks.push({
            name: "sprite: a visibility edit rebuilds the buckets live",
            pass: counts?.[0] === expected[0] - 1,
            detail: `screen bucket ${counts?.[0]} (expected ${expected[0] - 1})`,
        });
        Sprite.visible.set(hideable, 1);

        checks.push({
            name: "sprite: the shadow map renders (clip sprites cast)",
            pass: Profile.gpu.has("sear:cascadeshadow"),
            detail: `gpu passes: ${[...Profile.gpu.keys()].sort().join(", ")}`,
        });

        // the ortho top-down framing (the orrstead consumer): the frame keeps rendering and the
        // bucket routing is camera-independent. Restore the URL-resolved mode after, so the
        // post-run screenshot shows the camera the run was asked for
        const prior = params!.mode;
        params!.mode = "ortho";
        applyCamera();
        await settle(spriteArgs);
        counts = bucketCounts(spriteArgs);
        checks.push({
            name: "sprite: ortho top-down camera holds the same buckets",
            pass: counts !== null && expected.every((n, b) => counts?.[b] === n),
            detail: `counts ${counts?.join(",") ?? "(none)"} (expected ${expected.join(",")})`,
        });
        params!.mode = prior;
        applyCamera();
        await frames(2);
        return checks;
    },

    live(): string {
        applyCamera(); // the camera select is a live knob — re-applied so the dropdown takes effect
        return "sprite — drag to orbit; screen markers, y-locked trees, a ground decal, a ghost (alpha)";
    },
};

register(scenario);
