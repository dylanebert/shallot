// mesh-fixture — the release-prerequisite hardening fixture (spec shallot-typegpu 5b-2f-0): a deterministic
// mesh proving final pixels through both the canonical typed-surface path (a Part entity on the built-in
// `unlit` surface — the standard eids+transforms instancing convention) and the custom mesh-binding/
// custom-surface path Spindle's rope uses (a hand-registered surface reading its own published uniform,
// positioning geometry outside the Part convention). Both tags ride the SAME `cube` mesh, so mesh
// registration + index/vertex evidence is shared and the two paths differ only in how they publish
// position + color. `mode=blank` is the red-proof: it corrupts only the *content* published into the
// correctly-sized, correctly-bound resources (the class of defect 5b-2c hit — a real ropeFx publication
// miss survived every vertex/draw/timing check) while every upstream count stays exactly as green as
// `mode=ok`. The pixel gate itself is the published `PixelProbe` (`@dylanebert/shallot/harness`), checked
// by `shallot verify` against the post-run compositor screenshot, folded into the wire verdict outside
// this file's `assert`.

import {
    Camera,
    CameraMode,
    Color,
    Compute,
    GlazePlugin,
    InputPlugin,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import type { PixelProbe } from "@dylanebert/shallot/harness";
import {
    BeginFrameSystem,
    DrawIndexedIndirect,
    Draws,
    Meshes,
    Surfaces,
} from "@dylanebert/shallot/render/core";
import {
    fsCtxSchema,
    PrepassSystem,
    registerSurface,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "@dylanebert/shallot/sear/core";
import type { TgpuBuffer, UniformFlag } from "typegpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { type Check, type Params, register, type Scenario, settle } from "../gym";

const CANONICAL_POS: [number, number, number] = [-1.5, 0, 0];
const CUSTOM_POS: [number, number, number] = [1.5, 0, 0];
const RED: [number, number, number, number] = [1, 0, 0, 1];
const GREEN: [number, number, number, number] = [0, 1, 0, 1];
// the red-proof color: visibly nonzero (so the generic center-vs-corner `rendered` structure check the
// harness already runs stays green — this is the taste-ledger shape, real content that isn't the right
// content) but outside both PixelProbe bands below, so only the masked-tag check can catch it.
const BLANK: [number, number, number, number] = [0, 0, 1, 1];

// the custom surface's own published resource — a single uniform (position + color), the same shape as
// Spindle's `RopeFx`: not the standard eids/transforms per-instance convention.
const FixtureTag = d.struct({ pos: d.vec3f, pad: d.f32, color: d.vec4f }).$name("fixtureTag");

const fixtureLayout = surfaceLayout({
    fixtureTag: { type: "uniform", struct: FixtureTag },
});

const fixtureVs = tgpu.fn(
    [VsIn],
    vsPatchSchema(),
)((vsIn) => {
    "use gpu";
    const tag = fixtureLayout.$.fixtureTag;
    return {
        world: d.vec4f(std.add(tag.pos, vsIn.localPos), 1),
        worldNormal: vsIn.worldNormal,
        // unused — sear projects view.viewProj * world after the chunk for a non-`screen` surface, but
        // `vsPatchSchema`'s struct always carries the field (render.md "Surface authoring").
        clip: d.vec4f(0, 0, 0, 1),
    } as never;
});

const fixtureFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)(() => {
    "use gpu";
    return fixtureLayout.$.fixtureTag.color;
});

let fixtureTagBuf: (TgpuBuffer<typeof FixtureTag> & UniformFlag) | null = null;
let canonicalEid = -1;

const FixtureSystem: System = {
    name: "mesh-fixture",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    setup() {
        const cube = Meshes.get("cube");
        if (!cube) throw new Error("mesh-fixture: built-in cube mesh not registered");
        const args = Compute.root.createBuffer(DrawIndexedIndirect).$usage("indirect");
        args.write({
            indexCount: cube.indexCount,
            instanceCount: 1,
            firstIndex: cube.indexBase,
            baseVertex: 0,
            firstInstance: 0,
        });
        Draws.register({
            name: "fixtureCustom",
            surface: "fixtureCustom",
            mesh: "cube",
            args: { indirect: args },
        });
    },
};

const FixturePlugin: Plugin = {
    name: "MeshFixture",
    dependencies: [PartPlugin, SearPlugin],
    systems: [FixtureSystem],
    initialize(state) {
        registerSurface(state, {
            name: "fixtureCustom",
            layout: fixtureLayout,
            vs: fixtureVs,
            fs: fixtureFs,
        });
    },
    warm() {
        if (!Compute.device) return;
        fixtureTagBuf = Compute.root.createBuffer(FixtureTag).$usage("uniform");
        Compute.buffers.set("fixtureTag", Compute.root.unwrap(fixtureTagBuf));
        Compute.typed.set("fixtureTag", fixtureTagBuf);
    },
};

function writeTag(mode: string): void {
    if (!fixtureTagBuf) return;
    const color = mode === "blank" ? BLANK : GREEN;
    fixtureTagBuf.write({ pos: CUSTOM_POS, pad: 0, color });
}

async function build(
    _canvas: HTMLCanvasElement,
    params: Params,
): Promise<{ state: State; dispose: () => void }> {
    const { state, dispose } = await run({
        defaults: false,
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
            FixturePlugin,
        ],
    });

    writeTag(params.mode as string);

    canonicalEid = state.create();
    state.add(canonicalEid, Part);
    state.add(canonicalEid, Transform);
    state.add(canonicalEid, Color);
    Transform.pos.set(canonicalEid, ...CANONICAL_POS, 0);
    Part.surface.set(canonicalEid, Surfaces.id("unlit") ?? 0);
    Color.rgba.set(canonicalEid, ...(params.mode === "blank" ? BLANK : RED));

    const cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 55);
    Camera.near.set(cam, 0.1);
    Camera.far.set(cam, 50);
    Orbit.distance.set(cam, 6);
    Orbit.yaw.set(cam, 0);
    Orbit.pitch.set(cam, 0.12);

    return { state, dispose };
}

async function assert(state: State): Promise<Check[]> {
    const cube = Meshes.get("cube");
    const checks: Check[] = [
        {
            name: "fixture mesh evidence — nonzero index count",
            pass: !!cube && cube.indexCount > 0,
            detail: `indexCount ${cube?.indexCount ?? 0}`,
        },
        { name: "canonical Part entity present", pass: state.has(canonicalEid, Part) },
    ];
    const draw = Draws.get("fixtureCustom");
    if (!draw) {
        checks.push({
            name: "custom-surface draw registered",
            pass: false,
            detail: "fixtureCustom missing",
        });
        return checks;
    }
    const m = mirror(draw.args.indirect);
    try {
        await settle(m);
        const words = m.snapshot ? new Uint32Array(m.snapshot.bytes) : null;
        const indexCount = words?.[0] ?? 0;
        const instanceCount = words?.[1] ?? 0;
        checks.push({
            name: "custom-surface draw consumes geometry",
            pass: indexCount > 0 && instanceCount > 0,
            detail: `${indexCount} indices × ${instanceCount} instance`,
        });
    } finally {
        m.dispose();
    }
    return checks;
}

const CANONICAL_PROBE: PixelProbe = {
    name: "canonical typed-surface path reaches final compositor",
    minPixels: 200,
    minSpan: 12,
    r: [170, 255],
    g: [0, 60],
    b: [0, 60],
};

const CUSTOM_PROBE: PixelProbe = {
    name: "custom mesh-binding/custom-surface path reaches final compositor",
    minPixels: 200,
    minSpan: 12,
    r: [0, 60],
    g: [170, 255],
    b: [0, 60],
};

const scenario: Scenario = {
    name: "mesh-fixture",
    params: [
        { key: "mode", type: "select", options: ["ok", "blank"], default: "ok", rebuild: true },
    ],
    pixelProbe: [CANONICAL_PROBE, CUSTOM_PROBE],
    build,
    assert,
};

register(scenario);
