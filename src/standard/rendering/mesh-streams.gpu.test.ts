import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    NormalPrepass,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import {
    AlphaMode,
    extendedMaterial,
    MaterialPlugin,
    Materials,
    type MaterialType,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const tangent = new Float32Array([0.3, -0.4, Math.sqrt(0.75), 1]);
const uv = new Float32Array([0.125, 0.625]);
const uv1 = new Float32Array([0.2345, 0.7891]);
const color = new Float32Array([0.2, 0.4, 0.6, 0.8]);
// Half a unorm16 step plus at most half an f32 ULP on [0, 1].
const COLOR_ERROR = 0.5 / 65535 + 2 ** -25;
// Oct x error <= 0.5/32767, sign-biased/remapped y <= 2/32767.
// Unfolding costs sqrt(2) in L1 and normalization at most 2*sqrt(3):
// 2*sqrt(6)*2.5/32767 plus f32 rounding < 0.0004 in local direction.
// Instance normalization costs at most twice max/min scale (1.2/0.5),
// then the fragment maps [-1, 1] to [0, 1] with a factor of 0.5.
const TANGENT_COLOR_ERROR = 0.0004 * ((2 * 1.2) / 0.5) * 0.5;
const modes = ["uv1", "color", "tangent", "color alpha", "tangent w"] as const;
const subjects = modes.flatMap((label, mode) =>
    [false, true].map((extended) => {
        const name = `Streams_${mode}_${extended ? "extended" : "custom"}`;
        const fragment = (ctx: { uv1: d.v2f; color: d.v4f; tangent: d.v4f }) => {
            "use gpu";
            if (mode === 0) return d.vec4f(ctx.uv1, 0, 1);
            if (mode === 1) return d.vec4f(ctx.color.xyz, 1);
            if (mode === 2) return d.vec4f(std.add(std.mul(ctx.tangent.xyz, 0.5), d.vec3f(0.5)), 1);
            if (mode === 3) return d.vec4f(d.vec3f(ctx.color.w), 1);
            return d.vec4f(d.vec3f(ctx.tangent.w * 0.5 + 0.5), 1);
        };
        const type: MaterialType<any, any, any> = extended
            ? extendedMaterial(Materials, { name, fragment })
            : materialType({
                  name,
                  parameters: d.struct({ unused: d.u32 }),
                  defaults: { unused: 0 },
                  ...(mode === 1
                      ? {
                            varyings: { tag: d.f32 },
                            vertex: (input: { world: d.v4f; worldNormal: d.v3f }) => {
                                "use gpu";
                                return {
                                    world: input.world,
                                    worldNormal: input.worldNormal,
                                    tag: 1,
                                };
                            },
                        }
                      : {}),
                  fragment,
                  fragmentInputs: {
                      uv: true,
                      ...(mode === 0
                          ? { uv1: true as const }
                          : mode === 1 || mode === 3
                            ? { color: true as const }
                            : { tangent: true as const }),
                  },
              });
        return { label, mode, extended, type, camera: -1 };
    }),
);
let replaceStreamBuffers: () => () => void;
function scene(world: World) {
    const base = {
        name: "stream-triangle",
        vertices: new Float32Array([
            -0.6,
            -0.6,
            0,
            uv[0],
            0,
            0,
            1,
            uv[1],
            0.6,
            -0.6,
            0,
            uv[0],
            0,
            0,
            1,
            uv[1],
            0,
            0.6,
            0,
            uv[0],
            0,
            0,
            1,
            uv[1],
        ]),
        indices: new Uint32Array([0, 1, 2]),
    };
    const streams = {
        uv1: new Float32Array([...uv1, ...uv1, ...uv1]),
        color: new Float32Array([...color, ...color, ...color]),
        tangent: new Float32Array([...tangent, ...tangent, ...tangent]),
    };
    const meshes = world.resource(Meshes);
    const full = meshes.add({ ...base, ...streams });
    const absent = meshes.add({ ...base, name: "absent" });
    const singles = {
        uv1: meshes.add({ ...base, uv1: streams.uv1 }),
        color: meshes.add({ ...base, color: streams.color }),
        tangent: meshes.add({ ...base, tangent: streams.tangent }),
    };
    replaceStreamBuffers = () => {
        const target = meshes.get(singles.uv1)!;
        const saved = { ...target };
        Object.assign(target, meshes.get(full));
        return () => Object.assign(target, saved);
    };
    for (const [i, subject] of subjects.entries()) {
        const origin = i * 20;
        const camera = world.create();
        subject.camera = camera;
        world.add(camera, Transform, { translation: [origin, 0, 4] });
        world.add(camera, Camera, { mode: CameraMode.Orthographic, size: 1, antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, NormalPrepass);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: 132, height: 33 });
        const handle = world
            .resource(subject.type)
            .add(subject.extended ? StandardMaterial() : { unused: 0 }, {
                alphaMode: AlphaMode.Mask(0),
                cullMode: "none",
            });
        const single =
            singles[
                subject.mode === 0
                    ? "uv1"
                    : subject.mode === 1 || subject.mode === 3
                      ? "color"
                      : "tangent"
            ];
        for (const [column, mesh] of [full, single, absent, full].entries()) {
            const eid = world.create();
            world.add(eid, Transform, {
                translation: [origin - 3 + column * 2, 0, 0],
                scale: [column === 3 ? -1 : 1, 1.2, 0.5],
                rotation: [0, 0, Math.sin(0.2), Math.cos(0.2)],
            });
            world.add(eid, MeshInstance, { mesh });
            world.add(eid, MeshMaterial, handle);
        }
    }
}
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ...subjects.map(({ type }) => MaterialPlugin(type)),
            {
                name: "MeshStreamsScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                initialize: scene,
            },
        ],
    },
]);
beforeAll(() => {
    const { world } = apps()[0];
    console.log("mesh streams adapter:", world.gpu.adapter);
    world.step(1 / 60);
    world.step(1 / 60);
}, TIMEOUT.startup);
function tangentWorld(mirrored: boolean): number[] {
    const x = tangent[0] * (mirrored ? -1 : 1);
    const y = tangent[1] * 1.2;
    const z = tangent[2] * 0.5;
    const length = Math.hypot(x, y, z);
    return [
        (Math.cos(0.4) * x - Math.sin(0.4) * y) / length,
        (Math.sin(0.4) * x + Math.cos(0.4) * y) / length,
        z / length,
    ].map((v) => v * 0.5 + 0.5);
}
const display = (v: number) =>
    Math.round((v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055) * 255);
for (const subject of subjects)
    for (const column of [0, 1, 2, 3])
        test(`${subject.extended ? "extended StandardMaterial" : "custom material"} reads ${subject.label}: ${["full physical layout", "single physical layout", "absent in the same frame", "mirrored instance"][column]}`, async () => {
            const { world } = apps()[0];
            const image = await captureTexture(world, subject.camera);
            const absent = column === 2;
            const mirrored = column === 3;
            const expected =
                subject.mode === 0
                    ? [...(absent ? uv : uv1), 0]
                    : subject.mode === 1
                      ? absent
                          ? [1, 1, 1]
                          : [...color.subarray(0, 3)]
                      : subject.mode === 2
                        ? absent
                            ? [0.5, 0.5, 0.5]
                            : tangentWorld(mirrored)
                        : subject.mode === 3
                          ? Array(3).fill(absent ? 1 : color[3])
                          : Array(3).fill(absent ? 0.5 : mirrored ? 0 : 1);
            const at = (16 * 132 + 16 + column * 33) * 4;
            const error = absent
                ? 0
                : subject.mode === 1 || subject.mode === 3
                  ? COLOR_ERROR
                  : subject.mode === 2
                    ? TANGENT_COLOR_ERROR
                    : 0;
            // UV_1 and handedness expectations are exact source values. The interval admits only
            // the codec bound above, followed by HDR/display rounding of at most one output byte.
            expect(
                Math.max(
                    ...expected.map((v, c) =>
                        Math.max(
                            display(Math.max(0, v - error)) - image.rgba[at + c],
                            image.rgba[at + c] - display(Math.min(1, v + error)),
                        ),
                    ),
                ),
            ).toBeLessThanOrEqual(1);
        });

test("changing a mesh's physical buffers reselects its warmed pipeline variant", async () => {
    const { world } = apps()[0];
    const restore = replaceStreamBuffers();
    try {
        world.step(1 / 60);
        world.step(1 / 60);
        const errors: number[] = [];
        for (const subject of subjects.filter((subject) => subject.mode === 0)) {
            const image = await captureTexture(world, subject.camera);
            const at = (16 * 132 + 49) * 4;
            errors.push(
                Math.abs(image.rgba[at] - display(uv1[0])),
                Math.abs(image.rgba[at + 1] - display(uv1[1])),
            );
        }
        expect(Math.max(...errors)).toBeLessThanOrEqual(1);
    } finally {
        restore();
    }
});
