import {
    Transform,
    Part,
    Shape,
    Camera,
    Shadows,
    Tonemap,
    Orbit,
    AmbientLight,
    DirectionalLight,
    PointLight,
    Mesh,
    Dynamic,
    RenderPlugin,
    Render,
    Compute,
    beginComputePass,
    surface,
    property,
} from "@dylanebert/shallot";
import type { Plugin, State, System, ComputeNode } from "@dylanebert/shallot";
import type { FieldProxy } from "@dylanebert/shallot/ecs/core";
import { dynamicInfo, type ShapeAtlas } from "@dylanebert/shallot/render/core";
import {
    TextPlugin,
    Text,
    font,
    LinesPlugin,
    Line,
    ArrowsPlugin,
    Arrow,
} from "@dylanebert/shallot/extras";
import { BenchConfig, coneMeshId } from "../config";

export const RENDER_TEST_SHAPES = ["box", "sphere", "capsule", "plane", "cone"] as const;
export type RenderTestShape = (typeof RENDER_TEST_SHAPES)[number];

export const RENDER_TEST_VARIANTS = [
    "default",
    "transparent",
    "vertex",
    "fragment",
    "scaled",
    "reflective",
    "roughness",
    "refraction",
    "glow",
    "dynamic",
    "user",
] as const;
export type RenderTestVariant = (typeof RENDER_TEST_VARIANTS)[number];

export const RENDER_TEST_LIGHTING: { value: RenderTestLighting; label: string }[] = [
    { value: "directional", label: "dir" },
    { value: "point", label: "point" },
    { value: "dir+pt", label: "dir+pt" },
    { value: "multipoint", label: "multi" },
];
export type RenderTestLighting = "directional" | "point" | "dir+pt" | "multipoint";

const SHAPE_FACTORIES: Record<RenderTestShape, () => { shape: number; meshId?: number }> = {
    box: () => ({ shape: Shape.Box }),
    sphere: () => ({ shape: Shape.Sphere }),
    capsule: () => ({ shape: Shape.Capsule }),
    plane: () => ({ shape: Shape.Plane }),
    cone: () => ({ shape: Shape.Mesh, meshId: coneMeshId() }),
};

let renderTestState: State | null = null;
let renderTestEntity = -1;
let renderTestCurrentVariant: RenderTestVariant = "default";
let renderTestVertexSurface = 0;
let renderTestFragmentSurface = 0;
let renderTestSun = -1;
let renderTestPointLight = -1;
let renderTestPointBulb = -1;
let renderTestMultiLights: { light: number; bulb: number }[] = [];
let renderTestExtraEntities: number[] = [];
let renderTestTextEntity = -1;
let renderTestTextEnabled = false;
let renderTestFontId = 0;
let renderTestArrowEntity = -1;
let renderTestArrowEnabled = false;
let renderTestUserSurface = 0;
let renderTestWaveField: FieldProxy | null = null;

export function setRenderTestShape(name: RenderTestShape) {
    const state = renderTestState;
    if (!state) return;
    if (renderTestEntity >= 0) state.removeEntity(renderTestEntity);
    const cfg = SHAPE_FACTORIES[name]();
    const eid = state.addEntity();
    state.addComponent(eid, Part);
    state.addComponent(eid, Transform);
    Transform.posY[eid] = name === "capsule" ? -1 : -2;
    Part.shape[eid] = cfg.shape;
    Part.color[eid] = 0xdddddd;
    Part.sizeX[eid] = 2;
    Part.sizeY[eid] = 2;
    Part.sizeZ[eid] = 2;
    if (cfg.meshId !== undefined) {
        state.addComponent(eid, Mesh);
        Mesh.geometry[eid] = cfg.meshId;
    }
    renderTestEntity = eid;
    if (renderTestCurrentVariant !== "default") {
        setRenderTestVariant(renderTestCurrentVariant);
    }
}

export function setRenderTestVariant(name: RenderTestVariant) {
    renderTestCurrentVariant = name;
    if (renderTestEntity < 0) return;
    const state = renderTestState;
    for (const eid of renderTestExtraEntities) state?.removeEntity(eid);
    renderTestExtraEntities = [];
    Part.surface[renderTestEntity] = 0;
    Part.opacity[renderTestEntity] = 1;
    Part.sizeX[renderTestEntity] = 2;
    Part.sizeY[renderTestEntity] = 2;
    Part.sizeZ[renderTestEntity] = 2;
    Part.reflectivity[renderTestEntity] = 0;
    Part.roughness[renderTestEntity] = 1;
    Part.emission[renderTestEntity] = 0;
    Part.emissionIntensity[renderTestEntity] = 0;
    if (state?.hasComponent(renderTestEntity, Dynamic)) {
        state.removeComponent(renderTestEntity, Dynamic);
    }
    switch (name) {
        case "transparent":
            Part.opacity[renderTestEntity] = 0.5;
            break;
        case "vertex":
            Part.surface[renderTestEntity] = renderTestVertexSurface;
            break;
        case "fragment":
            Part.surface[renderTestEntity] = renderTestFragmentSurface;
            break;
        case "scaled":
            Part.sizeX[renderTestEntity] = 3;
            Part.sizeY[renderTestEntity] = 1;
            Part.sizeZ[renderTestEntity] = 2;
            break;
        case "reflective":
            Part.reflectivity[renderTestEntity] = 1.0;
            Part.roughness[renderTestEntity] = 0.1;
            break;
        case "roughness":
            Part.roughness[renderTestEntity] = 0.0;
            Part.reflectivity[renderTestEntity] = 0.0;
            break;
        case "refraction":
            Part.reflectivity[renderTestEntity] = 0.8;
            Part.opacity[renderTestEntity] = 0.3;
            Part.roughness[renderTestEntity] = 0.05;
            break;
        case "glow":
            Part.emission[renderTestEntity] = 0xffaa44;
            Part.emissionIntensity[renderTestEntity] = 10;
            break;
        case "dynamic":
            state?.addComponent(renderTestEntity, Dynamic);
            break;
        case "user":
            Part.surface[renderTestEntity] = renderTestUserSurface;
            break;
    }
}

export function setRenderTestDirectional(on: boolean) {
    if (renderTestSun >= 0) {
        DirectionalLight.intensity[renderTestSun] = on ? 1.5 : 0;
    }
}

export function setRenderTestPointLight(on: boolean) {
    const state = renderTestState;
    if (!state) return;

    if (renderTestPointLight >= 0) {
        state.removeEntity(renderTestPointLight);
        renderTestPointLight = -1;
    }
    if (renderTestPointBulb >= 0) {
        state.removeEntity(renderTestPointBulb);
        renderTestPointBulb = -1;
    }

    if (on) {
        const pl = state.addEntity();
        state.addComponent(pl, Transform);
        state.addComponent(pl, PointLight);
        Transform.posX[pl] = 0;
        Transform.posY[pl] = 2.5;
        Transform.posZ[pl] = 2;
        PointLight.color[pl] = 0xffeedd;
        PointLight.intensity[pl] = 1.5;
        PointLight.radius[pl] = 20;
        PointLight.shadows[pl] = 1;
        renderTestPointLight = pl;

        const bulb = state.addEntity();
        state.addComponent(bulb, Transform);
        state.addComponent(bulb, Part);
        Transform.posX[bulb] = 0;
        Transform.posY[bulb] = 2.5;
        Transform.posZ[bulb] = 2;
        Part.shape[bulb] = Shape.Sphere;
        Part.color[bulb] = 0xffeedd;
        Part.emission[bulb] = 0xffeedd;
        Part.emissionIntensity[bulb] = 2;
        Part.sizeX[bulb] = 0.3;
        Part.sizeY[bulb] = 0.3;
        Part.sizeZ[bulb] = 0.3;
        Part.shadows[bulb] = 0;
        renderTestPointBulb = bulb;
    }
}

export function setRenderTestDirectionalShadow(on: boolean) {
    if (renderTestSun >= 0) DirectionalLight.shadows[renderTestSun] = on ? 1 : 0;
}

export function setRenderTestPointShadow(on: boolean) {
    if (renderTestPointLight >= 0) PointLight.shadows[renderTestPointLight] = on ? 1 : 0;
}

export function setRenderTestShadows(on: boolean) {
    setRenderTestDirectionalShadow(on);
    setRenderTestPointShadow(on);
}

export function setRenderTestMultiPoint(on: boolean) {
    const state = renderTestState;
    if (!state) return;

    for (const { light, bulb } of renderTestMultiLights) {
        state.removeEntity(light);
        state.removeEntity(bulb);
    }
    renderTestMultiLights = [];

    if (!on) return;

    const lights: { x: number; y: number; z: number; color: number }[] = [
        { x: 1.5, y: 2.0, z: 1.5, color: 0xffd080 },
        { x: -1.5, y: 2.0, z: -1.5, color: 0x80b0ff },
        { x: 1.5, y: -1.0, z: -1.5, color: 0x80ff90 },
        { x: -1.5, y: -1.0, z: 1.5, color: 0xff80d0 },
    ];

    for (const l of lights) {
        const pl = state.addEntity();
        state.addComponent(pl, Transform);
        state.addComponent(pl, PointLight);
        Transform.posX[pl] = l.x;
        Transform.posY[pl] = l.y;
        Transform.posZ[pl] = l.z;
        PointLight.color[pl] = l.color;
        PointLight.intensity[pl] = 0.8;
        PointLight.radius[pl] = 15;
        PointLight.shadows[pl] = 1;

        const bulb = state.addEntity();
        state.addComponent(bulb, Transform);
        state.addComponent(bulb, Part);
        Transform.posX[bulb] = l.x;
        Transform.posY[bulb] = l.y;
        Transform.posZ[bulb] = l.z;
        Part.shape[bulb] = Shape.Sphere;
        Part.color[bulb] = l.color;
        Part.emission[bulb] = l.color;
        Part.emissionIntensity[bulb] = 2;
        Part.sizeX[bulb] = 0.3;
        Part.sizeY[bulb] = 0.3;
        Part.sizeZ[bulb] = 0.3;
        Part.shadows[bulb] = 0;

        renderTestMultiLights.push({ light: pl, bulb });
    }
}

export function setRenderTestLighting(name: RenderTestLighting) {
    setRenderTestDirectional(name === "directional" || name === "dir+pt");
    setRenderTestPointLight(name === "point" || name === "dir+pt");
    setRenderTestMultiPoint(name === "multipoint");
    if (renderTestSun >= 0)
        DirectionalLight.shadows[renderTestSun] =
            name === "directional" || name === "dir+pt" ? 1 : 0;
    if (renderTestPointLight >= 0) PointLight.shadows[renderTestPointLight] = 1;
}

function makeWall(
    state: State,
    x: number,
    y: number,
    z: number,
    sx: number,
    sy: number,
    sz: number,
    color: number,
) {
    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Part);
    Transform.posX[eid] = x;
    Transform.posY[eid] = y;
    Transform.posZ[eid] = z;
    Part.shape[eid] = Shape.Box;
    Part.sizeX[eid] = sx;
    Part.sizeY[eid] = sy;
    Part.sizeZ[eid] = sz;
    Part.color[eid] = color;
    Part.roughness[eid] = 0.9;
}

export function setRenderTestText(on: boolean) {
    const state = renderTestState;
    if (!state) return;
    renderTestTextEnabled = on;

    if (renderTestTextEntity >= 0) {
        state.removeEntity(renderTestTextEntity);
        renderTestTextEntity = -1;
    }

    if (!on) return;

    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Text);
    Transform.posX[eid] = 0;
    Transform.posY[eid] = 2;
    Transform.posZ[eid] = 0;
    Text.content[eid] = "RENDER TEST";
    Text.font[eid] = renderTestFontId;
    Text.fontSize[eid] = 0.6;
    Text.color[eid] = 0xffffff;
    Text.anchorX[eid] = 0.5;
    Text.anchorY[eid] = 1;
    Text.visible[eid] = 1;
    Text.opacity[eid] = 1;
    renderTestTextEntity = eid;
}

export function setRenderTestArrow(on: boolean) {
    const state = renderTestState;
    if (!state) return;
    renderTestArrowEnabled = on;

    if (renderTestArrowEntity >= 0) {
        state.removeEntity(renderTestArrowEntity);
        renderTestArrowEntity = -1;
    }

    if (!on) return;

    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Line);
    state.addComponent(eid, Arrow);
    Transform.posX[eid] = 0;
    Transform.posY[eid] = 0;
    Transform.posZ[eid] = 0;
    Line.offsetX[eid] = 0;
    Line.offsetY[eid] = 1;
    Line.offsetZ[eid] = 0;
    Line.thickness[eid] = 3;
    Line.color[eid] = 0xff4444;
    Line.visible[eid] = 1;
    Line.opacity[eid] = 1;
    Arrow.end[eid] = 1;
    Arrow.start[eid] = 0;
    Arrow.size[eid] = 1.5;
    renderTestArrowEntity = eid;
}

const RenderTestAnimSystem: System = {
    group: "draw",
    update(state: State) {
        const t = state.time.elapsed;

        if (renderTestTextEnabled && renderTestTextEntity >= 0) {
            Transform.posY[renderTestTextEntity] = 2 + Math.sin(t * 2) * 0.3;
        }

        if (renderTestCurrentVariant === "user" && renderTestEntity >= 0 && renderTestWaveField) {
            renderTestWaveField[renderTestEntity] = Math.sin(t) * 0.5 + 0.5;
        }

        if (renderTestArrowEnabled && renderTestArrowEntity >= 0) {
            Transform.posX[renderTestArrowEntity] = Math.sin(t * 0.8) * 0.5;
            const arc = Math.sin(t * 0.6) * 1.2;
            Line.offsetX[renderTestArrowEntity] = arc;
            Line.offsetZ[renderTestArrowEntity] = Math.cos(t * 0.6) * 0.8;
        }
    },
};

const DISPLACEMENT_SHADER = /* wgsl */ `
const STRIDE = 8u;

struct Params {
    baseOffset: u32,
    atlasOffset: u32,
    vertexCount: u32,
    timeBits: u32,
}

@group(0) @binding(0) var<storage, read> baseVerts: array<f32>;
@group(0) @binding(1) var<storage, read_write> atlasVerts: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.vertexCount) { return; }

    let t = bitcast<f32>(params.timeBits);
    let bi = params.baseOffset + gid.x * STRIDE;
    let ai = params.atlasOffset + gid.x * STRIDE;

    let px = baseVerts[bi]; let py = baseVerts[bi + 1u]; let pz = baseVerts[bi + 2u];
    let nx = baseVerts[bi + 3u]; let ny = baseVerts[bi + 4u]; let nz = baseVerts[bi + 5u];

    let wave = sin(px * 4.0 + t * 2.0) * cos(pz * 3.0 - t * 1.4) * 0.15;
    let ripple = sin(sqrt(px * px + pz * pz) * 8.0 - t * 3.0) * 0.08;
    let d = wave + ripple;

    atlasVerts[ai]      = px + nx * d;
    atlasVerts[ai + 1u] = py + ny * d;
    atlasVerts[ai + 2u] = pz + nz * d;
    atlasVerts[ai + 3u] = nx;
    atlasVerts[ai + 4u] = ny;
    atlasVerts[ai + 5u] = nz;
    atlasVerts[ai + 6u] = baseVerts[bi + 6u];
    atlasVerts[ai + 7u] = baseVerts[bi + 7u];
}
`;

function createDisplacementNode(
    atlas: ShapeAtlas,
    getTime: () => number,
    getEntity: () => number,
): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let paramsBuffer: GPUBuffer | null = null;
    let cachedBaseBuffer: GPUBuffer | null = null;
    let cachedVertexBuffer: GPUBuffer | null = null;
    const paramsData = new Uint32Array(4);
    const paramsF32 = new Float32Array(paramsData.buffer);

    return {
        name: "dynamic-displacement",
        scope: "frame",
        inputs: ["data"],
        outputs: ["dynamic-vertices"],

        async prepare(device: GPUDevice) {
            const module = device.createShaderModule({ code: DISPLACEMENT_SHADER });
            pipeline = await device.createComputePipelineAsync({
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });
            paramsBuffer = device.createBuffer({
                label: "displacement-params",
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        },

        execute(ctx) {
            if (!pipeline || !paramsBuffer) return;
            const eid = getEntity();
            if (eid < 0) return;
            const info = dynamicInfo(eid);
            if (!info) return;

            if (atlas.baseVertices !== cachedBaseBuffer || atlas.vertices !== cachedVertexBuffer) {
                cachedBaseBuffer = atlas.baseVertices;
                cachedVertexBuffer = atlas.vertices;
                bindGroup = null;
            }

            if (!bindGroup) {
                bindGroup = ctx.device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: atlas.baseVertices } },
                        { binding: 1, resource: { buffer: atlas.vertices } },
                        { binding: 2, resource: { buffer: paramsBuffer } },
                    ],
                });
            }

            paramsData[0] = info.baseFloatOffset;
            paramsData[1] = info.atlasFloatOffset;
            paramsData[2] = info.vertexCount;
            paramsF32[3] = getTime();
            ctx.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

            const pass = beginComputePass(
                ctx.encoder,
                ctx.timestampWrites?.("dynamic-displacement"),
            );
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(info.vertexCount / 64));
            pass.end();
        },
    };
}

export function buildRenderTestPlugin(): Plugin {
    return {
        name: "RenderTest",
        dependencies: [RenderPlugin, TextPlugin, LinesPlugin, ArrowsPlugin],
        systems: [RenderTestAnimSystem],
        initialize(state) {
            const render = Render.from(state);
            const compute = Compute.from(state);
            if (render && compute) {
                const node = createDisplacementNode(
                    render.meshAtlas,
                    () => state.time.elapsed,
                    () => (renderTestCurrentVariant === "dynamic" ? renderTestEntity : -1),
                );
                compute.graph.add(node);
            }

            renderTestFontId = font(
                "https://fonts.gstatic.com/s/pressstart2p/v16/e3t4euO8T-267oIAQAu6jDQyK0nS.ttf",
                "pixel",
            );
            renderTestState = state;

            renderTestVertexSurface = surface(
                {
                    vertex: `
    let t = scene.time * 2.0;
    let n = pos + normal * 0.5;
    let wave = sin(n.x * 4.0 + t) * cos(n.z * 3.0 - t * 0.7) * 0.15;
    let ripple = sin(length(n.xz) * 8.0 - t * 1.5) * 0.08;
    pos += normal * (wave + ripple);`,
                },
                "rt-vertex",
            );

            renderTestFragmentSurface = surface(
                {
                    fragment: `
    let uv = (*surface).uv;
    let checker = step(0.0, sin(uv.x * 20.0) * sin(uv.y * 20.0));
    let r = mix(uv.x, 0.9, checker);
    let g = mix(uv.y, 0.3, checker);
    let b = mix(0.3, 0.8, 1.0 - checker);
    (*surface).baseColor = vec3(r, g, b);`,
                },
                "rt-fragment",
            );

            renderTestUserSurface = surface(
                {
                    properties: [{ name: "wave", type: "f32" }],
                    fragment: `
    let w = inst.wave;
    (*surface).baseColor = vec3(w, 0.2, 1.0 - w);`,
                },
                "rt-user",
            );
            renderTestWaveField = property("wave");

            const cam = state.addEntity();
            state.addComponent(cam, Transform);
            state.addComponent(cam, Camera);
            state.addComponent(cam, Tonemap);
            state.addComponent(cam, Shadows);
            state.addComponent(cam, Orbit);
            state.addComponent(cam, BenchConfig);
            Orbit.distance[cam] = 12;
            Orbit.maxDistance[cam] = 30;
            Orbit.pitch[cam] = Math.PI / 12;

            const ambient = state.addEntity();
            state.addComponent(ambient, Transform);
            state.addComponent(ambient, AmbientLight);

            const sun = state.addEntity();
            state.addComponent(sun, Transform);
            state.addComponent(sun, DirectionalLight);
            DirectionalLight.directionX[sun] = -0.7;
            DirectionalLight.directionY[sun] = -0.5;
            DirectionalLight.directionZ[sun] = -0.5;
            renderTestSun = sun;

            const S = 6;
            const T = 0.2;
            const white = 0xcccccc;
            makeWall(state, 0, -S / 2 - T / 2, 0, S, T, S, white);
            makeWall(state, 0, S / 2 + T / 2, 0, S, T, S, white);
            makeWall(state, 0, 0, -S / 2 - T / 2, S, S, T, white);
            makeWall(state, -S / 2 - T / 2, 0, 0, T, S, S, 0xcc4444);
            makeWall(state, S / 2 + T / 2, 0, 0, T, S, S, 0x44cc44);

            setRenderTestShape("box");
        },
    };
}
