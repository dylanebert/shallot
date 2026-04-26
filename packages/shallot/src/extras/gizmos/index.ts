import { traits, type Plugin, type State } from "../../engine";
import { ComputePlugin, RenderPlugin, Render, ActiveCamera } from "../../standard";
import { SCENE_STRUCT_WGSL, type SharedPassContext } from "../../standard/render/core";

export const Gizmos = {
    grid: [] as number[],
};

traits(Gizmos, {
    defaults: () => ({ grid: 1 }),
});

const GRID_SHADER = /* wgsl */ `
${SCENE_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> scene: Scene;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) nearPoint: vec3<f32>,
    @location(1) farPoint: vec3<f32>,
}

fn unproject(p: vec3<f32>) -> vec3<f32> {
    let unprojected = scene.invViewProj * vec4(p, 1.0);
    return unprojected.xyz / unprojected.w;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    let pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    let p = pos[vi];
    var out: VSOut;
    out.position = vec4(p, 0.0, 1.0);
    out.nearPoint = unproject(vec3(p, 0.0));
    out.farPoint = unproject(vec3(p, 1.0));
    return out;
}

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) mask: f32,
}

fn grid(worldPos: vec3<f32>, scale: f32) -> f32 {
    let coord = worldPos.xz / scale;
    let d = fwidth(coord);
    let g = abs(fract(coord - 0.5) - 0.5) / d;
    return 1.0 - min(min(g.x, g.y), 1.0);
}

@fragment
fn fs(input: VSOut) -> FragOut {
    let t = -input.nearPoint.y / (input.farPoint.y - input.nearPoint.y);
    if (t < 0.0) { discard; }

    let worldPos = input.nearPoint + t * (input.farPoint - input.nearPoint);

    let clip = scene.viewProj * vec4(worldPos, 1.0);
    let depth = clip.z / clip.w - 0.0005;
    if (depth < 0.0 || depth > 1.0) { discard; }

    let dist = length(worldPos.xz - scene.cameraWorld[3].xz);
    let fade = 1.0 - smoothstep(20.0, 80.0, dist);
    if (fade <= 0.0) { discard; }

    let minor = grid(worldPos, 1.0);
    let major = grid(worldPos, 10.0);
    let line = max(minor * 0.15, major * 0.25);
    if (line < 0.02) { discard; }

    var color = vec3(1.0);
    var alpha = line * fade;

    let aw = fwidth(worldPos.xz);
    let xAxis = 1.0 - min(abs(worldPos.z) / aw.y, 1.0);
    let zAxis = 1.0 - min(abs(worldPos.x) / aw.x, 1.0);

    if (xAxis > 0.01) {
        color = mix(color, vec3(0.8, 0.2, 0.2), xAxis);
        alpha = max(alpha, xAxis * 0.8 * fade);
    }
    if (zAxis > 0.01) {
        color = mix(color, vec3(0.2, 0.4, 0.8), zAxis);
        alpha = max(alpha, zAxis * 0.8 * fade);
    }

    var out: FragOut;
    out.depth = depth;
    out.color = vec4(color, alpha);
    out.mask = select(0.0, 1.0, alpha > 0.01);
    return out;
}
`;

function createGridPipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
    maskFormat: GPUTextureFormat,
    eidFormat: GPUTextureFormat,
): GPURenderPipeline {
    const module = device.createShaderModule({ code: GRID_SHADER });
    return device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [
                {
                    format,
                    blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                    },
                },
                {
                    format: maskFormat,
                    writeMask: GPUColorWrite.RED,
                },
                {
                    format: eidFormat,
                    writeMask: 0,
                },
            ],
        },
        depthStencil: {
            format: "depth24plus",
            depthCompare: "less-equal",
            depthWriteEnabled: false,
        },
        primitive: { topology: "triangle-list" },
    });
}

function gridEnabled(state: State): boolean {
    const cam = ActiveCamera.from(state);
    if (!cam || cam.eid < 0) return false;
    if (!state.hasComponent(cam.eid, Gizmos)) return false;
    return Gizmos.grid[cam.eid] === 1;
}

export const GizmosPlugin: Plugin = {
    name: "Gizmos",
    components: { Gizmos },
    dependencies: [ComputePlugin, RenderPlugin],
    initialize(state) {
        const render = Render.from(state);
        if (!render) return;

        let pipeline: GPURenderPipeline | null = null;
        let bindGroup: GPUBindGroup | null = null;

        render.effects.overlay.push({
            order: -10,

            draw(pass: GPURenderPassEncoder, ctx: SharedPassContext) {
                if (!gridEnabled(state)) return;

                if (!pipeline) {
                    pipeline = createGridPipeline(
                        ctx.device,
                        ctx.format,
                        ctx.maskFormat,
                        ctx.eidFormat,
                    );
                }

                if (!bindGroup) {
                    bindGroup = ctx.device.createBindGroup({
                        layout: pipeline.getBindGroupLayout(0),
                        entries: [{ binding: 0, resource: { buffer: render.scene } }],
                    });
                }

                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(6);
            },
        });
    },
};
