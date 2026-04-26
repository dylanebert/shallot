import type { ComputeNode, ExecutionContext } from "../compute";
import { COLOR_FORMAT, EID_FORMAT, MASK_FORMAT } from "./scene";
import type { OverlayDraw, SharedPassContext } from "./pass";

export { MASK_FORMAT };

const depthInjectShader = /* wgsl */ `
@group(0) @binding(0) var depthTex: texture_2d<f32>;

struct FsOut {
    @location(0) color: vec4f,
    @location(1) mask: f32,
    @location(2) eid: u32,
    @builtin(frag_depth) depth: f32,
}

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
    return vec4f(p[i], 0, 1);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> FsOut {
    var out: FsOut;
    out.depth = textureLoad(depthTex, vec2i(pos.xy), 0).r;
    return out;
}

@fragment fn fsDepthOnly(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
    return textureLoad(depthTex, vec2i(pos.xy), 0).r;
}
`;

interface OverlayNodeConfig {
    overlays: OverlayDraw[];
    hasDepthWriter?: (subGraph: string) => boolean;
}

const _overlayColorAtt0: GPURenderPassColorAttachment = {
    view: null! as GPUTextureView,
    loadOp: "load",
    storeOp: "store",
};
const _overlayColorAtt1: GPURenderPassColorAttachment = {
    view: null! as GPUTextureView,
    clearValue: { r: 0, g: 0, b: 0, a: 0 },
    loadOp: "clear",
    storeOp: "store",
};
const _overlayColorAtt2: GPURenderPassColorAttachment = {
    view: null! as GPUTextureView,
    loadOp: "load",
    storeOp: "store",
};
const _overlayDepth: GPURenderPassDepthStencilAttachment = {
    view: null! as GPUTextureView,
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
};
const _overlayDesc: GPURenderPassDescriptor = {
    colorAttachments: [_overlayColorAtt0, _overlayColorAtt1, _overlayColorAtt2],
    depthStencilAttachment: _overlayDepth,
};

const _depthOnlyDepth: GPURenderPassDepthStencilAttachment = {
    view: null! as GPUTextureView,
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
};
const _depthOnlyDesc: GPURenderPassDescriptor = {
    colorAttachments: [],
    depthStencilAttachment: _depthOnlyDepth,
};

export function createOverlayNode(config: OverlayNodeConfig): ComputeNode {
    let sorted: OverlayDraw[] | null = null;
    let depthInjectPipeline: GPURenderPipeline | null = null;
    let depthInjectBG: GPUBindGroup | null = null;
    let cachedDepthView: GPUTextureView | null = null;
    let depthOnlyPipeline: GPURenderPipeline | null = null;
    let depthOnlyBG: GPUBindGroup | null = null;
    let cachedDepthOnlyView: GPUTextureView | null = null;

    return {
        name: "overlay",
        inputs: ["z", "eid", "depth"],
        outputs: ["color", "mask"],

        async prepare(device: GPUDevice) {
            const module = device.createShaderModule({ code: depthInjectShader });
            depthInjectPipeline = await device.createRenderPipelineAsync({
                label: "depth-inject",
                layout: "auto",
                vertex: { module, entryPoint: "vs" },
                fragment: {
                    module,
                    entryPoint: "fs",
                    targets: [
                        { format: COLOR_FORMAT, writeMask: 0 },
                        { format: MASK_FORMAT, writeMask: 0 },
                        { format: EID_FORMAT, writeMask: 0 },
                    ],
                },
                depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: true,
                    depthCompare: "always",
                },
                primitive: { topology: "triangle-list" },
            });
            depthOnlyPipeline = await device.createRenderPipelineAsync({
                label: "depth-only",
                layout: "auto",
                vertex: { module, entryPoint: "vs" },
                fragment: { module, entryPoint: "fsDepthOnly", targets: [] },
                depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: true,
                    depthCompare: "always",
                },
                primitive: { topology: "triangle-list" },
            });
        },

        execute(ctx: ExecutionContext) {
            const { device, encoder } = ctx;
            const targetView = ctx.getTextureView("color") ?? ctx.canvasView;
            const zView = ctx.getTextureView("z")!;
            const maskView = ctx.getTextureView("mask")!;
            const eidView = ctx.getTextureView("eid")!;
            const needsDepthInject = config.hasDepthWriter?.(ctx.subGraph) ?? false;

            const draws = config.overlays;
            if (draws.length === 0 && needsDepthInject && depthOnlyPipeline) {
                const depthView = ctx.getTextureView("depth");
                if (depthView) {
                    if (depthView !== cachedDepthOnlyView) {
                        depthOnlyBG = device.createBindGroup({
                            layout: depthOnlyPipeline.getBindGroupLayout(0),
                            entries: [{ binding: 0, resource: depthView }],
                        });
                        cachedDepthOnlyView = depthView;
                    }
                    _depthOnlyDepth.view = zView;
                    _depthOnlyDesc.timestampWrites = ctx.timestampWrites?.("raster-overlay");
                    const pass = encoder.beginRenderPass(_depthOnlyDesc);
                    pass.setPipeline(depthOnlyPipeline);
                    pass.setBindGroup(0, depthOnlyBG!);
                    pass.draw(3);
                    pass.end();
                }
            } else if (draws.length > 0) {
                if (!sorted || sorted.length !== draws.length) {
                    sorted = draws.slice().sort((a, b) => a.order - b.order);
                }

                _overlayColorAtt0.view = targetView;
                _overlayColorAtt1.view = maskView;
                _overlayColorAtt2.view = eidView;
                _overlayDepth.view = zView;
                _overlayDepth.depthLoadOp = needsDepthInject ? "clear" : "load";
                _overlayDesc.timestampWrites = ctx.timestampWrites?.("raster-overlay");

                const pass = encoder.beginRenderPass(_overlayDesc);

                if (needsDepthInject && depthInjectPipeline) {
                    const depthView = ctx.getTextureView("depth");
                    if (depthView) {
                        if (depthView !== cachedDepthView) {
                            depthInjectBG = device.createBindGroup({
                                layout: depthInjectPipeline.getBindGroupLayout(0),
                                entries: [{ binding: 0, resource: depthView }],
                            });
                            cachedDepthView = depthView;
                        }
                        pass.setPipeline(depthInjectPipeline);
                        pass.setBindGroup(0, depthInjectBG!);
                        pass.draw(3);
                    }
                }

                const sharedCtx: SharedPassContext = {
                    device,
                    format: COLOR_FORMAT,
                    maskFormat: MASK_FORMAT,
                    eidFormat: EID_FORMAT,
                };

                for (const draw of sorted) {
                    draw.draw(pass, sharedCtx);
                }

                pass.end();
            }
        },
    };
}
