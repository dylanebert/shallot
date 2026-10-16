import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Anchor } from "../../core/rendering";
import type { World } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups,
    precompile,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import {
    AlphaModeCode,
    AlphaPipelineKey,
    MeshInstanceInput,
    meshInstanceTable,
} from "../../standard/rendering";
import type { SpriteMaterialInput, spriteOffset } from "./material";
import { Sprite } from "./pack";

/** @internal Sprite's alpha and bounds publication uses three compute storage bindings. */
export function spritePublicationLayout(input: typeof SpriteMaterialInput) {
    return tgpu.bindGroupLayout({
        parameters: { storage: d.arrayOf(input), access: "readonly", visibility: ["compute"] },
        rows: { storage: d.arrayOf(d.vec2u), access: "readonly", visibility: ["compute"] },
        instances: {
            storage: d.arrayOf(MeshInstanceInput),
            access: "mutable",
            visibility: ["compute"],
        },
        params: { uniform: d.vec2u, visibility: ["compute"] },
    });
}

/** @internal Sprite owns its columns' table and GPU publication, independently of material assets. */
export function createSpriteSource(
    world: World,
    input: typeof SpriteMaterialInput,
    offset: typeof spriteOffset,
) {
    const table = world.table("sprite-material", input);
    table.bindComponent(Sprite, {
        image: "image",
        size: "size",
        color: "color",
        opacity: "opacity",
        billboard: "billboard",
        fillAmount: "fillAmount",
        fillMethod: "fillMethod",
        alphaMode: "alphaMode",
    });
    table.bindFields(Anchor, { anchor: "offset" });
    table.bindPresence(Anchor, "flags");
    const sprite = world.storage(Sprite);
    const terms = [Sprite];
    let keys = 1 << AlphaPipelineKey.Mask;
    let view: DataView | undefined;
    const alphaOffset = d.memoryLayoutOf(input, (row) => row.alphaMode).offset;
    const sizeOffset = d.memoryLayoutOf(input, (row) => row.size).offset;
    const anchorOffset = d.memoryLayoutOf(input, (row) => row.anchor).offset;
    const flagsOffset = d.memoryLayoutOf(input, (row) => row.flags).offset;
    const parameters = () => {
        const bytes = table.bytes;
        if (view?.buffer !== bytes.buffer)
            view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return view;
    };
    const layout = spritePublicationLayout(input);
    const kernel = tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((args) => {
        "use gpu";
        const i = args.gid.x;
        if (i >= layout.$.params.x) return;
        const row = layout.$.rows[i].y;
        if (layout.$.instances[row].materialType !== layout.$.params.y) return;
        const parameters = layout.$.parameters[layout.$.instances[row].material];
        layout.$.instances[row].alphaMode = d.u32(parameters.alphaMode.x);
        layout.$.instances[row].alphaCutoff = parameters.alphaMode.y;
        const corner = offset(parameters);
        const halfSize = std.max(std.abs(corner), std.abs(std.add(corner, parameters.size)));
        layout.$.instances[row].cullBounds = d.vec4f(0, 0, 0, std.length(halfSize));
    });
    const pipeline = world.gpu.root
        .createComputePipeline({ compute: kernel })
        .$name("sprite-publication");
    precompile(world, "sprite-publication", () => pipeline);
    const paramsBuffer = world.gpu.root.createBuffer(d.vec2u).$usage("uniform");
    world.own(world.gpu.root.unwrap(paramsBuffer));
    const params = d.vec2u(0);
    const generations = new Int32Array(3).fill(-1);
    const passDescriptor = { label: "sprite-publication" };
    let group: GPUBindGroup | undefined;
    return {
        table,
        get alphaKeys() {
            for (const eid of world.query(terms)) {
                const mode = sprite.alphaMode.x.get(eid);
                if (mode === AlphaModeCode.Mask) keys |= 1 << AlphaPipelineKey.Mask;
                else if (mode === AlphaModeCode.Blend) keys |= 1 << AlphaPipelineKey.Blend;
                else
                    throw new RangeError(
                        `Sprite ${eid}: alphaMode must be SpriteAlphaMode.Mask or Blend`,
                    );
            }
            return keys;
        },
        alphaMode(row: number) {
            return parameters().getFloat32(row * table.rowBytes + alphaOffset, true);
        },
        cullBounds(row: number, out: Float32Array) {
            const data = parameters();
            const base = row * table.rowBytes;
            const anchored = data.getUint32(base + flagsOffset, true) !== 0;
            const ax = anchored ? data.getFloat32(base + anchorOffset, true) + 0.5 : 0.5;
            const ay = anchored ? data.getFloat32(base + anchorOffset + 4, true) + 0.5 : 0.5;
            const width = data.getFloat32(base + sizeOffset, true);
            const height = data.getFloat32(base + sizeOffset + 4, true);
            out[0] = 0;
            out[1] = 0;
            out[2] = 0;
            out[3] = Math.hypot(
                Math.max(Math.abs(width * ax), Math.abs(width * (1 - ax))),
                Math.max(Math.abs(height * ay), Math.abs(height * (1 - ay))),
            );
        },
        publish(typeId: number) {
            const meshes = meshInstanceTable(world);
            if (!table.count || !meshes.count) return;
            if (
                generations[0] !== table.generation ||
                generations[1] !== meshes.activeGeneration ||
                generations[2] !== meshes.generation
            ) {
                group = world.gpu.root.unwrap(
                    world.gpu.root.createBindGroup(layout, {
                        parameters: table.typed,
                        rows: meshes.activeRowsTyped!,
                        instances: meshes.typed,
                        params: paramsBuffer,
                    } as never),
                );
                generations[0] = table.generation;
                generations[1] = meshes.activeGeneration;
                generations[2] = meshes.generation;
            }
            if (params.x !== meshes.count || params.y !== typeId) {
                params.x = meshes.count;
                params.y = typeId;
                paramsBuffer.write(params);
            }
            const pass = beginComputePass(world.frameEncoder()!, passDescriptor);
            setComputePipeline(pass, world.gpu.root.unwrap(pipeline));
            setComputeBindGroup(pass, 0, group!);
            dispatchWorkgroups(pass, Math.ceil(meshes.count / 64));
            pass.end();
        },
    };
}
