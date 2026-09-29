// Destination: engine; owner: engine-gpu-core.md.
import type { GpuTable, Plugin, State } from "../../engine";
import { field, vec4 } from "../../engine";
import { eulerAlias, Xform } from "../../engine/utils";

const TRANSFORM_TABLE = Symbol("shallot.transforms.table");

/** The dense world-pose table shared by rendering and physics. */
export function transformTable(state: State): GpuTable<typeof Xform> {
    return state.resource(TRANSFORM_TABLE, () => state.table("transforms", Xform));
}

/** Per-entity authored transform fields; the State-owned table mirrors them as one struct row. */
export const Transform = {
    pos: field(vec4),
    rot: field(vec4),
    scale: field(vec4),
};

/** Compose one entity's world matrix on CPU for low-count camera and query consumers. */
export function composeTransform(eid: number, out: Float32Array): Float32Array {
    const { pos, rot, scale } = Transform;

    const px = pos.x.get(eid);
    const py = pos.y.get(eid);
    const pz = pos.z.get(eid);
    const qx = rot.x.get(eid);
    const qy = rot.y.get(eid);
    const qz = rot.z.get(eid);
    const qw = rot.w.get(eid);
    const sx = scale.x.get(eid);
    const sy = scale.y.get(eid);
    const sz = scale.z.get(eid);

    const x2 = qx + qx;
    const y2 = qy + qy;
    const z2 = qz + qz;
    const xx = qx * x2;
    const xy = qx * y2;
    const xz = qx * z2;
    const yy = qy * y2;
    const yz = qy * z2;
    const zz = qz * z2;
    const wx = qw * x2;
    const wy = qw * y2;
    const wz = qw * z2;

    out[0] = (1 - yy - zz) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - xx - zz) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - xx - yy) * sz;
    out[11] = 0;
    out[12] = px;
    out[13] = py;
    out[14] = pz;
    out[15] = 1;
    return out;
}

/** Register Transform storage and its dense `Xform` record table. */
export const TransformsPlugin: Plugin = {
    name: "Transforms",
    components: { Transform },
    traits: {
        Transform: {
            defaults: () => ({
                pos: [0, 0, 0, 0],
                rot: [0, 0, 0, 1],
                scale: [1, 1, 1, 1],
            }),
            aliases: { rot: eulerAlias("rot") },
        },
    },
    initialize(state) {
        const table = transformTable(state);
        table.bindComponent(Transform, {
            pos: "pos",
            quat: "rot",
            scale: "scale",
        });
        table.enableEidLookup();
        const publishMap = (buffer: GPUBuffer) => {
            state.gpu.buffers.set("transformRows", buffer);
            state.gpu.typed.set("transformRows", table.eidToRowTyped!);
        };
        table.subscribeMap(publishMap);
        publishMap(table.eidToRowBuffer!);
    },
};
