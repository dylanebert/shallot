import * as d from "typegpu/data";
import type { World } from "../../engine";
import { component, u32 } from "../../engine";

/** A MeshInstance without MeshMaterial draws with the default StandardMaterial (the three.js Mesh fallback convention). */
export const MeshMaterial = component(
    "MeshMaterial",
    { material: u32 },
    { defaults: () => ({ material: 0 }) },
);

export interface StandardMaterial {
    /** Surfaces registry id in the same World; zero selects the built-in default surface. */
    surface: number;
    /** Linear RGBA; alpha is available to custom transparent surfaces. */
    baseColor: readonly [number, number, number, number];
    metallic: number;
    perceptualRoughness: number;
    /** Linear RGB radiance added by the lit surfaces; does not illuminate other objects. */
    emissive: readonly [number, number, number];
    /** Ambient-light multiplier in [0,1]. */
    occlusion: number;
    /** Shallot's blend in [0,1] from Lambert to Valve's squared half-Lambert. */
    diffuseWrap: number;
}

/** Linear colours; emissive is independent of baseColor. diffuseWrap mixes Lambert with squared half-Lambert: 1 preserves Shallot's diffuse look. */
export function StandardMaterial(values: Partial<StandardMaterial> = {}): StandardMaterial {
    return {
        surface: 0,
        baseColor: [1, 1, 1, 1],
        metallic: 0,
        perceptualRoughness: 0.5,
        emissive: [0, 0, 0],
        occlusion: 1,
        diffuseWrap: 1,
        ...values,
    };
}

export const MaterialInput = d
    .struct({
        surface: d.u32,
        baseColor: d.vec4f,
        params: d.vec4f,
        emissive: d.vec3f,
        diffuseWrap: d.f32,
    })
    .$name("MaterialInput");

const materialTableKey = { create: (world: World) => world.table("materials", MaterialInput) };
const materialOffsets = {
    surface: d.memoryLayoutOf(MaterialInput, (value) => value.surface).offset,
    baseColor: d.memoryLayoutOf(MaterialInput, (value) => value.baseColor).offset,
    params: d.memoryLayoutOf(MaterialInput, (value) => value.params).offset,
    emissive: d.memoryLayoutOf(MaterialInput, (value) => value.emissive).offset,
    diffuseWrap: d.memoryLayoutOf(MaterialInput, (value) => value.diffuseWrap).offset,
};
const MATERIAL_BYTES = d.sizeOf(MaterialInput);

class MaterialAssets {
    private readonly _world: World;
    private _view: DataView | undefined;

    constructor(world: World) {
        this._world = world;
        this.add(StandardMaterial());
    }

    /** Copy anonymous values and return an id stable until this World is disposed. Changes reach the next draw upload; mutating the supplied values does not publish changes. */
    add(values: StandardMaterial): number {
        const table = this._world.resource(materialTableKey);
        const id = table.highWater;
        table.reserveSlots(id + 1);
        this.update(id, values);
        return id;
    }

    /** Publish changed fields at an existing id for the next draw upload; omitted fields retain their values. Refuses unknown ids. */
    update(id: number, values: Partial<StandardMaterial>): void {
        const table = this._world.resource(materialTableKey);
        if (!Number.isSafeInteger(id) || id < 0 || id >= table.highWater) {
            throw new RangeError(`Materials.update: unknown material id ${id}`);
        }
        if (this._view?.buffer !== table.bytes.buffer)
            this._view = new DataView(table.bytes.buffer);
        const view = this._view;
        const offset = id * MATERIAL_BYTES;
        if (values.surface !== undefined)
            view.setUint32(offset + materialOffsets.surface, values.surface, true);
        if (values.baseColor !== undefined) {
            for (let lane = 0; lane < 4; lane++) {
                view.setFloat32(
                    offset + materialOffsets.baseColor + lane * 4,
                    values.baseColor[lane],
                    true,
                );
            }
        }
        const params = offset + materialOffsets.params;
        if (values.metallic !== undefined) view.setFloat32(params, values.metallic, true);
        if (values.perceptualRoughness !== undefined)
            view.setFloat32(params + 4, values.perceptualRoughness, true);
        view.setFloat32(params + 8, id, true);
        if (values.occlusion !== undefined) view.setFloat32(params + 12, values.occlusion, true);
        if (values.emissive !== undefined) {
            for (let lane = 0; lane < 3; lane++) {
                view.setFloat32(
                    offset + materialOffsets.emissive + lane * 4,
                    values.emissive[lane],
                    true,
                );
            }
        }
        if (values.diffuseWrap !== undefined)
            view.setFloat32(offset + materialOffsets.diffuseWrap, values.diffuseWrap, true);
        table.markRange(id, 1);
    }
}

/** Anonymous material ids belong to this World. Id zero is the shared default StandardMaterial. */
export const Materials = { create: (world: World) => new MaterialAssets(world) };

export function materialTable(world: World) {
    world.resource(Materials);
    return world.resource(materialTableKey);
}
