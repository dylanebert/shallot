import { writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import { u32 } from "../../engine";
import { Registry } from "../../engine/utils";

/** A Mesh3d without MeshMaterial3d draws with the default StandardMaterial (the three.js Mesh fallback convention). */
export const MeshMaterial3d = { material: u32 };

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

export interface MaterialRecord extends StandardMaterial {
    /** Registry key for stable replacement; shaders and mesh components use only the returned numeric id. */
    name: string;
}

const materialTableKey = { create: (world: World) => world.table("materials", MaterialInput) };

class MaterialRegistry extends Registry<MaterialRecord> {
    private readonly _world: World;

    constructor(world: World) {
        super();
        this._world = world;
        this.register({ name: "default", ...StandardMaterial() });
    }

    /** Register or replace values at a stable id; changes reach the next draw upload. Mutating a retained record does not publish changes. */
    override register(record: MaterialRecord): number {
        const id = super.register(record);
        const table = this._world.resource(materialTableKey);
        table.reserveSlots(id + 1);
        const bytes = new ArrayBuffer(d.sizeOf(MaterialInput));
        writeToArrayBuffer(bytes, MaterialInput, {
            surface: record.surface,
            baseColor: d.vec4f(...record.baseColor),
            params: d.vec4f(record.metallic, record.perceptualRoughness, id, record.occlusion),
            emissive: d.vec3f(...record.emissive),
            diffuseWrap: record.diffuseWrap,
        });
        table.bytes.set(new Uint8Array(bytes), id * d.sizeOf(MaterialInput));
        table.markRange(id, 1);
        return id;
    }
}

/** Material ids belong to this World. Id zero is the shared default; register or replace records to publish values. */
export const Materials = { create: (world: World) => new MaterialRegistry(world) };

export function materialTable(world: World) {
    world.resource(Materials);
    return world.resource(materialTableKey);
}
