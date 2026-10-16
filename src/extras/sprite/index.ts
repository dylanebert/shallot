import { Meshes, type MeshHandle, MeshInstance, MeshPlugin } from "../../core/mesh";
import { BeginFrameSystem, RenderingPlugin } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { ClearChangeMarksSystem, type Plugin, type System } from "../../engine";
import {
    MaterialPlugin,
    MeshMaterial,
    MeshPreprocessSystem,
    MeshRenderPlugin,
    materialTypeId,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { SpriteMaterialType, SpriteSource } from "./material";
import { Sprite } from "./pack";

export { SpriteMaterialInput, SpriteMaterialType } from "./material";
export { Sprite, SpriteAlphaMode, SpriteFill } from "./pack";

const SPRITE_QUERY = [Sprite];
const QUAD_VERTS = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);
const spriteGpuKey = {
    create: () => ({ quad: null as MeshHandle | null, instances: new Set<number>() }),
};

// Membership reconciliation follows terminal simulation writes and precedes column upload.
const SpriteSystem: System = {
    name: "sprite",
    group: "draw",
    first: true,
    after: [BeginFrameSystem],
    before: [ClearChangeMarksSystem],
    update(world) {
        const state = world.resource(spriteGpuKey);
        if (state.quad === null) return;
        for (const eid of world.query(SPRITE_QUERY)) {
            if (state.instances.has(eid) || !world.has(eid, GlobalTransform)) continue;
            if (world.has(eid, MeshInstance) || world.has(eid, MeshMaterial))
                throw new Error(
                    `Sprite ${eid} cannot share its entity with another MeshInstance or MeshMaterial`,
                );
            world.add(eid, MeshInstance, { mesh: state.quad });
            world.add(eid, MeshMaterial, {
                type: materialTypeId(world, SpriteMaterialType),
                material: world.resource(SpriteSource).table.rowIndex(eid),
            });
            state.instances.add(eid);
        }
    },
};
const SpriteMaterialPlugin = MaterialPlugin(SpriteMaterialType);
const SpritePublicationSystem: System = {
    name: "sprite-publication",
    group: "draw",
    after: [ClearChangeMarksSystem],
    before: [MeshPreprocessSystem],
    update(world) {
        world.resource(SpriteSource).publish(materialTypeId(world, SpriteMaterialType));
    },
};

/** Retained textured quads; Mask sprites cast shadows through StandardRenderingPlugin. */
export const SpritePlugin: Plugin = {
    gpu: {},
    name: "Sprite",
    components: [Sprite],
    systems: [SpriteSystem, SpritePublicationSystem],
    dependencies: [
        MeshPlugin,
        RenderingPlugin,
        StandardRenderingPlugin,
        MeshRenderPlugin,
        SpriteMaterialPlugin,
    ],
    initialize(world) {
        world.onDispose(
            world.observeMembership(Sprite, (eid, present) => {
                if (present || !world.resource(spriteGpuKey).instances.delete(eid)) return;
                if (world.has(eid, MeshMaterial)) world.remove(eid, MeshMaterial);
                if (world.has(eid, MeshInstance)) world.remove(eid, MeshInstance);
            }),
        );
        world.resource(spriteGpuKey).quad = world
            .resource(Meshes)
            .add({ name: "spriteQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });
    },
    dispose(world) {
        const state = world.resource(spriteGpuKey);
        state.quad = null;
        state.instances.clear();
    },
};
