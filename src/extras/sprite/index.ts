import { Meshes, type MeshHandle, MeshInstance, MeshPlugin } from "../../core/mesh";
import { BeginFrameSystem, RenderingPlugin } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { ClearChangeMarksSystem, type Plugin, type System, type World } from "../../engine";
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
    create: () => ({ quad: null as MeshHandle | null }),
};

/** Whether `eid` holds a MeshMaterial of `type`, the sprite material, which only a Sprite publishes.
 * Read from ECS state, so it agrees with a restored image. */
function published(world: World, eid: number, type: number): boolean {
    return world.has(eid, MeshMaterial) && world.storage(MeshMaterial).type.get(eid) === type;
}

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
        const type = materialTypeId(world, SpriteMaterialType);
        for (const eid of world.query(SPRITE_QUERY)) {
            if (published(world, eid, type) || !world.has(eid, GlobalTransform)) continue;
            if (world.has(eid, MeshInstance) || world.has(eid, MeshMaterial))
                throw new Error(
                    `Sprite ${eid} cannot share its entity with another MeshInstance or MeshMaterial`,
                );
            world.add(eid, MeshInstance, { mesh: state.quad });
            world.add(eid, MeshMaterial, {
                type,
                material: world.resource(SpriteSource).table.rowIndex(eid),
            });
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
                if (present || !published(world, eid, materialTypeId(world, SpriteMaterialType)))
                    return;
                world.remove(eid, MeshMaterial);
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
    },
};
