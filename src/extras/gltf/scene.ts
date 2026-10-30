import { Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import { registerImage } from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine/ecs";
import { composeMat4, decomposeMat4, multiplyMat4 } from "../../engine/utils";
import {
    AlphaMode,
    type MaterialHandle,
    Materials,
    MeshMaterial,
    StandardMaterial,
} from "../../standard/rendering";
import { loadGltf } from "./load";

export interface GltfPrimitive {
    readonly mesh: MeshHandle;
    readonly material: MaterialHandle;
    readonly node: string;
    /** Node world transform in model space, before placement; treat its elements as immutable. */
    readonly matrix: Float32Array;
}

export interface Gltf {
    readonly world: World;
    readonly primitives: readonly GltfPrimitive[];
    readonly unsupported: readonly string[];
}

export interface GltfPlacement {
    translation?: readonly [number, number, number];
    rotation?: readonly [number, number, number, number];
    scale?: readonly [number, number, number];
}

/** Registers the default scene before warm, from the game's async initialize.
 * The handle and its mesh, image and material registrations belong to this world until disposal.
 */
export async function loadGltfScene(
    world: World,
    url: string | URL,
    onProgress?: (progress: number) => void,
): Promise<Gltf> {
    onProgress?.(0);
    const scene = await loadGltf(url);
    onProgress?.(0.5);
    const textures = scene.textures.map((texture) => {
        if (texture.image === undefined) return undefined;
        const image = scene.images[texture.image];
        return registerImage(
            world,
            image.uri ?? new Blob([image.bytes!.slice().buffer], { type: image.mimeType }),
            { sampler: texture.sampler },
        );
    });
    const materials = world.resource(Materials);
    const handles = scene.materials.map((material) =>
        materials.add(
            StandardMaterial({
                baseColor: material.color,
                baseColorTexture:
                    material.texture === undefined ? undefined : textures[material.texture],
                metallic: material.metallic,
                perceptualRoughness: material.roughness,
                emissive: material.emissive,
                unlit: material.unlit,
                doubleSided: material.doubleSided,
            }),
            {
                cullMode: material.doubleSided ? "none" : "back",
                alphaMode:
                    material.alphaMode === "MASK"
                        ? AlphaMode.Mask(material.cutoff)
                        : material.alphaMode === "BLEND"
                          ? AlphaMode.Blend
                          : AlphaMode.Opaque,
            },
        ),
    );
    let defaultMaterial: MaterialHandle | undefined;
    const meshes = scene.meshes.map((mesh) => world.resource(Meshes).add(mesh));
    const primitives = scene.instances.map((instance) => {
        const materialIndex = scene.meshes[instance.mesh].material;
        if (materialIndex < 0)
            defaultMaterial ??= materials.add(
                StandardMaterial({ metallic: 1, perceptualRoughness: 1 }),
            );
        return {
            mesh: meshes[instance.mesh],
            material: handles[materialIndex] ?? defaultMaterial!,
            node: instance.node,
            matrix: instance.matrix,
        };
    });
    onProgress?.(1);
    return { world, primitives, unsupported: scene.unsupported };
}

/** Spawns one entity per primitive instance, composing placement onto the node's world transform.
 * Refuses a foreign-world handle or a non-TRS composition, naming the node, before spawning anything.
 */
export function spawnGltfScene(world: World, gltf: Gltf, placement: GltfPlacement = {}): number[] {
    if (gltf.world !== world) throw new Error("glTF handle belongs to another world");
    const matrix = composeMat4(
        ...(placement.translation ?? [0, 0, 0]),
        ...(placement.rotation ?? [0, 0, 0, 1]),
        ...(placement.scale ?? [1, 1, 1]),
    );
    const transforms = gltf.primitives.map((primitive) => {
        const composed = multiplyMat4(matrix, primitive.matrix);
        const t = decomposeMat4(composed);
        const recomposed = composeMat4(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9]);
        if (
            composed.some(
                (value, i) =>
                    !Number.isFinite(value) ||
                    Math.abs(value - recomposed[i]) > 1e-5 * Math.max(1, Math.abs(value)),
            )
        )
            throw new Error(
                `glTF node "${primitive.node}": composed transform has shear or cannot be represented as TRS`,
            );
        return {
            translation: [t[0], t[1], t[2]] as [number, number, number],
            rotation: [t[3], t[4], t[5], t[6]] as [number, number, number, number],
            scale: [t[7], t[8], t[9]] as [number, number, number],
        };
    });
    return gltf.primitives.map((primitive, i) => {
        const eid = world.create();
        world.add(eid, Transform, transforms[i]);
        world.add(eid, MeshInstance, { mesh: primitive.mesh });
        world.add(eid, MeshMaterial, primitive.material);
        return eid;
    });
}
