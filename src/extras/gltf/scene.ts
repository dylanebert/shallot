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
import {
    evaluateGltfAnimation,
    type GltfAnimation,
    type GltfNode,
    type GltfNodePose,
} from "./animation";
import { loadGltf } from "./load";

export interface GltfPrimitive {
    readonly mesh: MeshHandle;
    readonly material: MaterialHandle;
    readonly node: string;
    readonly nodeIndex: number;
    /** Node world transform in model space, before placement; treat its elements as immutable. */
    readonly matrix: Float32Array;
}

export interface Gltf {
    readonly world: World;
    readonly primitives: readonly GltfPrimitive[];
    readonly nodes: readonly GltfNode[];
    readonly animations: readonly GltfAnimation[];
    /** Default scene's axis-aligned bounds, including node transforms, before spawn placement. */
    readonly bounds: {
        min: readonly [number, number, number];
        max: readonly [number, number, number];
    };
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
    const textures = new Map<string, number>();
    const imageId = (index: number | undefined, isSrgb = true) => {
        if (index === undefined) return undefined;
        const key = `${index}:${isSrgb}`;
        const previous = textures.get(key);
        if (previous !== undefined) return previous;
        const texture = scene.textures[index];
        if (texture.image === undefined) return undefined;
        const image = scene.images[texture.image];
        const id = registerImage(
            world,
            image.uri ?? new Blob([image.bytes!.slice().buffer], { type: image.mimeType }),
            { sampler: texture.sampler, isSrgb },
        );
        textures.set(key, id);
        return id;
    };
    const materials = world.resource(Materials);
    const addMaterial = (material: (typeof scene.materials)[number], inverted = false) =>
        materials.add(
            StandardMaterial({
                baseColor: material.color,
                baseColorTexture: imageId(material.texture),
                baseColorChannel: material.baseColorChannel,
                normalMapTexture: imageId(material.normalMapTexture, false),
                normalMapChannel: material.normalMapChannel,
                normalMapScale: material.normalMapScale,
                metallicRoughnessTexture: imageId(material.metallicRoughnessTexture, false),
                metallicRoughnessChannel: material.metallicRoughnessChannel,
                occlusionTexture: imageId(material.occlusionTexture, false),
                occlusionChannel: material.occlusionChannel,
                occlusionStrength: material.occlusionStrength,
                emissiveTexture: imageId(material.emissiveTexture),
                emissiveChannel: material.emissiveChannel,
                metallic: material.metallic,
                perceptualRoughness: material.roughness,
                emissive: material.emissive,
                unlit: material.unlit,
                doubleSided: material.doubleSided,
            }),
            {
                cullMode: material.doubleSided ? "none" : inverted ? "front" : "back",
                alphaMode:
                    material.alphaMode === "MASK"
                        ? AlphaMode.Mask(material.cutoff)
                        : material.alphaMode === "BLEND"
                          ? AlphaMode.Blend
                          : AlphaMode.Opaque,
            },
        );
    const handles = scene.materials.map((material) => addMaterial(material));
    const invertedHandles = new Map<number, MaterialHandle>();
    const defaultValues = {
        color: [1, 1, 1, 1],
        metallic: 1,
        roughness: 1,
        emissive: [0, 0, 0],
        alphaMode: "OPAQUE",
        cutoff: 0.5,
        unlit: false,
        doubleSided: false,
    } satisfies (typeof scene.materials)[number];
    let defaultMaterial: MaterialHandle | undefined;
    const meshes = scene.meshes.map((mesh) => world.resource(Meshes).add(mesh));
    const primitives = scene.instances.map((instance) => {
        const materialIndex = scene.meshes[instance.mesh].material;
        if (materialIndex < 0) defaultMaterial ??= addMaterial(defaultValues);
        let material = handles[materialIndex] ?? defaultMaterial!;
        const m = instance.matrix;
        const determinant =
            m[0] * (m[5] * m[10] - m[6] * m[9]) -
            m[4] * (m[1] * m[10] - m[2] * m[9]) +
            m[8] * (m[1] * m[6] - m[2] * m[5]);
        if (determinant < 0 && material.cullMode !== 1) {
            if (!invertedHandles.has(materialIndex))
                invertedHandles.set(
                    materialIndex,
                    addMaterial(scene.materials[materialIndex] ?? defaultValues, true),
                );
            material = invertedHandles.get(materialIndex)!;
        }
        return {
            mesh: meshes[instance.mesh],
            material,
            node: instance.node,
            nodeIndex: scene.nodes.findIndex((node) => node.index === instance.nodeIndex),
            matrix: instance.matrix,
        };
    });
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const instance of scene.instances) {
        const vertices = scene.meshes[instance.mesh].vertices;
        const m = instance.matrix;
        for (let i = 0; i < vertices.length; i += 8) {
            for (let axis = 0; axis < 3; axis++) {
                const value =
                    m[axis] * vertices[i] +
                    m[axis + 4] * vertices[i + 1] +
                    m[axis + 8] * vertices[i + 2] +
                    m[axis + 12];
                min[axis] = Math.min(min[axis], value);
                max[axis] = Math.max(max[axis], value);
            }
        }
    }
    if (!scene.instances.length) {
        min.fill(0);
        max.fill(0);
    }
    onProgress?.(1);
    return {
        world,
        primitives,
        nodes: scene.nodes,
        animations: scene.animations,
        bounds: { min, max },
        unsupported: scene.unsupported,
    };
}

/** Spawns one entity per primitive instance, composing placement onto the node's world transform.
 * Refuses a foreign-world handle or a non-TRS composition, naming the node, before spawning anything.
 */
export function spawnGltfScene(
    world: World,
    gltf: Gltf,
    placement: GltfPlacement = {},
): GltfSceneInstance {
    if (gltf.world !== world) throw new Error("glTF handle belongs to another world");
    const matrix = composeMat4(
        ...(placement.translation ?? [0, 0, 0]),
        ...(placement.rotation ?? [0, 0, 0, 1]),
        ...(placement.scale ?? [1, 1, 1]),
    );
    const transforms = gltf.primitives.map((primitive) => {
        const composed = multiplyMat4(matrix, primitive.matrix);
        const t = decomposeMat4(composed);
        validateTransform(composed, t, new Float32Array(16), primitive.node);
        return {
            translation: [t[0], t[1], t[2]] as [number, number, number],
            rotation: [t[3], t[4], t[5], t[6]] as [number, number, number, number],
            scale: [t[7], t[8], t[9]] as [number, number, number],
        };
    });
    const entities = gltf.primitives.map((primitive, i) => {
        const eid = world.create();
        world.add(eid, Transform, transforms[i]);
        world.add(eid, MeshInstance, { mesh: primitive.mesh });
        world.add(eid, MeshMaterial, primitive.material);
        return eid;
    });
    return {
        entities,
        gltf,
        placement: matrix,
        poses: gltf.nodes.map(({ translation, rotation, scale, weights }) => ({
            translation: [...translation],
            rotation: [...rotation],
            scale: [...scale],
            weights: [...weights],
        })),
        globals: gltf.nodes.map(() => new Float32Array(16)),
        transforms: entities.map(() => new Float32Array(10)),
        composed: new Float32Array(16),
        recomposed: new Float32Array(16),
    };
}

/** A spawn and its independent pose working storage, overwritten by poseGltfScene.
 * Keep entity order, array lengths and storage references unchanged while posing.
 */
export interface GltfSceneInstance {
    /** Primitive eids in spawn order. */
    entities: number[];
    gltf: Gltf;
    /** Captured spawn placement, composed onto node globals. */
    placement: Float32Array;
    /** Local poses in scene-node order. */
    poses: GltfNodePose[];
    /** Model-space global matrices in scene-node order. */
    globals: Float32Array[];
    /** Entity TRS values in primitive order, validated before any entity write. */
    transforms: Float32Array[];
    /** Matrix multiplication scratch. */
    composed: Float32Array;
    /** TRS recomposition scratch for shear validation. */
    recomposed: Float32Array;
}

function validateTransform(
    matrix: Float32Array,
    t: Float32Array,
    scratch: Float32Array,
    node: string,
) {
    composeMat4(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], scratch);
    const magnitude = Math.max(
        1,
        Math.hypot(matrix[0], matrix[1], matrix[2]),
        Math.hypot(matrix[4], matrix[5], matrix[6]),
        Math.hypot(matrix[8], matrix[9], matrix[10]),
    );
    for (let i = 0; i < 16; i++)
        if (!Number.isFinite(matrix[i]) || Math.abs(matrix[i] - scratch[i]) > 1e-5 * magnitude)
            throw new Error(
                `glTF node "${node}": composed transform has shear or cannot be represented as TRS`,
            );
}

/** Poses a spawned scene at time in seconds, clamped per sampler, without playback or blending.
 * Writes authored Transform; GlobalTransform derives at the next placement boundary.
 * Refuses foreign-world instances and names a sheared entity placement before writing any entity.
 * The instance's eid order and length must remain unchanged and its entities must remain alive.
 */
export function poseGltfScene(
    world: World,
    instance: GltfSceneInstance,
    animation: GltfAnimation,
    time: number,
): void {
    const { entities, gltf, poses, globals, transforms, composed, recomposed } = instance;
    if (gltf.world !== world) throw new Error("glTF handle belongs to another world");
    evaluateGltfAnimation(gltf.nodes, animation, time, poses);
    for (let i = 0; i < poses.length; i++) {
        const { translation: t, rotation: r, scale: s } = poses[i];
        const local = composeMat4(
            t[0],
            t[1],
            t[2],
            r[0],
            r[1],
            r[2],
            r[3],
            s[0],
            s[1],
            s[2],
            composed,
        );
        const parent = gltf.nodes[i].parent;
        if (parent !== -1) multiplyMat4(globals[parent], local, globals[i]);
        else globals[i].set(local);
    }
    for (let i = 0; i < entities.length; i++) {
        const primitive = gltf.primitives[i];
        multiplyMat4(instance.placement, globals[primitive.nodeIndex], composed);
        decomposeMat4(composed, transforms[i]);
        validateTransform(composed, transforms[i], recomposed, primitive.node);
    }
    const storage = world.storage(Transform);
    for (let i = 0; i < entities.length; i++) {
        const eid = entities[i],
            t = transforms[i];
        storage.translation.set(eid, t[0], t[1], t[2]);
        storage.rotation.set(eid, t[3], t[4], t[5], t[6]);
        storage.scale.set(eid, t[7], t[8], t[9]);
    }
}
