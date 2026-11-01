import { Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import { DirectionalLight, PointLight, registerImage, SpotLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine/ecs";
import { composeMat4, decomposeMat4, linearToSrgb, multiplyMat4 } from "../../engine/utils";
import {
    AlphaMode,
    type MaterialHandle,
    Materials,
    MeshMaterial,
    StandardMaterial,
} from "../../standard/rendering";
import type { GltfAnimation, GltfNode, GltfNodePose } from "./animation";
import type { GltfScene, GltfSkeleton, GltfSkin } from "./gltf";
import { initializeGltfGlobals, poseKernel, registerGltfKernel, writeKernelPose } from "./kernel";
import { loadGltf } from "./load";
import {
    MorphedMesh,
    SkinnedMaterial,
    SkinnedMesh,
    SkinPalette,
    VertexDeformation,
} from "./skinning";

export interface GltfPrimitive {
    readonly mesh: MeshHandle;
    readonly material: MaterialHandle;
    readonly node: string;
    readonly nodeIndex: number;
    /** Index into skins, or -1 for an unskinned primitive. */
    readonly skin: number;
    /** Bind-space min/max boxes, six floats per skin-local joint; empty joints have infinite minima. */
    readonly jointBounds?: Float32Array;
    /** Maximum position displacement per target, before skinning or placement. */
    readonly morphReach?: Float32Array;
    /** Node world transform in model space, ignored for skinned placement; treat its elements as immutable. */
    readonly matrix: Float32Array;
}

export interface Gltf {
    readonly world: World;
    readonly primitives: readonly GltfPrimitive[];
    /** Default-scene lights; nodeIndex addresses nodes, and matrix is the rest-pose model-space placement. */
    readonly lights: GltfScene["lights"];
    readonly nodes: readonly GltfNode[];
    readonly animations: readonly GltfAnimation[];
    readonly skins: readonly GltfSkin[];
    readonly skeletons: readonly GltfSkeleton[];
    /** Default scene's rest-pose axis-aligned bounds before placement; skinned vertices use joint globals and inverse binds, not the mesh node transform. */
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
 * The handle's decoded data, including animations, is immutable and copied to the kernel.
 * glTF emissive (cd/m²) reaches StandardMaterial.emissive unscaled and is added outside camera
 * exposure, as Bevy's default; glTF leaves its screen brightness to the client's exposure.
 * Punctual lights preserve lux and convert candela to lumens with 4π, matching standard rendering's
 * inverse conversion for both point and spot lights. Linear light colors are encoded as sRGB hex.
 * An omitted infinite light range becomes max(1, sqrt(candela / 0.01)) world units: a finite
 * cluster bound at 0.01 lux, rather than Bevy's fixed 20, so brighter lights keep their reach.
 * To change a clip, pass a new animation object. All kernel data is freed when the world disposes.
 */
export async function loadGltfScene(
    world: World,
    url: string | URL,
    onProgress?: (progress: number) => void,
): Promise<Gltf> {
    onProgress?.(0);
    const scene = await loadGltf(url);
    onProgress?.(0.5);
    if (
        scene.instances.some(
            (instance) => instance.skin >= 0 || scene.meshes[instance.mesh].morphTargets?.length,
        ) &&
        !world.resource(SkinPalette).enabled
    )
        throw new Error(`glTF "${url}" requires GltfPlugin for skinning or morphing`);
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
    const addMaterial = (
        material: (typeof scene.materials)[number],
        inverted = false,
        skinned = false,
    ) =>
        (skinned ? world.resource(SkinnedMaterial) : materials).add(
            StandardMaterial({
                baseColor: material.color,
                baseColorTexture: imageId(material.texture),
                baseColorChannel: material.baseColorChannel,
                baseColorTransform: material.baseColorTransform,
                normalMapTexture: imageId(material.normalMapTexture, false),
                normalMapChannel: material.normalMapChannel,
                normalMapTransform: material.normalMapTransform,
                normalMapScale: material.normalMapScale,
                metallicRoughnessTexture: imageId(material.metallicRoughnessTexture, false),
                metallicRoughnessChannel: material.metallicRoughnessChannel,
                metallicRoughnessTransform: material.metallicRoughnessTransform,
                occlusionTexture: imageId(material.occlusionTexture, false),
                occlusionChannel: material.occlusionChannel,
                occlusionTransform: material.occlusionTransform,
                occlusionStrength: material.occlusionStrength,
                emissiveTexture: imageId(material.emissiveTexture),
                emissiveChannel: material.emissiveChannel,
                emissiveTransform: material.emissiveTransform,
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
    const invertedHandles = new Map<string, MaterialHandle>();
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
    const meshes = scene.meshes.map((mesh) => {
        const morphed = !!mesh.morphTargets?.length;
        if ((!mesh.joints || !mesh.weights) && !morphed) return world.resource(Meshes).add(mesh);
        const morph = morphed ? world.resource(SkinPalette).morph.register(mesh) : undefined;
        const data = new ArrayBuffer((mesh.vertices.length / 8) * 32);
        const joints = new Uint32Array(data),
            weights = new Float32Array(data);
        for (let v = 0; v < mesh.vertices.length / 8; v++) {
            if (mesh.joints && mesh.weights) {
                joints[v * 8] = mesh.joints[v * 4] | (mesh.joints[v * 4 + 1] << 16);
                joints[v * 8 + 1] = mesh.joints[v * 4 + 2] | (mesh.joints[v * 4 + 3] << 16);
                for (let k = 0; k < 4; k++) weights[v * 8 + 4 + k] = mesh.weights[v * 4 + k];
            }
            if (morph) {
                joints[v * 8 + 2] = morph.first + v * mesh.morphTargets!.length * morph.stride;
                joints[v * 8 + 3] = (mesh.morphTargets!.length << 3) | morph.mask;
            }
        }
        return world.resource(Meshes).add({
            ...mesh,
            attributes: { deformation: { element: VertexDeformation, data: new Uint8Array(data) } },
        });
    });
    const skinnedHandles = new Map<number, MaterialHandle>();
    const jointBoxes = new Map<string, Float32Array>();
    const primitives = scene.instances.map((instance) => {
        const sourceMesh = scene.meshes[instance.mesh];
        const morphReach = sourceMesh.morphTargets?.length
            ? Float32Array.from(sourceMesh.morphTargets, (target) => {
                  let reach = 0;
                  for (let v = 0; v < (target.position?.length ?? 0); v += 3)
                      reach = Math.max(
                          reach,
                          Math.hypot(
                              target.position![v],
                              target.position![v + 1],
                              target.position![v + 2],
                          ),
                      );
                  return reach;
              })
            : undefined;
        const materialIndex = sourceMesh.material;
        if (materialIndex < 0) defaultMaterial ??= addMaterial(defaultValues);
        let material = handles[materialIndex] ?? defaultMaterial!;
        let jointBounds: Float32Array | undefined;
        if (instance.skin >= 0 || morphReach) {
            if (!skinnedHandles.has(materialIndex))
                skinnedHandles.set(
                    materialIndex,
                    addMaterial(scene.materials[materialIndex] ?? defaultValues, false, true),
                );
            material = skinnedHandles.get(materialIndex)!;
        }
        if (instance.skin >= 0) {
            const key = `${instance.mesh}:${instance.skin}`;
            jointBounds = jointBoxes.get(key);
            if (!jointBounds) {
                const mesh = scene.meshes[instance.mesh];
                jointBounds = new Float32Array(scene.skins[instance.skin].joints.length * 6);
                for (let j = 0; j < jointBounds.length; j += 6) {
                    jointBounds.fill(Infinity, j, j + 3);
                    jointBounds.fill(-Infinity, j + 3, j + 6);
                }
                for (let v = 0; v < mesh.vertices.length / 8; v++)
                    for (let k = 0; k < 4; k++) {
                        if (mesh.weights![v * 4 + k] <= 0) continue;
                        const j = mesh.joints![v * 4 + k] * 6;
                        for (let a = 0; a < 3; a++) {
                            jointBounds[j + a] = Math.min(
                                jointBounds[j + a],
                                mesh.vertices[v * 8 + a],
                            );
                            jointBounds[j + 3 + a] = Math.max(
                                jointBounds[j + 3 + a],
                                mesh.vertices[v * 8 + a],
                            );
                        }
                    }
                jointBoxes.set(key, jointBounds);
            }
        }
        const m = instance.matrix;
        const determinant =
            m[0] * (m[5] * m[10] - m[6] * m[9]) -
            m[4] * (m[1] * m[10] - m[2] * m[9]) +
            m[8] * (m[1] * m[6] - m[2] * m[5]);
        if (instance.skin < 0 && determinant < 0 && material.cullMode !== 1) {
            const key = `${materialIndex}:${!!morphReach}`;
            if (!invertedHandles.has(key))
                invertedHandles.set(
                    key,
                    addMaterial(
                        scene.materials[materialIndex] ?? defaultValues,
                        true,
                        !!morphReach,
                    ),
                );
            material = invertedHandles.get(key)!;
        }
        return {
            mesh: meshes[instance.mesh],
            material,
            node: instance.node,
            skin: instance.skin,
            jointBounds,
            morphReach,
            nodeIndex: scene.nodes.findIndex((node) => node.index === instance.nodeIndex),
            matrix: instance.matrix,
        };
    });
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const globals: Float32Array[] = [];
    for (const node of scene.nodes) {
        const local = composeMat4(
            ...(node.translation as [number, number, number]),
            ...(node.rotation as [number, number, number, number]),
            ...(node.scale as [number, number, number]),
        );
        globals.push(node.parent < 0 ? local : multiplyMat4(globals[node.parent], local));
    }
    for (const instance of scene.instances) {
        const mesh = scene.meshes[instance.mesh];
        const vertices = mesh.vertices;
        const skin = scene.skins[instance.skin];
        const palette = skin?.joints.map((node, joint) =>
            multiplyMat4(
                globals[node],
                skin.inverseBindMatrices.subarray(joint * 16, joint * 16 + 16),
            ),
        );
        const m = new Float32Array(16);
        const morphWeights = scene.nodes.find((node) => node.index === instance.nodeIndex)!.weights;
        for (let i = 0; i < vertices.length; i += 8) {
            if (palette) {
                m.fill(0);
                for (let influence = 0; influence < 4; influence++) {
                    const offset = i / 2 + influence;
                    const weight = mesh.weights![offset];
                    if (weight === 0) continue;
                    const joint = palette[mesh.joints![offset]];
                    for (let component = 0; component < 16; component++)
                        m[component] += weight * joint[component];
                }
            } else m.set(instance.matrix);
            let px = vertices[i],
                py = vertices[i + 1],
                pz = vertices[i + 2];
            for (let target = 0; target < (mesh.morphTargets?.length ?? 0); target++) {
                const delta = mesh.morphTargets![target].position;
                if (!delta) continue;
                px += morphWeights[target] * delta[(i / 8) * 3];
                py += morphWeights[target] * delta[(i / 8) * 3 + 1];
                pz += morphWeights[target] * delta[(i / 8) * 3 + 2];
            }
            for (let axis = 0; axis < 3; axis++) {
                const value = m[axis] * px + m[axis + 4] * py + m[axis + 8] * pz + m[axis + 12];
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
    const handle: Gltf = {
        world,
        primitives,
        lights: scene.lights,
        nodes: scene.nodes,
        animations: scene.animations,
        skins: scene.skins,
        skeletons: scene.skeletons,
        bounds: { min, max },
        unsupported: scene.unsupported,
    };
    registerGltfKernel(handle);
    return handle;
}

/** Spawns one entity per primitive instance and light. Skinned entities use placement alone; others compose the node's world transform.
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
        const composed = primitive.skin >= 0 ? matrix : multiplyMat4(matrix, primitive.matrix);
        const t = decomposeMat4(composed);
        validateTransform(composed, t, new Float32Array(16), primitive.node);
        return {
            translation: [t[0], t[1], t[2]] as [number, number, number],
            rotation: [t[3], t[4], t[5], t[6]] as [number, number, number, number],
            scale: [t[7], t[8], t[9]] as [number, number, number],
        };
    });
    const lightTransforms = gltf.lights.map((light) => {
        const composed = multiplyMat4(matrix, light.matrix);
        const t = decomposeMat4(composed);
        validateTransform(composed, t, new Float32Array(16), gltf.nodes[light.nodeIndex].name);
        return t;
    });
    const entities = gltf.primitives.map((primitive, i) => {
        const eid = world.create();
        world.add(eid, Transform, transforms[i]);
        world.add(eid, MeshInstance, { mesh: primitive.mesh });
        world.add(eid, MeshMaterial, primitive.material);
        if (primitive.skin >= 0)
            world.add(eid, SkinnedMesh, { jointCount: gltf.skins[primitive.skin].joints.length });
        if (primitive.morphReach)
            world.add(eid, MorphedMesh, { weightCount: primitive.morphReach.length });
        return eid;
    });
    const lightEntities = gltf.lights.map((light, i) => {
        const eid = world.create();
        const t = lightTransforms[i];
        world.add(eid, Transform, {
            translation: [t[0], t[1], t[2]],
            rotation: [t[3], t[4], t[5], t[6]],
            scale: [t[7], t[8], t[9]],
        });
        const color = (light.color ?? [1, 1, 1]).reduce(
            (hex, c) => (hex << 8) | Math.round(linearToSrgb(c) * 255),
            0,
        );
        const intensity = light.intensity ?? 1;
        if (light.type === "directional")
            world.add(eid, DirectionalLight, { color, illuminance: intensity });
        else {
            const fields = {
                color,
                intensity: intensity * 4 * Math.PI,
                range: light.range ?? Math.max(1, Math.sqrt(intensity / 0.01)),
                radius: 0,
            };
            if (light.type === "point") world.add(eid, PointLight, fields);
            else
                world.add(eid, SpotLight, {
                    ...fields,
                    innerAngle: light.spot?.innerConeAngle ?? 0,
                    outerAngle: light.spot?.outerConeAngle ?? Math.PI / 4,
                });
        }
        return eid;
    });
    const instance: GltfSceneInstance = {
        entities,
        lightEntities,
        lightTransforms,
        gltf,
        placement: matrix,
        poses: gltf.nodes.map(({ translation, rotation, scale, weights }) => ({
            translation: [...translation],
            rotation: [...rotation],
            scale: [...scale],
            weights: [...weights],
        })),
        globals: new Array<Float32Array>(gltf.nodes.length),
        transforms: transforms.map(
            (t) => new Float32Array([...t.translation, ...t.rotation, ...t.scale]),
        ),
        composed: new Float32Array(16),
        recomposed: new Float32Array(16),
    };
    initializeGltfGlobals(instance);
    poseKernel(world, instance);
    for (let i = 0; i < gltf.primitives.length; i++) writeKernelPose(world, instance, i);
    return instance;
}

/** A spawn and its independent pose working storage, overwritten by poseGltfScene.
 * Keep entity order, array lengths and storage references unchanged while posing.
 */
export interface GltfSceneInstance {
    /** Primitive eids in spawn order. */
    entities: number[];
    /** Light eids and reusable TRS storage in decoded light order. */
    lightEntities: number[];
    lightTransforms: Float32Array[];
    gltf: Gltf;
    /** Captured spawn placement; composed onto node globals for lights and unskinned primitives. */
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
    const { entities, gltf, globals, transforms, composed, recomposed } = instance;
    if (gltf.world !== world) throw new Error("glTF handle belongs to another world");
    poseKernel(world, instance, animation, time);
    for (let i = 0; i < entities.length; i++) {
        const primitive = gltf.primitives[i];
        if (primitive.skin >= 0) continue;
        multiplyMat4(instance.placement, globals[primitive.nodeIndex], composed);
        decomposeMat4(composed, transforms[i]);
        validateTransform(composed, transforms[i], recomposed, primitive.node);
    }
    for (let i = 0; i < instance.lightEntities.length; i++) {
        const light = gltf.lights[i];
        multiplyMat4(instance.placement, globals[light.nodeIndex], composed);
        decomposeMat4(composed, instance.lightTransforms[i]);
        validateTransform(
            composed,
            instance.lightTransforms[i],
            recomposed,
            gltf.nodes[light.nodeIndex].name,
        );
    }
    const storage = world.storage(Transform);
    for (let i = 0; i < instance.lightEntities.length; i++) {
        const eid = instance.lightEntities[i],
            t = instance.lightTransforms[i];
        storage.translation.set(eid, t[0], t[1], t[2]);
        storage.rotation.set(eid, t[3], t[4], t[5], t[6]);
        storage.scale.set(eid, t[7], t[8], t[9]);
    }
    for (let i = 0; i < entities.length; i++) {
        const eid = entities[i],
            t = transforms[i];
        storage.translation.set(eid, t[0], t[1], t[2]);
        storage.rotation.set(eid, t[3], t[4], t[5], t[6]);
        storage.scale.set(eid, t[7], t[8], t[9]);
        writeKernelPose(world, instance, i);
    }
}
