import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    MeshRenderPlugin,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { bakeGltfAnimation, releaseGltfBake } from "./bake";
import { morphFixture, morphFixtureUrl } from "./morph.fixture";
import { type Gltf, loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";
import { GltfPlugin, MorphedMesh, SkinnedMesh, SkinPalette } from "./skinning";

setDefaultTimeout(TIMEOUT.node);
const cases = ["weights", "rotation"].flatMap((path) =>
    [true, false].map((warm) => ({ path, warm, camera: 0, gltf: undefined as Gltf | undefined })),
);
const subjects = gpuApps(
    import.meta.path,
    cases.map((subject) => ({
        defaults: false,
        plugins: [
            GltfPlugin,
            StandardRenderingPlugin,
            MeshRenderPlugin,
            {
                name: "RecoveryScene",
                dependencies: [GltfPlugin, StandardRenderingPlugin, MeshRenderPlugin],
                async initialize(world) {
                    const camera = (subject.camera = world.create());
                    world.add(camera, Transform, { translation: [0, 0, 4] });
                    world.add(camera, Camera);
                    world.add(camera, StandardRenderer);
                    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, camera, { width: 64, height: 64 });
                    if (subject.warm)
                        subject.gltf = await loadGltfScene(world, source(subject.path));
                },
            },
        ],
    })),
);
function source(path: string): string {
    const { json } = morphFixture();
    json.materials = [{ doubleSided: true, emissiveFactor: [1, 0.2, 0.1] }];
    json.images = [];
    json.textures = [];
    for (const mesh of json.meshes!)
        for (const primitive of mesh.primitives) primitive.material = 0;
    json.animations![0].channels = json.animations![0].channels.filter(
        (channel) => channel.target.path === path,
    );
    return morphFixtureUrl(json);
}
for (const [index, subject] of cases.entries()) {
    test(`${subject.warm ? "warm" : "late-loaded"} glTF ${subject.path} pose recovers before and after its first draw`, async () => {
        const { world } = subjects()[index];
        const gltf = subject.gltf ?? (await loadGltfScene(world, source(subject.path)));
        const empty = world.snapshot();
        const instance = spawnGltfScene(world, gltf);
        const palette = world.resource(SkinPalette);
        const values = () =>
            instance.entities.map((eid) => {
                if (subject.path === "weights") {
                    const storage = world.storage(MorphedMesh);
                    const first = storage.firstWeight.get(eid);
                    return palette.morph.weights.slice(first, first + storage.weightCount.get(eid));
                }
                const storage = world.storage(SkinnedMesh);
                const first = storage.firstJoint.get(eid) * 16;
                return palette.matrices.slice(first, first + storage.jointCount.get(eid) * 16);
            });
        const draw = async () => {
            world.step(0);
            return (await captureTexture(world, subject.camera)).rgba;
        };
        poseGltfScene(world, instance, gltf.animations[0], 0.25);
        const savedValues = values();
        const before = world.snapshot();
        const saved = await draw();
        const after = world.snapshot();
        for (const snapshot of [before, after]) {
            poseGltfScene(world, instance, gltf.animations[0], 0.75);
            expect(await draw()).not.toEqual(saved);
            world.restore(snapshot);
            world.tick();
            expect(await draw()).toEqual(saved);
            expect(values()).toEqual(savedValues);
        }
        // Assets stay current, including bake ranges created after the snapshot. Entity membership
        // may be entirely replaced before recovery, so saved firstJoint/firstWeight are not owners.
        const bake = bakeGltfAnimation(world, gltf, gltf.animations[0], 4);
        const rows = [...palette.bakes.values()][0];
        const bakedJoints = () =>
            palette.matrices.slice(
                rows.firstJoint * 16,
                (rows.firstJoint + rows.jointCount * rows.frameCount) * 16,
            );
        const bakedWeights = () =>
            palette.morph.weights.slice(
                rows.firstWeight,
                rows.firstWeight + rows.weightCount * rows.frameCount,
            );
        const joints = bakedJoints();
        const weights = bakedWeights();
        for (let cycle = 0; cycle < 3; cycle++) {
            for (const eid of instance.entities) world.destroy(eid);
            const replacement = spawnGltfScene(world, gltf);
            poseGltfScene(world, replacement, gltf.animations[0], 0.75);
            await draw();
            world.restore(after);
            world.tick();
            expect(await draw()).toEqual(saved);
            expect(values()).toEqual(savedValues);
            expect(bakedJoints()).toEqual(joints);
            expect(bakedWeights()).toEqual(weights);
        }
        world.restore(empty);
        expect(Array.from(world.query([SkinnedMesh]))).toHaveLength(0);
        expect(Array.from(world.query([MorphedMesh]))).toHaveLength(0);
        releaseGltfBake(world, bake);
        expect(palette.ranges.extent).toBe(0);
        expect(palette.morph.ranges.extent).toBe(0);
    });
}
