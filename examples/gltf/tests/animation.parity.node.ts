import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
    composeMat4,
    createApp,
    GlobalTransform,
    MeshPlugin,
    multiplyMat4,
    TransformPlugin,
} from "@dylanebert/shallot";
import {
    evaluateGltfAnimation,
    type GltfJson,
    loadGltfScene,
    parse,
    poseGltfScene,
    spawnGltfScene,
} from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { animateScene, Viewer } from "../src/app";
import gold from "./animations.gold.json";
import { fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
await syncAssets();

const subjects = gold.records.map((record) => {
    // Fixture identifiers are relative to the record generator's directory.
    const path = resolve(import.meta.dir, "../../../diagnostics/gltf-records", record.fixture);
    const json = JSON.parse(readFileSync(path, "utf8")) as GltfJson;
    const buffers = (json.buffers ?? []).map((buffer) => {
        const bytes = buffer.uri!.startsWith("data:")
            ? Buffer.from(buffer.uri!.split(",")[1], "base64")
            : readFileSync(resolve(dirname(path), buffer.uri!));
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    });
    return { record, path, json, scene: parse(json, buffers) };
});

// Float32 accessor/interpolant storage and Float32 world composition, relative to magnitude.
function close(actual: ArrayLike<number>, expected: ArrayLike<number>): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++)
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(
            3e-5 * Math.max(1, Math.abs(expected[i])),
        );
}

for (const mode of ["STEP", "LINEAR", "CUBICSPLINE", "clamping"]) {
    test(`recorded animation ${mode} follows Animations / Appendix C`, () => {
        let assertions = 0;
        for (const { record, scene } of subjects) {
            for (const animation of record.animations) {
                const decoded = scene.animations[animation.index];
                for (const sample of animation.samples) {
                    const poses = evaluateGltfAnimation(scene.nodes, decoded, sample.time);
                    animation.channels.forEach((channel, i) => {
                        const sampler = decoded.samplers[decoded.channels[i].sampler];
                        const outside =
                            sample.time < sampler.input[0] ||
                            sample.time > sampler.input[sampler.input.length - 1];
                        if (
                            mode === "clamping"
                                ? !outside
                                : outside || channel.interpolation !== mode
                        )
                            return;
                        const index = scene.nodes.findIndex((node) => node.index === channel.node);
                        close(poses[index][decoded.channels[i].path], sample.values[i]);
                        assertions++;
                    });
                }
            }
        }
        expect(assertions).toBeGreaterThan(0);
    });
}

test("recorded default-scene nodes retain authored local values and parent-first order", () => {
    for (const { record, scene } of subjects) {
        expect(scene.nodes.map((node) => node.index)).toEqual(
            record.nodes.map((node) => node.index),
        );
        scene.nodes.forEach((node, i) => {
            expect(node.parent).toBeLessThan(i);
            const expected = record.nodes[i];
            for (const path of ["translation", "rotation", "scale", "weights"] as const)
                close(node[path], expected[path]);
        });
        expect(scene.unsupported).not.toContain("animation");
    }
});

test("skinned corpus joints name scene nodes and RecursiveSkeletons merges into four independent skeletons", () => {
    for (const { json, scene } of subjects) {
        expect(
            scene.skins.map((skin) => skin.joints.map((joint) => scene.nodes[joint].index)),
        ).toEqual((json.skins ?? []).map((skin) => skin.joints));
        expect(scene.unsupported).not.toContain("skin");
        expect(scene.unsupported).not.toContain("JOINTS_0");
        expect(scene.unsupported).not.toContain("WEIGHTS_0");
    }
    const recursive = subjects.find((subject) =>
        subject.record.fixture.includes("RecursiveSkeletons"),
    )!.scene;
    expect(recursive.skins).toHaveLength(84);
    expect(recursive.skeletons.map((skeleton) => skeleton.bones.length)).toEqual([
        210, 210, 210, 210,
    ]);
    expect(new Set(recursive.skins.map((skin) => skin.skeleton)).size).toBe(4);
    for (const skeleton of recursive.skeletons)
        expect(skeleton.bones).toEqual([...new Set(skeleton.bones)].sort((a, b) => a - b));
});

test("sampled local poses compose to recorded node world matrices", () => {
    for (const { record, scene } of subjects) {
        for (const animation of record.animations) {
            for (const sample of animation.samples) {
                const poses = evaluateGltfAnimation(
                    scene.nodes,
                    scene.animations[animation.index],
                    sample.time,
                );
                const worlds: Float32Array[] = [];
                poses.forEach((pose, i) => {
                    const { translation: t, rotation: r, scale: s } = pose;
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
                    );
                    const parent = scene.nodes[i].parent;
                    worlds.push(parent === -1 ? local : multiplyMat4(worlds[parent], local));
                    close(worlds[i], sample.worlds[i].matrix);
                    for (const path of ["translation", "rotation", "scale", "weights"] as const)
                        if (
                            !animation.channels.some(
                                (channel) =>
                                    channel.node === scene.nodes[i].index && channel.path === path,
                            )
                        )
                            expect(pose[path]).toEqual(scene.nodes[i][path]);
                });
            }
        }
    }
});

await setupGlobals();
for (const { record, path, scene } of subjects) {
    test(`spawned animation ${record.fixture} matches recorded world transforms`, async () => {
        const fetch = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
        const app = await createApp({ defaults: false, plugins: [MeshPlugin, TransformPlugin] });
        try {
            const gltf = await loadGltfScene(app.world, new URL(`file://${path}`));
            const instance = spawnGltfScene(app.world, gltf);
            if (record.fixture.includes("AnimatedTriangle")) {
                const stretched = spawnGltfScene(app.world, gltf, { scale: [2, 1, 1] });
                const before = app.world.storage(GlobalTransform);
                app.world.tick();
                const x = before.scale.x.get(stretched.entities[0]);
                expect(() =>
                    poseGltfScene(app.world, stretched, gltf.animations[0], 0.125),
                ).toThrow(/Node0.*shear/);
                app.world.tick();
                expect(before.scale.x.get(stretched.entities[0])).toBe(x);
                const foreign = await createApp({ defaults: false, plugins: [] });
                try {
                    expect(() =>
                        poseGltfScene(foreign.world, instance, gltf.animations[0], 0),
                    ).toThrow("another world");
                } finally {
                    foreign.dispose();
                }
                const viewer = app.world.resource(Viewer);
                viewer.active = "AnimatedTriangle";
                viewer.models.set("AnimatedTriangle", gltf);
                viewer.instance = instance;
                animateScene(app.world, 1.125);
                expect(viewer.clocks.get("AnimatedTriangle")).toBe(0.125);
                viewer.active = null;
                animateScene(app.world, 0.5);
                expect(viewer.clocks.get("AnimatedTriangle")).toBe(0.125);
                viewer.active = "AnimatedTriangle";
                animateScene(app.world, 0.125);
                expect(viewer.clocks.get("AnimatedTriangle")).toBe(0.25);
            }
            const placed = spawnGltfScene(app.world, gltf, {
                translation: [3, -2, 1],
                scale: [2, 2, 2],
                rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
            });
            const placement = composeMat4(3, -2, 1, 0, 0, Math.SQRT1_2, Math.SQRT1_2, 2, 2, 2);
            for (const animation of record.animations) {
                for (const sample of animation.samples) {
                    poseGltfScene(
                        app.world,
                        instance,
                        gltf.animations[animation.index],
                        sample.time,
                    );
                    poseGltfScene(app.world, placed, gltf.animations[animation.index], sample.time);
                    for (const skin of gltf.skins)
                        for (const joint of skin.joints)
                            close(instance.globals[joint], sample.worlds[joint].matrix);
                    app.world.tick();
                    const global = app.world.storage(GlobalTransform);
                    for (const [entities, offset] of [
                        [instance.entities, null],
                        [placed.entities, placement],
                    ] as const) {
                        entities.forEach((eid, i) => {
                            const t = global.translation,
                                r = global.rotation,
                                s = global.scale;
                            const actual = composeMat4(
                                t.x.get(eid),
                                t.y.get(eid),
                                t.z.get(eid),
                                r.x.get(eid),
                                r.y.get(eid),
                                r.z.get(eid),
                                r.w.get(eid),
                                s.x.get(eid),
                                s.y.get(eid),
                                s.z.get(eid),
                            );
                            const index = scene.nodes.findIndex(
                                (node) => node.index === scene.instances[i].nodeIndex,
                            );
                            const expected =
                                gltf.primitives[i].skin >= 0
                                    ? composeMat4(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)
                                    : new Float32Array(sample.worlds[index].matrix);
                            close(actual, offset ? multiplyMat4(offset, expected) : expected);
                        });
                    }
                }
            }
        } finally {
            app.dispose();
            fetch.mockRestore();
        }
    });
}
