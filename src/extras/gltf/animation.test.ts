import { expect, test } from "bun:test";
import { evaluateGltfAnimation, type GltfAnimation, type GltfNode, parse } from ".";

const nodes: GltfNode[] = [
    {
        index: 4,
        parent: -1,
        name: "node",
        translation: [2, 3, 4],
        rotation: [0, 0, 0, 1],
        scale: [1, 2, 3],
        weights: [0.2, 0.8],
    },
];
function animation(
    path: "translation" | "rotation" | "weights",
    output: number[],
    interpolation: "STEP" | "LINEAR" | "CUBICSPLINE" = "LINEAR",
): GltfAnimation {
    return {
        name: "pose",
        channels: [{ node: 4, path, sampler: 0 }],
        samplers: [
            { input: new Float32Array([1, 3]), output: new Float32Array(output), interpolation },
        ],
    };
}

test("evaluation is pure, keeps untargeted properties, and clamps each sampler", () => {
    const clip = animation("translation", [0, 0, 0, 4, 8, 12]);
    const before = structuredClone(nodes);
    expect(evaluateGltfAnimation(nodes, clip, 2)[0]).toEqual({
        translation: [2, 4, 6],
        rotation: [0, 0, 0, 1],
        scale: [1, 2, 3],
        weights: [0.2, 0.8],
    });
    expect(evaluateGltfAnimation(nodes, clip, -2)[0].translation).toEqual([0, 0, 0]);
    expect(evaluateGltfAnimation(nodes, clip, 9)[0].translation).toEqual([4, 8, 12]);
    expect(nodes).toEqual(before);
});

test("single-key and differently timed samplers clamp independently; off-scene channels do not create poses", () => {
    const clip = animation("translation", [0, 0, 0, 4, 8, 12]);
    clip.samplers.push({
        input: new Float32Array([5]),
        output: new Float32Array([0.4, 0.6]),
        interpolation: "LINEAR",
    });
    clip.channels.push(
        { sampler: 1, node: 4, path: "weights" },
        { sampler: 0, node: 99, path: "translation" },
    );
    const poses = evaluateGltfAnimation(nodes, clip, 2);
    expect(poses).toHaveLength(1);
    expect(poses[0].translation).toEqual([2, 4, 6]);
    expect(poses[0].weights[0]).toBeCloseTo(0.4, 6);
    expect(poses[0].weights[1]).toBeCloseTo(0.6, 6);
});

test("caller-owned pose arrays are reused and reset when the animation changes", () => {
    const clip = animation("translation", [0, 0, 0, 4, 8, 12]);
    const poses = evaluateGltfAnimation(nodes, clip, 2);
    const translation = poses[0].translation;
    const weights = poses[0].weights;
    expect(evaluateGltfAnimation(nodes, clip, 3, poses)).toBe(poses);
    expect(poses[0].translation).toBe(translation);
    expect(translation).toEqual([4, 8, 12]);
    evaluateGltfAnimation(nodes, animation("weights", [0, 1, 1, 0]), 2, poses);
    expect(poses[0].translation).toBe(translation);
    expect(translation).toEqual(nodes[0].translation);
    expect(poses[0].weights).toBe(weights);
    expect(weights).toEqual([0.5, 0.5]);
});

test("STEP changes at the key itself", () => {
    const clip = animation("weights", [0, 1, 1, 0], "STEP");
    expect(evaluateGltfAnimation(nodes, clip, 2.99)[0].weights).toEqual([0, 1]);
    expect(evaluateGltfAnimation(nodes, clip, 3)[0].weights).toEqual([1, 0]);
});

test("LINEAR rotations follow the short great-circle arc, including antipodal and orthogonal keys", () => {
    const clip = animation("rotation", [0, 0, 0, 1, 0, 0, -Math.sqrt(0.75), -0.5]);
    const rotation = evaluateGltfAnimation(nodes, clip, 1.5)[0].rotation;
    expect(rotation[2]).toBeCloseTo(Math.sin(Math.PI / 12), 6);
    expect(rotation[3]).toBeCloseTo(Math.cos(Math.PI / 12), 6);
    expect(
        evaluateGltfAnimation(nodes, animation("rotation", [0, 0, 0, 1, 0, 0, 0, -1]), 2)[0]
            .rotation,
    ).toEqual([0, 0, 0, 1]);
    const orthogonal = evaluateGltfAnimation(
        nodes,
        animation("rotation", [1, 0, 0, 0, 0, 1, 0, 0]),
        2,
    )[0].rotation;
    expect(orthogonal[0]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(orthogonal[1]).toBeCloseTo(Math.SQRT1_2, 6);
});

test("CUBICSPLINE uses duration-scaled outgoing and incoming tangents and normalizes only interpolated rotations", () => {
    const clip = animation("weights", [0, 0, 0, 1, 2, 0, -1, 0, 1, 0, 0, 0], "CUBICSPLINE");
    expect(evaluateGltfAnimation(nodes, clip, 2)[0].weights).toEqual([1.25, 0.5]);
    const quaternion = animation(
        "rotation",
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "CUBICSPLINE",
    );
    const value = evaluateGltfAnimation(nodes, quaternion, 2)[0].rotation;
    expect(value[2]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(value[3]).toBeCloseTo(Math.SQRT1_2, 6);
});

test("default scene keeps node indices, parent order and decomposed matrix locals; channels without nodes are ignored", () => {
    const scene = parse(
        {
            asset: { version: "2.0" },
            nodes: [
                { translation: [9, 9, 9] },
                { children: [2], matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1] },
                {},
            ],
            scenes: [{ nodes: [1] }],
            animations: [
                { channels: [{ sampler: 0, target: { path: "translation" } }], samplers: [] },
            ],
        },
        [],
    );
    expect(scene.nodes.map((node) => [node.index, node.parent])).toEqual([
        [1, -1],
        [2, 0],
    ]);
    expect(scene.nodes[0].translation).toEqual([2, 3, 4]);
    expect(scene.animations[0].channels).toEqual([]);
});
