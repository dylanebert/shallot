export type GltfAnimationPath = "translation" | "rotation" | "scale" | "weights";

export interface GltfNodePose {
    translation: number[];
    rotation: number[];
    scale: number[];
    weights: number[];
}

export interface GltfNode extends GltfNodePose {
    /** Index in the asset's nodes array. */
    index: number;
    /** Index in the decoded scene's nodes array, or -1 for a root. */
    parent: number;
    name: string;
}

export interface GltfAnimation {
    name: string;
    /** Channel node indices address the asset's nodes, not the selected scene's ordered array. */
    channels: { sampler: number; node: number; path: GltfAnimationPath }[];
    samplers: {
        input: Float32Array;
        output: Float32Array;
        interpolation: "STEP" | "LINEAR" | "CUBICSPLINE";
    }[];
}

/** Evaluates local poses in scene-node order at time in seconds, without changing the scene.
 * Each sampler clamps independently; properties without a channel retain their authored values.
 * Channels outside the selected scene do not affect its poses. No looping or blending is applied.
 */
export function evaluateGltfAnimation(
    nodes: readonly GltfNode[],
    animation: GltfAnimation,
    time: number,
): GltfNodePose[] {
    const poses = nodes.map(({ translation, rotation, scale, weights }) => ({
        translation: [...translation],
        rotation: [...rotation],
        scale: [...scale],
        weights: [...weights],
    }));
    for (const channel of animation.channels) {
        const index = nodes.findIndex((node) => node.index === channel.node);
        if (index === -1) continue;
        const { input, output, interpolation } = animation.samplers[channel.sampler];
        const cubic = interpolation === "CUBICSPLINE";
        const width = output.length / input.length / (cubic ? 3 : 1);
        let low = 0;
        let high = input.length - 1;
        while (low < high) {
            const middle = Math.ceil((low + high) / 2);
            if (input[middle] <= time) low = middle;
            else high = middle - 1;
        }
        const key = low;
        const offset = (key * (cubic ? 3 : 1) + (cubic ? 1 : 0)) * width;
        let value = Array.from(output.subarray(offset, offset + width));
        if (
            time > input[0] &&
            time < input[input.length - 1] &&
            time !== input[key] &&
            interpolation !== "STEP"
        ) {
            const duration = input[key + 1] - input[key];
            const t = (time - input[key]) / duration;
            const next = offset + width * (cubic ? 3 : 1);
            if (cubic) {
                const t2 = t * t,
                    t3 = t2 * t;
                value = value.map(
                    (v, c) =>
                        (2 * t3 - 3 * t2 + 1) * v +
                        duration * (t3 - 2 * t2 + t) * output[offset + width + c] +
                        (-2 * t3 + 3 * t2) * output[next + c] +
                        duration * (t3 - t2) * output[next - width + c],
                );
                if (channel.path === "rotation") {
                    const length = Math.hypot(...value);
                    value = value.map((v) => v / length);
                }
            } else if (channel.path === "rotation") {
                const dot = value.reduce((sum, v, c) => sum + v * output[next + c], 0);
                const sign = dot < 0 ? -1 : 1;
                const angle = Math.acos(Math.min(1, Math.abs(dot)));
                const a = angle < 1e-6 ? 1 - t : Math.sin((1 - t) * angle) / Math.sin(angle);
                const b = angle < 1e-6 ? t : Math.sin(t * angle) / Math.sin(angle);
                value = value.map((v, c) => a * v + sign * b * output[next + c]);
            } else value = value.map((v, c) => (1 - t) * v + t * output[next + c]);
        }
        poses[index][channel.path] = value;
    }
    return poses;
}
