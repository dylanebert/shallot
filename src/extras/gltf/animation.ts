import { evaluateKernel } from "./kernel";

export type GltfAnimationPath = "translation" | "rotation" | "scale" | "weights";

export interface GltfNodePose {
    translation: [number, number, number];
    rotation: [number, number, number, number];
    scale: [number, number, number];
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
 * Optional output must have one independent pose per node with matching property lengths; it is reused.
 */
export function evaluateGltfAnimation(
    nodes: readonly GltfNode[],
    animation: GltfAnimation,
    time: number,
    poses: GltfNodePose[] = nodes.map(({ translation, rotation, scale, weights }) => ({
        translation: [...translation],
        rotation: [...rotation],
        scale: [...scale],
        weights: [...weights],
    })),
): GltfNodePose[] {
    evaluateKernel(nodes, animation, time, poses);
    return poses;
}
