import type { ManifoldPose } from "../src/manifold";

export function manifoldPoseRecord(step: number, pose: ManifoldPose): string {
    return [step, pose.p.x, pose.p.y, pose.p.z, pose.q.v.x, pose.q.v.y, pose.q.v.z, pose.q.s].join(
        "\t",
    );
}

export function manifoldPoses(text: string): ManifoldPose[] {
    return text
        .trimEnd()
        .split("\n")
        .map((line, step) => {
            const values = line.split(/\s+/).map(Number);
            if (values.length !== 8 || values[0] !== step || !values.every(Number.isFinite))
                throw new Error(`invalid manifold pose record at step ${step}`);
            return {
                p: { x: values[1]!, y: values[2]!, z: values[3]! },
                q: { v: { x: values[4]!, y: values[5]!, z: values[6]! }, s: values[7]! },
            };
        });
}
