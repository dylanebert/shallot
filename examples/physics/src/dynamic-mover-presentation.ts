import type { World } from "@dylanebert/shallot";
import { DynamicMoverState } from "./dynamic-mover-scene";
import { add, mul, pose } from "./mover";

type Point = [number, number, number];
export function dynamicMoverDrawing(world: World) {
    const s = world.resource(DynamicMoverState);
    const lines: { a: Point; b: Point; color: number; point?: boolean; width?: number }[] = [];
    if (!s.active) return lines;
    if (s.drawProbes)
        for (const [i, probe] of s.mover.probes.entries()) {
            const end = add(probe.origin, mul(probe.fraction, probe.translation));
            lines.push({
                a: [probe.origin.x, probe.origin.y, probe.origin.z],
                b: [end.x, end.y, end.z],
                point: probe.hit,
                color: probe.hit ? (i === s.mover.selectedProbe ? 0x4dbac5 : 0xd9a441) : 0x746a60,
            });
        }
    if (s.keys.Aim) {
        const shoulder = add(
            pose(world, s.mover.body).p,
            add(
                s.mover.capsule.center2,
                mul(Math.fround(s.mover.capsule.radius + Math.fround(0.03)), s.right),
            ),
        );
        const muzzle = add(
            shoulder,
            mul(Math.fround(Math.fround(0.55) + Math.fround(0.03)), s.aimDirection),
        );
        lines.push({
            a: [shoulder.x, shoulder.y, shoulder.z],
            b: [muzzle.x, muzzle.y, muzzle.z],
            color: 0x746a60,
            width: 4,
        });
    }
    return lines;
}
