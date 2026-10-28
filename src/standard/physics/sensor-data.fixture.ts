import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";

export function sensorVisitors(world: World): number[] {
    const shapes: number[] = [];
    for (let i = 0; i < 4; i++) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [i === 0 ? 0 : i - 2, 0, 0] });
        world.add(eid, Body, {
            type: i === 0 ? BodyType.Static : BodyType.Dynamic,
            gravityScale: 0,
        });
        world.add(eid, Shape, {
            kind: ShapeKind.Sphere,
            radius: i === 0 ? 3 : 0.25,
            isSensor: i === 0 ? 1 : 0,
            enableSensorEvents: 1,
        });
        shapes.push(eid);
    }
    return shapes;
}
