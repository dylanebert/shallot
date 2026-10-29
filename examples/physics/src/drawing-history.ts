import {
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    type System,
    type World,
} from "@dylanebert/shallot";

/** Fixed query drawing stays separate from the records inspected by parity. A topology change snaps. */
export function drawingHistory(read: (world: World) => number[][], name = "query") {
    let previous: number[][] = [],
        current: number[][] = [];
    const systems: System[] = [
        {
            name: `${name}-drawing-start`,
            group: "fixed",
            boundary: "before",
            after: [GlobalTransformTickStartSystem],
            update(world) {
                previous = read(world).map((record) => [...record]);
            },
        },
        {
            name: `${name}-drawing-end`,
            group: "fixed",
            boundary: "after",
            after: [GlobalTransformTickEndSystem],
            update(world) {
                current = read(world).map((record) => [...record]);
            },
        },
    ];
    return {
        systems,
        records(world: World): number[][] {
            if (!current.length) return read(world);
            if (
                previous.length !== current.length ||
                previous.some(
                    (record, i) =>
                        record.length !== current[i]!.length || record[0] !== current[i]![0],
                )
            )
                return current;
            return current.map((record, i) =>
                record.map((value, j) =>
                    j === 0
                        ? value
                        : previous[i]![j]! + (value - previous[i]![j]!) * world.time.fixedAlpha,
                ),
            );
        },
    };
}
