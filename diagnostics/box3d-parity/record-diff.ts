import { sampleGold } from "../../examples/physics/tests/gold";

function records(scene: string, text: string): string[] {
    if (scene !== "benchmarkSensor") return text.trimEnd() ? text.trimEnd().split("\n") : [];
    const { hash, events } = sampleGold(text)[0]!;
    return [
        `world ${hash.toString(16).padStart(16, "0")}`,
        ...events.flatMap((channel, index) =>
            channel.map((event) => `channel ${index} ${JSON.stringify(event)}`),
        ),
    ];
}

/** Indices are zero-based; Sensor's world hash is record 0, followed by events in channel order. */
export function compareRecords(scene: string, step: number, native: string, actual: string): { equal: boolean; message: string } {
    const a = records(scene, native);
    const b = records(scene, actual);
    const label = `${scene} step ${step}`;
    for (let index = 0; index < Math.max(a.length, b.length); index++) {
        if (a[index] === b[index]) continue;
        const counts = a.length === b.length ? "" : `\ncounts: native ${a.length}, Shallot ${b.length}`;
        return {
            equal: false,
            message: `${label} record ${index} differs${counts}\nnative: ${a[index] ?? "<missing>"}\nShallot: ${b[index] ?? "<missing>"}`,
        };
    }
    return { equal: true, message: `${label}: ${a.length} records equal` };
}
