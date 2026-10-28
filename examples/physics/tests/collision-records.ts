import type { QueryRecord } from "../src/collision";

const bits = (n: number) =>
    new Uint32Array(new Float32Array([n]).buffer)[0]!.toString(16).padStart(8, "0");
export function collisionRecords(records: readonly QueryRecord[]): string {
    const lines: string[] = [];
    for (const record of records) {
        lines.push(
            record.kind === "body" || record.kind === "closest"
                ? `${record.kind} ${Number(record.hit)}`
                : record.kind,
        );
        for (const { hit: h, result } of record.hits) {
            lines.push(
                `hit ${[h.point.x, h.point.y, h.point.z, h.normal.x, h.normal.y, h.normal.z, h.fraction].map(bits).join(" ")} ${h.userMaterialId} ${h.triangleIndex} ${h.childIndex}${result === undefined ? "" : ` ${bits(result)}`}`,
            );
        }
        for (const result of record.overlaps) lines.push(`overlapHit ${Number(result)}`);
        lines.push("end");
    }
    return `${lines.join("\n")}\n`;
}
