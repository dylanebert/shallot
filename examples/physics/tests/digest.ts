import { sampleGold } from "./gold";

/** FNV-1a's u64 seed, xor and wrapping multiply, as in hashWorldState.
 * Input is UTF-8 of the gold's canonical records: world hash plus JSON event lists,
 * or newline-terminated cast/hull records whose float fields are already raw f32 hex.
 */
export function recordDigest(records: string): string {
    let hash = 14695981039346656037n;
    for (const byte of new TextEncoder().encode(records)) {
        hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 1099511628211n);
    }
    return hash.toString(16).padStart(16, "0");
}

export function eventRecord(hash: bigint, events: (number | string)[][][]): string {
    return `${hash.toString(16).padStart(16, "0")} ${JSON.stringify(events)}\n`;
}

export function eventDigests(text: string): string[] {
    return sampleGold(text).map(({ hash, events }) => recordDigest(eventRecord(hash, events)));
}

export function querySteps(scene: "heightField" | "hull", text: string): string[] {
    const lines = text.trimEnd().split("\n");
    if (scene === "hull") return [`${lines.join("\n")}\n`];
    if (lines.length !== 28556)
        throw new Error(`heightField: expected 28556 cast records, got ${lines.length}`);
    return [lines.slice(0, 14278), lines.slice(14278)].map((records) => `${records.join("\n")}\n`);
}

export function digestMismatch(scene: string, step: number): string {
    return `${scene} step ${step}; compare native and Shallot records: bun diagnostics/box3d-parity/records.ts ${scene} ${step} (set BOX3D to the native checkout; --full dumps both)`;
}
