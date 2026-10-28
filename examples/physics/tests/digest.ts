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

/** One native drawing callback per row, with every numeric field encoded as its exact f32 bits. */
export function drawingDigest(records: readonly (readonly number[])[]): string {
    return recordDigest(
        records
            .map((row) => {
                const floats = new Float32Array(row);
                return Array.from(new Uint32Array(floats.buffer), (bits) =>
                    bits.toString(16).padStart(8, "0"),
                ).join(" ");
            })
            .join("\n") + "\n",
    );
}

export function eventRecord(hash: bigint, events: (number | string)[][][]): string {
    return `${hash.toString(16).padStart(16, "0")} ${JSON.stringify(events)}\n`;
}

export function eventDigests(text: string): string[] {
    return sampleGold(text).map(({ hash, events }) => recordDigest(eventRecord(hash, events)));
}

export function querySteps(scene: "cast" | "heightField" | "hull", text: string): string[] {
    const lines = text.trimEnd().split("\n");
    if (scene === "cast") {
        const steps: string[] = [];
        for (let i = 0; i < lines.length; ) {
            const count = Number(lines[i + 3]?.split(" ")[1]);
            if (
                !lines[i]?.startsWith("cast ") ||
                !lines[i + 3]?.startsWith("mover ") ||
                !Number.isInteger(count) ||
                count < 0 ||
                i + 4 + count > lines.length
            )
                throw new Error(`cast: invalid query records at line ${i}`);
            steps.push(`${lines.slice(i, i + 4 + count).join("\n")}\n`);
            i += 4 + count;
        }
        return steps;
    }
    if (scene === "hull") return [`${lines.join("\n")}\n`];
    if (lines.length !== 28556)
        throw new Error(`heightField: expected 28556 cast records, got ${lines.length}`);
    return [lines.slice(0, 14278), lines.slice(14278)].map((records) => `${records.join("\n")}\n`);
}

export function digestMismatch(scene: string, step: number): string {
    return `${scene} step ${step}; compare native and Shallot records: bun diagnostics/box3d-parity/records.ts ${scene} ${step} (set BOX3D to the native checkout; --full dumps both)`;
}
