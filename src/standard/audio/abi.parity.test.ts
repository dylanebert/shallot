/**
 * Cross-language ABI parity gate: the TS audio layer hand-mirrors five capacity
 * constants and the 15-entry NodeType discriminant map from the Rust worklet
 * source. A mismatch silently corrupts the worklet ABI, and `cargo test` sees
 * only the Rust side. This arm parses the Rust source and asserts the TS mirrors
 * equal it — resolving each mirror by importing the module, never by re-parsing
 * or re-spelling its value.
 *
 * Witnessed red (mutation proof, run in place):
 *   - MAX_VOICES in rust/audio/src/lib.rs changed 64 → 65: arm red, exit 1.
 *   - NODE_TYPE_ID discriminant reorder: swapped `phaser: 14` and `tremolo: 15`
 *     in the Rust enum (graph.rs): arm red, exit 1.
 *   Both restored with `git show HEAD:<path> > <path>`.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { MAX_VOICES } from "./core";
import { MAX_BUFFERS, MAX_INSTRUMENTS, NODE_TYPE_ID } from "./instrument";
import { MAX_SAMPLES } from "./sample";
import { MAX_TRANSPORTS } from "./worklet";

const libRs = readFileSync(new URL("../../../rust/audio/src/lib.rs", import.meta.url), "utf8");
const graphRs = readFileSync(new URL("../../../rust/audio/src/graph.rs", import.meta.url), "utf8");

/** Parse `pub const NAME: TYPE = VALUE;` from Rust source. */
function parseRustConst(source: string, name: string): number {
    const re = new RegExp(`pub\\s+const\\s+${name}\\s*:\\s*\\w+\\s*=\\s*(\\d+)`);
    const m = source.match(re);
    if (!m) throw new Error(`Rust constant ${name} not found in source`);
    return Number(m[1]);
}

/**
 * Parse the Rust `NodeType` enum's variant→discriminant mapping from graph.rs.
 * Returns a Map<PascalCase, number> for every variant including `None = 0`.
 */
function parseNodeTypeEnum(source: string): Map<string, number> {
    const enumMatch = source.match(/pub\s+enum\s+NodeType\s*\{([^}]+)\}/);
    if (!enumMatch) throw new Error("NodeType enum not found in graph.rs");
    const body = enumMatch[1];
    const variants = new Map<string, number>();
    for (const line of body.split("\n")) {
        const m = line.match(/^\s*(\w+)\s*=\s*(\d+)\s*,/);
        if (m) variants.set(m[1], Number(m[2]));
    }
    return variants;
}

describe("audio ABI parity (Rust source ↔ TS mirrors)", () => {
    test("five capacity constants match the Rust source", () => {
        const rustMaxVoices = parseRustConst(libRs, "MAX_VOICES");
        const rustMaxSamples = parseRustConst(libRs, "MAX_SAMPLES");
        const rustMaxTransports = parseRustConst(libRs, "MAX_TRANSPORTS");
        const rustMaxInstruments = parseRustConst(graphRs, "MAX_INSTRUMENTS");
        const rustMaxBuffers = parseRustConst(graphRs, "MAX_BUFFERS");

        // Scalar equality is symmetric — one direction per constant is the whole
        // check; the load-bearing half is that each left side is the *imported*
        // mirror, so a module move or rename reds here rather than at a re-spelling.
        expect(MAX_VOICES).toBe(rustMaxVoices);
        expect(MAX_SAMPLES).toBe(rustMaxSamples);
        expect(MAX_TRANSPORTS).toBe(rustMaxTransports);
        expect(MAX_INSTRUMENTS).toBe(rustMaxInstruments);
        expect(MAX_BUFFERS).toBe(rustMaxBuffers);
    });

    test("NODE_TYPE_ID map mirrors the Rust NodeType discriminant order", () => {
        const rustVariants = parseNodeTypeEnum(graphRs);

        // Derive the expected TS entry count from the Rust parse: every variant
        // except `None` (discriminant 0, which the TS union type omits).
        const expectedCount = [...rustVariants.values()].filter((v) => v !== 0).length;
        expect(Object.keys(NODE_TYPE_ID).length).toBe(expectedCount);

        // Bidirectional whole-equality: every non-None Rust variant appears in
        // the TS map with the same discriminant, and every TS map entry appears
        // in the Rust enum with the same value.
        const tsEntries = Object.entries(NODE_TYPE_ID);

        // Rust → TS: every non-None variant must be in the TS map.
        for (const [rustName, rustVal] of rustVariants) {
            if (rustVal === 0) continue;
            const tsKey = rustName.toLowerCase();
            expect(NODE_TYPE_ID).toHaveProperty(tsKey);
            expect(NODE_TYPE_ID[tsKey as keyof typeof NODE_TYPE_ID]).toBe(rustVal);
        }

        // TS → Rust: every TS map entry must be in the Rust enum with the same value.
        for (const [tsKey, tsVal] of tsEntries) {
            const rustName = tsKey.charAt(0).toUpperCase() + tsKey.slice(1);
            expect(rustVariants.has(rustName)).toBe(true);
            expect(rustVariants.get(rustName)).toBe(tsVal);
        }
    });
});
