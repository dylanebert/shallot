import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { relative, resolve } from "node:path";

/** the result vocabulary printed by the surface reporter. */
export type VerdictResult = "pass" | "fail" | "refused" | "unrun";

/** optional host measurements returned by a process-tier check. */
export interface VerdictMetadata {
    runtime?: string;
    hardware?: string;
    reason?: string;
}

interface QuarantineRow {
    file: string;
    claim: string;
    reason: string;
}

const require = createRequire(import.meta.url);
const ROOT = resolve(import.meta.dir, "../..");

function readQuarantine(): QuarantineRow[] {
    const path = resolve(ROOT, "quarantine.json");
    if (!existsSync(path)) return [];
    try {
        const value = JSON.parse(readFileSync(path, "utf8"));
        if (!Array.isArray(value)) return [];
        return value.filter(
            (row): row is QuarantineRow =>
                row !== null &&
                typeof row === "object" &&
                typeof row.file === "string" &&
                typeof row.claim === "string" &&
                typeof row.reason === "string",
        );
    } catch {
        return [];
    }
}

/** return a quarantine reason for a runtime registration, or null when the row is live. */
export function quarantineReason(file: string, claim: string): string | null {
    const relativeFile = relative(ROOT, file).split("\\").join("/");
    return (
        readQuarantine().find((row) => row.file === relativeFile && row.claim === claim)?.reason ??
        null
    );
}

/** attempt to load each named package premise; absence is a refusal, never a pass. */
export function missingPremise(premises: readonly string[]): string | null {
    for (const premise of premises) {
        try {
            require(premise);
        } catch {
            return `${premise} is unavailable`;
        }
    }
    return null;
}

/** the default runtime/hardware labels for checks that do not return host measurements. */
export function defaultMetadata(): VerdictMetadata {
    return { runtime: `bun ${Bun.version}`, hardware: "none" };
}

function metadataFrom(value: unknown): VerdictMetadata {
    if (value === null || typeof value !== "object") return {};
    const candidate = value as Record<string, unknown>;
    return {
        runtime: typeof candidate.runtime === "string" ? candidate.runtime : undefined,
        hardware: typeof candidate.hardware === "string" ? candidate.hardware : undefined,
        reason: typeof candidate.reason === "string" ? candidate.reason : undefined,
    };
}

/** extract host labels from a returned value or an error carrying process-run diagnostics. */
export function verdictMetadata(value: unknown): VerdictMetadata {
    return metadataFrom(value);
}

/** print the one-line, structured verdict for a non-step or refused check. */
export function emitVerdict(
    claim: string,
    tier: string,
    started: number,
    result: VerdictResult,
    metadata: VerdictMetadata = {},
): void {
    const defaults = defaultMetadata();
    const runtime =
        metadata.runtime ??
        (tier === "browser" ? `${defaults.runtime} + chromium unavailable` : defaults.runtime);
    const line = {
        claim,
        tier,
        runtime,
        hardware: metadata.hardware ?? defaults.hardware,
        duration: Number((performance.now() - started).toFixed(2)),
        result,
        ...(metadata.reason === undefined ? {} : { reason: metadata.reason }),
    };
    console.log(`shallot verdict ${JSON.stringify(line)}`);
}
