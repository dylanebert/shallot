import { existsSync, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";

export interface QuarantineRow {
    file: string;
    claim: string;
    reason: string;
    expires: string;
    spec: string;
}

export interface DeclarationFile<Row> {
    rows: Row[];
    errors: string[];
}

function isIsoDate(value: unknown): value is string {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const date = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(date.valueOf()) && date.toISOString().slice(0, 10) === value;
}

/** Read the project's quarantine declaration; absent means no quarantined claims. */
export function readQuarantine(root: string): DeclarationFile<QuarantineRow> {
    const path = resolve(root, "quarantine.json");
    if (!existsSync(path)) return { rows: [], errors: [] };
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
        return { rows: [], errors: [`invalid quarantine.json: ${(error as Error).message}`] };
    }
    if (!Array.isArray(parsed))
        return { rows: [], errors: ["invalid quarantine.json: expected an array"] };
    const rows: QuarantineRow[] = [];
    const errors: string[] = [];
    for (const [index, raw] of parsed.entries()) {
        const row = (raw ?? {}) as Record<string, unknown>;
        const missing = ["file", "claim", "reason", "expires", "spec"].filter(
            (field) => typeof row[field] !== "string" || (row[field] as string).trim() === "",
        );
        if (missing.length > 0) {
            errors.push(
                `invalid quarantine row ${index + 1}: fields must be strings: ${missing.join(", ")}`,
            );
            continue;
        }
        if (!isIsoDate(row.expires)) {
            errors.push(`invalid quarantine row ${index + 1}: expires must be an ISO date`);
            continue;
        }
        rows.push(row as unknown as QuarantineRow);
    }
    return { rows, errors };
}

/** Return a quarantine reason for a registration, or null when the row is live. */
export function quarantineReason(root: string, file: string, claim: string): string | null {
    const relativeFile = relative(resolve(root), file).split("\\").join("/");
    return (
        readQuarantine(root).rows.find((row) => row.file === relativeFile && row.claim === claim)
            ?.reason ?? null
    );
}
