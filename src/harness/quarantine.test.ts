import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "./check";
import { quarantineReason, readQuarantine } from "./quarantine";

check(
    "quarantine registration uses the explicit project root",
    {
        claim: "check registration reads the matching quarantine from its declared project root and leaves other claims live",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-check-quarantine-"));
        const file = join(root, "src/row.test.ts");
        try {
            writeFileSync(
                join(root, "quarantine.json"),
                JSON.stringify([
                    {
                        file: "src/row.test.ts",
                        claim: "quarantined claim",
                        reason: "retained reason",
                        expires: "2099-01-01",
                        spec: "test seam",
                    },
                ]),
            );
            expect(readQuarantine(root).errors).toEqual([]);
            expect(quarantineReason(root, file, "quarantined claim")).toBe("retained reason");
            expect(quarantineReason(root, file, "live claim")).toBeNull();
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);
