import { expect } from "bun:test";
import { cpSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { Glob } from "bun";
import { readDeviceTierViolations } from "./check-device-tiers";
import { unfixture } from "./unfixture";

const ROOT = resolve(import.meta.dir, "..");
const FIXTURE = resolve(ROOT, "scripts/fixtures/surface/undeclared-device");

check(
    "device declaration gate: undeclared fixture reds",
    {
        claim: "the device declaration gate refuses a standard module that reads Compute.device without a plugin device declaration",
    },
    () => {
        const tree = mkdtempSync(join(Bun.env.TMPDIR ?? "/tmp", "shallot-device-tier-"));
        try {
            cpSync(FIXTURE, tree, { recursive: true });
            unfixture(tree);
            const violations = readDeviceTierViolations(tree);
            expect(violations).toEqual([
                "src/standard/rogue/index.ts: references Compute.device/root/buffers but its plugin has no device declaration",
            ]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "device declaration gate: shipped tree is green",
    {
        claim: "every shipped standard and extras module that reads Compute.device, Compute.root or Compute.buffers belongs to a plugin with a device declaration",
        size: "integration",
        subject: "src/standard",
    },
    () => {
        // Floor: the gate's own glob and device-read pattern find modules to judge, so an empty scan
        // cannot pass as green.
        const scanned = ["src/standard/**/*.ts", "src/extras/**/*.ts"]
            .flatMap((pattern) => [...new Glob(pattern).scanSync(ROOT)])
            .filter((file) =>
                /\bCompute\.(?:device|root|buffers)\b/.test(
                    readFileSync(resolve(ROOT, file), "utf8"),
                ),
            );
        expect(scanned.length).toBeGreaterThan(0);
        expect(readDeviceTierViolations(ROOT)).toEqual([]);
    },
);
