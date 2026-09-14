import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import {
    type CargoArtifact,
    selectCargoTestExecutable,
    selectCargoTestTargetExecutables,
} from "./verdict";

check(
    "the Cargo carrier recognizes declared cdylib and rlib test targets",
    {
        claim: "the Cargo carrier admits the current libtest executable when Cargo declares cdylib and rlib kinds",
    },
    () => {
        const root = mkdtempSync(join(resolve(import.meta.dir, "../.."), ".verdict-cargo-"));
        const executable = join(root, "shallot_physics-current");
        writeFileSync(executable, "current");
        const target = {
            kind: ["cdylib", "rlib"],
            name: "shallot_physics",
            test: true,
        };
        const fixture: CargoArtifact[] = [
            { target, profile: { test: false }, executable: join(root, "stale") },
            { target, profile: { test: true }, executable },
        ];
        try {
            // This is the old refusal premise: Cargo's declared kinds do not contain the literal
            // `lib`, even though the second artifact is the current libtest executable.
            expect(target.kind.includes("lib")).toBe(false);
            expect(selectCargoTestExecutable("shallot-physics", fixture)).toEqual({ executable });
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);

check(
    "the Cargo carrier refuses stale and ambiguous test targets",
    {
        claim: "the Cargo carrier refuses missing current executables and more than one current libtest executable",
    },
    () => {
        const root = mkdtempSync(join(resolve(import.meta.dir, "../.."), ".verdict-cargo-"));
        const first = join(root, "shallot_physics-first");
        const second = join(root, "shallot_physics-second");
        writeFileSync(first, "first");
        writeFileSync(second, "second");
        const target = {
            kind: ["cdylib", "rlib"],
            name: "shallot_physics",
            test: true,
        };
        const artifact = (executable: string): CargoArtifact => ({
            target,
            profile: { test: true },
            executable,
        });
        const targetArtifact = (name: string, executable: string): CargoArtifact => ({
            target: { kind: ["test"], name, test: true },
            profile: { test: true },
            executable,
        });
        const targetReason =
            "cargo test --no-run -p shallot-physics produced missing, stale, or ambiguous named test executables";
        try {
            expect(
                selectCargoTestExecutable("shallot-physics", [artifact(join(root, "missing"))]),
            ).toEqual({
                reason: "cargo test --no-run -p shallot-physics produced multiple or missing libtest executables",
            });
            expect(
                selectCargoTestExecutable("shallot-physics", [artifact(first), artifact(second)]),
            ).toEqual({
                reason: "cargo test --no-run -p shallot-physics produced multiple or missing libtest executables",
            });
            expect(
                selectCargoTestTargetExecutables(
                    "shallot-physics",
                    ["gold"],
                    [targetArtifact("gold", join(root, "missing"))],
                ),
            ).toEqual({ reason: targetReason });
            expect(
                selectCargoTestTargetExecutables(
                    "shallot-physics",
                    ["gold"],
                    [targetArtifact("gold", first), targetArtifact("gold", second)],
                ),
            ).toEqual({ reason: targetReason });
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);
