import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import {
    type CargoArtifact,
    currentHost,
    hostMismatch,
    nodeVersionMismatch,
    selectCargoTestExecutable,
    selectCargoTestTargetExecutables,
} from "./verdict";

check(
    "the node requirement holds Node to its exact pin",
    {
        claim: "the node requirement admits only the exact pinned version and refuses a loose pin or another version",
    },
    () => {
        expect(nodeVersionMismatch("26.8.1", "v26.8.1\n")).toBeNull();
        expect(nodeVersionMismatch("26.8.1", "v26.8.0\n")).toBe(
            "node v26.8.0 does not match the pinned 26.8.1",
        );
        expect(nodeVersionMismatch("26.8.1", "v126.8.1")).not.toBeNull();
        expect(nodeVersionMismatch("26.8.1", "")).toBe(
            "node (no version) does not match the pinned 26.8.1",
        );
        expect(nodeVersionMismatch("26", "v26.8.1")).toBe(
            "node pin must be an exact version, not 26",
        );
    },
);

check(
    "the Cargo carrier recognizes declared cdylib and rlib test targets",
    {
        claim: "the Cargo carrier admits the current libtest executable when Cargo declares cdylib and rlib kinds",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-verdict-cargo-"));
        try {
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
        const root = mkdtempSync(join(tmpdir(), "shallot-verdict-cargo-"));
        try {
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

check(
    "a Hyprland session is the omarchy host unless the host names itself",
    {
        claim: "a Linux session under Hyprland reports host other, or a hosted Linux runner or SHALLOT_HOST is overridden, so every omarchy row goes unrun on its own seat",
        subject: "src/harness/verdict.ts",
    },
    () => {
        expect(currentHost({ HYPRLAND_INSTANCE_SIGNATURE: "abc" }, "linux")).toBe("omarchy");
        expect(currentHost({}, "linux")).toBe("other");
        expect(currentHost({ HYPRLAND_INSTANCE_SIGNATURE: "abc" }, "darwin")).toBe("mac");
        expect(
            currentHost({ HYPRLAND_INSTANCE_SIGNATURE: "abc", SHALLOT_HOST: "mac" }, "linux"),
        ).toBe("mac");
        expect(hostMismatch(["mac", "omarchy"], "omarchy")).toBeNull();
        expect(hostMismatch(["mac", "omarchy"], "other")).toBe(
            "declared for hosts mac, omarchy; this host is other",
        );
    },
);
