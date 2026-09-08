import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

export const harnessContract = `
import * as harness from "@dylanebert/shallot/harness";
import { REAL_GPU_LAUNCH as leaf } from "@dylanebert/shallot/harness/browser";
const names = ["installHarness", "isDegradedBootMessage", "assertMotion", "frameDifference", "pixelProbePass", "probePixels", "REAL_GPU_LAUNCH"].sort();
if (JSON.stringify(Object.keys(harness).sort()) !== JSON.stringify(names)) throw new Error("HARNESS_SURFACE");
const launch = { channel: "chromium", args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures", "--class=kex-gate"] };
for (const value of [harness.REAL_GPU_LAUNCH, leaf]) {
    if (JSON.stringify(value) !== JSON.stringify(launch)) throw new Error("HARNESS_LAUNCH");
}
if (harness.assertMotion([0], [2], 1) !== 2) throw new Error("HARNESS_MOTION");
let refused = false;
try { harness.assertMotion([0], [0], 1); } catch (error) { refused = /samples are parked/.test(String(error)); }
if (!refused) throw new Error("HARNESS_MOTION_REFUSAL");
console.log("HARNESS_CONTRACT_OK");
`;

const types = `
import type { Check, Verdict, PoseState, HarnessTarget, PixelProbe, PixelProbeResult, RealGpuLaunch } from "@dylanebert/shallot/harness";
import type { RealGpuLaunch as LeafLaunch } from "@dylanebert/shallot/harness/browser";
import { REAL_GPU_LAUNCH } from "@dylanebert/shallot/harness";
const channel: "chromium" = REAL_GPU_LAUNCH.channel;
const launch: RealGpuLaunch = REAL_GPU_LAUNCH;
const leaf: LeafLaunch = launch;
leaf.args.push("consumer-owned-option");
type Protocol = [Check, Verdict, PoseState, HarnessTarget, PixelProbe, PixelProbeResult];
`;

/** Exercise the raw public composition and its independent Node/type leaves in a physical install. */
export function harnessArms(project: string): void {
    const shipped = join(project, "node_modules/@dylanebert/shallot");
    const exec = (name: string, cmd: string[], pass: boolean, message?: RegExp) => {
        const result = Bun.spawnSync(cmd, { cwd: project, stdout: "pipe", stderr: "pipe" });
        const output = `${result.stdout.toString()}\n${result.stderr.toString()}`;
        assert.equal(result.exitCode === 0, pass, `${name}: ${output}`);
        if (message) assert.match(output, message, name);
        console.log(`harness: ${name} exit=${result.exitCode}`);
    };
    writeFileSync(join(project, "harness-contract.ts"), harnessContract);
    writeFileSync(join(project, "harness-types.ts"), types);
    writeFileSync(
        join(project, "harness-preload.ts"),
        `import { installGpuGlobals } from "./node_modules/@dylanebert/shallot/bin/gpu-globals.ts"; installGpuGlobals();\n`,
    );
    writeFileSync(
        join(project, "harness-node.mjs"),
        `import { REAL_GPU_LAUNCH } from "@dylanebert/shallot/harness/browser";
if (JSON.stringify(REAL_GPU_LAUNCH) !== '${JSON.stringify({ channel: "chromium", args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures", "--class=kex-gate"] })}') throw new Error("NODE_LAUNCH");
console.log("NODE_LEAF_OK");\n`,
    );
    const raw = ["bun", "--preload", "./harness-preload.ts", "harness-contract.ts"];
    const node = ["node", "harness-node.mjs"];
    const typecheck = [
        "bun",
        resolve(import.meta.dir, "../../node_modules/typescript/bin/tsc"),
        "--noEmit",
        "--skipLibCheck",
        "--moduleResolution",
        "bundler",
        "--module",
        "esnext",
        "--target",
        "esnext",
        "--types",
        "@webgpu/types,vite/client",
        "harness-types.ts",
    ];
    exec("raw value surface", raw, true, /HARNESS_CONTRACT_OK/);
    exec("Node leaf", node, true, /NODE_LEAF_OK/);
    exec("raw and leaf types", typecheck, true);
    const mutate = (file: string, change: (source: string) => string, body: () => void) => {
        const path = join(shipped, file);
        const before = readFileSync(path, "utf8");
        const after = change(before);
        assert.notEqual(after, before, `unreached mutation: ${file}`);
        try {
            writeFileSync(path, after);
            body();
        } finally {
            writeFileSync(path, before);
        }
    };
    mutate(
        "src/harness/index.ts",
        (s) => s.replace("REAL_GPU_LAUNCH, ", ""),
        () => {
            exec("missing raw launch", raw, false, /HARNESS_SURFACE/);
            exec("independent Node leaf survives", node, true, /NODE_LEAF_OK/);
        },
    );
    mutate(
        "src/harness/index.ts",
        (s) => s.replace(", type RealGpuLaunch", ""),
        () => exec("missing raw type", typecheck, false, /RealGpuLaunch/),
    );
    for (const [before, after, name] of [
        ['channel: "chromium";', "channel: string;", "widened channel"],
        ["args: string[];", "args: readonly string[];", "immutable args"],
    ])
        mutate(
            "src/harness/browser.ts",
            (s) => s.replace(before, after),
            () => exec(name, typecheck, false),
        );
    mutate(
        "package.json",
        (s) => {
            const pkg = JSON.parse(s);
            pkg.exports["./harness/browser"].default = "./src/harness/browser.ts";
            return JSON.stringify(pkg);
        },
        () => exec("Node raw-TS default", node, false, /node_modules|TypeScript|typescript/),
    );
    for (const [file, command, name] of [
        ["dist/harness-browser.js", node, "missing compiled leaf"],
        ["src/harness/browser.ts", typecheck, "missing type projection"],
        ["src/harness/runtime.ts", raw, "missing runtime projection"],
    ] as const) {
        const path = join(shipped, file);
        try {
            renameSync(path, `${path}.absent`);
            exec(name, [...command], false);
        } finally {
            renameSync(`${path}.absent`, path);
        }
    }
    mutate(
        "package.json",
        (s) => {
            const pkg = JSON.parse(s);
            pkg.exports["./harness"] = "./dist/harness-browser.js";
            return JSON.stringify(pkg);
        },
        () => exec("compiled aggregate replacement", raw, false, /HARNESS_SURFACE/),
    );
    const record = JSON.parse(readFileSync(join(shipped, "dist/tooling-inputs.json"), "utf8"));
    const hash = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
    const owner = resolve(import.meta.dir, "../../packages/shallot-tooling");
    const tracked = Bun.spawnSync(["git", "ls-files", "--", "packages/shallot-tooling"], {
        cwd: resolve(import.meta.dir, "../.."),
    });
    assert.equal(tracked.exitCode, 0, "canonical source inventory");
    const expectedInputs = tracked.stdout
        .toString()
        .trim()
        .split("\n")
        .map((file) => file.replace("packages/shallot-tooling/", ""))
        .filter((file) =>
            /^(bin\/|src\/project\/|src\/harness\/browser\.ts$|rust\/window\/|assets\/|scripts\/build\.ts$|package\.json$)/.test(
                file,
            ),
        )
        .filter((file) => !/\.(test|probes)\.ts$|\/\.gitignore$/.test(file))
        .sort();
    assert(expectedInputs.length > 40, "nonempty tooling source population");
    assert.deepEqual(
        Object.keys(record.inputs).sort(),
        expectedInputs,
        "two-way canonical projection population",
    );
    for (const [file, expected] of Object.entries(record.inputs))
        assert.equal(hash(join(owner, file)), expected, `canonical input ${file}`);
    for (const [file, expected] of Object.entries(record.outputs))
        assert.equal(hash(join(shipped, file)), expected, `installed projection ${file}`);
    exec("restored raw surface", raw, true, /HARNESS_CONTRACT_OK/);
    console.log(
        `harness: ${Object.keys(record.inputs).length} canonical hashes, ${Object.keys(record.outputs).length} installed projections`,
    );
}
