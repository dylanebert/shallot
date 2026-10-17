// Prints each warm window's unadmitted allocation rows of voxel turnover: sample count, bytes and the stack
// below the scheduler. argv: probe.entry.ts knobs, `all` to print admitted wrapper rows too, and
// `edit=<name>[,<name>]` to sample with source edits below, restored when the run ends.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { sampleAllocation } from "../first-person-allocation/allocation";
import { isAdmittedAllocationSite } from "../first-person-allocation/allocation-admissions";

const root = resolve(import.meta.dir, "../..");
// Each edit removes one named allocation without changing what the frame does, so the row it moves names its cause.
// A shared module-scope value is reached through globalThis so the edit stays on the lines it replaces.
const G = "(globalThis as any)";
const edits: Record<string, { file: string; pairs: [from: string, to: string][] }> = {
    "add-attached": {
        file: "src/engine/ecs/world.ts",
        pairs: [["const attached: GpuTable[] = [];", `const attached: GpuTable[] = (${G}.__attached ??= []);`], ["attached.push(table);", ""]],
    },
    "notify-closure": {
        file: "src/engine/ecs/world.ts",
        pairs: [[
            "this._membershipObservers.get(idOf(component))?.forEach((observer) => {\n            observer(eid, present);\n        });",
            "const observers = this._membershipObservers.get(idOf(component));\n        if (observers) for (const observer of observers) observer(eid, present);",
        ]],
    },
    "requirements-empty": {
        file: "src/engine/ecs/registry.ts",
        pairs: [["?.requires ?? [];", `?.requires ?? (${G}.__none ??= []);`]],
    },
    "remove-find": {
        file: "src/core/mesh/mesh.ts",
        pairs: [[
            "const index = pending.findIndex((entry) => entry.handle === handle);",
            "let index = -1; for (let i = 0; i < pending.length; i++) if (pending[i].handle === handle) { index = i; break; }",
        ]],
    },
    "backdrop-pick": {
        file: "src/standard/rendering/forward.ts",
        pairs: [["return { bg, ct };", `const pick = (${G}.__pick ??= { bg, ct }); pick.bg = bg; pick.ct = ct; return pick;`]],
    },
    "background-cached": {
        file: "src/standard/rendering/forward.ts",
        pairs: [['if (Object.keys(entries).length === 0) return "none";', 'if (ct.group2) return ct.group2.group; if (Object.keys(entries).length === 0) return "none";']],
    },
    "orbit-terms": {
        file: "src/extras/orbit/index.ts",
        pairs: [
            ["world.query([Orbit, not(OrbitSmooth)])", `world.query((${G}.__orbitAdd ??= [Orbit, not(OrbitSmooth)]))`],
            ["world.query([not(Orbit), OrbitSmooth])", `world.query((${G}.__orbitRemove ??= [not(Orbit), OrbitSmooth]))`],
            ["world.query([Orbit, OrbitSmooth, Transform])", `world.query((${G}.__orbitPose ??= [Orbit, OrbitSmooth, Transform]))`],
        ],
    },
    "orbit-accelerate": {
        file: "src/extras/orbit/index.ts",
        pairs: [
            [
                "const accelerate = (velocity: number, direction: number): number => {\n                if (direction === 0) return velocity * Math.exp(-keyDamping * dt);\n                const target = direction * keyRate;\n                const step = keyAcceleration * dt;\n                return velocity < target\n                    ? Math.min(target, velocity + step)\n                    : Math.max(target, velocity - step);\n            };",
                "",
            ],
            [
                "const OrbitSystem: System = {",
                "function accelerateKey(velocity: number, direction: number, keyDamping: number, keyRate: number, keyAcceleration: number, dt: number): number {\n    if (direction === 0) return velocity * Math.exp(-keyDamping * dt);\n    const target = direction * keyRate;\n    const step = keyAcceleration * dt;\n    return velocity < target ? Math.min(target, velocity + step) : Math.max(target, velocity - step);\n}\nconst OrbitSystem: System = {",
            ],
            ["keyYawVelocity = accelerate(keyYawVelocity, locked ? 0 : keyYaw);", "keyYawVelocity = accelerateKey(keyYawVelocity, locked ? 0 : keyYaw, keyDamping, keyRate, keyAcceleration, dt);"],
            ["keyPitchVelocity = accelerate(keyPitchVelocity, locked ? 0 : keyPitch);", "keyPitchVelocity = accelerateKey(keyPitchVelocity, locked ? 0 : keyPitch, keyDamping, keyRate, keyAcceleration, dt);"],
        ],
    },
    "orbit-look": {
        file: "src/extras/orbit/index.ts",
        pairs: [["const r = lookAtRotation(camX, camY, camZ, targetX, targetY, targetZ);", "const r = lookAtRotation(0, 1, 5, 0, 0, 0);"]],
    },
    "orbit-spread": {
        file: "src/extras/orbit/index.ts",
        pairs: [["world.storage(Transform).rotation.set(eid, ...r);", "world.storage(Transform).rotation.set(eid, r[0], r[1], r[2], r[3]);"]],
    },
    // Bisection cuts: each skips the rest of the pose loop's body from its line on.
    "orbit-cut-pan": { file: "src/extras/orbit/index.ts", pairs: [["            if (!flying && panHeld) {", "            if (eid >= 0) continue;\n            if (!flying && panHeld) {"]] },
    "orbit-cut-smooth": { file: "src/extras/orbit/index.ts", pairs: [["            const t = smoothLerp(smoothness, dt);", "            if (eid >= 0) continue;\n            const t = smoothLerp(smoothness, dt);"]] },
    "orbit-cut-write": { file: "src/extras/orbit/index.ts", pairs: [["            world.storage(Orbit).yaw.set(eid, yawO);", "            if (eid >= 0) continue;\n            world.storage(Orbit).yaw.set(eid, yawO);"]] },
    "sky-terms": {
        file: "src/extras/sky/index.ts",
        pairs: [["world.only([Sky]);", `world.only((${G}.__sky ??= [Sky]));`]],
    },
    // probe.entry.ts publishes the admitted pass wrappers as globals for these two.
    "emit-wrapped": {
        file: "examples/voxel/src/terrain.ts",
        pairs: [
            ["pass = world.frameEncoder()!.beginComputePass(s.emitPass);", `pass = ${G}.__beginComputePass(world.frameEncoder()!, s.emitPass);`],
            ["pass?.end();", `if (pass) ${G}.__endComputePass(pass);`],
        ],
    },
    "place-wrapped": {
        file: "examples/voxel/src/placement.ts",
        pairs: [
            ["const pass = encoder.beginComputePass(descriptor);", `const pass = ${G}.__beginComputePass(encoder, descriptor);`],
            ["            pass.end();", `            ${G}.__endComputePass(pass);`],
        ],
    },
};

const words = process.argv.slice(2);
const names = words.find((word) => word.startsWith("edit="))?.slice(5).split(",") ?? [];
const originals = new Map<string, string>();
const edited = new Map<string, string>();
for (const name of names) {
    const edit = edits[name];
    if (!edit) throw new Error(`unknown edit ${name}; one of ${Object.keys(edits).join(", ")}`);
    const path = resolve(root, edit.file);
    if (!originals.has(path)) originals.set(path, readFileSync(path, "utf8"));
    let text = edited.get(path) ?? originals.get(path)!;
    for (const [from, to] of edit.pairs) {
        if (text.split(from).length !== 2) throw new Error(`edit ${name}: ${from} is not unique in ${edit.file}`);
        text = text.replace(from, to);
    }
    edited.set(path, text);
}
try {
    for (const [path, text] of edited) writeFileSync(path, text);
    const sample = await sampleAllocation(resolve(import.meta.dir, "probe.entry.ts"), {
        warm: 2_400,
        frames: 60,
        input: words.filter((word) => word !== "all" && !word.startsWith("edit=")).join(" "),
        retainOptimizedSites: true,
    });
    for (const window of sample.windows) {
        console.log(`== ${window.label}; optimized ${window.optimizations?.join(", ") || "nothing"}`);
        for (const row of window.sites) {
            if (!words.includes("all") && isAdmittedAllocationSite(row)) continue;
            const stack = row.stack?.map((frame) => frame.site.replace(/^(\S+) src\//, "$1 ")) ?? [];
            const from = stack.findLastIndex((site) => /^runGroup /.test(site));
            console.log(`${String(row.count).padStart(5)} ${String(row.bytes).padStart(7)} B  ${stack.slice(from + 1).join(" <- ")}`);
        }
    }
    console.log(`control rows ${sample.control.length}`);
} finally {
    for (const [path, text] of originals) writeFileSync(path, text);
}
