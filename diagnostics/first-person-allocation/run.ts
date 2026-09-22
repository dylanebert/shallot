import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { sampleAllocation } from "../../src/harness/allocation";

const root = resolve(import.meta.dir);
const scenePath = resolve(root, "../../examples/first-person/public/scenes/first-person.scene");
const scene = readFileSync(scenePath, "utf8");
const baseline = resolve(root, "../../examples/first-person/src/allocation.entry.ts");
const entries = {
    baseline,
    noopStep: resolve(root, "entries/no-op-step.ts"),
    knownSubjectAllocation: resolve(root, "entries/known-subject.ts"),
    noCharacterSystem: resolve(root, "entries/no-character-system.ts"),
    noPhysicsStep: resolve(root, "entries/no-physics-step.ts"),
};
const sceneControls = {
    empty: "<scene></scene>",
    character:
        '<scene><a id="player" body="pos: 0 1.4 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0" character /></scene>',
    characterGround:
        '<scene><a id="player" body="pos: 0 1.4 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0" character /><a id="ground" body="pos: 0 0 0; half-extents: 16 0.5 26; mass: 0" /></scene>',
};

const sample = async (entry: string, input: string = scene) =>
    sampleAllocation(entry, { warm: 6000, frames: 600, input });
const all: Record<string, unknown> = {
    metadata: {
        baseline: "c6a7fc16",
        bun: Bun.version,
        node: "sampler child runtime is recorded in each sample.runtime",
        warm: 6000,
        frames: 600,
        input: "examples/first-person/public/scenes/first-person.scene",
    },
    composition: {},
    sceneControls: {},
};
for (const [name, entry] of Object.entries(entries)) {
    console.error(`RUN ${name}`);
    (all.composition as Record<string, unknown>)[name] = await sample(entry);
}
for (const [name, input] of Object.entries(sceneControls)) {
    console.error(`RUN scene:${name}`);
    (all.sceneControls as Record<string, unknown>)[name] = await sample(baseline, input);
}
writeFileSync(resolve(root, "all-results.json"), `${JSON.stringify(all, null, 2)}\n`);
console.log(JSON.stringify(all, null, 2));
