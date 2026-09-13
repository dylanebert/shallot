import { readFileSync } from "node:fs";
import { check } from "../../../harness/check";
import { getBodySim, getBodyState } from "../world/body";
import { hashWorldState } from "../world/hash";
import { buildLegacyScene } from "../solver/step.fixture";
import { loadScenarioCorpus, runScenario, type ScenarioOutput } from "./scenario";

type NumberBody = { p: string[]; q: string[]; v?: string[]; w?: string[] };
const hex = (value: number): string => {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, true);
    return `0x${view.getUint32(0, true).toString(16).padStart(8, "0")}`;
};
function legacyOutput(name: string, enableSleep: boolean, enableContinuous: boolean, steps: number): { observations: Array<{ step: number; bodies: NumberBody[] }>; hashes: string[] } {
    const world = buildLegacyScene(name, enableSleep, enableContinuous);
    const observations: Array<{ step: number; bodies: NumberBody[] }> = [];
    const hashes: string[] = [];
    for (let step = 0; step < steps; step++) {
        world.step(Math.fround(1 / 60), 4);
        const bodies: NumberBody[] = [];
        for (let index = 0; index < world.state.bodies.length; index++) {
            const body = world.state.bodies[index];
            if (body.id !== index) continue;
            const sim = getBodySim(world.state, body);
            const state = getBodyState(world.state, body);
            bodies.push({ p: [hex(sim.transform.p.x), hex(sim.transform.p.y), hex(sim.transform.p.z)], q: [hex(sim.transform.q.v.x), hex(sim.transform.q.v.y), hex(sim.transform.q.v.z), hex(sim.transform.q.s)], v: state ? [hex(state.linearVelocity.x), hex(state.linearVelocity.y), hex(state.linearVelocity.z)] : ["0x00000000", "0x00000000", "0x00000000"], w: state ? [hex(state.angularVelocity.x), hex(state.angularVelocity.y), hex(state.angularVelocity.z)] : ["0x00000000", "0x00000000", "0x00000000"] });
        }
        observations.push({ step, bodies });
        hashes.push(`0x${hashWorldState(world.state).toString(16).padStart(16, "0")}`);
    }
    world.destroy();
    return { observations, hashes };
}
function compareOutputs(generic: ScenarioOutput, legacy: ReturnType<typeof legacyOutput>): void {
    if (JSON.stringify(generic.hashes.map((item) => item.value)) !== JSON.stringify(legacy.hashes)) throw new Error(`${generic.name}: world hash mismatch`);
    if (generic.observations.length !== legacy.observations.length) throw new Error(`${generic.name}: observation count mismatch`);
    for (let step = 0; step < legacy.observations.length; step++) {
        const got = generic.observations[step] as { bodies: Array<Record<string, unknown>> };
        const withoutLogicalIds = got.bodies.map(({ id: _id, ...body }) => body);
        if (JSON.stringify(withoutLogicalIds) !== JSON.stringify(legacy.observations[step].bodies)) throw new Error(`${generic.name}: body observation mismatch at step ${step}`);
    }
}

const FOUNDATION_ROSTER = ["free-fall", "sphere-drop", "box-stack", "sphere-sleep", "box-sleep", "wake-drop", "split-slide"];
const JOINT_ROSTER = ["revolute-dd", "revolute-pendulum", "revolute-motor", "revolute-limit", "revolute-chain", "weld-dd", "parallel", "joint-contacts", "motor", "motor-spring", "distance", "distance-spring", "prismatic", "prismatic-motor", "spherical", "spherical-limits", "spherical-motor", "wheel", "wheel-spin", "wheel-steer", "ragdoll"];
function compareFamily(roster: string[]): void {
    const { corpus, digest } = loadScenarioCorpus();
    const cumulative = [...FOUNDATION_ROSTER, ...JOINT_ROSTER];
    if (JSON.stringify(corpus.scenarios.map((scenario) => scenario.name)) !== JSON.stringify(cumulative) || JSON.stringify(corpus.scenarios.map((scenario) => scenario.id)) !== JSON.stringify(cumulative.map((name) => `s1.${name}.v1`))) throw new Error("scenario corpus roster and IDs are not the exact cumulative O5b order");
    const selected = corpus.scenarios.filter((scenario) => roster.includes(scenario.name));
    if (JSON.stringify(selected.map((scenario) => scenario.name)) !== JSON.stringify(roster)) throw new Error("scenario family roster is not exact");
    for (const scenario of selected) {
        const world = scenario.commands.find((command) => command.op === "world.create");
        compareOutputs(runScenario(scenario, digest), legacyOutput(scenario.name, world?.enableSleep === true, world?.enableContinuous === true, scenario.stepCount));
    }
}
check("O5a Shallot command interpreter migration", { claim: "box3d-scenario-migration-foundation", size: "integration" }, () => compareFamily(FOUNDATION_ROSTER));
check("O5b Shallot command interpreter migration", { claim: "box3d-scenario-migration-joints", size: "integration" }, () => compareFamily(JOINT_ROSTER));

check("O5a command receipt and adversarial gates", { claim: "box3d-scenario-command-corpus", size: "integration" }, () => {
    const { corpus, digest } = loadScenarioCorpus();
    const freeFall = structuredClone(corpus.scenarios[0]);
    const baseline = runScenario(freeFall, digest);
    if (JSON.stringify(baseline.receipt.consumedCommands) !== JSON.stringify(freeFall.commands.map((command) => command.id))) throw new Error("receipt omitted a consumed command");
    if (baseline.receipt.observationIds.length !== freeFall.stepCount) throw new Error("receipt omitted an observation ID");
    const mutation = structuredClone(freeFall);
    const body = mutation.commands.find((command) => command.op === "body.create");
    if (!body) throw new Error("free-fall body command is missing");
    body.angularVelocity = ["0x40000000", "0x40a00000", "0x40000000"];
    const mutated = runScenario(mutation, digest, true);
    if (JSON.stringify(mutated.observations) === JSON.stringify(baseline.observations) || JSON.stringify(mutated.hashes) === JSON.stringify(baseline.hashes)) throw new Error("angular-velocity mutation did not reach the TypeScript adapter");
    const revoluteMotor = structuredClone(corpus.scenarios.find((scenario) => scenario.name === "revolute-motor"));
    if (!revoluteMotor) throw new Error("revolute-motor mutation target is missing");
    const motorCommand = revoluteMotor.commands.find((command) => command.op === "joint.revolute");
    if (!motorCommand) throw new Error("revolute-motor joint command is missing");
    const motorBaseline = runScenario(revoluteMotor, digest);
    motorCommand.motorSpeed = "0xc0400000";
    const motorChanged = runScenario(revoluteMotor, digest);
    if (JSON.stringify(motorBaseline.observations) === JSON.stringify(motorChanged.observations) || JSON.stringify(motorBaseline.hashes) === JSON.stringify(motorChanged.hashes)) throw new Error("revolute-motor motor-speed mutation did not reach the TypeScript adapter");
    const deletedJoint = structuredClone(revoluteMotor);
    deletedJoint.commands = deletedJoint.commands.filter((command) => command.op !== "joint.revolute");
    try { runScenario(deletedJoint, digest); throw new Error("deleting a joint command unexpectedly succeeded"); } catch (error) { if (!(error instanceof Error) || !error.message.includes("missing joint commands")) throw error; }
    const unknown = structuredClone(freeFall);
    unknown.commands[0].op = "scenario-name-dispatch";
    try { runScenario(unknown, digest); throw new Error("unknown command unexpectedly succeeded"); } catch (error) { if (!(error instanceof Error) || !error.message.includes("unknown command op")) throw error; }
    const unconsumed = structuredClone(freeFall);
    unconsumed.commands = unconsumed.commands.filter((command) => command.op !== "hash" || command.step !== 0);
    try { runScenario(unconsumed, digest); throw new Error("unconsumed command unexpectedly succeeded"); } catch (error) { if (!(error instanceof Error) || !error.message.includes("unconsumed schedule")) throw error; }
    const source = readFileSync(new URL("./scenario.ts", import.meta.url), "utf8");
    if (/legacyBuilder|setupKind|builders\s*\[|scenario\.(name|id)\s*===/.test(source)) throw new Error("generic interpreter contains forbidden name dispatch");
});
