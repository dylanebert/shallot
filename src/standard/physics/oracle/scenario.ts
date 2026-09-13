import { Body } from "../api/body";
import { World } from "../api/world";
import type { Pos, Quat, Vec3 } from "../common/math";
import { buildScenario, createScenarioStepper, type Fixture } from "../solver/step.fixture";
import { makeBodyId } from "../world/body";
import { hashWorldState } from "../world/hash";
import corpus from "./scenarios/v2.json";

export type ScenarioCommand = (typeof corpus.scenarios)[number];
export const SCENARIO_CORPUS = corpus;
export const SCENARIO_NAMES = corpus.scenarios.map((scenario) => scenario.name);

function fromBits(bits: string): number {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, Number.parseInt(bits.slice(2), 16), true);
    return view.getFloat32(0, true);
}
function bits(value: number): string {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, true);
    return `0x${view.getUint32(0, true).toString(16).padStart(8, "0")}`;
}
function vecBits(value: Vec3): string[] {
    return [bits(value.x), bits(value.y), bits(value.z)];
}
function quatBits(value: Quat): string[] {
    return [bits(value.v.x), bits(value.v.y), bits(value.v.z), bits(value.s)];
}
function publicObservation(
    body: {
        getPosition(): Pos;
        getRotation(): Quat;
        getLinearVelocity(): Vec3;
        getAngularVelocity(): Vec3;
    },
    step: number,
) {
    return {
        step: `0x${step.toString(16).padStart(8, "0")}`,
        position: vecBits(body.getPosition()),
        rotation: quatBits(body.getRotation()),
        linearVelocity: vecBits(body.getLinearVelocity()),
        angularVelocity: vecBits(body.getAngularVelocity()),
    };
}

/** Run the shared command language through Shallot. It owns setup, actions, and observation timing;
 * it never consumes the official case file or an old fixture/gold. */
export function runScenario(command: ScenarioCommand, gravityYOverride?: number) {
    const setup = command.setup as {
        gravity: string[];
        enableSleep: boolean;
        enableContinuous: boolean;
        fixtureInput: Fixture;
    };
    const action = (
        command.actions as Array<{ timeStep: string; subStepCount: number; repeat: number }>
    )[0];
    const gravityY = gravityYOverride ?? fromBits(setup.gravity[1]);
    const world = new World({
        gravity: { x: fromBits(setup.gravity[0]), y: gravityY, z: fromBits(setup.gravity[2]) },
        enableSleep: setup.enableSleep,
        enableContinuous: setup.enableContinuous,
    });
    buildScenario(world, command.name, setup.fixtureInput);
    const stepFn = createScenarioStepper(command.name);
    // Every live body is an owned logical observation ID in creation order.
    const observeBodies = () =>
        world.state.bodies
            .filter((candidate, index) => candidate.id === index)
            .map((_record, index) => new Body(world.state, makeBodyId(world.state, index)));
    const publicObservations = [];
    const whiteBoxObservations = [];
    for (let step = 0; step < action.repeat; step += 1) {
        stepFn(world, step);
        world.step(fromBits(action.timeStep), action.subStepCount);
        publicObservations.push({
            step: `0x${step.toString(16).padStart(8, "0")}`,
            bodies: observeBodies().map((body) => publicObservation(body, step)),
        });
        whiteBoxObservations.push({
            step: `0x${step.toString(16).padStart(8, "0")}`,
            hash: `0x${hashWorldState(world.state).toString(16).padStart(16, "0")}`,
        });
    }
    const result = {
        id: command.id,
        family: "scenario",
        symbol: "World.step",
        input: {
            name: command.name,
            gravityY: bits(gravityY),
            timeStep: action.timeStep,
            subStepCount: action.subStepCount,
            repeat: action.repeat,
        },
        output: { publicObservations, whiteBoxObservations, whiteBoxSource: "hashWorldState" },
    };
    world.destroy();
    return result;
}

export function mutateScenarioInput(command: ScenarioCommand): ScenarioCommand {
    const copy = structuredClone(command) as ScenarioCommand;
    const setup = copy.setup as { gravity: string[] };
    setup.gravity[1] = setup.gravity[1] === "0xc1200000" ? "0xc1100000" : "0xc1200000";
    return copy;
}

export function assertExactMembership(): void {
    if (SCENARIO_CORPUS.scenarios.length !== 53)
        throw new Error(
            `scenario corpus has ${SCENARIO_CORPUS.scenarios.length} entries, expected 53`,
        );
    const ids = new Set(SCENARIO_CORPUS.scenarios.map((scenario) => scenario.id));
    const names = new Set(SCENARIO_NAMES);
    if (ids.size !== 53 || names.size !== 53)
        throw new Error("scenario corpus contains duplicate IDs or names");
}
