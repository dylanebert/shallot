import { State } from "@dylanebert/shallot/ecs";
import { BrowserInputPlugin, devices, InputPlugin } from "@dylanebert/shallot/input";

const state = new State();
for (const system of InputPlugin.systems ?? []) state.addSystem(system, InputPlugin.name);
for (const system of BrowserInputPlugin.systems ?? [])
    state.addSystem(system, BrowserInputPlugin.name);
state.step(0);

const observations: { focused?: boolean; unfocused?: boolean } = {};
const inputCheck = {
    held: () => devices(state).keys.held.has("KeyW"),
    observations,
};
(globalThis as typeof globalThis & { __inputCheck: typeof inputCheck }).__inputCheck = inputCheck;

(
    globalThis as typeof globalThis & {
        __harness?: {
            ready: boolean;
            run(): Promise<{ ok: boolean; checks: { name: string; ok: boolean }[] }>;
        };
    }
).__harness = {
    ready: true,
    async run() {
        const checks = [
            {
                name: "a real key press on the focused canvas is held in its State record",
                ok: observations.focused === true,
            },
            {
                name: "the same real key press is ignored when the canvas is unfocused",
                ok: observations.unfocused === false,
            },
        ];
        state.dispose();
        return { ok: checks.every((check) => check.ok), checks };
    },
};
