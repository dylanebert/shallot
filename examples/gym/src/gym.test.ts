import { expect, test } from "bun:test";
import type { State } from "@dylanebert/shallot";
import { getScenario, installHarness, resolveNoRender } from "./gym";
import "./scenarios/render";

const installedNoRender = (mode: string): boolean => {
    const scenario = getScenario("render");
    if (!scenario) throw new Error("render scenario was not registered");
    const savedWindow = globalThis.window;
    const fakeWindow = {} as typeof window;
    try {
        globalThis.window = fakeWindow;
        installHarness(scenario, {} as State, () => true, { mode });
        return fakeWindow.__harness?.noRender === true;
    } finally {
        globalThis.window = savedWindow;
    }
};

test("the registered render scenario forwards only its two reference-probe rows", () => {
    expect(installedNoRender("spec")).toBe(true);
    expect(installedNoRender("cascade-boundary")).toBe(true);
    expect(installedNoRender("cull")).toBe(false);
    expect(installedNoRender("cascade")).toBe(false);
});

test("static noRender declarations keep their diagnostic-scenario behavior", () => {
    expect(resolveNoRender(true, {})).toBe(true);
    expect(resolveNoRender(undefined, {})).toBe(false);
});
