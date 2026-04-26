import { mount } from "svelte";
import { run } from "@dylanebert/shallot";
import { config, noUI, activeScenarioName, stepCount, Body, Transform, Compute } from "./lib";
import { gymState } from "./state.svelte";
import App from "./App.svelte";

if (noUI) {
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100vw;height:100vh";
    document.getElementById("app")!.appendChild(canvas);
} else {
    mount(App, { target: document.getElementById("app")! });
}

const state = await run(config);
gymState.ecs = state;

const compute = Compute.from(state);

const shallotExport: Record<string, unknown> = {
    state,
    Transform,
    Body,
    gpuResourceCount: () => {
        const registry = compute ? (compute.device as any).__gpuRegistry : null;
        return registry ? registry.count() : -1;
    },
};

if (activeScenarioName) {
    shallotExport.getStepCount = () => stepCount;
}

(window as any).__shallot = shallotExport;

if (import.meta.hot) {
    import.meta.hot.dispose(() => state.dispose());
}
