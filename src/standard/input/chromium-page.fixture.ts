import { devices, InputPlugin, State, Time } from "@dylanebert/shallot";
import type { Check, Verdict } from "@dylanebert/shallot/harness";

const canvas = document.createElement("canvas");
canvas.width = 1280;
canvas.height = 720;
canvas.tabIndex = 0;
canvas.style.width = "1280px";
canvas.style.height = "720px";
document.body.style.margin = "0";
document.body.append(canvas);
canvas.focus();

const state = new State();
for (const system of InputPlugin.systems ?? []) state.addSystem(system, InputPlugin.name);
state.step(0);

window.__harness = {
    ready: true,
    async run(): Promise<Verdict> {
        canvas.focus();
        canvas.dispatchEvent(
            new KeyboardEvent("keydown", { code: "KeyW", key: "w", bubbles: true }),
        );
        state.step(Time.FIXED_DT);
        const held = devices(state).keys.held.has("KeyW");
        const checks: Check[] = [
            {
                name: "focused canvas keydown reaches the device record",
                ok: held,
                detail: held
                    ? "KeyW is held after the browser keydown"
                    : "KeyW did not reach the record",
            },
        ];
        state.dispose();
        return { ok: held, checks, tick: state.time.fixedTick };
    },
};
