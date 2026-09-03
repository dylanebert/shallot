import { mountOverlay, type Plugin, type State, type System } from "../../engine";
import { Inputs } from "../../standard/input";
import { Orbit } from "./index";

const controls = () =>
    [
        {
            field: Orbit.keyRate,
            label: "rate",
            unit: "rad/s",
            step: 0.1,
            minus: "Digit1",
            plus: "Digit2",
        },
        {
            field: Orbit.keyAcceleration,
            label: "acceleration",
            unit: "rad/s²",
            step: 0.5,
            minus: "Digit3",
            plus: "Digit4",
        },
        {
            field: Orbit.keyDamping,
            label: "release damping",
            unit: "s⁻¹",
            step: 0.5,
            minus: "Digit5",
            plus: "Digit6",
        },
    ] as const;

function text(eid: number): string {
    return controls()
        .map(({ field, label, unit }) => `${label} ${field.get(eid).toFixed(1)} ${unit}`)
        .join(" · ");
}

let cleanup: (() => void) | null = null;
let last = "";

const TuningSystem: System = {
    group: "draw",
    last: true,
    update(state: State) {
        const eid = [...state.query([Orbit])][0];
        if (eid === undefined) return;
        for (const control of controls()) {
            const direction =
                Number(Inputs.isKeyPressed(control.plus)) -
                Number(Inputs.isKeyPressed(control.minus));
            if (direction)
                control.field.set(
                    eid,
                    Math.max(0, control.field.get(eid) + direction * control.step),
                );
        }
        const value = text(eid);
        const hasDom =
            typeof document !== "undefined" && typeof document.createElement === "function";
        if (value !== last) {
            if (!hasDom) console.error(`orbit tuning: ${value}`);
            last = value;
        }
        if (cleanup || !hasDom) return;
        const parent = mountOverlay(document.querySelector("canvas"), state);
        const root = document.createElement("div");
        Object.assign(root.style, {
            position: "absolute",
            top: "12px",
            right: "12px",
            padding: "8px",
            background: "rgba(14,13,12,0.9)",
            color: "#f0ece8",
            font: "11px 'JetBrains Mono', monospace",
            pointerEvents: "auto",
        });
        const readout = document.createElement("div");
        readout.textContent = value;
        root.append(readout);
        for (const control of controls()) {
            const row = document.createElement("div");
            row.textContent = `${control.minus.slice(-1)}/${control.plus.slice(-1)} ${control.label} `;
            for (const [label, direction] of [
                ["−", -1],
                ["+", 1],
            ] as const) {
                const button = document.createElement("button");
                button.textContent = label;
                button.addEventListener(
                    "click",
                    () => {
                        control.field.set(
                            eid,
                            Math.max(0, control.field.get(eid) + direction * control.step),
                        );
                        readout.textContent = text(eid);
                    },
                    { signal: state.signal },
                );
                row.append(button);
            }
            root.append(row);
        }
        parent.append(root);
        cleanup = () => parent.remove();
        state.onDispose(cleanup);
    },
};

/** Temporary live instrument for tuning keyboard orbit with keys 1–6 or the on-canvas buttons. */
export const OrbitTuningPlugin: Plugin = {
    name: "OrbitTuning",
    systems: [TuningSystem],
    warm() {
        cleanup?.();
        cleanup = null;
        last = "";
    },
    dispose() {
        cleanup?.();
        cleanup = null;
        last = "";
    },
};
