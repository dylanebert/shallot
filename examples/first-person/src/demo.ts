import {
    Body,
    Character,
    CharacterPlugin,
    CharacterSweepSystem,
    InputPlugin,
    mountOverlay,
    PhysicsPlugin,
    type Plugin,
    pointerLockRefusal,
    pointerLockStatus,
    type State,
    type System,
    setKinematic,
} from "@dylanebert/shallot";

// The built-in Player keeps its default WASD, look, and jump controls. These two Character values make the
// ascent's step rhythm and lift transfer feel deliberate without replacing the controller.
function tune(state: State): void {
    for (const eid of state.query([Character])) {
        if (state.identity.id(eid) !== "player") continue;
        Character.jumpSpeed.set(eid, 7);
        Character.gravity.set(eid, -30);
    }
}

// The scene owns the lift's size and starting height. This role only gives the small trajectory system a
// declarative target; the lift is the sole moving object in the recipe.
export const Lift = {};

const TRAVEL = 1.5;
const RATE = 0.65;
const liftBases = new WeakMap<State, Map<number, readonly [number, number, number]>>();

const lift: System = {
    name: "lift",
    group: "fixed",
    before: [CharacterSweepSystem],
    update(state: State): void {
        const bases = liftBases.get(state);
        if (!bases) return;
        const phase = state.time.elapsed * RATE;
        const offset = 0.5 * TRAVEL * (1 - Math.cos(2 * phase));
        const vy = RATE * TRAVEL * Math.sin(2 * phase);
        for (const eid of state.query([Lift, Body])) {
            const base =
                bases.get(eid) ??
                ([Body.pos.x.get(eid), Body.pos.y.get(eid), Body.pos.z.get(eid)] as const);
            bases.set(eid, base);
            setKinematic(state, eid, [base[0], base[1] + offset, base[2]], [0, 0, 0, 1], false, [
                0,
                vy,
                0,
            ]);
        }
    },
};

const controlPanels = new WeakMap<State, { panel: HTMLDivElement; status: HTMLDivElement }>();

function mountControls(state: State): void {
    if (typeof document === "undefined" || controlPanels.has(state)) return;
    const overlay = mountOverlay(document.querySelector("canvas"), state);
    const panel = document.createElement("div");
    panel.dataset.recipeControls = "";
    panel.style.cssText =
        "position:absolute;top:20px;left:20px;pointer-events:none;padding:10px 12px;" +
        "display:grid;row-gap:6px;column-gap:16px;border:1px solid rgba(255,255,255,0.12);" +
        "border-radius:6px;background:rgba(14,17,20,0.72);color:#ffffff;" +
        "font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";
    for (const [control, action] of [
        ["WASD", "Move"],
        ["MOUSE", "Look"],
        ["SPACE", "Jump"],
    ] as const) {
        const row = document.createElement("div");
        row.dataset.controlRow = "";
        row.style.cssText =
            "display:grid;grid-template-columns:max-content max-content;column-gap:16px";
        for (const text of [control, action]) {
            const cell = document.createElement("span");
            cell.textContent = text;
            cell.style.color = "#ffffff";
            row.append(cell);
        }
        panel.append(row);
    }
    const status = document.createElement("div");
    status.dataset.pointerLockStatus = "";
    status.style.cssText = "margin-top:4px;color:#ffffff";
    panel.append(status);
    overlay.append(panel);
    controlPanels.set(state, { panel, status });
    state.onDispose(() => controlPanels.delete(state));
}

const controls: System = {
    name: "first-person-controls",
    group: "draw",
    update(state) {
        mountControls(state);
        const current = controlPanels.get(state);
        if (!current) return;
        const status = pointerLockStatus(state);
        current.status.hidden = status === "locked";
        if (status === "locked") return;
        if (status === "unsupported" || status === "refused") {
            const refusal = pointerLockRefusal(state);
            current.status.textContent = `Mouse look unavailable.${refusal ? ` ${refusal}` : ""}`;
        } else {
            current.status.textContent = "Click the scene to enable mouse look.";
        }
    },
};

export const Demo = {
    name: "Demo",
    components: { Lift },
    dependencies: [CharacterPlugin, InputPlugin, PhysicsPlugin],
    warm(state: State) {
        tune(state);
        liftBases.set(state, new Map());
        state.onDispose(() => liftBases.delete(state));
    },
    systems: [lift, controls],
} satisfies Plugin;

export default Demo;
