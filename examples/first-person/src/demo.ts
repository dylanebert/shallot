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
const RECIPE_STATE = Symbol.for("shallot.examples.first-person.state");
type DemoBag = {
    // one slot per lift: its body eid, and its authored base at `slot * 3`. Two held arrays rather than a
    // Map, so the per-tick walk indexes instead of iterating and the base reads stay unboxed doubles.
    liftEids: number[];
    liftBases: number[];
    liftCount: number;
    panel: HTMLDivElement | null;
    look: HTMLDivElement | null;
};
type DemoState = State & { [RECIPE_STATE]?: DemoBag };

function stateBag(state: State): DemoBag {
    return (state as DemoState)[RECIPE_STATE] ?? createBag(state);
}

// The bag's creation, apart from the per-frame lookup: its dispose closure would otherwise make every
// lookup allocate a context.
function createBag(state: State): DemoBag {
    const owner = state as DemoState;
    const bag: DemoBag = { liftEids: [], liftBases: [], liftCount: 0, panel: null, look: null };
    owner[RECIPE_STATE] = bag;
    state.onDispose(() => {
        if (owner[RECIPE_STATE] !== bag) return;
        // the slot arrays keep their capacity; the count is what empties them
        bag.liftCount = 0;
        bag.panel = null;
        bag.look = null;
        delete owner[RECIPE_STATE];
    });
    return bag;
}

// The lift's pose and velocity registers, written in place each tick; setKinematic copies them.
const liftPos: [number, number, number] = [0, 0, 0];
const LIFT_QUAT = [0, 0, 0, 1] as const;
const liftVel: [number, number, number] = [0, 0, 0];

const lift: System = {
    name: "lift",
    group: "fixed",
    before: [CharacterSweepSystem],
    // Every lift shares one trajectory, so the phase, the rise and the velocity are the tick's, not each
    // lift's: they are computed once here and the slot walk only adds each lift's base to them.
    update(state: State): void {
        const bag = stateBag(state);
        const phase = 2 * (state.time.elapsed * RATE);
        const rise = 0.5 * TRAVEL * (1 - Math.cos(phase));
        liftVel[1] = RATE * TRAVEL * Math.sin(phase);
        for (let slot = 0; slot < bag.liftCount; slot++) {
            const base = slot * 3;
            liftPos[0] = bag.liftBases[base];
            liftPos[1] = bag.liftBases[base + 1] + rise;
            liftPos[2] = bag.liftBases[base + 2];
            setKinematic(state, bag.liftEids[slot], liftPos, LIFT_QUAT, false, liftVel);
        }
    },
};

function mountControls(state: State): void {
    if (typeof document === "undefined") return;
    const bag = stateBag(state);
    if (bag.panel) return;
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
        if (control === "MOUSE") bag.look = row;
    }
    overlay.append(panel);
    bag.panel = panel;
}

// Mouse look states itself on the control it governs, never as a sentence: the MOUSE row is dim until the
// pointer locks and brightens when it does, so the affordance and its outcome are one mark. A refusal reads
// as a struck row, with the browser's reason kept on the title so the cause stays recoverable without copy.
const controls: System = {
    name: "first-person-controls",
    group: "draw",
    update(state) {
        mountControls(state);
        const bag = stateBag(state);
        if (!bag.panel || !bag.look) return;
        const status = pointerLockStatus(state);
        const look =
            status === "locked" ? "locked" : status === "unlocked" ? "idle" : "unavailable";
        if (bag.look.dataset.pointerLook === look) return;
        bag.look.dataset.pointerLook = look;
        bag.look.style.opacity = look === "locked" ? "1" : "0.45";
        bag.look.style.textDecoration = look === "unavailable" ? "line-through" : "none";
        bag.look.title = look === "unavailable" ? (pointerLockRefusal(state) ?? "") : "";
    },
};

export const Demo = {
    name: "Demo",
    components: { Lift },
    dependencies: [CharacterPlugin, InputPlugin, PhysicsPlugin],
    warm(state: State) {
        tune(state);
        const bag = stateBag(state);
        bag.liftCount = 0;
        for (const eid of state.query([Lift, Body])) {
            const base = bag.liftCount * 3;
            bag.liftEids[bag.liftCount] = eid;
            bag.liftBases[base] = Body.pos.x.get(eid);
            bag.liftBases[base + 1] = Body.pos.y.get(eid);
            bag.liftBases[base + 2] = Body.pos.z.get(eid);
            bag.liftCount++;
        }
    },
    systems: [lift, controls],
} satisfies Plugin;

export default Demo;
