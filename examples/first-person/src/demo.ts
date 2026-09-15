import {
    Body,
    Character,
    CharacterPlugin,
    CharacterSweepSystem,
    PhysicsPlugin,
    type Plugin,
    type State,
    type System,
    setKinematic,
} from "@dylanebert/shallot";

// The built-in Player keeps its default WASD, look, and jump controls. These two Character values make the
// ascent's step rhythm and lift transfer feel deliberate without replacing the controller.
function tune(state: State): void {
    for (const eid of state.query([Character])) {
        Character.jumpSpeed.set(eid, 7);
        Character.gravity.set(eid, -30);
    }
}

// The scene owns the lift's size and starting height. This role only gives the small trajectory system a
// declarative target; the lift is the sole moving object in the recipe.
export const Lift = {};

const TRAVEL = 1.5;
const RATE = 0.65;

const lift: System = {
    name: "lift",
    group: "fixed",
    before: [CharacterSweepSystem],
    update(state: State): void {
        const phase = state.time.elapsed * RATE;
        for (const eid of state.query([Lift, Body])) {
            const y = Body.pos.y.get(eid) + Math.sin(phase) * TRAVEL;
            const vy = Math.cos(phase) * RATE * TRAVEL;
            setKinematic(
                state,
                eid,
                [Body.pos.x.get(eid), y, Body.pos.z.get(eid)],
                [0, 0, 0, 1],
                false,
                [0, vy, 0],
            );
        }
    },
};

export const Demo = {
    name: "Demo",
    components: { Lift },
    dependencies: [CharacterPlugin, PhysicsPlugin],
    warm: tune,
    systems: [lift],
} satisfies Plugin;

export default Demo;
