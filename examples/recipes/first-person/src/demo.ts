import {
    Character,
    Physics,
    Player,
    type Plugin,
    type State,
    type System,
} from "@dylanebert/shallot";

// move/look tuning lives on `Player`; walk physics (jump height, gravity, walkable slope) lives on
// `Character`. Set both once on load.
function tune(state: State) {
    for (const eid of state.query([Player])) {
        Player.speed.set(eid, 7); // walk speed, m/s
        Player.sensitivity.set(eid, 1.5); // mouse look, radians per 1080 px of motion
    }
    for (const eid of state.query([Character])) {
        Character.jumpSpeed.set(eid, 7); // jump launch speed
        Character.gravity.set(eid, -30); // per-character gravity, snappier than the world's
    }
}

// a `mass: 0` body is kinematic: the solver never moves it, you do. Drive its pose each fixed tick with
// `setKinematic`, and a character standing on it rides along; the scene tags it `moving` so this system
// finds it.
const Moving = {};

const slide: System = {
    name: "slide",
    group: "fixed",
    update(state: State) {
        const backend = Physics.backend;
        if (!backend) return;
        const x = -4 + Math.sin(state.time.elapsed) * 3;
        for (const eid of state.query([Moving])) {
            backend.setKinematic(eid, [x, 0.75, 0], [0, 0, 0, 1]);
        }
    },
};

export const Demo = {
    name: "Demo",
    components: { Moving },
    warm: tune,
    systems: [slide],
} satisfies Plugin;

export default Demo;
