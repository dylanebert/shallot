import {
    Character,
    mountOverlay,
    Physics,
    Player,
    type Plugin,
    pointerLockRefusal,
    pointerLockStatus,
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
        const backend = Physics;
        const x = -4 + Math.sin(state.time.elapsed) * 3;
        for (const eid of state.query([Moving])) {
            backend.setKinematic(eid, [x, 0.75, 0], [0, 0, 0, 1]);
        }
    },
};

// mouse look needs Pointer Lock, and a browser can lack it or refuse the capture. `pointerLockStatus()`
// reports that as data, so say it on screen instead of leaving a view that never turns.
let notice: HTMLElement | null = null;
let overlay: HTMLElement | null = null;

const lockNotice: System = {
    name: "lock-notice",
    group: "simulation",
    setup() {
        notice = null;
        overlay = null; // a rebuilt State mounts its own overlay
    },
    update(state: State) {
        const status = pointerLockStatus();
        const refused = status === "unsupported" || status === "refused";
        if (refused === !!notice) return;
        if (!refused) {
            notice?.remove();
            notice = null;
            return;
        }
        const el = document.createElement("div");
        el.style.cssText =
            "position:absolute;top:16px;left:50%;transform:translateX(-50%);padding:8px 14px;" +
            "border-radius:6px;background:rgba(0,0,0,0.7);color:#fff;font:12px system-ui;";
        el.textContent = `Mouse look unavailable — ${pointerLockRefusal() ?? "pointer lock refused"}. WASD still walks.`;
        overlay ??= mountOverlay(document.querySelector("canvas"), state);
        overlay.appendChild(el);
        notice = el;
    },
};

export const Demo = {
    name: "Demo",
    components: { Moving },
    warm: tune,
    systems: [slide, lockNotice],
} satisfies Plugin;

export default Demo;
