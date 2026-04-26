import {
    Transform,
    rotate,
    traits,
    minimalLight,
    type State,
    type System,
    type Plugin,
    type Config,
} from "@dylanebert/shallot";
import { OrbitPlugin, record, download } from "@dylanebert/shallot/extras";

export const OrbitParent = { speed: [] as number[] };
traits(OrbitParent, { defaults: () => ({ speed: 30 }) });

export const SelfSpin = { speed: [] as number[] };
traits(SelfSpin, { defaults: () => ({ speed: 90 }) });

const OrbitParentSystem: System = {
    group: "simulation",
    update(state: State) {
        const dt = state.time.deltaTime;
        for (const eid of state.query([OrbitParent, Transform])) {
            const q = rotate(
                Transform.quatX[eid],
                Transform.quatY[eid],
                Transform.quatZ[eid],
                Transform.quatW[eid],
                0,
                OrbitParent.speed[eid] * dt,
                0,
            );
            Transform.quatX[eid] = q.x;
            Transform.quatY[eid] = q.y;
            Transform.quatZ[eid] = q.z;
            Transform.quatW[eid] = q.w;
        }
    },
};

const SelfSpinSystem: System = {
    group: "simulation",
    update(state: State) {
        const dt = state.time.deltaTime;
        for (const eid of state.query([SelfSpin, Transform])) {
            const q = rotate(
                Transform.quatX[eid],
                Transform.quatY[eid],
                Transform.quatZ[eid],
                Transform.quatW[eid],
                0,
                SelfSpin.speed[eid] * dt,
                0,
            );
            Transform.quatX[eid] = q.x;
            Transform.quatY[eid] = q.y;
            Transform.quatZ[eid] = q.z;
            Transform.quatW[eid] = q.w;
        }
    },
};

export const HierarchyPlugin: Plugin = {
    name: "Hierarchy",
    systems: [OrbitParentSystem, SelfSpinSystem],
    components: { OrbitParent, SelfSpin },
};

const R = 11;
const CIRC = 2 * Math.PI * R;

function recordUI(container: HTMLElement, state: State): () => void {
    const style = document.createElement("style");
    style.textContent = `
        @keyframes rec-pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(0.85); }
        }
        .rec-btn {
            position: absolute; bottom: 14px; right: 14px;
            display: flex; align-items: center; justify-content: center;
            width: 32px; height: 32px; padding: 0;
            background: #fff; border: 1px solid #e0e0e0;
            border-radius: 50%; cursor: pointer;
            pointer-events: auto; outline: none;
            font: 500 12px/1 system-ui, sans-serif;
            color: #444; letter-spacing: 0.01em;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
        }
        .rec-btn:hover {
            border-color: #ccc;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .rec-btn:active { background: #fafafa; }
        .rec-btn:disabled { cursor: default; }
        .rec-btn:disabled:hover {
            border-color: #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .rec-dot { transition: r 0.25s cubic-bezier(0.34, 1.4, 0.64, 1); }
        .rec-btn.recording .rec-dot {
            animation: rec-pulse 1.4s ease-in-out infinite;
        }
        .rec-ring {
            transform: rotate(-90deg); transform-origin: center;
            transition: opacity 0.25s;
        }
    `;
    container.appendChild(style);

    const btn = document.createElement("button");
    btn.className = "rec-btn";
    btn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24">
            <circle class="rec-ring" cx="12" cy="12" r="${R}"
                fill="none" stroke="#dc2626" stroke-width="2"
                stroke-dasharray="${CIRC}" stroke-dashoffset="${CIRC}"
                stroke-linecap="round" opacity="0"/>
            <circle class="rec-dot" cx="12" cy="12" r="5" fill="#dc2626"/>
        </svg>`;
    container.appendChild(btn);

    const ring = btn.querySelector(".rec-ring") as SVGCircleElement;
    const dot = btn.querySelector(".rec-dot") as SVGCircleElement;

    btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.classList.add("recording");
        dot.setAttribute("r", "3.5");
        ring.style.opacity = "1";

        const blob = await record(state, 5, undefined, (p) => {
            ring.style.strokeDashoffset = String(CIRC * (1 - p));
        });
        download(blob, "hierarchy.mp4");

        btn.classList.remove("recording");
        ring.style.opacity = "0";
        ring.style.strokeDashoffset = String(CIRC);
        dot.setAttribute("r", "5");
        btn.disabled = false;
    });

    return () => {};
}

export const config: Config = {
    plugins: [OrbitPlugin, HierarchyPlugin],
    scene: "/hierarchy/scenes/hierarchy.scene",
    loading: minimalLight(),
    ui: recordUI,
};
