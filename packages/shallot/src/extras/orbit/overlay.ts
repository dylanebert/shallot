import { mountOverlay, type Plugin, type State, type System } from "../../engine";
import { Inputs } from "../../standard/input";
import { Orbit } from "./index";
import { OrbitSmooth } from "./smooth";

// palette + font match the profile HUD (extras/profile) so the two overlays read as one toolset
const BG = "rgba(14,13,12,0.88)";
const FG = "#f0ece8";
const ACCENT = "#d49560";
const BORDER = "rgba(255,255,255,0.06)";
const FONT = "'JetBrains Mono', ui-monospace, 'Cascadia Code', 'Fira Code', monospace";

// seconds the readout lingers after the last speed change before fading out
const HoldSeconds = 1;

interface Overlay {
    set(speed: number, boost: number, shift: boolean, visible: boolean): void;
    destroy(): void;
}

function createOverlay(canvas: HTMLElement | null): Overlay {
    // the readout lives in the engine's sandboxed overlay (canvas-bounded, can't spill into an
    // embedding host page), the same surface `config.ui` hands an app
    const parent = mountOverlay(canvas);
    const root = document.createElement("div");
    Object.assign(root.style, {
        position: "absolute",
        // lower-third / title-safe band, horizontally centered: near the gaze for transient feedback,
        // inside the safe margin so it clears the very bottom where game UIs (hotbars, action bars) live
        left: "50%",
        bottom: "12%",
        transform: "translateX(-50%)",
        zIndex: "9999",
        pointerEvents: "none",
        opacity: "0",
        transition: "opacity 0.25s ease",
        background: BG,
        color: FG,
        fontFamily: FONT,
        fontSize: "11px",
        padding: "5px 9px",
        borderRadius: "4px",
        border: `1px solid ${BORDER}`,
        fontVariantNumeric: "tabular-nums",
    });
    root.setAttribute("data-orbit-overlay", "");

    const speedEl = document.createElement("span");
    const boostEl = document.createElement("span");
    Object.assign(boostEl.style, { color: ACCENT, marginLeft: "6px" });
    root.append(speedEl, boostEl);
    parent.append(root);

    return {
        set(speed, boost, shift, visible) {
            // keep the text current even while fading, so the last value reads as it dims out
            speedEl.textContent = `fly ${speed.toFixed(1)} u/s`;
            speedEl.style.color = shift ? ACCENT : FG;
            boostEl.textContent = shift ? `×${boost}` : "";
            root.style.opacity = visible ? "1" : "0";
        },
        destroy() {
            parent.remove(); // removes the sandboxed host (root lives inside it)
        },
    };
}

// module-scoped so it survives a rebuild (the State is reused; the DOM node should be too). _lastSpeed
// tracks the previous frame's speed to detect a scroll change; -1 means "not flying / uninitialized", so
// entering fly doesn't flash the readout. _shownUntil is the elapsed time the fade-out begins.
let _overlay: Overlay | null = null;
let _lastSpeed = -1;
let _shownUntil = 0;

const OrbitOverlaySystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    last: true,
    update(state: State) {
        if (typeof document === "undefined") return;

        // first flying camera in query order owns the readout — one shared HUD, like the profile overlay
        let flying = 0;
        for (const eid of state.query([Orbit, OrbitSmooth])) {
            if (OrbitSmooth.flyActive.get(eid) === 1) {
                flying = eid;
                break;
            }
        }
        if (!flying) {
            _lastSpeed = -1;
            _overlay?.set(0, 0, false, false);
            return;
        }

        const speed = Orbit.flySpeed.get(flying);
        const shift = Inputs.isKeyDown("ShiftLeft") || Inputs.isKeyDown("ShiftRight");
        const elapsed = state.time.elapsed;
        if (_lastSpeed < 0)
            _lastSpeed = speed; // just started flying — arm without showing
        else if (speed !== _lastSpeed) {
            _shownUntil = elapsed + HoldSeconds;
            _lastSpeed = speed;
        }
        // visible while a scroll change is fresh, or while shift is actively boosting
        const visible = shift || elapsed < _shownUntil;

        if (!_overlay) _overlay = createOverlay(document.querySelector("canvas"));
        _overlay.set(speed, Orbit.flyBoost.get(flying), shift, visible);
    },
};

/** opt-in HUD that flashes the fly speed while you adjust it (scroll or shift); add alongside {@link OrbitPlugin} */
export const OrbitOverlayPlugin: Plugin = {
    name: "OrbitOverlay",
    systems: [OrbitOverlaySystem],
    dispose() {
        _overlay?.destroy();
        _overlay = null;
        _lastSpeed = -1;
        _shownUntil = 0;
    },
};
