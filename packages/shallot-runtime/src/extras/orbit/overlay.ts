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

function createOverlay(canvas: HTMLElement | null, state: State): Overlay {
    // the readout lives in the engine's sandboxed overlay (canvas-bounded, can't spill into an
    // embedding host page), the same surface `config.ui` hands an app. Passing `state` ties the
    // overlay's removal to `state.onDispose` (auto-registers `overlay.remove()`), so a direct
    // `state.dispose()` — which never fires the plugin `dispose` hook — still cleans up the DOM node.
    // The module-scope cleanup cleared at top-of-warm (below) is the fallback for a host that re-warms
    // without disposing (`swap()`), which `onDispose` doesn't fire for (collapse exemplar's shape).
    const parent = mountOverlay(canvas, state);
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
            // only update the text while visible, so the last value reads as it dims out — the
            // not-flying path fades the overlay without rewriting to "fly 0.0 u/s"
            if (visible) {
                speedEl.textContent = `fly ${speed.toFixed(1)} u/s`;
                speedEl.style.color = shift ? ACCENT : FG;
                boostEl.textContent = shift ? `×${boost}` : "";
            }
            root.style.opacity = visible ? "1" : "0";
        },
        destroy() {
            parent.remove(); // removes the sandboxed host (root lives inside it)
        },
    };
}

// module-scoped runtime state, keyed to the canvas/State it was built against. _lastSpeed tracks the
// previous frame's speed to detect a scroll change; -1 means "not flying / uninitialized", so entering
// fly doesn't flash the readout. _shownUntil is the elapsed time the fade-out begins. Both reset on a
// rebuild (warm), so a fresh State can't inherit a stale visible window or a stale speed sentinel.
let _overlay: Overlay | null = null;
let _overlayCanvas: HTMLElement | null = null;
let _lastSpeed = -1;
let _shownUntil = 0;

const OrbitOverlaySystem: System = {
    group: "draw",
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
            _overlay?.set(0, 0, false, false); // fade out, keeping the last text
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

        const canvas = document.querySelector("canvas");
        // a rebuild against a new canvas leaves the old overlay parented to the old canvas's host —
        // detect the canvas change and tear down the stale overlay so update re-creates it on the new one
        if (_overlay && _overlayCanvas !== canvas) {
            _overlay.destroy();
            _overlay = null;
            _overlayCanvas = null;
            _lastSpeed = -1;
            _shownUntil = 0;
        }
        if (!_overlay) {
            _overlay = createOverlay(canvas, state);
            _overlayCanvas = canvas;
        }
        _overlay.set(speed, Orbit.flyBoost.get(flying), shift, visible);
    },
};

/** opt-in HUD that flashes the fly speed while you adjust it (scroll or shift); add alongside {@link OrbitPlugin} */
export const OrbitOverlayPlugin: Plugin = {
    name: "OrbitOverlay",
    systems: [OrbitOverlaySystem],
    // swap fallback: a host that re-warms without disposing never fires onDispose, so clear the
    // module-scope overlay here before update re-creates it (collapse's panelCleanup-at-top-of-warm)
    warm() {
        _overlay?.destroy();
        _overlay = null;
        _overlayCanvas = null;
        _lastSpeed = -1;
        _shownUntil = 0;
    },
    dispose() {
        _overlay?.destroy();
        _overlay = null;
        _overlayCanvas = null;
        _lastSpeed = -1;
        _shownUntil = 0;
    },
};
