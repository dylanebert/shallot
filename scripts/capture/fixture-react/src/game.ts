import { type Plugin, type System, sparse, u32 } from "@dylanebert/shallot";
import { vec3 } from "wgpu-matrix";
import { aliasValue } from "$game/values";
import { mountHud } from "./Hud";

// The framework-merge capture fixture (scripts/capture/flows/framework-react.pw.ts): a standard vite+react
// project the editor opens through its OWN toolchain. Everything here resolves only because the host merged
// the project's vite.config — `alias` (resolve.alias $game → lib), `dep` (a bare project dep, wgpu-matrix),
// and the .tsx HUD (transformed by the project's own @vitejs/plugin-react, rendered by react + react-dom).
// The fields read back through /__api/entities; the HUD renders `define` as its text in the editor.
export const Game = { alias: sparse(u32), dep: sparse(u32) };

// |(2,3,6)| = 7 — computed BY the project dep, so a nonzero `dep` proves wgpu-matrix resolved, not a
// constant we inlined. BUMP is edited by the flow's HMR leg to prove the merged dev server recompiles the
// plugin; like the ticker fixture, the value is written by a system (a swap doesn't re-run warm).
const depValue = Math.round(vec3.length(vec3.fromValues(2, 3, 6)));
const BUMP = 0;

const GameSystem: System = {
    name: "game",
    group: "simulation",
    annotations: { mode: "always" },
    update: (state) => {
        for (const eid of state.query([Game])) {
            Game.dep.set(eid, depValue + BUMP);
        }
    },
};

export const GamePlugin: Plugin = {
    name: "game",
    components: { Game },
    systems: [GameSystem],
    warm: (state) => {
        const eid = state.create();
        state.add(eid, Game);
        Game.alias.set(eid, aliasValue);
        // idempotent: the editor re-runs warm on a State rebuild, and a DOM mount isn't tracked by State
        // dispose — so clear any prior host first, leaving exactly one HUD however many times warm runs.
        for (const n of document.querySelectorAll("[data-game-host]")) n.remove();
        const host = document.createElement("div");
        host.setAttribute("data-game-host", "");
        document.body.appendChild(host);
        mountHud(host);
    },
};

export default GamePlugin;
