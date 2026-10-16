import type { System, World } from "@dylanebert/shallot";
import { Hint } from "./interact";

export function hud(container: HTMLElement, world: World): () => void {
    const root = document.createElement("div");
    root.className = "interact-hud";
    root.innerHTML = `<span class="reticle"><span class="reticle-dot" hidden></span></span><div class="hint" hidden><span class="key-slot"><kbd>E</kbd><svg class="hold-ring" viewBox="0 0 48 48"><circle class="track" cx="24" cy="24" r="22"/><circle class="progress" cx="24" cy="24" r="22" pathLength="1"/></svg></span><span class="action"></span></div>`;
    container.append(root);
    const hint = root.querySelector<HTMLElement>(".hint")!;
    const dot = root.querySelector<HTMLElement>(".reticle-dot")!;
    const ring = root.querySelector<SVGElement>(".hold-ring")!;
    const progress = root.querySelector<SVGElement>(".progress")!;
    const action = root.querySelector<HTMLElement>(".action")!;
    const read: System = {
        name: "Hud/Read",
        group: "draw",
        update(world) {
            const state = world.resource(Hint);
            hint.hidden = !state.target;
            dot.hidden = !state.target;
            action.textContent = state.action;
            ring.style.visibility = state.hold ? "visible" : "hidden";
            progress.style.strokeDasharray = `${state.progress} 1`;
        },
    };
    world.addSystem(read);
    return () => {
        world.removeSystem(read);
        root.remove();
    };
}
