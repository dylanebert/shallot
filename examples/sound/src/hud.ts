import type { System, World } from "@dylanebert/shallot";
import { Hint } from "./sound";

export function hud(container: HTMLElement, world: World): () => void {
    const root = document.createElement("div");
    root.className = "interact-hud";
    root.innerHTML = `<span class="reticle"><span class="reticle-dot" hidden></span></span><div class="hint" hidden><span class="key-slot"><kbd>E</kbd></span><span class="action"></span></div>`;
    container.append(root);
    const hint = root.querySelector<HTMLElement>(".hint")!;
    const dot = root.querySelector<HTMLElement>(".reticle-dot")!;
    const action = root.querySelector<HTMLElement>(".action")!;
    const read: System = {
        name: "Hud/Read",
        group: "draw",
        update(world) {
            const state = world.resource(Hint);
            hint.hidden = !state.target;
            dot.hidden = !state.target;
            action.textContent = state.action;
        },
    };
    world.addSystem(read);
    return () => {
        world.removeSystem(read);
        root.remove();
    };
}
