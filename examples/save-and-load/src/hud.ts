import type { System, World } from "@dylanebert/shallot";
import { count, Game, loadGame, newGame, saveGame } from "./game";

export function hud(container: HTMLElement, world: World): () => void {
    const root = document.createElement("section");
    root.className = "save-hud";
    root.setAttribute("aria-label", "Game controls");
    root.innerHTML = `<h1><output aria-label="Coins">0 coins</output></h1><div class="controls"><button>New game <kbd>N</kbd></button><button>Save <kbd>S</kbd></button><button>Load <kbd>L</kbd></button></div><span role="status" aria-label="Save feedback"></span>`;
    const buttons = root.querySelectorAll("button");
    buttons[0]!.onclick = () => newGame(world);
    buttons[1]!.onclick = () => saveGame(world);
    buttons[2]!.onclick = () => loadGame(world);
    const coins = root.querySelector("output")!;
    const feedback = root.querySelector("[role=status]")!;
    const read: System = {
        name: "Hud/Read",
        group: "draw",
        update(world) {
            const game = world.resource(Game);
            coins.textContent = `${count(world)} coins`;
            buttons[2]!.disabled = !game.saved;
            feedback.textContent = game.feedback > 0 ? "Saved" : "";
        },
    };
    container.append(root);
    world.addSystem(read);
    read.update!(world);
    return () => {
        world.removeSystem(read);
        root.remove();
    };
}
