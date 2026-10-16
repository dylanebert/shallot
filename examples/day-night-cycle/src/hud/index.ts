import type { World } from "@dylanebert/shallot";
import { mount, unmount } from "svelte";
import Hud from "./Clock.svelte";

/** Mounts the HUD into the app's overlay; `runApp`'s `ui`. */
export function hud(container: HTMLElement, world: World): () => void {
    const component = mount(Hud, { target: container, props: { world } });
    return () => void unmount(component);
}
