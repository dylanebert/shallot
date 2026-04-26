import { mount, unmount } from "svelte";
import type { State } from "@dylanebert/shallot";
import Panel from "./Panel.svelte";
import { RotateCube } from "./lib";
import { Part } from "@dylanebert/shallot";

export function svelteOverlay(container: HTMLElement, state: State): () => void {
    const app = mount(Panel, {
        target: container,
        props: {
            onspeedx(value: number) {
                const eid = state.only([RotateCube]);
                if (eid >= 0) RotateCube.speedX[eid] = value;
            },
            onspeedy(value: number) {
                const eid = state.only([RotateCube]);
                if (eid >= 0) RotateCube.speedY[eid] = value;
            },
            oncolor(hex: string) {
                const eid = state.only([RotateCube]);
                if (eid >= 0) Part.color[eid] = parseInt(hex.slice(1), 16);
            },
        },
    });

    return () => unmount(app);
}
