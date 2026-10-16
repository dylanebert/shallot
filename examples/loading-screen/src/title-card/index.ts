import type { Loading } from "@dylanebert/shallot";
import { completeIntro } from "./dwell";

export function titleCard(bootstrap: HTMLElement): Loading {
    let card = bootstrap;
    let shownAt = 0;
    let intro: Promise<unknown> = Promise.resolve();

    return {
        show() {
            // Reuse the painted card; only its ownership changes at import time.
            card.removeAttribute("data-shallot-bootstrap");
            card.dataset.titleCard = "";
            shownAt = performance.now();
            if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
                intro = card
                    .querySelector<HTMLElement>("h1")!
                    .animate([{ opacity: 0 }, { opacity: 1 }], {
                        duration: 1100,
                        easing: "ease-out",
                        fill: "both",
                    }).finished;
            }
            return () => card.remove();
        },
        update(progress) {
            const value = Math.min(1, Math.max(0, progress));
            card.querySelector<HTMLElement>(".progress")!.style.width = `${value * 100}%`;
            card.dataset.progress = String(value);
        },

        complete() {
            return completeIntro(shownAt, intro);
        },
        notice(verdict) {
            card.querySelector<HTMLElement>(".notice")!.textContent =
                `${verdict.class} adapter: ${verdict.identity}`;
        },
        error(error) {
            if (!card.isConnected) {
                card = bootstrap.cloneNode(true) as HTMLElement;
                document.body.append(card);
            }
            card.dataset.state = "error";
            card.setAttribute("role", "alert");
            const panel = card.querySelector<HTMLElement>(".error")!;
            panel.hidden = false;
            const details = error instanceof Error ? error.message : String(error);
            const remedy = details.includes("No WebGPU adapter") ? "WebGPU not supported.\n" : "";
            panel.textContent = `Something went wrong\nAn error occurred in the app. The details below can help identify the cause.\n${details}\n${remedy}Reloading the page may resolve this issue.`;
        },
    };
}
