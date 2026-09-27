import { DARK } from "@dylanebert/shallot/brand";

const MUTED = "#a08c78";
const LINE = "#2a2420";

export interface HostFrame {
    frame: HTMLElement;
    canvas: HTMLCanvasElement;
    reveal(): void;
    fail(error: unknown): void;
}

export function mountHost(): HostFrame {
    const root = document.documentElement;
    root.style.setProperty("--bg", DARK.bg);
    root.style.setProperty("--ink", DARK.ink);
    root.style.setProperty("--gold", DARK.gold);
    root.style.setProperty("--muted", MUTED);
    root.style.setProperty("--line", LINE);

    const app = document.querySelector<HTMLElement>("#app")!;
    app.innerHTML = `
        <main class="page">
            <p class="description">lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
            <section class="scene" aria-label="Embedded Shallot scene">
                <div class="frame" id="frame">
                    <canvas id="scene" aria-label="Rendered Shallot scene"></canvas>
                </div>
            </section>
            <section class="section" id="details">
                <h2>aliquam erat</h2>
                <p>lorem ipsum dolor sit amet, <a id="section-link" href="#details">consectetur</a> adipiscing elit.</p>
            </section>
        </main>`;

    const frame = app.querySelector<HTMLElement>("#frame")!;
    const canvas = app.querySelector<HTMLCanvasElement>("#scene")!;

    return {
        frame,
        canvas,
        reveal() {
            canvas.classList.add("visible");
        },
        fail(error) {
            console.error(error);
        },
    };
}
