import { DARK } from "@dylanebert/shallot/brand";

const MUTED = "#a08c78";
const LINE = "#2a2420";

export interface HostFrame {
    frame: HTMLElement;
    canvas: HTMLCanvasElement;
    status: HTMLElement;
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
            <header class="intro">
                <p class="eyebrow">SHALLOT / FIELD NOTE 01</p>
                <h1>A small world, held in view.</h1>
                <p class="lede">A scene can find its place inside an ordinary page. The space around it stays quiet; the world arrives only when its first frame is ready.</p>
                <p>Curabitur blandit tempus porttitor. Integer posuere erat a ante venenatis dapibus posuere velit aliquet. Maecenas faucibus mollis interdum.</p>
                <p>Donec sed odio dui. Aenean lacinia bibendum nulla sed consectetur. Vestibulum id ligula porta felis euismod semper.</p>
            </header>
            <section class="scene" aria-label="Embedded Shallot scene">
                <div class="frame" id="frame">
                    <canvas id="scene" aria-label="Rendered Shallot scene"></canvas>
                </div>
                <p class="frame-caption"><span>01 / ORBIT STUDY</span><span id="scene-status">Preparing scene</span></p>
            </section>
            <section class="after" aria-label="More about the scene">
                <p>Praesent commodo cursus magna, vel scelerisque nisl consectetur et. Nulla vitae elit libero, a pharetra augue. Aenean eu leo quam.</p>
                <p>Nullam quis risus eget urna mollis ornare vel eu leo. Cras mattis consectetur purus sit amet fermentum.</p>
                <p class="page-note">The rest of the page stays available while the scene prepares. <button id="page-action" type="button">Still usable <span aria-hidden="true">↗</span></button><span id="page-action-result" aria-live="polite"></span></p>
            </section>
            <footer><span>SHALLOT / EMBEDDED SCENE</span><span>FIELD NOTE 01</span></footer>
        </main>`;

    const frame = app.querySelector<HTMLElement>("#frame")!;
    const canvas = app.querySelector<HTMLCanvasElement>("#scene")!;
    const status = app.querySelector<HTMLElement>("#scene-status")!;
    const actionResult = app.querySelector<HTMLElement>("#page-action-result")!;
    app.querySelector<HTMLButtonElement>("#page-action")!.addEventListener("click", () => {
        actionResult.textContent = "Page action works.";
    });

    return {
        frame,
        canvas,
        status,
        reveal() {
            canvas.classList.add("visible");
            status.textContent = "Scene ready";
        },
        fail(error) {
            status.textContent = "Scene could not be prepared";
            console.error(error);
        },
    };
}
