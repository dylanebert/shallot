export interface HostFrame {
    frame: HTMLElement;
    poster: HTMLElement;
    canvas: HTMLCanvasElement;
    status: HTMLElement;
    reveal(): void;
    fail(error: unknown): void;
}

export function mountHost(): HostFrame {
    const app = document.querySelector<HTMLElement>("#app")!;
    app.innerHTML = `
        <main class="page">
            <header><a href="#top" class="wordmark">FIELD / NOTES</a><span>INTERACTIVE STUDY&nbsp; 01</span></header>
            <section class="intro" id="top">
                <p class="eyebrow">A SMALL SCENE, INSIDE A PAGE</p>
                <h1>Make room<br />for the world.</h1>
                <p class="lede">A Shallot scene can live inside an ordinary page. The host keeps its poster in place while the scene prepares, and reveals the canvas only after its first frame is ready.</p>
            </section>
            <section class="demo" aria-label="Embedded Shallot scene">
                <div class="frame" id="frame">
                    <div class="poster" id="poster" aria-label="Host-owned scene poster">
                        <span class="poster-index">SCENE&nbsp; / &nbsp;01</span>
                        <span class="poster-title">Quiet geometry</span>
                        <span class="poster-caption">The scene is being prepared.</span>
                    </div>
                    <canvas id="scene" aria-label="Rendered Shallot scene"></canvas>
                </div>
                <p class="frame-caption"><span>01 / BOX STUDY</span><span id="scene-status">Preparing scene</span></p>
            </section>
            <section class="below">
                <div><p class="eyebrow">THE PAGE GOES ON</p><h2>A frame, not a takeover.</h2></div>
                <div><p>The loading treatment belongs to this frame. The rest of the page remains available while the app builds.</p><button id="page-action" type="button">Still usable <span aria-hidden="true">↗</span></button><span id="page-action-result" aria-live="polite"></span></div>
            </section>
            <footer><span>FIELD / NOTES</span><span>AN EMBEDDED ENGINE STUDY</span></footer>
        </main>`;

    const frame = app.querySelector<HTMLElement>("#frame")!;
    const poster = app.querySelector<HTMLElement>("#poster")!;
    const canvas = app.querySelector<HTMLCanvasElement>("#scene")!;
    const status = app.querySelector<HTMLElement>("#scene-status")!;
    const actionResult = app.querySelector<HTMLElement>("#page-action-result")!;
    app.querySelector<HTMLButtonElement>("#page-action")!.addEventListener("click", () => {
        actionResult.textContent = "Page action works.";
    });

    return {
        frame,
        poster,
        canvas,
        status,
        reveal() {
            poster.hidden = true;
            canvas.classList.add("visible");
            status.textContent = "Scene ready";
        },
        fail(error) {
            status.textContent = "Scene could not be prepared";
            console.error(error);
        },
    };
}
