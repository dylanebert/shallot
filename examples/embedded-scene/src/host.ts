export interface HostFrame {
    frame: HTMLElement;
    canvas: HTMLCanvasElement;
    reveal(): void;
    fail(error: unknown): void;
}

export function mountHost(): HostFrame {
    const app = document.querySelector<HTMLElement>("#app")!;
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
            const loadingOverlay = Array.from(frame.children).some(
                (child) => (child as HTMLElement).style.zIndex === "10000",
            );
            if (loadingOverlay) return;

            let line = frame.querySelector<HTMLElement>(".frame-error");
            if (!line) {
                line = document.createElement("p");
                line.className = "frame-error";
                line.setAttribute("role", "alert");
                frame.appendChild(line);
            }
            const detail =
                error instanceof Error ? `${error.name}: ${error.message}` : String(error);
            line.textContent = `Something went wrong: ${detail}`;
        },
    };
}
