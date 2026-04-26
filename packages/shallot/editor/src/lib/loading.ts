import type { Loading } from "@dylanebert/shallot";

export function editorLoading(container: HTMLElement): Loading {
    let overlay: HTMLDivElement | null = null;
    let spinner: SVGCircleElement | null = null;

    const Size = 32;
    const Stroke = 2.5;
    const Radius = (Size - Stroke) / 2;
    const Circumference = 2 * Math.PI * Radius;

    return {
        show() {
            overlay = document.createElement("div");
            overlay.style.cssText = `
                position: absolute;
                inset: 0;
                background: rgb(14, 13, 12);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: editorLoadFadeIn 0.1s ease;
            `;

            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("width", String(Size));
            svg.setAttribute("height", String(Size));
            svg.setAttribute("viewBox", `0 0 ${Size} ${Size}`);
            svg.style.cssText = "animation: editorLoadSpin 1s linear infinite;";

            const track = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            track.setAttribute("cx", String(Size / 2));
            track.setAttribute("cy", String(Size / 2));
            track.setAttribute("r", String(Radius));
            track.setAttribute("fill", "none");
            track.setAttribute("stroke", "#252220");
            track.setAttribute("stroke-width", String(Stroke));
            svg.appendChild(track);

            const arc = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            arc.setAttribute("cx", String(Size / 2));
            arc.setAttribute("cy", String(Size / 2));
            arc.setAttribute("r", String(Radius));
            arc.setAttribute("fill", "none");
            arc.setAttribute("stroke", "#d49560");
            arc.setAttribute("stroke-width", String(Stroke));
            arc.setAttribute("stroke-linecap", "round");
            arc.setAttribute("stroke-dasharray", String(Circumference));
            arc.setAttribute("stroke-dashoffset", String(Circumference));
            arc.style.cssText = "transition: stroke-dashoffset 0.2s ease-out;";
            arc.setAttribute("transform", `rotate(-90 ${Size / 2} ${Size / 2})`);
            svg.appendChild(arc);
            spinner = arc;

            overlay.appendChild(svg);

            if (!document.getElementById("editor-load-style")) {
                const style = document.createElement("style");
                style.id = "editor-load-style";
                style.textContent = `
                    @keyframes editorLoadFadeIn {
                        from { opacity: 0; }
                        to { opacity: 1; }
                    }
                    @keyframes editorLoadSpin {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                `;
                document.head.appendChild(style);
            }

            if (getComputedStyle(container).position === "static") {
                container.style.position = "relative";
            }
            container.appendChild(overlay);

            return () => {
                if (overlay) {
                    overlay.style.opacity = "0";
                    overlay.style.transition = "opacity 0.08s ease";
                    setTimeout(() => {
                        overlay?.remove();
                        overlay = null;
                        spinner = null;
                    }, 80);
                }
            };
        },

        update(progress) {
            if (spinner) {
                const offset = Circumference * (1 - progress);
                spinner.setAttribute("stroke-dashoffset", String(offset));
            }
        },
    };
}
