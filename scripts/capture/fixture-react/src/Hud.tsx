import { createRoot } from "react-dom/client";

// The HUD proves the project's React toolchain works in the editor: @vitejs/plugin-react (the project's
// own framework plugin) transforms this JSX, and react + react-dom render it over the canvas. Static label,
// not a `define` — vite's dev server doesn't text-replace `define` in client modules (it applies at build
// time), so the flow reads this literal.
export function mountHud(target: HTMLElement): void {
    createRoot(target).render(
        <div
            className="game-hud"
            data-framework="react"
            style={{
                position: "fixed",
                top: "8px",
                left: "8px",
                zIndex: 9999,
                font: "12px monospace",
                color: "#fff",
                pointerEvents: "none",
            }}
        >
            react
        </div>,
    );
}
