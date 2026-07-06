// The HUD — a center crosshair that morphs with the gun mode (cross / ring on hover / dot while
// holding) and bottom-right mouse prompts that surface contextually. Ported from the legacy sandbox,
// desktop-only (F3 stats is engine-standard via ProfilePlugin, no custom debug plumbing).

import type { GunMode } from "./gun";

let crosshairEl: HTMLElement | null = null;
let grabPrompt: HTMLElement | null = null;
let launchPrompt: HTMLElement | null = null;
let dropPrompt: HTMLElement | null = null;

const HUD_CSS = `
.sandbox-prompts {
    position: absolute;
    bottom: 28px;
    right: 28px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    pointer-events: none;
    font-family: 'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace;
}
.sandbox-prompt {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 12px 7px 10px;
    background: rgba(10, 12, 14, 0.5);
    backdrop-filter: blur(12px) saturate(1.2);
    -webkit-backdrop-filter: blur(12px) saturate(1.2);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 3px;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 6px 20px rgba(0, 0, 0, 0.28);
    color: rgba(255, 255, 255, 0.92);
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0;
    transform: translateX(6px);
    transition: opacity 130ms ease, transform 130ms ease;
}
.sandbox-prompt[data-show="1"] {
    opacity: 1;
    transform: translateX(0);
}
.sandbox-prompt__key {
    display: inline-flex;
    width: 14px;
    height: 18px;
    align-items: center;
    justify-content: center;
    color: rgba(255, 255, 255, 0.95);
}
.sandbox-prompt__key svg { display: block; }
.sandbox-prompt__label {
    color: rgba(255, 255, 255, 0.86);
    font-weight: 500;
    line-height: 1;
}
`;

function mouseSvg(side: "left" | "right"): string {
    const fill =
        side === "left"
            ? "M 7 1.8 C 4.4 1.8 2.2 3.5 2.2 6 V 7.4 H 7 Z"
            : "M 7 1.8 C 9.6 1.8 11.8 3.5 11.8 6 V 7.4 H 7 Z";
    return `<svg viewBox="0 0 14 18" width="14" height="18">
        <path d="M 7 1 C 4 1 1.5 3 1.5 6 V 12 C 1.5 15.5 4 17 7 17 C 10 17 12.5 15.5 12.5 12 V 6 C 12.5 3 10 1 7 1 Z M 7 1 V 8 M 1.5 8 H 12.5"
              fill="none" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/>
        <path d="${fill}" fill="currentColor"/>
    </svg>`;
}

function makePrompt(side: "left" | "right", label: string): HTMLElement {
    const el = document.createElement("div");
    el.className = "sandbox-prompt";
    el.dataset.show = "0";
    el.innerHTML = `<span class="sandbox-prompt__key">${mouseSvg(side)}</span><span class="sandbox-prompt__label">${label}</span>`;
    return el;
}

let lastMode: GunMode | null = null;

export function setCrosshair(mode: GunMode): void {
    if (mode === lastMode) return;
    lastMode = mode;
    if (crosshairEl) {
        const circles = crosshairEl.querySelectorAll("circle");
        const ring = circles[0];
        const dot = circles[1];
        const lines = Array.from(crosshairEl.querySelectorAll("line"));
        if (ring && dot) {
            ring.setAttribute("display", mode === "default" ? "none" : "");
            dot.setAttribute("display", mode === "grab" ? "" : "none");
            for (const l of lines) l.setAttribute("display", mode === "default" ? "" : "none");
        }
    }
    if (grabPrompt) grabPrompt.dataset.show = mode === "hover" ? "1" : "0";
    if (launchPrompt) launchPrompt.dataset.show = mode === "grab" ? "1" : "0";
    if (dropPrompt) dropPrompt.dataset.show = mode === "grab" ? "1" : "0";
}

export function hud(container: HTMLElement): () => void {
    let style = document.querySelector<HTMLStyleElement>("style[data-sandbox-hud]");
    if (!style) {
        style = document.createElement("style");
        style.dataset.sandboxHud = "";
        style.textContent = HUD_CSS;
        document.head.appendChild(style);
    }

    const el = document.createElement("div");
    el.style.cssText =
        "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;width:24px;height:24px;";
    el.innerHTML = `<svg viewBox="0 0 24 24" width="24" height="24" stroke="white" stroke-width="2" fill="none" opacity="0.8">
        <line x1="12" y1="4" x2="12" y2="10"/><line x1="12" y1="14" x2="12" y2="20"/>
        <line x1="4" y1="12" x2="10" y2="12"/><line x1="14" y1="12" x2="20" y2="12"/>
        <circle cx="12" cy="12" r="6" display="none"/>
        <circle cx="12" cy="12" r="2" fill="white" stroke="none" display="none"/>
    </svg>`;
    crosshairEl = el;
    container.appendChild(el);

    const prompts = document.createElement("div");
    prompts.className = "sandbox-prompts";
    grabPrompt = makePrompt("left", "Grab");
    launchPrompt = makePrompt("left", "Launch");
    dropPrompt = makePrompt("right", "Drop");
    prompts.append(grabPrompt, launchPrompt, dropPrompt);
    container.appendChild(prompts);

    return () => {
        el.remove();
        prompts.remove();
        crosshairEl = null;
        grabPrompt = null;
        launchPrompt = null;
        dropPrompt = null;
        lastMode = null;
    };
}
