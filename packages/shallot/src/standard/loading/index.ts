import { version } from "../../../package.json";
import { UnsupportedError } from "../../engine";
import type { Loading } from "../../engine/app";

interface Theme {
    bg: string;
    surface: string;
    track: string;
    bar: string;
    text: string;
    muted: string;
    amber: string;
    red: string;
}

const dark: Theme = {
    bg: "#1a1816",
    surface: "#252220",
    track: "#252220",
    bar: "#d49560",
    text: "#e8e0d8",
    muted: "#8a7d70",
    amber: "#e8a86b",
    red: "#c4574b",
};

const light: Theme = {
    bg: "#f5f5f5",
    surface: "#e8e0d8",
    track: "#ddd",
    bar: "#B87654",
    text: "#3D2415",
    muted: "#6B4230",
    amber: "#a55c2c",
    red: "#9b3528",
};

const SUPPORT_URL = "https://caniuse.com/webgpu";

const LOGO_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 285 80">
  <defs>
    <radialGradient id="baseGradient" cx="35%" cy="30%" r="70%" fx="25%" fy="20%">
      <stop offset="0%" stop-color="#F5D4B8"/>
      <stop offset="45%" stop-color="#E8A86B"/>
      <stop offset="100%" stop-color="#B87654"/>
    </radialGradient>
  </defs>
  <g id="Icon" transform="rotate(35 40 40)">
    <path id="Background" d="M40,2 C44,10 66,28 66,46 C66,60 48,70 40,78 C32,70 14,60 14,46 C14,28 36,10 40,2 Z" fill="#E8A86B"/>
    <path id="CloveLeft" d="M40,6 C37,14 22,28 20,44 C20,52 28,62 36,70 C34,58 26,46 26,38 C26,26 38,12 40,6 Z" fill="#D49560"/>
    <path id="CloveRight" d="M40,6 C43,14 58,28 60,44 C60,52 52,62 44,70 C46,58 54,46 54,38 C54,26 42,12 40,6 Z" fill="#D49560"/>
    <path id="CenterCrease" d="M40,8 C40,20 40,50 40,72" stroke="#6B4230" stroke-width="1" stroke-opacity="0.4" fill="none" stroke-linecap="round"/>
    <path id="BottomEdge" d="M40,78 C48,70 66,60 66,46 C61,58 44,70 40,73 Z" fill="#D49560"/>
    <path id="Outline" d="M40,2 C44,10 66,28 66,46 C66,60 48,70 40,78 C32,70 14,60 14,46 C14,28 36,10 40,2 Z" fill="none" stroke="#6B4230" stroke-width="2"/>
  </g>
  <g id="Text" transform="translate(80 59)">
    <path d="M13.37 0.73Q10.88 0.73 8.47 0.07Q6.06 -0.58 4.02 -1.75Q1.97 -2.93 0.52 -4.52L5.54 -9.63Q6.96 -8.09 8.87 -7.26Q10.79 -6.44 13.05 -6.44Q14.62 -6.44 15.44 -6.89Q16.27 -7.34 16.27 -8.18Q16.27 -9.22 15.27 -9.77Q14.27 -10.32 12.7 -10.74Q11.14 -11.17 9.4 -11.7Q7.66 -12.24 6.08 -13.17Q4.5 -14.09 3.51 -15.73Q2.52 -17.37 2.52 -19.95Q2.52 -22.65 3.92 -24.66Q5.31 -26.68 7.86 -27.83Q10.41 -28.97 13.86 -28.97Q17.43 -28.97 20.49 -27.74Q23.55 -26.51 25.46 -24.04L20.42 -18.94Q19.08 -20.5 17.43 -21.16Q15.78 -21.81 14.18 -21.81Q12.67 -21.81 11.93 -21.36Q11.19 -20.91 11.19 -20.13Q11.19 -19.23 12.18 -18.7Q13.17 -18.18 14.73 -17.78Q16.3 -17.37 18.02 -16.81Q19.75 -16.24 21.32 -15.24Q22.88 -14.24 23.87 -12.59Q24.85 -10.93 24.85 -8.29Q24.85 -4.15 21.75 -1.71Q18.65 0.73 13.37 0.73Z M48.31 0V-16.04Q48.31 -18.27 46.95 -19.62Q45.59 -20.97 43.48 -20.97Q42.04 -20.97 40.93 -20.36Q39.82 -19.75 39.19 -18.63Q38.57 -17.52 38.57 -16.04L35.12 -17.72Q35.12 -21.05 36.53 -23.53Q37.93 -26.01 40.42 -27.39Q42.91 -28.77 46.15 -28.77Q49.45 -28.77 51.94 -27.39Q54.43 -26.01 55.81 -23.61Q57.19 -21.2 57.19 -18.04V0ZM29.7 0V-42.11H38.57V0Z M74.65 0.58Q70.76 0.58 67.7 -1.33Q64.64 -3.25 62.89 -6.55Q61.13 -9.86 61.13 -14.07Q61.13 -18.3 62.89 -21.62Q64.64 -24.94 67.7 -26.85Q70.76 -28.77 74.65 -28.77Q77.49 -28.77 79.78 -27.67Q82.07 -26.56 83.51 -24.62Q84.94 -22.68 85.14 -20.18V-8Q84.94 -5.51 83.52 -3.57Q82.1 -1.62 79.79 -0.52Q77.49 0.58 74.65 0.58ZM76.44 -7.42Q79.29 -7.42 81.03 -9.29Q82.77 -11.17 82.77 -14.09Q82.77 -16.07 81.98 -17.56Q81.2 -19.05 79.78 -19.91Q78.36 -20.76 76.47 -20.76Q74.62 -20.76 73.2 -19.91Q71.78 -19.05 70.95 -17.55Q70.12 -16.04 70.12 -14.09Q70.12 -12.15 70.93 -10.64Q71.75 -9.13 73.18 -8.28Q74.62 -7.42 76.44 -7.42ZM82.39 0V-7.57L83.72 -14.44L82.39 -21.26V-28.19H91.12V0Z M97.38 0V-42.11H106.26V0Z M112.52 0V-42.11H121.39V0Z M141.23 0.64Q136.85 0.64 133.36 -1.31Q129.86 -3.25 127.83 -6.61Q125.8 -9.98 125.8 -14.15Q125.8 -18.33 127.82 -21.63Q129.83 -24.94 133.33 -26.88Q136.82 -28.83 141.2 -28.83Q145.61 -28.83 149.09 -26.9Q152.57 -24.97 154.6 -21.65Q156.63 -18.33 156.63 -14.15Q156.63 -9.98 154.61 -6.61Q152.6 -3.25 149.12 -1.31Q145.64 0.64 141.23 0.64ZM141.2 -7.42Q143.12 -7.42 144.56 -8.27Q146.02 -9.11 146.81 -10.63Q147.61 -12.15 147.61 -14.12Q147.61 -16.1 146.78 -17.59Q145.96 -19.08 144.54 -19.92Q143.12 -20.76 141.2 -20.76Q139.34 -20.76 137.9 -19.91Q136.45 -19.05 135.63 -17.56Q134.82 -16.07 134.82 -14.09Q134.82 -12.15 135.63 -10.63Q136.45 -9.11 137.9 -8.27Q139.34 -7.42 141.2 -7.42Z M165.07 0V-39.85H173.94V0ZM158.69 -20.65V-28.19H180.32V-20.65Z" fill="#3D2415" transform="translate(2.5 3)"/>
    <path d="M13.37 0.73Q10.88 0.73 8.47 0.07Q6.06 -0.58 4.02 -1.75Q1.97 -2.93 0.52 -4.52L5.54 -9.63Q6.96 -8.09 8.87 -7.26Q10.79 -6.44 13.05 -6.44Q14.62 -6.44 15.44 -6.89Q16.27 -7.34 16.27 -8.18Q16.27 -9.22 15.27 -9.77Q14.27 -10.32 12.7 -10.74Q11.14 -11.17 9.4 -11.7Q7.66 -12.24 6.08 -13.17Q4.5 -14.09 3.51 -15.73Q2.52 -17.37 2.52 -19.95Q2.52 -22.65 3.92 -24.66Q5.31 -26.68 7.86 -27.83Q10.41 -28.97 13.86 -28.97Q17.43 -28.97 20.49 -27.74Q23.55 -26.51 25.46 -24.04L20.42 -18.94Q19.08 -20.5 17.43 -21.16Q15.78 -21.81 14.18 -21.81Q12.67 -21.81 11.93 -21.36Q11.19 -20.91 11.19 -20.13Q11.19 -19.23 12.18 -18.7Q13.17 -18.18 14.73 -17.78Q16.3 -17.37 18.02 -16.81Q19.75 -16.24 21.32 -15.24Q22.88 -14.24 23.87 -12.59Q24.85 -10.93 24.85 -8.29Q24.85 -4.15 21.75 -1.71Q18.65 0.73 13.37 0.73Z M48.31 0V-16.04Q48.31 -18.27 46.95 -19.62Q45.59 -20.97 43.48 -20.97Q42.04 -20.97 40.93 -20.36Q39.82 -19.75 39.19 -18.63Q38.57 -17.52 38.57 -16.04L35.12 -17.72Q35.12 -21.05 36.53 -23.53Q37.93 -26.01 40.42 -27.39Q42.91 -28.77 46.15 -28.77Q49.45 -28.77 51.94 -27.39Q54.43 -26.01 55.81 -23.61Q57.19 -21.2 57.19 -18.04V0ZM29.7 0V-42.11H38.57V0Z M74.65 0.58Q70.76 0.58 67.7 -1.33Q64.64 -3.25 62.89 -6.55Q61.13 -9.86 61.13 -14.07Q61.13 -18.3 62.89 -21.62Q64.64 -24.94 67.7 -26.85Q70.76 -28.77 74.65 -28.77Q77.49 -28.77 79.78 -27.67Q82.07 -26.56 83.51 -24.62Q84.94 -22.68 85.14 -20.18V-8Q84.94 -5.51 83.52 -3.57Q82.1 -1.62 79.79 -0.52Q77.49 0.58 74.65 0.58ZM76.44 -7.42Q79.29 -7.42 81.03 -9.29Q82.77 -11.17 82.77 -14.09Q82.77 -16.07 81.98 -17.56Q81.2 -19.05 79.78 -19.91Q78.36 -20.76 76.47 -20.76Q74.62 -20.76 73.2 -19.91Q71.78 -19.05 70.95 -17.55Q70.12 -16.04 70.12 -14.09Q70.12 -12.15 70.93 -10.64Q71.75 -9.13 73.18 -8.28Q74.62 -7.42 76.44 -7.42ZM82.39 0V-7.57L83.72 -14.44L82.39 -21.26V-28.19H91.12V0Z M97.38 0V-42.11H106.26V0Z M112.52 0V-42.11H121.39V0Z M141.23 0.64Q136.85 0.64 133.36 -1.31Q129.86 -3.25 127.83 -6.61Q125.8 -9.98 125.8 -14.15Q125.8 -18.33 127.82 -21.63Q129.83 -24.94 133.33 -26.88Q136.82 -28.83 141.2 -28.83Q145.61 -28.83 149.09 -26.9Q152.57 -24.97 154.6 -21.65Q156.63 -18.33 156.63 -14.15Q156.63 -9.98 154.61 -6.61Q152.6 -3.25 149.12 -1.31Q145.64 0.64 141.23 0.64ZM141.2 -7.42Q143.12 -7.42 144.56 -8.27Q146.02 -9.11 146.81 -10.63Q147.61 -12.15 147.61 -14.12Q147.61 -16.1 146.78 -17.59Q145.96 -19.08 144.54 -19.92Q143.12 -20.76 141.2 -20.76Q139.34 -20.76 137.9 -19.91Q136.45 -19.05 135.63 -17.56Q134.82 -16.07 134.82 -14.09Q134.82 -12.15 135.63 -10.63Q136.45 -9.11 137.9 -8.27Q139.34 -7.42 141.2 -7.42Z M165.07 0V-39.85H173.94V0ZM158.69 -20.65V-28.19H180.32V-20.65Z" fill="none" stroke="#6B4230" stroke-width="3.5" stroke-linejoin="round"/>
    <path d="M13.37 0.73Q10.88 0.73 8.47 0.07Q6.06 -0.58 4.02 -1.75Q1.97 -2.93 0.52 -4.52L5.54 -9.63Q6.96 -8.09 8.87 -7.26Q10.79 -6.44 13.05 -6.44Q14.62 -6.44 15.44 -6.89Q16.27 -7.34 16.27 -8.18Q16.27 -9.22 15.27 -9.77Q14.27 -10.32 12.7 -10.74Q11.14 -11.17 9.4 -11.7Q7.66 -12.24 6.08 -13.17Q4.5 -14.09 3.51 -15.73Q2.52 -17.37 2.52 -19.95Q2.52 -22.65 3.92 -24.66Q5.31 -26.68 7.86 -27.83Q10.41 -28.97 13.86 -28.97Q17.43 -28.97 20.49 -27.74Q23.55 -26.51 25.46 -24.04L20.42 -18.94Q19.08 -20.5 17.43 -21.16Q15.78 -21.81 14.18 -21.81Q12.67 -21.81 11.93 -21.36Q11.19 -20.91 11.19 -20.13Q11.19 -19.23 12.18 -18.7Q13.17 -18.18 14.73 -17.78Q16.3 -17.37 18.02 -16.81Q19.75 -16.24 21.32 -15.24Q22.88 -14.24 23.87 -12.59Q24.85 -10.93 24.85 -8.29Q24.85 -4.15 21.75 -1.71Q18.65 0.73 13.37 0.73Z M48.31 0V-16.04Q48.31 -18.27 46.95 -19.62Q45.59 -20.97 43.48 -20.97Q42.04 -20.97 40.93 -20.36Q39.82 -19.75 39.19 -18.63Q38.57 -17.52 38.57 -16.04L35.12 -17.72Q35.12 -21.05 36.53 -23.53Q37.93 -26.01 40.42 -27.39Q42.91 -28.77 46.15 -28.77Q49.45 -28.77 51.94 -27.39Q54.43 -26.01 55.81 -23.61Q57.19 -21.2 57.19 -18.04V0ZM29.7 0V-42.11H38.57V0Z M74.65 0.58Q70.76 0.58 67.7 -1.33Q64.64 -3.25 62.89 -6.55Q61.13 -9.86 61.13 -14.07Q61.13 -18.3 62.89 -21.62Q64.64 -24.94 67.7 -26.85Q70.76 -28.77 74.65 -28.77Q77.49 -28.77 79.78 -27.67Q82.07 -26.56 83.51 -24.62Q84.94 -22.68 85.14 -20.18V-8Q84.94 -5.51 83.52 -3.57Q82.1 -1.62 79.79 -0.52Q77.49 0.58 74.65 0.58ZM76.44 -7.42Q79.29 -7.42 81.03 -9.29Q82.77 -11.17 82.77 -14.09Q82.77 -16.07 81.98 -17.56Q81.2 -19.05 79.78 -19.91Q78.36 -20.76 76.47 -20.76Q74.62 -20.76 73.2 -19.91Q71.78 -19.05 70.95 -17.55Q70.12 -16.04 70.12 -14.09Q70.12 -12.15 70.93 -10.64Q71.75 -9.13 73.18 -8.28Q74.62 -7.42 76.44 -7.42ZM82.39 0V-7.57L83.72 -14.44L82.39 -21.26V-28.19H91.12V0Z M97.38 0V-42.11H106.26V0Z M112.52 0V-42.11H121.39V0Z M141.23 0.64Q136.85 0.64 133.36 -1.31Q129.86 -3.25 127.83 -6.61Q125.8 -9.98 125.8 -14.15Q125.8 -18.33 127.82 -21.63Q129.83 -24.94 133.33 -26.88Q136.82 -28.83 141.2 -28.83Q145.61 -28.83 149.09 -26.9Q152.57 -24.97 154.6 -21.65Q156.63 -18.33 156.63 -14.15Q156.63 -9.98 154.61 -6.61Q152.6 -3.25 149.12 -1.31Q145.64 0.64 141.23 0.64ZM141.2 -7.42Q143.12 -7.42 144.56 -8.27Q146.02 -9.11 146.81 -10.63Q147.61 -12.15 147.61 -14.12Q147.61 -16.1 146.78 -17.59Q145.96 -19.08 144.54 -19.92Q143.12 -20.76 141.2 -20.76Q139.34 -20.76 137.9 -19.91Q136.45 -19.05 135.63 -17.56Q134.82 -16.07 134.82 -14.09Q134.82 -12.15 135.63 -10.63Q136.45 -9.11 137.9 -8.27Q139.34 -7.42 141.2 -7.42Z M165.07 0V-39.85H173.94V0ZM158.69 -20.65V-28.19H180.32V-20.65Z" fill="#E8A86B"/>
  </g>
</svg>`;

const WARNING_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" width="32" height="32" aria-hidden="true">
  <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
  <line x1="12" y1="9" x2="12" y2="13"/>
  <line x1="12" y1="17" x2="12.01" y2="17"/>
</svg>`;

const ERROR_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" width="32" height="32" aria-hidden="true">
  <circle cx="12" cy="12" r="10"/>
  <line x1="15" y1="9" x2="9" y2="15"/>
  <line x1="9" y1="9" x2="15" y2="15"/>
</svg>`;

const ARROW_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true">
  <line x1="7" y1="17" x2="17" y2="7"/>
  <polyline points="7 7 17 7 17 17"/>
</svg>`;

function createOverlay(bg: string, container?: HTMLElement): HTMLDivElement | null {
    if (typeof document === "undefined") return null;

    const overlay = document.createElement("div");

    const parent = container ?? document.querySelector("canvas")?.parentElement ?? document.body;

    // a scroll container, not a fixed center: the panel centers itself with `margin: auto`, which
    // collapses to the scroll start when content is taller than the viewport (small phones) instead
    // of clipping the top — `justify-content: center` would make the overflow unreachable.
    //
    // `inset: 0` fills the parent, but a full-page parent is `100vh` — the *large* viewport, which on
    // mobile spans behind the dynamic URL bar. A card centered there has its lower edge hidden behind
    // the browser chrome and, fitting within `100vh`, never becomes scrollable. `max-height: 100dvh`
    // caps the overlay to the *visible* viewport, so centering and scroll both stay on-screen.
    overlay.style.cssText = `
        position: absolute;
        inset: 0;
        max-height: 100dvh;
        display: flex;
        overflow: auto;
        background: ${bg};
        z-index: 10000;
    `;

    if (getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
    }
    parent.appendChild(overlay);

    return overlay;
}

// the single centered column every overlay state fills. `margin: auto` centers it in both axes when
// it fits and resolves to the scroll start when it doesn't (see createOverlay). `align` is `center`
// for the loading logo/bar, `stretch` for the left-aligned error cards.
function panel(maxWidth: number, align: string): HTMLDivElement {
    const el = document.createElement("div");
    el.style.cssText = `
        margin: auto;
        width: 100%;
        max-width: ${maxWidth}px;
        display: flex;
        flex-direction: column;
        align-items: ${align};
        gap: 16px;
        padding: 32px 24px;
        box-sizing: border-box;
    `;
    return el;
}

function createProgressBar(theme: Theme): { track: HTMLDivElement; bar: HTMLDivElement } {
    const track = document.createElement("div");
    track.style.cssText = `
        width: 228px;
        max-width: 100%;
        height: 4px;
        background: ${theme.track};
        overflow: hidden;
    `;

    const bar = document.createElement("div");
    bar.style.cssText = `
        width: 0%;
        height: 100%;
        background: ${theme.bar};
        transition: width 0.15s ease-out;
    `;
    track.appendChild(bar);

    return { track, bar };
}

function fontStack(): string {
    return "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
}

function monoStack(): string {
    return "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
}

function diagnosticText(error: Error): string {
    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
    const stack = error.stack ?? `${error.name}: ${error.message}`;
    return `shallot v${version}\n${stack}\n\nUser agent: ${ua}`;
}

function createLink(label: string, href: string, accent: string): HTMLAnchorElement {
    const a = document.createElement("a");
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.style.cssText = `
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: ${accent};
        font: 500 13px/1 ${fontStack()};
        text-decoration: none;
        padding: 8px 12px;
        border: 1px solid ${accent}55;
        border-radius: 4px;
        transition: background 0.12s ease, border-color 0.12s ease;
    `;
    a.innerHTML = `<span>${label}</span>${ARROW_ICON}`;
    a.addEventListener("mouseenter", () => {
        a.style.background = `${accent}14`;
        a.style.borderColor = `${accent}aa`;
    });
    a.addEventListener("mouseleave", () => {
        a.style.background = "transparent";
        a.style.borderColor = `${accent}55`;
    });
    return a;
}

function createButton(label: string, accent: string): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.style.cssText = `
        appearance: none;
        background: transparent;
        color: ${accent};
        font: 500 13px/1 ${fontStack()};
        padding: 8px 12px;
        border: 1px solid ${accent}55;
        border-radius: 4px;
        cursor: pointer;
        transition: background 0.12s ease, border-color 0.12s ease;
    `;
    b.addEventListener("mouseenter", () => {
        b.style.background = `${accent}14`;
        b.style.borderColor = `${accent}aa`;
    });
    b.addEventListener("mouseleave", () => {
        b.style.background = "transparent";
        b.style.borderColor = `${accent}55`;
    });
    return b;
}

function renderUnsupported(overlay: HTMLDivElement, error: UnsupportedError, theme: Theme): void {
    overlay.replaceChildren();

    const card = panel(440, "stretch");

    const head = document.createElement("div");
    head.style.cssText = `display: flex; align-items: center; gap: 12px; color: ${theme.amber};`;
    head.innerHTML = `${WARNING_ICON}<div style="font: 600 18px/1.2 ${fontStack()};">Unsupported configuration</div>`;
    card.appendChild(head);

    const body = document.createElement("p");
    body.style.cssText = `margin: 0; color: ${theme.text}; font: 14px/1.5 ${fontStack()};`;
    body.textContent = error.message;
    card.appendChild(body);

    if (error.missing.length > 0) {
        const list = document.createElement("div");
        list.style.cssText = `display: flex; flex-wrap: wrap; gap: 6px;`;
        for (const feat of error.missing) {
            const pill = document.createElement("code");
            pill.textContent = feat;
            pill.style.cssText = `
                background: ${theme.surface};
                color: ${theme.amber};
                font: 12px/1 ${monoStack()};
                padding: 6px 8px;
                border-radius: 3px;
            `;
            list.appendChild(pill);
        }
        card.appendChild(list);
    }

    const hint = document.createElement("p");
    hint.style.cssText = `margin: 0; color: ${theme.muted}; font: 13px/1.5 ${fontStack()};`;
    hint.textContent = "Use a recent Chromium browser (Chrome, Edge, Brave) on desktop or Android.";
    card.appendChild(hint);

    const actions = document.createElement("div");
    actions.style.cssText = `display: flex; gap: 8px;`;
    actions.appendChild(createLink("Browser support", SUPPORT_URL, theme.amber));
    card.appendChild(actions);

    overlay.appendChild(card);
}

function renderEngineError(overlay: HTMLDivElement, error: Error, theme: Theme): void {
    overlay.replaceChildren();

    const card = panel(520, "stretch");

    const head = document.createElement("div");
    head.style.cssText = `display: flex; align-items: center; gap: 12px; color: ${theme.red};`;
    head.innerHTML = `${ERROR_ICON}<div style="font: 600 18px/1.2 ${fontStack()};">Something went wrong</div>`;
    card.appendChild(head);

    const body = document.createElement("p");
    body.style.cssText = `margin: 0; color: ${theme.text}; font: 14px/1.5 ${fontStack()};`;
    body.textContent =
        "An error occurred during startup. The details below can help identify the cause.";
    card.appendChild(body);

    const detail = document.createElement("pre");
    detail.style.cssText = `
        margin: 0;
        background: ${theme.surface};
        color: ${theme.text};
        font: 12px/1.5 ${monoStack()};
        padding: 12px;
        border-radius: 4px;
        max-height: 200px;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
    `;
    detail.textContent = `${error.name}: ${error.message}`;
    card.appendChild(detail);

    const actions = document.createElement("div");
    actions.style.cssText = `display: flex; gap: 8px; margin-top: 4px; flex-wrap: wrap;`;

    const copy = createButton("Copy details", theme.red);
    let resetTimer: ReturnType<typeof setTimeout> | null = null;
    copy.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(diagnosticText(error));
            copy.textContent = "Copied";
            if (resetTimer) clearTimeout(resetTimer);
            resetTimer = setTimeout(() => {
                copy.textContent = "Copy details";
            }, 1500);
        } catch {
            copy.textContent = "Copy failed";
        }
    });
    actions.appendChild(copy);
    card.appendChild(actions);

    overlay.appendChild(card);
}

function renderError(overlay: HTMLDivElement, error: unknown, theme: Theme): void {
    if (error instanceof UnsupportedError) {
        renderUnsupported(overlay, error, theme);
        return;
    }
    const wrapped = error instanceof Error ? error : new Error(String(error));
    renderEngineError(overlay, wrapped, theme);
}

function shallotLoading(theme: Theme, container?: HTMLElement): Loading {
    let overlay: HTMLDivElement | null = null;
    let bar: HTMLDivElement | null = null;

    return {
        show() {
            overlay = createOverlay(theme.bg, container);
            if (!overlay) return;

            const content = panel(276, "center");

            const logo = document.createElement("div");
            logo.innerHTML = LOGO_SVG;
            logo.style.cssText = "width: 228px; max-width: 100%; height: auto;";
            content.appendChild(logo);

            const progressBar = createProgressBar(theme);
            bar = progressBar.bar;
            content.appendChild(progressBar.track);
            overlay.appendChild(content);

            return () => {
                overlay?.remove();
                overlay = null;
                bar = null;
            };
        },

        update(progress) {
            if (bar) bar.style.width = `${progress * 100}%`;
        },

        error(error) {
            if (overlay) renderError(overlay, error, theme);
        },
    };
}

function minimalLoading(theme: Theme, container?: HTMLElement): Loading {
    let overlay: HTMLDivElement | null = null;
    let bar: HTMLDivElement | null = null;

    return {
        show() {
            overlay = createOverlay(theme.bg, container);
            if (!overlay) return;

            const content = panel(276, "center");
            const progressBar = createProgressBar(theme);
            bar = progressBar.bar;
            content.appendChild(progressBar.track);
            overlay.appendChild(content);

            return () => {
                overlay?.remove();
                overlay = null;
                bar = null;
            };
        },

        update(progress) {
            if (bar) bar.style.width = `${progress * 100}%`;
        },

        error(error) {
            if (overlay) renderError(overlay, error, theme);
        },
    };
}

/** dark-theme startup screen: the Shallot logo over a progress bar. the engine default. */
export const shallotDark = (container?: HTMLElement): Loading => shallotLoading(dark, container);
/** light-theme startup screen: the Shallot logo over a progress bar */
export const shallotLight = (container?: HTMLElement): Loading => shallotLoading(light, container);
/** dark-theme startup screen: a bare progress bar, no logo */
export const minimalDark = (container?: HTMLElement): Loading => minimalLoading(dark, container);
/** light-theme startup screen: a bare progress bar, no logo */
export const minimalLight = (container?: HTMLElement): Loading => minimalLoading(light, container);
