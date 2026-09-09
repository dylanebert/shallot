// Both halves of this import are load-bearing under Node, and bun tolerates neither absence — which is
// why no bun-side gate could see the defect and `scripts/check-node-import.ts` is the arm that does.
// Without the attribute Node ≥26 throws `TypeError: Module "…/package.json" needs an import attribute of
// "type: json"` at load; with it, a *named* import throws `SyntaxError: The requested module
// '../../../package.json' does not provide an export named 'version'`, since a Node JSON module exposes
// its object as the default export only (bun synthesizes named ones). Either way a Node-side consumer of
// this package — a Playwright driver, a vite/playwright config — dies before running.
import pkg from "../../../package.json" with { type: "json" };
import { UnsupportedError } from "../../engine";
import type { Loading } from "../../engine/app";
import {
    DARK,
    HIT_TICK,
    LIGHT,
    type Palette,
    progressTick,
    type Splash,
    splash,
    TICK_MS,
    toSvg,
} from "./mark";

/** Ticks of dead air after the lockup lands, before the overlay is dismissed. */
const HOLD_TICKS = 6;

/** The track fades over the hold, so the lockup is alone on screen when the overlay goes. */
const FADE_MS = Math.round(HOLD_TICKS * TICK_MS);

interface Theme {
    bg: string;
    surface: string;
    track: string;
    bar: string;
    text: string;
    muted: string;
    amber: string;
    red: string;
    /** The mark palette the splash draws its pixels with. */
    mark: Palette;
}

const dark: Theme = {
    bg: DARK.bg,
    surface: "#1c1917",
    track: "#1c1917",
    bar: DARK.gold,
    text: DARK.ink,
    muted: "#a08c78",
    amber: DARK.gold,
    red: "#c4574b",
    mark: DARK,
};

const light: Theme = {
    bg: LIGHT.bg,
    surface: "#efe9df",
    track: "#efe9df",
    bar: LIGHT.gold,
    text: LIGHT.ink,
    muted: "#6e655c",
    amber: LIGHT.gold,
    red: "#9b3528",
    mark: LIGHT,
};

const SUPPORT_URL = "https://caniuse.com/webgpu";

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
// for the loading splash/bar, `stretch` for the left-aligned error cards.
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
        transition: opacity ${FADE_MS}ms ease-out;
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

// named, never fetched: the overlay runs offline and inside native windows, so the brand faces are
// asked for and the platform stack carries the surface when they are absent.
function fontStack(): string {
    return "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
}

function monoStack(): string {
    return "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
}

function prefersReducedMotion(): boolean {
    return (
        typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches
    );
}

// the splash the site ships, driven by progress rather than a clock: the lockup lands pixel by
// pixel as the build advances, then `complete` plays the name typing in. Reduced motion rests on
// the finished lockup instead. At 4px a square the 52-pixel lockup is 208px wide, just inside the
// 228px track below it, so the two read as one thing settling into a load.
function createSplash(theme: Theme): { el: HTMLDivElement; driver: Splash; reduced: boolean } {
    const el = document.createElement("div");
    el.style.cssText = "width: 208px; max-width: 100%;";
    const reduced = prefersReducedMotion();
    const driver = splash(el, (grid) => toSvg(grid, theme.mark, 4), reduced);
    return { el, driver, reduced };
}

/** Resolves `ticks` after the call, on the same clock the splash runs on. */
function hold(ticks: number): Promise<void> {
    if (typeof requestAnimationFrame !== "function") return Promise.resolve();
    const until = performance.now() + ticks * TICK_MS;
    return new Promise<void>((resolve) => {
        const frame = () => {
            if (performance.now() >= until) {
                resolve();
                return;
            }
            requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    });
}

function diagnosticText(error: Error): string {
    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
    const stack = error.stack ?? `${error.name}: ${error.message}`;
    return `shallot v${pkg.version}\n${stack}\n\nUser agent: ${ua}`;
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

    // one line, and it is the link: the caniuse table stays current where a browser list would not
    const actions = document.createElement("div");
    actions.style.cssText = `display: flex; gap: 8px;`;
    const what =
        error.missing.length > 0
            ? "Required WebGPU features not supported."
            : "WebGPU not supported.";
    actions.appendChild(createLink(what, SUPPORT_URL, theme.amber));
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

function loading(theme: Theme, container: HTMLElement | undefined, withSplash: boolean): Loading {
    let overlay: HTMLDivElement | null = null;
    let bar: HTMLDivElement | null = null;
    let track: HTMLDivElement | null = null;
    let driver: Splash | null = null;
    let reduced = false;
    let tick = 0;

    const screen: Loading = {
        show() {
            overlay = createOverlay(theme.bg, container);
            if (!overlay) return;

            const content = panel(276, "center");

            if (withSplash) {
                const made = createSplash(theme);
                driver = made.driver;
                reduced = made.reduced;
                tick = 0;
                driver.seek(0);
                content.appendChild(made.el);
            }

            const progressBar = createProgressBar(theme);
            bar = progressBar.bar;
            track = progressBar.track;
            content.appendChild(progressBar.track);
            overlay.appendChild(content);

            return () => {
                overlay?.remove();
                overlay = null;
                bar = null;
                track = null;
                driver = null;
                tick = 0;
            };
        },

        update(progress) {
            if (bar) bar.style.width = `${progress * 100}%`;
            // the landing only ever gains pixels: a lower progress never unlights one
            if (driver) {
                tick = Math.max(tick, progressTick(progress));
                driver.seek(tick);
            }
        },

        error(error) {
            if (overlay) renderError(overlay, error, theme);
        },
    };

    // only the splash variants hold: the bar-only screens dismiss the moment the build finishes
    if (withSplash) {
        screen.complete = async () => {
            if (!driver) return;
            // the bar has said all it can; it clears so the outro plays against the ground alone
            if (track) {
                if (reduced) track.style.transition = "none";
                track.style.opacity = "0";
            }
            await driver.play(HIT_TICK);
            if (!reduced) await hold(HOLD_TICKS);
        };
    }

    return screen;
}

function shallotLoading(theme: Theme, container?: HTMLElement): Loading {
    return loading(theme, container, true);
}

function minimalLoading(theme: Theme, container?: HTMLElement): Loading {
    return loading(theme, container, false);
}

/** dark-theme startup screen: the shallot splash over a progress bar. the engine default. */
export const shallotDark = (container?: HTMLElement): Loading => shallotLoading(dark, container);
/** light-theme startup screen: the shallot splash over a progress bar */
export const shallotLight = (container?: HTMLElement): Loading => shallotLoading(light, container);
/** dark-theme startup screen: a bare progress bar, no splash */
export const minimalDark = (container?: HTMLElement): Loading => minimalLoading(dark, container);
/** light-theme startup screen: a bare progress bar, no splash */
export const minimalLight = (container?: HTMLElement): Loading => minimalLoading(light, container);
