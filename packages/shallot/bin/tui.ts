#!/usr/bin/env bun
// `shallot tui <dir>` boots a project headless, drives the frame loop, and encodes its cell grid.

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
// type-only — erased at compile time, so this carries no runtime import and no GPU touch (unlike a
// value import, verified already at this file's own module doc: `verify.ts` does the same with
// `GpuDiagnostics`/`ShaderArtifact`).
import type { Plugin } from "../src/engine";
import { requireProject } from "./toolchain";
import type { RGB, Tier, Cell as TuiCell, Grid as TuiGrid } from "./tui/index";
// type-only — erased at compile time, so this carries no runtime import and no encoder touch (unlike a
// value import, verified already at this file's own module doc: `verify.ts` does the same with
// `GpuDiagnostics`/`ShaderArtifact`).
import {
    ALT_SCREEN_ENTER,
    detectTier,
    Encoder,
    installTeardown,
    makeGrid,
    onResize,
    terminalSize,
} from "./tui/index";

const PACKAGE_ROOT = resolve(import.meta.dir, "..");

export const usage = `
  shallot tui [dir] — boot a project headless and render its cell grid to this terminal

  Boots the project the same way \`shallot dev\`/\`shallot build\` do (its shallot.json manifest), minus
  the swapchain compositor (GlazePlugin never runs headless) — the project must enable "Cells": true.
  Drives the frame loop itself (no browser, no rAF) and feeds the cell grid to the
  private encoder every frame. On a real terminal, arrow keys drive any project system
  reading Inputs.isKeyDown the normal way (KeyboardEvent.code) — mouse/pointer input is out of scope
  (specs/shallot-tui.md).

  Options
    --frames <n>   exit automatically after N simulated frames (default: run until Ctrl-C/q or SIGINT/SIGTERM)
    --fps <n>      fixed simulation step rate, frames per second (default: 30)
    --tier <tier>  force the encoder tier: plain, glyph, ansi256, or truecolor (default: auto-detected)
    -h, --help     show this help

  Requires bun-webgpu (optional): bun add -d bun-webgpu
`;

export interface TuiArgs {
    dir: string;
    /** undefined = run until interrupted (SIGINT/SIGTERM/'q'/Ctrl-C on a real tty) */
    frames?: number;
    fps: number;
    tier?: Tier;
    help: boolean;
}

const VALID_TIERS: ReadonlySet<string> = new Set(["plain", "glyph", "ansi256", "truecolor"]);

/** parse `shallot tui` flags. Pure — the CLI wiring and the tests share it, the shape `parseVerifyArgs`
 *  (verify.ts) and `parseCliArgs` (cli.ts) both model. Throws on an unrecognized `-`-prefixed option or
 *  an invalid flag value, same policy as every sibling command's parser. */
export function parseTuiArgs(raw: string[]): TuiArgs {
    const num = (flag: string, v: string): number => {
        if (v.trim() === "") throw new Error(`invalid ${flag} value "${v}" — must not be empty`);
        const n = Number(v);
        if (!Number.isFinite(n) || n <= 0) {
            throw new Error(`invalid ${flag} value "${v}" — expected a positive number`);
        }
        if (!Number.isInteger(n))
            throw new Error(`invalid ${flag} value "${v}" — expected an integer`);
        return n;
    };
    const args: TuiArgs = { dir: ".", fps: 30, help: false };
    let sawDir = false;
    for (let i = 0; i < raw.length; i++) {
        const a = raw[i];
        if (a === "--help" || a === "-h") args.help = true;
        else if (a === "--frames" && raw[i + 1]) args.frames = num("--frames", raw[++i]);
        else if (a?.startsWith("--frames="))
            args.frames = num("--frames", a.slice("--frames=".length));
        else if (a === "--fps" && raw[i + 1]) args.fps = num("--fps", raw[++i]);
        else if (a?.startsWith("--fps=")) args.fps = num("--fps", a.slice("--fps=".length));
        else if (a === "--tier" && raw[i + 1]) args.tier = tier(raw[++i]);
        else if (a?.startsWith("--tier=")) args.tier = tier(a.slice("--tier=".length));
        else if (a?.startsWith("-")) throw new Error(`unknown option: ${a}`);
        else if (!sawDir) {
            args.dir = a;
            sawDir = true;
        } else throw new Error(`unexpected argument: ${a}`);
    }
    return args;

    function tier(v: string): Tier {
        if (!VALID_TIERS.has(v)) {
            throw new Error(
                `invalid --tier value "${v}" — expected plain, glyph, ansi256, or truecolor`,
            );
        }
        return v as Tier;
    }
}

/**
 * Owns two signals that used to be one shared `stopped` boolean, which was the bug: "the frame loop
 * should stop" and "the dispose sequence already ran" are different once-only facts, and collapsing
 * them into one flag meant a quit request (`q`/Ctrl-C, set directly by the stdin bridge's callback)
 * satisfied `disposeAll`'s own idempotency guard before `disposeAll` ever ran once — so the clean-exit
 * path (`teardown(); disposeAll();` after the loop falls out) silently no-op'd: `stdinBridge.stop()`
 * (raw mode off, `stdin.pause()`) never ran, held-key timers were never cleared, `app.dispose()` never
 * ran, leaving the real terminal in raw mode with a resumed stdin holding the event loop open. Regression
 * covered directly against this type in `tui.test.ts`.
 */
export interface QuitGuard {
    /** true once quit has been requested — the frame loop's own stop condition. */
    readonly stopped: boolean;
    /** request the frame loop stop. Idempotent; never affects disposal. */
    requestStop(): void;
    /** run `dispose` exactly once, however many times this is called, independent of {@link stopped}. */
    disposeOnce(dispose: () => void): void;
}

/** Constructs a fresh {@link QuitGuard} — one per `runTui` call, never shared across invocations. */
export function createQuitGuard(): QuitGuard {
    let stopped = false;
    let disposed = false;
    return {
        get stopped() {
            return stopped;
        },
        requestStop() {
            stopped = true;
        },
        disposeOnce(dispose) {
            if (disposed) return;
            disposed = true;
            dispose();
        },
    };
}

/**
 * Builds `runTui`'s own `disposeAll` from an independent {@link QuitGuard} and the actual teardown
 * steps — extracted out of `runTui`'s inline wiring (rather than left as a closure only `runTui` can
 * construct) so this exact composition, not just `QuitGuard`'s own contract in isolation, is unit-
 * testable without booting the engine. The original defect (see the `QuitGuard` docblock above) was a
 * composition defect: the frame loop's stop condition and `disposeAll`'s own idempotency guard read the
 * same shared boolean. `QuitGuard` fixes that at the type level by keeping `stopped` and the dispose
 * guard as two independent facts — but a *future* edit to this composition could still re-collapse them
 * without ever touching `createQuitGuard`, e.g. by reading `quitGuard.stopped` here instead of calling
 * `quitGuard.disposeOnce`. This function calling `disposeOnce` (never `stopped`) is what such an edit
 * would have to change, and it's exercised directly — not via a hand-copied stand-in — by `runTui`
 * itself. Regression covered against this exact function in `tui.test.ts`.
 */
export function buildDisposeAll(quitGuard: QuitGuard, dispose: () => void): () => void {
    return () => quitGuard.disposeOnce(dispose);
}

/**
 * Runs `loop`, then always runs `teardown` and `disposeAll` — including when `loop` throws.
 * Extracted out of `runTui`'s own control flow (rather than a bare `try`/`finally` inline) so this
 * exact composition is unit-testable without booting the engine. Before this wrapper existed,
 * `teardown()`/`disposeAll()` sat after the frame loop as a normal-completion-only step: a throw from
 * inside the loop (`app.state.step`, `cellsGridFor`, `staging.mapAsync`, `encoder.encode`) propagated
 * straight out of `runTui`, skipping `disposeAll()` entirely — `stdinBridge.stop()` never ran, so
 * `setRawMode(false)` was never called and held-key timers were never cleared, the same user-visible
 * symptom (raw mode left on, the process held open) the q/Ctrl-C `createQuitGuard` fix removed, through
 * a different door. `installTeardown`'s own `"exit"` listener only restores the alt screen on process
 * exit — it never calls `disposeAll`'s work (see `screen.ts`'s own docblock: `opts.exit` is never
 * invoked for the `"exit"` event), so it doesn't cover this gap either. The thrown error still
 * propagates once teardown/disposeAll have run — this wrapper changes *when* cleanup happens, not
 * whether the error is reported.
 */
export async function runLoopWithTeardown(
    loop: () => Promise<void>,
    teardown: () => void,
    disposeAll: () => void,
): Promise<void> {
    try {
        await loop();
    } finally {
        teardown();
        disposeAll();
    }
}

// exit codes: distinct so a caller (CI, an agent) can tell a real failure from a missing tool — mirrors
// verify.ts's EXIT_SETUP / EXIT_NO_PLAYWRIGHT convention, its own numbering (this command's exit space,
// not shared with verify's).
export const EXIT_PASS = 0;
export const EXIT_SETUP = 2; // bad flags, no project — never reached the frame loop
export const EXIT_NO_BUN_WEBGPU = 3; // bun-webgpu isn't installed

const INSTALL_BUN_WEBGPU = "bun add -d bun-webgpu";

/** the refusal diagnostic for a missing `bun-webgpu` — names the install command, never a stack trace
 *  (criterion 7's own wording). Mirrors `verify.ts`'s `displayGateMessage` / playwright-remedy shape. */
export function noBunWebgpuMessage(): string {
    return (
        `shallot tui needs bun-webgpu to boot the engine headless, but it isn't installed. ` +
        `Install it: ${INSTALL_BUN_WEBGPU}`
    );
}

/**
 * resolve the optional `bun-webgpu` module, or `null` when it isn't installed — the caller then prints
 * {@link noBunWebgpuMessage} and exits {@link EXIT_NO_BUN_WEBGPU} rather than letting a bare
 * `Cannot find module` stack trace reach the user. `loader` is DI'd (default: the real dynamic import) so
 * the failure path is unit-testable with no filesystem trickery and no real absence required — the
 * two-sided proof criterion 7 asks for: a `loader` that resolves proves the success path reaches the
 * engine, and a `loader` that rejects proves the exit code + message rather than a thrown stack trace
 * (`bin/tui.test.ts`). The genuine-absence arm (a real project with no bun-webgpu installed at all) lives
 * in `scripts/install-test.ts`, mirroring its existing `importPlaywright` sibling check exactly.
 */
export async function importBunWebgpu(
    loader: () => Promise<typeof import("bun-webgpu")> = () => import("bun-webgpu"),
): Promise<typeof import("bun-webgpu") | null> {
    try {
        return await loader();
    } catch {
        return null;
    }
}

// ---------------------------------------------------------------------------------------------------
// headless DOM shim — the minimum surface `InputSystem` (`standard/input/index.ts`) and `attachCanvas`
// (`standard/render/view.ts`) read to bind keyboard + a canvas-shaped render target with no browser.
// `document`/`window` are bare identifiers those modules reference directly (not `globalThis.x`), so
// this shim is installed on `globalThis` — the same trick `dump-cells-ascii.ts` uses for `ResizeObserver`.
// ---------------------------------------------------------------------------------------------------

/** the one KeyboardEvent field `InputSystem`'s handlers read (`e.code`) — Bun has no `KeyboardEvent`
 *  global (a DOM class), so this is the minimal `Event` subclass that carries it. */
class TuiKeyEvent extends Event {
    constructor(
        type: string,
        readonly code: string,
    ) {
        super(type);
    }
}

/**
 * a minimal `HTMLCanvasElement`-shaped object satisfying exactly what `attachCanvas` / `sizeView` /
 * `BeginFrameSystem` (the render path) and `InputSystem` (the keyboard-registration path) read — the
 * same reasoning `dump-cells-ascii.ts`'s own `DumpCanvas` docblock gives for hand-writing this rather than
 * reaching for `bun-webgpu`'s internal mock (no published types, and it configures real
 * `STORAGE_BINDING` usage for Glaze's compute composite, a capability this command never needs since it
 * drops `GlazePlugin` the same way that script does). `addEventListener`/`style` additionally satisfy
 * `InputSystem`'s own per-canvas pointer-listener registration (pointer events are never dispatched here
 * — mouse/pointer input is out of scope, `specs/shallot-tui.md`'s Out of scope — so those listeners just
 * sit unused).
 */
class TuiCanvas {
    width: number;
    height: number;
    readonly style: Record<string, string> = {};
    private _device: GPUDevice | null = null;
    private _format: GPUTextureFormat = "bgra8unorm";
    private _texture: GPUTexture | null = null;
    private readonly _resizeListeners = new Set<() => void>();

    constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
    }

    resize(width: number, height: number): void {
        this.width = width;
        this.height = height;
        for (const listener of this._resizeListeners) listener();
    }

    observeResize(listener: () => void): () => void {
        this._resizeListeners.add(listener);
        return () => this._resizeListeners.delete(listener);
    }

    getContext(kind: string): TuiCanvas | null {
        return kind === "webgpu" ? this : null;
    }

    configure(descriptor: { device: GPUDevice; format: GPUTextureFormat }): void {
        this._device = descriptor.device;
        this._format = descriptor.format;
    }

    getCurrentTexture(): GPUTexture {
        this._texture?.destroy();
        if (!this._device) throw new Error("TuiCanvas: getCurrentTexture before configure");
        this._texture = this._device.createTexture({
            label: "tui-swapchain",
            size: { width: this.width, height: this.height },
            format: this._format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        return this._texture;
    }

    getBoundingClientRect(): DOMRect {
        return {
            width: this.width,
            height: this.height,
            top: 0,
            left: 0,
            right: this.width,
            bottom: this.height,
            x: 0,
            y: 0,
            toJSON: () => ({}),
        } as DOMRect;
    }

    addEventListener(..._args: unknown[]): void {}
    removeEventListener(..._args: unknown[]): void {}
}

/**
 * install the globals `InputSystem.setup` and `attachCanvas`'s render path need — idempotent (a no-op
 * where they already exist), the same guard `dump-cells-ascii.ts` uses for `ResizeObserver`. Returns the
 * one canvas object both paths share, so `attachCanvas` (render) and `InputSystem` (keyboard) bind the
 * same element a real browser would present as one `<canvas>`.
 *
 * **Ordering matters.** `InputSystem.setup` runs during `build()`'s plugin registration, before the
 * caller can call the render path's `attachCanvas` (which needs a live camera entity that doesn't exist
 * until after `build()` returns) — so `document.querySelectorAll("canvas")` must already report this
 * canvas *before* `build()` is called, or `InputSystem` never binds. The render `attachCanvas` call
 * happens later, post-build, against the same object.
 */
function installHeadlessDom(
    width: number,
    height: number,
): { canvas: TuiCanvas; window: EventTarget } {
    const canvas = new TuiCanvas(width, height);
    const g = globalThis as unknown as {
        document?: unknown;
        window?: unknown;
        // biome-ignore lint/style/useNamingConvention: ResizeObserver is the DOM global's own name.
        ResizeObserver?: unknown;
    };
    if (typeof g.document === "undefined") {
        g.document = {
            querySelectorAll: (sel: string) => (sel === "canvas" ? [canvas] : []),
            pointerLockElement: null,
        };
    }
    let win = g.window as EventTarget | undefined;
    if (typeof win === "undefined") {
        win = new EventTarget();
        g.window = win;
    }
    // Feed the render view's normal ResizeObserver seam when terminal dimensions change.
    if (typeof g.ResizeObserver === "undefined") {
        g.ResizeObserver = class {
            private _unsubscribe: (() => void) | null = null;
            constructor(private readonly _callback: () => void) {}
            observe(target: TuiCanvas): void {
                this._unsubscribe = target.observeResize(this._callback);
            }
            disconnect(): void {
                this._unsubscribe?.();
                this._unsubscribe = null;
            }
        };
    }
    return { canvas, window: win };
}

// ---------------------------------------------------------------------------------------------------
// raw-mode stdin → KeyboardEvent.code bridge. A terminal reports key PRESSES only (no release byte), so
// "held" is approximated by OS key-repeat: a code stays "down" while repeats keep arriving inside a short
// quiet window, and releases once they stop — the standard TUI approximation (blessed/ink-style readers
// use the same shape). Arrow keys (CSI cursor sequences) map to the same `KeyboardEvent.code` values a
// browser would report, so any project system reading `Inputs.isKeyDown` (the recipe's own KeyOrbit,
// `examples/recipes/render-to-a-terminal/src/keys.ts`) drives identically off either input source — this
// bridge is general, not special-cased to that one recipe.
// ---------------------------------------------------------------------------------------------------

const ARROW_CODES: Readonly<Record<string, string>> = {
    A: "ArrowUp",
    B: "ArrowDown",
    C: "ArrowRight",
    D: "ArrowLeft",
};

const KEY_HOLD_MS = 150; // quiet window before a code with no repeat reads as released

/** one decoded raw-stdin byte chunk: the `KeyboardEvent.code` values it named (arrow keys only today —
 *  WASD/other printable-key mapping is a natural follow-on, not required by this unit's own criteria),
 *  plus whether it asked to quit (Ctrl-C `\x03`, or `q`/`Q`). Pure — testable with no real tty. */
export function decodeStdinChunk(chunk: string): { codes: string[]; quit: boolean } {
    const codes: string[] = [];
    let quit = false;
    for (let i = 0; i < chunk.length; i++) {
        const c = chunk[i];
        if (c === "\x03") {
            quit = true;
            continue;
        }
        if (c === "q" || c === "Q") {
            quit = true;
            continue;
        }
        if (c === "\x1b" && chunk[i + 1] === "[" && chunk[i + 2] in ARROW_CODES) {
            codes.push(ARROW_CODES[chunk[i + 2]]);
            i += 2;
        }
    }
    return { codes, quit };
}

/** owns the raw-mode stdin listener and its key-repeat-decay timers; `stop()` tears both down. Only
 *  installed on a real tty (`runTui` guards it) — a piped/non-tty stdin has no raw mode to enter and no
 *  keys to read, which is also what keeps the deterministic-into-a-pipe path (criterion 6) free of any
 *  input nondeterminism. */
function installStdinBridge(win: EventTarget, onQuit: () => void): { stop: () => void } {
    const held = new Map<string, ReturnType<typeof setTimeout>>();
    const release = (code: string) => {
        held.delete(code);
        win.dispatchEvent(new TuiKeyEvent("keyup", code));
    };
    const press = (code: string) => {
        const existing = held.get(code);
        if (existing) clearTimeout(existing);
        else win.dispatchEvent(new TuiKeyEvent("keydown", code));
        held.set(
            code,
            setTimeout(() => release(code), KEY_HOLD_MS),
        );
    };
    const onData = (chunk: string) => {
        const { codes, quit } = decodeStdinChunk(chunk);
        for (const code of codes) press(code);
        if (quit) onQuit();
    };
    // only ever installed on a real tty (`runTui`'s own guard), so unconditionally entering/leaving raw
    // mode here is safe — there is no "was it already raw" state to preserve.
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", onData);
    return {
        stop() {
            process.stdin.off("data", onData);
            for (const [code, timer] of held) {
                clearTimeout(timer);
                win.dispatchEvent(new TuiKeyEvent("keyup", code));
            }
            held.clear();
            process.stdin.setRawMode(false);
            process.stdin.pause();
        },
    };
}

// ---------------------------------------------------------------------------------------------------
// cell grid decode — the CPU-side counterpart of `packCell` (`extras/cells/cell.ts`), already sRGB per
// that file's own locked contract, so the encoder reads these bytes directly with no conversion.
// ---------------------------------------------------------------------------------------------------

/** decode a raw cell-grid readback into the `private encoder` `Grid` shape the encoder consumes.
 *  Pure — no GPU, no engine import, no `private encoder` import (only the three functions passed
 *  in), so this is unit-testable against a hand-built buffer with neither optional dependency installed.
 *  `unpack`/`glyphChar`/`makeGrid` are all DI'd rather than imported at module top, keeping this module's
 *  own top-level import-free of the engine and of `private encoder` (see the module doc's N-1
 *  note). */
export function cellsBytesToGrid(
    bytes: ArrayBuffer,
    cols: number,
    rows: number,
    unpack: (bytes: ArrayBuffer, index: number) => { glyph: number; fg: number[]; bg: number[] },
    glyphChar: (glyph: number) => string,
    makeGrid: (cols: number, rows: number, fn: (x: number, y: number) => TuiCell) => TuiGrid,
): TuiGrid {
    const rgb = (bytes4: number[]): RGB => ({ r: bytes4[0], g: bytes4[1], b: bytes4[2] });
    return makeGrid(cols, rows, (x, y) => {
        const cell = unpack(bytes, y * cols + x);
        const out: TuiCell = { glyph: glyphChar(cell.glyph), fg: rgb(cell.fg), bg: rgb(cell.bg) };
        return out;
    });
}

// ---------------------------------------------------------------------------------------------------
// project plugin resolution — the manifest → real Plugin objects a Node process needs, generalizing
// `dump-cells-ascii.ts`'s hand-copied plugin list via `../src/project/generate.ts`'s `plan()` (the same
// classifier the browser build's `virtual:project` generator uses) so this command boots ANY project's
// own shallot.json rather than one fixed scene's plugin set.
// ---------------------------------------------------------------------------------------------------

/** resolve one engine plugin name (`plan()`'s `engine` list) to its real Plugin object: the main barrel
 *  first (every default + most extras), falling back to its declared subpath module (today only `Avbd` —
 *  `SUBPATH_PLUGIN_MODULES`) read off this package's own `exports` map rather than hand-copied, so a
 *  future subpath-only plugin resolves with no edit here. Throws naming the plugin on total failure —
 *  never a silent `undefined` plugin object. */
async function resolveEnginePlugin(
    name: string,
    barrel: Record<string, unknown>,
    subpathModules: Readonly<Record<string, string>>,
    pkgExports: Record<string, unknown>,
): Promise<Plugin> {
    const direct = barrel[`${name}Plugin`];
    if (direct) return direct as Plugin;
    const subpathSpec = subpathModules[name];
    if (subpathSpec) {
        const key = `.${subpathSpec.slice("@dylanebert/shallot".length)}`;
        const target = pkgExports[key];
        if (typeof target === "string") {
            const mod = (await import(resolve(PACKAGE_ROOT, target))) as Record<string, unknown>;
            const plugin = mod[`${name}Plugin`];
            if (plugin) return plugin as Plugin;
        }
    }
    throw new Error(
        `shallot tui: shallot.json enables unknown engine plugin "${name}" (no export named "${name}Plugin")`,
    );
}

/** resolve one local plugin entry (`plan()`'s `locals` list) — a module whose default export is the
 *  Plugin, the same runtime contract `generate.ts`'s emitted browser module enforces (its own docblock:
 *  "A wrong/missing default is silently `undefined`... the runtime guard below is what makes a mistake
 *  loud"). Mirrored here rather than reused, since that guard lives inline in generated module source
 *  text, not as a callable function. */
async function resolveLocalPlugin(name: string, path: string): Promise<Plugin> {
    const mod = (await import(path)) as { default?: Plugin };
    const plugin = mod.default;
    if (!plugin || typeof plugin.name !== "string") {
        throw new Error(
            `shallot tui: shallot.json plugin "${name}": its module must default-export a Plugin`,
        );
    }
    return plugin;
}

// The measured terminal payload ceiling: at 200x50 the worst-case truecolor frame is 182 KiB, or
// 5.3 MB/s at 30 fps. Web has no equivalent cap because it never reads the grid back.
const TERMINAL_MAX_COLS = 200;
const TERMINAL_MAX_ROWS = 50;
const TERMINAL_CELL_DEVICE_PX = 11;

export function terminalGridSize(size: { width: number; height: number }): {
    width: number;
    height: number;
} {
    return {
        width: Math.min(size.width, TERMINAL_MAX_COLS),
        height: Math.min(size.height, TERMINAL_MAX_ROWS),
    };
}

/**
 * `shallot tui [dir]` — the real entry, reached only past  {@link importBunWebgpu}'s success. Returns the process exit code rather than calling `process.exit`
 * itself (mirrors `runVerify`/`runRecipe`; `cli.ts` calls `process.exit(await runTui(...))`), except for
 * `requireProject` — reused verbatim from `toolchain.ts`, and like `dev.ts`/`build.ts` already do, it
 * exits the process directly on a missing project rather than threading a second return convention through
 * this one command.
 *
 * `bunWebgpuLoader` forwards to {@link importBunWebgpu}; tests can inject its absence.
 */
export async function runTui(
    argv: string[],
    bunWebgpuLoader?: () => Promise<typeof import("bun-webgpu")>,
): Promise<number> {
    let args: TuiArgs;
    try {
        args = parseTuiArgs(argv);
    } catch (e) {
        console.error(e instanceof Error ? e.message : String(e));
        return EXIT_SETUP;
    }
    if (args.help) {
        console.log(usage);
        return EXIT_PASS;
    }

    const projectDir = resolve(args.dir);
    requireProject(projectDir); // exits the process on a missing project (toolchain.ts's own contract)

    const bunWebgpu = await importBunWebgpu(bunWebgpuLoader);
    if (!bunWebgpu) {
        console.error(noBunWebgpuMessage());
        return EXIT_NO_BUN_WEBGPU;
    }

    // Past this point the run is committed to the engine — register the TGSL transform (exactly once
    // this process; see the module doc) before any import below reaches a TGSL-bearing module. `Bun.plugin`
    // (the global, not `import { plugin } from "bun"`) — `scripts/wsl-bridge.ts` bundles the whole
    // `cli.ts` graph to a `--target node` node-runnable bundle for its `verify --connect` browser launch,
    // and a static `import("bun")` anywhere in that graph fails that unrelated build ("Browser build
    // cannot import() Bun builtin: 'bun'") even though this line never runs under node — dump-cells-
    // ascii.ts's own top-level `import { plugin } from "bun"` is safe only because that script sits
    // outside the bundled graph entirely.
    const typegpuBunPlugin = (await import("unplugin-typegpu/bun")).default;
    Bun.plugin(typegpuBunPlugin({ include: /\.tsx?$/ }));

    await bunWebgpu.setupGlobals();

    const outputSize = terminalGridSize(
        terminalSize(process.stdout as unknown as Parameters<typeof terminalSize>[0]),
    );
    const { canvas, window: win } = installHeadlessDom(
        outputSize.width * TERMINAL_CELL_DEVICE_PX,
        outputSize.height * TERMINAL_CELL_DEVICE_PX,
    );

    // `cellsGridFor` rides the main barrel (`extras/cells/index.ts`'s author-facing
    // surface); `unpackCell`/`cellGlyphChar` are the `*/core` subpath's decode functions
    // (`dump-cells-ascii.ts`'s own import split, mirrored here).
    const engine = await import("../src");
    const { cellsGridFor } = engine;
    const { attachCanvas } = await import("../src/standard/render/core");
    const { unpackCell, cellGlyphChar } = await import("../src/extras/cells/core");
    const { readManifest } = await import("../src/project/assets");
    const { plan } = await import("../src/project/generate");
    const { SUBPATH_PLUGIN_MODULES } = await import("../src/project/engine");

    const manifest = readManifest(projectDir);
    const { engine: engineNames, locals } = plan(manifest, projectDir);
    // Glaze composites the rendered scene to a swapchain — never applicable headless, the same reason
    // `dump-cells-ascii.ts` bypasses it via `defaults: false` + an explicit list. Dropped unconditionally
    // rather than left to the manifest, since a headless run has no swapchain to disable it against.
    const enabledNames = engineNames.filter((n) => n !== "Glaze");
    if (!enabledNames.includes("Cells")) {
        console.error(
            `shallot tui: ${projectDir}/shallot.json does not enable "Cells" — add "Cells": true to its plugins`,
        );
        return EXIT_SETUP;
    }

    const pkgExports = (
        JSON.parse(readFileSync(resolve(PACKAGE_ROOT, "package.json"), "utf8")) as {
            exports: Record<string, unknown>;
        }
    ).exports;
    const enginePlugins = await Promise.all(
        enabledNames.map((n) => resolveEnginePlugin(n, engine, SUBPATH_PLUGIN_MODULES, pkgExports)),
    );
    const localPlugins = await Promise.all(locals.map((l) => resolveLocalPlugin(l.name, l.path)));

    const sceneXml = manifest.scene
        ? readFileSync(join(projectDir, "public", manifest.scene), "utf8")
        : undefined;

    const app = await engine.build({
        plugins: [...enginePlugins, ...localPlugins],
        defaults: false,
        scene: sceneXml,
        capacity: manifest.capacity,
        // the default loading screen (`standard/loading`) mounts a DOM overlay via
        // `document.createElement`; `installHeadlessDom`'s shim only implements what `InputSystem`/
        // `attachCanvas` read, not a real DOM, and a terminal command has no browser overlay to show
        // regardless — a silent no-op screen, never the default.
        loading: { show: () => undefined, update: () => undefined },
    });
    if (app.skipped.length > 0) {
        console.error(
            `shallot tui: plugins skipped at build (missing dependency): ${app.skipped.join(", ")}`,
        );
    }

    const cameraEid = [...app.state.query([engine.Camera])][0];
    if (cameraEid === undefined) {
        console.error(`shallot tui: no camera in ${projectDir}'s scene`);
        app.dispose();
        return EXIT_SETUP;
    }
    attachCanvas(cameraEid, canvas as unknown as HTMLCanvasElement, app.state);

    // `interactive` gates everything terminal-presentation-specific (alt-screen, raw-mode stdin, resize
    // tracking, real-time pacing) on stdin being a real tty — a piped run (criterion 6) hits none of it,
    // which is what keeps that path free of any input/timing nondeterminism.
    const interactive = !!process.stdin.isTTY;
    const tier = args.tier ?? detectTier({ isTTY: !!process.stdout.isTTY, env: process.env });
    const encoder = new Encoder(tier);
    const write = (s: string) => {
        if (s.length > 0) process.stdout.write(s);
    };

    let stdinBridge: { stop: () => void } | null = null;
    let unsubscribeResize: (() => void) | null = null;
    const quitGuard = createQuitGuard();

    const disposeAll = buildDisposeAll(quitGuard, () => {
        stdinBridge?.stop();
        unsubscribeResize?.();
        app.dispose();
    });

    // Owns the "process killed externally" path (SIGINT/SIGTERM) — it writes the alt-screen restore
    // itself (gated the same way `write` above is, so it's a no-op in piped mode) and this `exit`
    // callback disposes engine/stdin state before actually ending the process. The clean path below
    // (loop finished, or `q`/Ctrl-C read from the stdin bridge) never goes through `exit` — it falls out
    // of the loop and calls the returned `teardown()` directly, which restores the screen exactly once
    // (idempotent — see `screen.ts`'s own docblock) without touching the exit obligation a live SIGINT
    // still owes afterward.
    const teardown = installTeardown({
        write: (bytes) => {
            if (interactive) process.stdout.write(bytes);
        },
        signals: process,
        exit: (code) => {
            disposeAll();
            process.exit(code);
        },
    });

    if (interactive) {
        write(ALT_SCREEN_ENTER);
        stdinBridge = installStdinBridge(win, () => {
            quitGuard.requestStop(); // 'q' / Ctrl-C: fall out of the loop below, the normal clean-exit path
        });
        unsubscribeResize = onResize(
            process.stdout as unknown as Parameters<typeof onResize>[0],
            (size) => {
                const next = terminalGridSize(size);
                canvas.resize(
                    next.width * TERMINAL_CELL_DEVICE_PX,
                    next.height * TERMINAL_CELL_DEVICE_PX,
                );
                encoder.invalidate();
            },
        );
    }

    const dt = 1 / args.fps;
    let frame = 0;
    await runLoopWithTeardown(
        async () => {
            while (!quitGuard.stopped && (args.frames === undefined || frame < args.frames)) {
                app.state.step(dt);
                const grid = cellsGridFor(cameraEid);
                if (grid) {
                    const raw = engine.Compute.root.unwrap(grid.buffer);
                    const staging = engine.Compute.device.createBuffer({
                        label: "tui-staging",
                        size: raw.size,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                    });
                    const cmd = engine.Compute.device.createCommandEncoder({
                        label: "tui-readback",
                    });
                    cmd.copyBufferToBuffer(raw, 0, staging, 0, raw.size);
                    engine.Compute.device.queue.submit([cmd.finish()]);
                    await engine.Compute.device.queue.onSubmittedWorkDone();
                    await staging.mapAsync(GPUMapMode.READ, 0, raw.size);
                    const bytes = staging.getMappedRange(0, raw.size).slice(0);
                    staging.destroy();
                    const tuiGrid = cellsBytesToGrid(
                        bytes,
                        grid.cols,
                        grid.rows,
                        unpackCell,
                        cellGlyphChar,
                        makeGrid,
                    );
                    write(encoder.encode(tuiGrid));
                }
                frame++;
                if (interactive) await Bun.sleep(dt * 1000);
            }
        },
        teardown,
        disposeAll,
    );
    return EXIT_PASS;
}

if (import.meta.main) {
    process.exit(await runTui(process.argv.slice(2)));
}
