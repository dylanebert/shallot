// The tier-0 ASCII dump — criterion 8's own artifact promoted into the repo (s3r item 2,
// specs/shallot-tui.md): "the grid is readable as text at a stock 80x24 with color removed... read from
// the recipe" (examples/recipes/render-to-a-terminal). Boots that recipe's scene headlessly through
// bun-webgpu — no browser, no `shallot verify`, no dev/preview server — drives the orbit camera to a
// named yaw, steps the real `CellsPlugin` pipeline (`select.ts`'s structure-first glyph selection over
// the actual rendered cube), and prints the 80x24 glyph-only grid: exactly what criterion 8's human read
// judges, with the color the criterion strips already absent.
//
// Lives beside `generate-ramp.ts`, not under `src/extras/cells/` — the same reasoning that file's own
// header gives: `check-imports.ts` only walks `src/`, so a script tree is where cross-module, Node-only,
// real-device tooling lives rather than in a module whose barrel a browser consumer imports. This one
// additionally needs a canvas-shaped `GPUCanvasContext` to drive the engine's real per-camera render path
// headlessly (`BeginFrameSystem` only populates `view.framebuffer` for a canvas-bound camera) — a
// minimal hand-written mock (below) rather than `bun-webgpu`'s own internal `GPUCanvasContextMock`, which
// carries no published type declarations and configures real `STORAGE_BINDING` usage on the canvas
// texture for Glaze's compute composite, a capability this diagnostic never needs since it drops
// `GlazePlugin` entirely (it only reads the pre-Glaze offscreen `CellsPlugin` selects from, never the
// swapchain).
//
// Run from the `packages/shallot` package root (the registered `dump-cells-ascii` script, so the TGSL
// preload below — required, `bun run` alone does not apply a `bunfig.toml` `[test]`-only preload — is
// never forgotten): `bun run dump-cells-ascii -- [--yaw <radians>] [--pitch <radians>]`, or equivalently
// from the shallot repo root: `bun run --cwd packages/shallot dump-cells-ascii -- --yaw <radians>`.
// Defaults match the recipe scene's own authored orbit (`yaw: 0.6; pitch: 0.55`).

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { plugin } from "bun";
import { setupGlobals } from "bun-webgpu";
import typegpuBunPlugin from "unplugin-typegpu/bun";

// TGSL function bodies are transpiled at load time (`tests/tgsl.ts`'s own docblock: without this
// transform every kernel resolves with no metadata and CPU-called kernels return NaN). `bunfig.toml`'s
// `[test]` preload only reaches `bun test`, never a plain `bun run`, so this script registers the same
// loader itself — first, before any import below pulls in a TGSL-bearing module (`../src`).
plugin(typegpuBunPlugin({ include: /\.tsx?$/ }));

import {
    build,
    Camera,
    CellsPlugin,
    COLS,
    Compute,
    cellsGridFor,
    InputPlugin,
    Orbit,
    OrbitPlugin,
    PartPlugin,
    RenderPlugin,
    ROWS,
    SearPlugin,
    SlabPlugin,
    TransformsPlugin,
} from "../src";
import { cellGlyphChar, unpackCell } from "../src/extras/cells/core";
import { attachCanvas } from "../src/standard/render/core";

const SCENE_URL = new URL(
    "../../../examples/recipes/render-to-a-terminal/public/scenes/render-to-a-terminal.scene",
    import.meta.url,
);

// the render resolution CellsPlugin's compute pass block-samples down to the fixed 80x24 grid — a
// diagnostic detail, not part of what criterion 8 reads (only the glyph grid is).
const RENDER_W = 640;
const RENDER_H = 360;

// `BeginFrameSystem` calls `new ResizeObserver(...)` on attach; bun-webgpu has no DOM, so this script
// supplies the trivial stand-in `attachCanvas`'s own resize-tracking needs (`sizeView` reads the cached
// `clientWidth`/`clientHeight` the observer would update on a real resize — this canvas never resizes).
// biome-ignore lint/style/useNamingConvention: ResizeObserver is the DOM global's own name.
if (typeof (globalThis as { ResizeObserver?: unknown }).ResizeObserver === "undefined") {
    // biome-ignore lint/style/useNamingConvention: ResizeObserver is the DOM global's own name.
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
        observe(): void {}
        disconnect(): void {}
    };
}

/**
 * a minimal `HTMLCanvasElement`-shaped object satisfying exactly what `attachCanvas` / `sizeView` /
 * `BeginFrameSystem` read (module doc above names why this is hand-written rather than `bun-webgpu`'s own
 * mock). `getCurrentTexture()` hands back a plain `RENDER_ATTACHMENT` texture nothing ever composites
 * into or reads — this diagnostic drops `GlazePlugin`, so the swapchain texture exists only to satisfy
 * `BeginFrameSystem`'s per-frame `getCurrentTexture()` call, never sampled or presented.
 */
class DumpCanvas {
    readonly width: number;
    readonly height: number;
    readonly style: Record<string, string> = {};
    private _device: GPUDevice | null = null;
    private _format: GPUTextureFormat = "bgra8unorm";
    private _texture: GPUTexture | null = null;

    constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
    }

    getContext(kind: string): DumpCanvas | null {
        return kind === "webgpu" ? this : null;
    }

    configure(descriptor: { device: GPUDevice; format: GPUTextureFormat }): void {
        this._device = descriptor.device;
        this._format = descriptor.format;
    }

    getCurrentTexture(): GPUTexture {
        this._texture?.destroy();
        if (!this._device) throw new Error("DumpCanvas: getCurrentTexture before configure");
        this._texture = this._device.createTexture({
            label: "dump-cells-ascii-swapchain",
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
}

interface Args {
    yaw: number;
    pitch: number;
}

function parseArgs(argv: readonly string[]): Args {
    const args: Args = { yaw: 0.6, pitch: 0.55 };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === "--yaw" && argv[i + 1]) args.yaw = Number(argv[++i]);
        else if (argv[i] === "--pitch" && argv[i + 1]) args.pitch = Number(argv[++i]);
    }
    return args;
}

/** the 24x80 glyph-only text {@link cellsGridFor}'s grid decodes to — one row per line, no color, no
 *  trailing padding beyond the grid's own fixed shape. @internal exported for the dump's own test. */
export function renderAsciiGrid(bytes: ArrayBuffer, cols: number, rows: number): string {
    const lines: string[] = [];
    for (let y = 0; y < rows; y++) {
        let line = "";
        for (let x = 0; x < cols; x++) {
            const cell = unpackCell(bytes, y * cols + x);
            line += cellGlyphChar(cell.glyph);
        }
        lines.push(line);
    }
    return lines.join("\n");
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));
    await setupGlobals();

    const sceneXml = readFileSync(fileURLToPath(SCENE_URL), "utf8");
    const app = await build({
        // the recipe's own `shallot.json` plugin list (Orbit, Cells) plus the render substrate its
        // `defaults:false` bypasses `GlazePlugin` for (module doc above) — everything else mirrors
        // `standard/defaults.ts`'s `DEFAULT_PLUGINS`.
        plugins: [
            SlabPlugin,
            TransformsPlugin,
            InputPlugin,
            RenderPlugin,
            PartPlugin,
            SearPlugin,
            OrbitPlugin,
            CellsPlugin,
        ],
        defaults: false,
        scene: sceneXml,
    });
    if (app.skipped.length > 0) {
        throw new Error(`dump-cells-ascii: plugins skipped at build: ${app.skipped.join(", ")}`);
    }

    const cameraEid = [...app.state.query([Camera])][0];
    if (cameraEid === undefined) throw new Error("dump-cells-ascii: no camera in the recipe scene");

    const canvas = new DumpCanvas(RENDER_W, RENDER_H);
    attachCanvas(cameraEid, canvas as unknown as HTMLCanvasElement, app.state);

    // the named yaw/pitch — set before the first step so `OrbitSystem`'s smoothing state initializes
    // directly at this pose instead of lerping toward it (`extras/orbit`'s own `OrbitSmooth` seeding).
    Orbit.yaw.set(cameraEid, args.yaw);
    Orbit.pitch.set(cameraEid, args.pitch);

    app.state.step();
    app.state.step();

    const grid = cellsGridFor(cameraEid);
    if (!grid) throw new Error("dump-cells-ascii: CellsPlugin produced no grid for the camera");
    const raw = Compute.root.unwrap(grid.buffer);
    const staging = Compute.device.createBuffer({
        label: "dump-cells-ascii-staging",
        size: raw.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = Compute.device.createCommandEncoder({ label: "dump-cells-ascii-readback" });
    encoder.copyBufferToBuffer(raw, 0, staging, 0, raw.size);
    Compute.device.queue.submit([encoder.finish()]);
    await Compute.device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ, 0, raw.size);
    const bytes = staging.getMappedRange(0, raw.size).slice(0);
    staging.destroy();

    console.log(`# dump-cells-ascii yaw=${args.yaw} pitch=${args.pitch}`);
    console.log(renderAsciiGrid(bytes, COLS, ROWS));

    app.dispose();
}

if (import.meta.main) {
    main().catch((err) => {
        console.error(err);
        process.exit(1);
    });
}
