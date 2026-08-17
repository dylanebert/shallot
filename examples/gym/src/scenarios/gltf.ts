import {
    Compute,
    GlazePlugin,
    InputPlugin,
    MirrorPlugin,
    OrbitPlugin,
    PartPlugin,
    type Plugin,
    RenderPlugin,
    run,
    SearPlugin,
    SlabPlugin,
    type State,
    TransformsPlugin,
} from "@dylanebert/shallot";
import {
    type GltfImport,
    GltfPlugin,
    loadGltf,
    ProfilePlugin,
    placeScene,
} from "@dylanebert/shallot/extras";
import {
    ALBEDO_NAMES,
    gltfCacheStats,
    invalidate,
    unionPending,
} from "@dylanebert/shallot/gltf/core";
import { now, requestGPU } from "@dylanebert/shallot/runtime";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// gltf — the asset lifecycle atom: load → dispose/rebuild (cache hit) → unload (invalidate) → reload, the
// substrate of a live host's play/stop and a standalone HMR. The render scenario's gltf-* modes cover a single
// build's import correctness; this one covers what survives a State rebuild. The bug it pins: `build()` calls
// `requestGPU(device)` which wipes `Compute.textures`/`buffers` to empty maps every build (the engine's
// clear-then-rebuild contract — each producer re-publishes its slots), but the glTF union memo
// (`ensureUnion`) re-points the surviving textures only on a miss — so a cache-hit rebuild left the 1×1
// fallback bound and the textured scene went solid black on the second build onward (the editor's first
// play/stop). The gate reads the published artifacts + the decode counter directly (no GPU readback): the
// rebuild must re-publish the real union with no re-decode, no re-upload.

const SOURCES: Record<string, string> = {
    sponza: "sponza/Sponza-KTX-Draco.glb", // textured union path — the black-on-replay case
    fox: "gltf-samples/Fox/glTF/Fox.gltf", // skinned: also exercises the per-mesh VAT republish
};

// the render + import stack the rebuild cycles (render's proven corePlugins minus Profile, plus the importer).
// Profile leads the live leg only, so its device-method patches wrap once, not once per build.
const renderStack: Plugin[] = [
    SlabPlugin,
    MirrorPlugin,
    TransformsPlugin,
    InputPlugin,
    OrbitPlugin,
    RenderPlugin,
    PartPlugin,
    SearPlugin,
    GlazePlugin,
    GltfPlugin,
];
const liveStack: Plugin[] = [ProfilePlugin, ...renderStack];

// the env without the asset — the import loads imperatively after build (the editor's build-then-load runtime
// path), so the first-load cost lands on a measurable `loadGltf`, not buried in warm behind a loading screen.
const envScene = `<scene>
    <a ambient-light="color: 0x6a7290; intensity: 1.0" />
    <a directional-light="intensity: 3.0; direction: -0.4 -0.8 -0.45" shadow="distance: 30" />
    <a camera="clear-color: 0x0a0c12" sear
       orbit="pan: 0 4 0; distance: 16; max-distance: 40; yaw: 0.6; pitch: 0.1" transform />
</scene>`;

let src = SOURCES.sponza;
// the cache trace the assert reads — recorded across the rebuild cycle in build()
const trace = {
    firstDecodes: 0,
    unloadAssets: -1,
    redecodes: 0,
    rebuildDecodes: 0,
    rebuildAssets: 0,
};
// load wall-time: the cold first load (decode + upload) vs the cache-hit rebuild (place + pointer-republish)
let loadMs = { first: 0, rebuild: 0 };
// the worst main-thread stall *during* each load — the actual user-felt "lag spike". The worker pool decodes
// off-thread, so a cold load that doesn't block the main thread reads ≈ one frame here (the cost is delay +
// pop-in, a loading-cover concern); a big number is a synchronous spike worth spreading.
let loadJankMs = { first: 0, rebuild: 0 };

// time one imperative load while sampling the largest gap between animation frames — the main-thread stall the
// load induces (the loop keeps running through an off-thread decode; only synchronous work widens a gap).
async function timedLoad(
    state: State,
    url: string,
): Promise<{ ms: number; jankMs: number; asset: GltfImport }> {
    let maxGap = 0;
    let last = now();
    let sampling = true;
    const sample = () => {
        const t = now();
        maxGap = Math.max(maxGap, t - last);
        last = t;
        if (sampling) requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
    const t0 = now();
    const asset = await loadGltf(state, url);
    // the union uploads across frames (the cold path stages it); drain it before stopping the clock so the jank
    // sampler measures the spread per-frame cost (not just the cheap begin) and `ms` covers the full load
    while (unionPending()) await new Promise<void>((r) => requestAnimationFrame(() => r()));
    const ms = now() - t0;
    sampling = false;
    return { ms, jankMs: maxGap, asset };
}

// one rebuild step — a fresh State on the shared device with the env scene (the caller disposes it)
function cycle(device: GPUDevice, stack: Plugin[]): Promise<{ state: State; dispose: () => void }> {
    return run({ defaults: false, plugins: stack, scene: envScene, device });
}

const scenario: Scenario = {
    name: "gltf",
    params: [
        {
            key: "source",
            type: "select",
            default: "sponza",
            options: ["sponza", "fox"],
            rebuild: true,
        },
    ],

    async build(_canvas, p: Params) {
        src = SOURCES[(p.source as string) ?? "sponza"] ?? SOURCES.sponza;
        loadMs = { first: 0, rebuild: 0 };
        loadJankMs = { first: 0, rebuild: 0 };

        // reuse ONE device across every build — a live host's play/stop rebuild reuses the device, so the
        // module-level asset cache + union survive it. Acquiring per build (run's default) would hand each
        // build a fresh device and the cache's GPU resources would belong to a dead one.
        //
        // Acquiring by hand means unioning the stack's feature needs by hand too — `run({ device })` adopts
        // an external device as-is, so a plugin's `features` / `preferredFeatures` never reach it. Both are
        // load-bearing here: ProfilePlugin *requires* timestamp-query, and the importer *prefers* all three
        // compression families (a family the device never requested reads false in `pickTargets`, so a
        // KTX2 asset would throw on hardware that supports it).
        const device = (
            await requestGPU(
                undefined,
                liveStack.flatMap((p) => p.features ?? []),
                liveStack.flatMap((p) => p.preferredFeatures ?? []),
            )
        ).device;

        // warm the once-per-process costs (decode worker pool + Draco/Basis WASM + sear/blit pipeline compiles)
        // so the measured leg A reflects the per-load UNION BUILD, not the one-time cold-start a live host pays
        // at startup behind a loading cover. Without this, leg A's stall is dominated by ~175ms of worker+pipeline
        // warm-up that has nothing to do with the texture upload this scenario gates.
        const warm = await cycle(device, renderStack);
        await loadGltf(warm.state, src);
        while (unionPending()) await frames(1);
        warm.dispose();
        invalidate(src);

        // leg A — cold UNION load (worker + pipelines warm, union cache-cold): a live host's runtime-load path. The
        // staged build spreads the upload across frames, so the stall is per-frame budget, not one 168ms freeze.
        const a = await cycle(device, renderStack);
        const first = await timedLoad(a.state, src);
        loadMs.first = first.ms;
        loadJankMs.first = first.jankMs;
        trace.firstDecodes = gltfCacheStats().decodes;
        a.dispose();

        // unload — invalidate evicts the source from the cache + drops the accumulated union (the HMR / asset-swap
        // seam). The next load must re-decode, proving the eviction freed the entry.
        invalidate(src);
        trace.unloadAssets = gltfCacheStats().assets;

        // leg C — reload after the unload: a cache miss, so decodes advances. Re-seats the cache + union for B.
        const c = await cycle(device, renderStack);
        await loadGltf(c.state, src);
        while (unionPending()) await frames(1); // let the staged upload finish before disposing this leg
        trace.redecodes = gltfCacheStats().decodes;
        c.dispose();

        // leg B — the cache-hit rebuild (a live host's play/stop), the returned live state. requestGPU wiped
        // Compute.textures at this build and the union memo survives from C: the exact path the black-on-replay
        // bug lived on. The union must re-publish into the wiped map (no re-decode, no re-upload).
        const b = await cycle(device, liveStack);
        const rebuild = await timedLoad(b.state, src);
        loadMs.rebuild = rebuild.ms;
        loadJankMs.rebuild = rebuild.jankMs;
        trace.rebuildDecodes = gltfCacheStats().decodes;
        trace.rebuildAssets = gltfCacheStats().assets;
        // place the cache-hit-rebuilt union so the returned live state actually renders the model — the pixel
        // gate's honest end-to-end proof that the republish bound the real textures (a black canvas here is the
        // black-on-replay bug's own symptom, the very thing the resource-state checks above verify indirectly).
        placeScene(b.state, rebuild.asset);
        await frames(2);
        return b;
    },

    async assert(): Promise<Check[]> {
        // a real baseColor bucket is >1×1 (a 1×1 is the fallback that stays bound when the publish is skipped) —
        // the material-count-independent signal. publishTextures re-points the albedo arrays + the palette
        // atomically, so a real bucket means the whole union republished (the palette's own size is ambiguous: a
        // real 1-material palette is 64B, same as the fallback, so it's shown but not gated on).
        const buckets = ALBEDO_NAMES.map((n) => Compute.textures.get(n)).filter(
            (t): t is GPUTexture => !!t && (t.width > 1 || t.depthOrArrayLayers > 1),
        );
        const layers = buckets.reduce((s, t) => s + t.depthOrArrayLayers, 0);
        const palette = Compute.buffers.get("materialData");

        return [
            {
                // the black-on-replay gate: red before the ensureUnion republish fix (only the 1×1 fallback bound)
                name: "union re-published after cache-hit rebuild",
                pass: layers > 0,
                detail: buckets.length
                    ? `${buckets.length} real bucket(s), ${layers} layers, palette ${palette?.size}B`
                    : "only the 1×1 fallback is bound — union not re-published (black)",
            },
            {
                name: "cache-hit rebuild re-decoded + re-uploaded nothing",
                pass: trace.rebuildDecodes === trace.redecodes && trace.rebuildAssets === 1,
                detail: `decodes ${trace.redecodes} → ${trace.rebuildDecodes} across the rebuild, ${trace.rebuildAssets} cached asset`,
            },
            {
                name: "invalidate evicted the source + forced a re-decode",
                pass: trace.unloadAssets === 0 && trace.redecodes === trace.firstDecodes + 1,
                detail: `assets → ${trace.unloadAssets} on invalidate, decodes ${trace.firstDecodes} → ${trace.redecodes} on reload`,
            },
            {
                // the spike instrument: a cache hit skips decode + upload, so the rebuild load is far cheaper than
                // the cold first load (orders of magnitude, not a tuned margin)
                name: "cache-hit rebuild load is cheaper than the cold first load",
                pass: loadMs.rebuild < loadMs.first,
                detail: `first ${loadMs.first.toFixed(1)}ms / rebuild ${loadMs.rebuild.toFixed(1)}ms wall · main-thread stall first ${loadJankMs.first.toFixed(1)}ms / rebuild ${loadJankMs.rebuild.toFixed(1)}ms`,
            },
        ];
    },

    live(): string {
        const s = gltfCacheStats();
        return [
            "gltf — asset lifecycle",
            `decodes ${s.decodes}  assets ${s.assets}  inflight ${s.inflight}`,
            `load    first ${loadMs.first.toFixed(1)}ms / rebuild ${loadMs.rebuild.toFixed(1)}ms`,
            `stall   first ${loadJankMs.first.toFixed(1)}ms / rebuild ${loadJankMs.rebuild.toFixed(1)}ms`,
        ].join("\n");
    },
};

register(scenario);
