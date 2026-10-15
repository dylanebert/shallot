import type { Plugin, Resource, System, World } from "../../engine";
import { beginRenderPass, endRenderPass } from "../../engine/runtime";
import { Camera, computeViewFromWorld } from "./camera";
import { BeginFrameSystem, OverlaySystem, RenderingPlugin } from "./substrate";
import {
    colorPassDescriptor,
    colorTargets,
    DepthPrepass,
    DepthPrepassRequests,
    disposeViewTargets,
    initializeViewTargets,
    prepassDescriptor,
} from "./targets";
import { TonemappingPlugin, TonemappingSystem } from "./tonemapping-state";
import { type View, Views } from "./view";

/** One renderer-owned draw submitted to the shared, per-view transparent phase. */
export interface TransparentPhaseItem {
    /** Row 2 of the view-from-world matrix dotted with the item's world-space bounds center. */
    readonly sortKey: number;
    /** Renderer-owned stable identity, distinct from `draw`; unique with `stablePart` per renderer. */
    readonly stableId: number;
    /** Stable subidentity for several draws owned by one item. */
    readonly stablePart: number;
    /** Index of the contributing renderer in `RenderPhases`. */
    readonly renderer: number;
    /** Renderer-owned identity used to prepare and issue the draw. */
    readonly draw: number;
    /** Renderer-owned key for state and geometry that may batch with adjacent items. */
    readonly batch: number;
}

type MutableTransparentItem = {
    sortKey: number;
    stableId: number;
    stablePart: number;
    renderer: number;
    draw: number;
    batch: number;
};

function invalidTransparentProducer(): never {
    throw new Error("TransparentPhase.add must run from a RenderPhases transparent producer");
}
function invalidTransparentStableId(stableId: number): never {
    throw new Error(`TransparentPhase.add received an invalid stable id: ${stableId}`);
}
function invalidTransparentStablePart(stablePart: number): never {
    throw new Error(`TransparentPhase.add received an invalid stable part: ${stablePart}`);
}
function invalidTransparentSortKey(sortKey: number): never {
    throw new Error(`TransparentPhase.add received a non-finite view-space Z: ${sortKey}`);
}

/** Per-view transparent queue, reused across frames and grown only when capacity is exceeded. */
export class TransparentPhase {
    private _items: MutableTransparentItem[] = [];
    private _scratch: MutableTransparentItem[] = [];
    private _countValue = 0;
    private _rendererValue = -1;
    private readonly _viewFromWorld = new Float32Array(16);
    /** @internal Reusable position input for renderers with their own typed staging. */
    readonly positionScratch = new Float64Array(3);
    private _z0 = 0;
    private _z1 = 0;
    private _z2 = 1;
    private _z3 = 0;

    /** @internal Enable clock reads for the explicit rendering oracle, not ordinary frames. */
    measureTimings = false;

    readonly metrics = {
        items: 0,
        batches: 0,
        draws: 0,
        sortComparisons: 0,
        keyMs: 0,
        sortMs: 0,
        prepareMs: 0,
        drawMs: 0,
        uploadMs: 0,
        uploadBytes: 0,
        pipelineChanges: 0,
        bindGroupChanges: 0,
    };

    get length(): number {
        return this._countValue;
    }

    begin(world: World, eid: number, view: View): void {
        if (view.viewFromWorld) this._viewFromWorld.set(view.viewFromWorld);
        else computeViewFromWorld(world, eid, this._viewFromWorld);
        this._z0 = this._viewFromWorld[2];
        this._z1 = this._viewFromWorld[6];
        this._z2 = this._viewFromWorld[10];
        this._z3 = this._viewFromWorld[14];
        this._countValue = 0;
        this._rendererValue = -1;
        this.metrics.items = 0;
        this.metrics.batches = 0;
        this.metrics.draws = 0;
        this.metrics.sortComparisons = 0;
        this.metrics.keyMs = 0;
        this.metrics.sortMs = 0;
        this.metrics.prepareMs = 0;
        this.metrics.drawMs = 0;
        this.metrics.uploadMs = 0;
        this.metrics.uploadBytes = 0;
        this.metrics.pipelineChanges = 0;
        this.metrics.bindGroupChanges = 0;
    }

    /** @internal Sets the renderer namespace for the producer callback currently adding items. */
    useRenderer(renderer: number): void {
        this._rendererValue = renderer;
    }

    /** Add an item by world-space bounds center. Smaller view-space Z sorts first; renderer namespace and stable identity break ties. */
    add(
        draw: number,
        batch: number,
        x: number,
        y: number,
        z: number,
        stableId: number,
        stablePart = 0,
    ): void {
        if (this._rendererValue < 0) invalidTransparentProducer();
        const sortKey = this._z0 * x + this._z1 * y + this._z2 * z + this._z3;
        if (!Number.isSafeInteger(stableId) || stableId < 0) invalidTransparentStableId(stableId);
        if (!Number.isSafeInteger(stablePart) || stablePart < 0)
            invalidTransparentStablePart(stablePart);
        if (!Number.isFinite(sortKey)) invalidTransparentSortKey(sortKey);
        this.ensureCapacity(this._countValue + 1);
        const item = this._items[this._countValue++];
        item.sortKey = sortKey;
        item.stableId = stableId;
        item.stablePart = stablePart;
        item.renderer = this._rendererValue;
        item.draw = draw;
        item.batch = batch;
        this.metrics.items++;
    }

    /** @internal Add an item from positionScratch using trusted built-in identities and batch keys. */
    addFromScratch(draw: number, batch: number, stableId: number, stablePart = 0): void {
        if (this._rendererValue < 0) invalidTransparentProducer();
        if (!Number.isSafeInteger(stableId) || stableId < 0) invalidTransparentStableId(stableId);
        if (!Number.isSafeInteger(stablePart) || stablePart < 0)
            invalidTransparentStablePart(stablePart);
        const sortKey =
            this._z0 * this.positionScratch[0]! +
            this._z1 * this.positionScratch[1]! +
            this._z2 * this.positionScratch[2]! +
            this._z3;
        if (!Number.isFinite(sortKey)) invalidTransparentSortKey(sortKey);
        this.ensureCapacity(this._countValue + 1);
        const item = this._items[this._countValue++];
        item.sortKey = sortKey;
        item.stableId = stableId;
        item.stablePart = stablePart;
        item.renderer = this._rendererValue;
        item.draw = draw;
        item.batch = batch;
        this.metrics.items++;
    }

    /** Read a sorted item without creating a per-frame view. */
    item(index: number): TransparentPhaseItem {
        if (index < 0 || index >= this._countValue)
            throw new RangeError(`transparent item index ${index} is out of range`);
        return this._items[index];
    }

    /** Stable order by view Z, renderer namespace, stable id and stable part. */
    sort(): void {
        const started = this.measureTimings ? performance.now() : 0;
        const count = this._countValue;
        for (let width = 1; width < count; width *= 2) {
            for (let left = 0; left < count; left += width * 2) {
                const middle = Math.min(left + width, count);
                const right = Math.min(left + width * 2, count);
                let a = left;
                let b = middle;
                let out = left;
                while (a < middle && b < right) {
                    this.metrics.sortComparisons++;
                    const first = this._items[a];
                    const second = this._items[b];
                    if (
                        first.sortKey < second.sortKey ||
                        (first.sortKey === second.sortKey &&
                            (first.renderer < second.renderer ||
                                (first.renderer === second.renderer &&
                                    (first.stableId < second.stableId ||
                                        (first.stableId === second.stableId &&
                                            first.stablePart <= second.stablePart)))))
                    ) {
                        this._scratch[out++] = first;
                        a++;
                    } else {
                        this._scratch[out++] = second;
                        b++;
                    }
                }
                while (a < middle) this._scratch[out++] = this._items[a++];
                while (b < right) this._scratch[out++] = this._items[b++];
            }
            const previous = this._items;
            this._items = this._scratch;
            this._scratch = previous;
        }
        if (this.measureTimings) this.metrics.sortMs += performance.now() - started;
    }

    /** @internal Record CPU time and bytes passed through World's ordered upload path. */
    recordUpload(bytes: number, milliseconds: number): void {
        this.metrics.uploadBytes += bytes;
        this.metrics.uploadMs += milliseconds;
    }

    private ensureCapacity(required: number): void {
        if (required <= this._items.length) return;
        let capacity = Math.max(32, this._items.length * 2);
        while (capacity < required) capacity *= 2;
        for (let i = this._items.length; i < capacity; i++) {
            this._items[i] = {
                sortKey: 0,
                stableId: 0,
                stablePart: 0,
                renderer: 0,
                draw: 0,
                batch: 0,
            };
            this._scratch[i] = this._items[i];
        }
    }
}

interface TransparentPhaseResources {
    views: Map<number, TransparentPhase>;
}
const transparentPhaseKey: Resource<TransparentPhaseResources> = {
    create: () => ({ views: new Map() }),
};

/** Reusable transparent phases, one for each presenting camera. */
export const TransparentPhases: Resource<Map<number, TransparentPhase>> = {
    create: (world) => world.resource(transparentPhaseKey).views,
};

/** Records into a core-owned pass. Records must not end the pass. Renderers contribute transparent items to the shared sorted phase, prepare view-specific data before the pass, then draw only consecutive compatible items. */
export interface PhaseRenderer {
    prepass?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
    opaque?(world: World, eid: number, view: View, pass: GPURenderPassEncoder): void;
    transparent?(world: World, eid: number, view: View, phase: TransparentPhase): void;
    prepareTransparent?(
        world: World,
        eid: number,
        view: View,
        phase: TransparentPhase,
        renderer: number,
    ): void;
    drawTransparent?(
        world: World,
        eid: number,
        view: View,
        phase: TransparentPhase,
        start: number,
        end: number,
        pass: GPURenderPassEncoder,
        renderer: number,
    ): void;
}
export const RenderPhases = { create: (): PhaseRenderer[] => [] };

function recordDepthPrepass(this: World, view: View, eid: number): void {
    if (!view.framebuffer) return;
    const encoder = this.frameEncoder()!;
    view.depth = null;
    let requested = this.has(eid, DepthPrepass);
    if (!requested) {
        const requests = this.resource(DepthPrepassRequests);
        for (let i = 0; i < requests.length; i++) {
            if (requests[i]!(this, eid, view)) {
                requested = true;
                break;
            }
        }
    }
    if (!requested) return;
    const pass = beginRenderPass(encoder, prepassDescriptor(this, eid, view));
    const renderers = this.resource(RenderPhases);
    for (let i = 0; i < renderers.length; i++) renderers[i].prepass?.(this, eid, view, pass);
    endRenderPass(pass);
}

function pruneTransparentPhase(this: World, _phase: TransparentPhase, eid: number): void {
    if (!this.resource(Views).has(eid)) this.resource(TransparentPhases).delete(eid);
}

function recordMainPass(this: World, view: View, eid: number): void {
    if (!view.framebuffer) return;
    const renderers = this.resource(RenderPhases);
    const phases = this.resource(TransparentPhases);
    let phase = phases.get(eid);
    if (!phase) phases.set(eid, (phase = new TransparentPhase()));
    phase.begin(this, eid, view);
    let started = phase.measureTimings ? performance.now() : 0;
    for (let renderer = 0; renderer < renderers.length; renderer++) {
        phase.useRenderer(renderer);
        renderers[renderer].transparent?.(this, eid, view, phase);
    }
    if (phase.measureTimings) phase.metrics.keyMs = performance.now() - started;
    phase.sort();
    started = phase.measureTimings ? performance.now() : 0;
    for (let renderer = 0; renderer < renderers.length; renderer++)
        renderers[renderer].prepareTransparent?.(this, eid, view, phase, renderer);
    if (phase.measureTimings) phase.metrics.prepareMs = performance.now() - started;

    const encoder = this.frameEncoder()!;
    const targets = colorTargets(
        this,
        eid,
        view.width,
        view.height,
        this.storage(Camera).antialias.get(eid) !== 0,
    );
    const pass = beginRenderPass(
        encoder,
        colorPassDescriptor(this, eid, targets, view.framebuffer),
    );
    for (let i = 0; i < renderers.length; i++) renderers[i].opaque?.(this, eid, view, pass);
    let index = 0;
    started = phase.measureTimings ? performance.now() : 0;
    while (index < phase.length) {
        const item = phase.item(index);
        let end = index + 1;
        while (end < phase.length) {
            const next = phase.item(end);
            if (next.renderer !== item.renderer || next.batch !== item.batch) break;
            end++;
        }
        const renderer = renderers[item.renderer];
        if (!renderer.drawTransparent)
            throw new Error(
                `RenderPhases renderer ${item.renderer} queued transparent work without drawTransparent`,
            );
        phase.metrics.batches++;
        renderer.drawTransparent(this, eid, view, phase, index, end, pass, item.renderer);
        index = end;
    }
    endRenderPass(pass);
    if (phase.measureTimings) phase.metrics.drawMs = performance.now() - started;
}

export const PrepassSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        world.resource(Views).forEach(recordDepthPrepass, world);
    },
};
export const MainPassSystem: System = {
    group: "draw",
    after: [PrepassSystem],
    before: [OverlaySystem],
    update(world) {
        const phases = world.resource(TransparentPhases);
        phases.forEach(pruneTransparentPhase, world);
        world.resource(Views).forEach(recordMainPass, world);
    },
};

/** Optional shared view pipeline: clear, targets, depth prepass and opaque/transparent records. */
export const CorePipelinePlugin: Plugin = {
    gpu: {},
    name: "CorePipeline",
    dependencies: [RenderingPlugin],
    systems: [PrepassSystem, MainPassSystem, TonemappingSystem],
    components: [...(TonemappingPlugin.components ?? []), DepthPrepass],
    initialize(world) {
        initializeViewTargets(world);
        world.resource(RenderPhases);
        world.resource(DepthPrepassRequests);
        TonemappingPlugin.initialize?.(world);
    },
    warm: TonemappingPlugin.warm,
    dispose(world) {
        disposeViewTargets(world);
        TonemappingPlugin.dispose?.(world);
    },
};
