import type { Plugin, Resource, System, World } from "../../engine";
import { Camera, computeViewProj } from "./camera";
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
    /** Stable contribution order; only distinguishes items with equal sort keys. */
    readonly tie: number;
    /** Index of the contributing renderer in `RenderPhases`. */
    readonly renderer: number;
    /** Renderer-owned identity used to prepare and issue the draw. */
    readonly draw: number;
    /** Renderer-owned key for state and geometry that may batch with adjacent items. */
    readonly batch: number;
}

type MutableTransparentItem = {
    sortKey: number;
    tie: number;
    renderer: number;
    draw: number;
    batch: number;
};

/** Per-view transparent queue, reused across frames and grown only when capacity is exceeded. */
export class TransparentPhase {
    private _items: MutableTransparentItem[] = [];
    private _scratch: MutableTransparentItem[] = [];
    private _countValue = 0;
    private _rendererValue = -1;
    private _sequence = 0;
    private readonly _viewProjection = new Float32Array(16);
    private readonly _viewFromWorld = new Float32Array(16);
    private _z0 = 0;
    private _z1 = 0;
    private _z2 = 1;
    private _z3 = 0;

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

    /** Prepare this view's view-space depth row and reset its reusable item queue. */
    begin(world: World, eid: number, view: View): void {
        computeViewProj(
            world,
            eid,
            view.width / view.height,
            this._viewProjection,
            this._viewFromWorld,
        );
        this._z0 = this._viewFromWorld[2];
        this._z1 = this._viewFromWorld[6];
        this._z2 = this._viewFromWorld[10];
        this._z3 = this._viewFromWorld[14];
        this._countValue = 0;
        this._rendererValue = -1;
        this._sequence = 0;
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

    /** @internal Select the producer whose callback is currently contributing items. */
    useRenderer(renderer: number): void {
        this._rendererValue = renderer;
    }

    /** Add one item at its world-space bounds center. Smaller view-space Z sorts first. */
    add(draw: number, batch: number, x: number, y: number, z: number): void {
        if (this._rendererValue < 0)
            throw new Error(
                "TransparentPhase.add must run from a RenderPhases transparent producer",
            );
        const sortKey = this._z0 * x + this._z1 * y + this._z2 * z + this._z3;
        if (!Number.isFinite(sortKey))
            throw new Error(`TransparentPhase.add received a non-finite view-space Z: ${sortKey}`);
        this.ensureCapacity(this._countValue + 1);
        const item = this._items[this._countValue++];
        item.sortKey = sortKey;
        item.tie = this._sequence++;
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

    /** Stable bottom-up merge sort; the reusable scratch array avoids per-frame sort allocation. */
    sort(): void {
        const started = performance.now();
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
                        (first.sortKey === second.sortKey && first.tie <= second.tie)
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
        this.metrics.sortMs += performance.now() - started;
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
            this._items[i] = { sortKey: 0, tie: 0, renderer: 0, draw: 0, batch: 0 };
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
export const PrepassSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        for (const [eid, view] of world.resource(Views)) {
            if (!view.framebuffer) continue;
            const encoder = world.frameEncoder()!;
            view.depth = null;
            let requested = world.has(eid, DepthPrepass);
            if (!requested) {
                const requests = world.resource(DepthPrepassRequests);
                for (let i = 0; i < requests.length; i++) {
                    if (requests[i]!(world, eid, view)) {
                        requested = true;
                        break;
                    }
                }
            }
            if (!requested) continue;
            const pass = encoder.beginRenderPass(prepassDescriptor(world, eid, view));
            for (const renderer of world.resource(RenderPhases))
                renderer.prepass?.(world, eid, view, pass);
            pass.end();
        }
    },
};
export const MainPassSystem: System = {
    group: "draw",
    after: [PrepassSystem],
    before: [OverlaySystem],
    update(world) {
        const renderers = world.resource(RenderPhases);
        const phases = world.resource(TransparentPhases);
        const views = world.resource(Views);
        for (const eid of phases.keys()) if (!views.has(eid)) phases.delete(eid);
        for (const [eid, view] of views) {
            if (!view.framebuffer) continue;
            let phase = phases.get(eid);
            if (!phase) phases.set(eid, (phase = new TransparentPhase()));
            phase.begin(world, eid, view);
            let started = performance.now();
            for (let renderer = 0; renderer < renderers.length; renderer++) {
                phase.useRenderer(renderer);
                renderers[renderer].transparent?.(world, eid, view, phase);
            }
            phase.metrics.keyMs = performance.now() - started;
            phase.sort();
            started = performance.now();
            for (let renderer = 0; renderer < renderers.length; renderer++)
                renderers[renderer].prepareTransparent?.(world, eid, view, phase, renderer);
            phase.metrics.prepareMs = performance.now() - started;

            const encoder = world.frameEncoder()!;
            const targets = colorTargets(
                world,
                eid,
                view.width,
                view.height,
                world.storage(Camera).antialias.get(eid) !== 0,
            );
            const pass = encoder.beginRenderPass(
                colorPassDescriptor(world, eid, targets, view.framebuffer),
            );
            for (const renderer of renderers) renderer.opaque?.(world, eid, view, pass);
            let index = 0;
            started = performance.now();
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
                renderer.drawTransparent(world, eid, view, phase, index, end, pass, item.renderer);
                index = end;
            }
            pass.end();
            phase.metrics.drawMs = performance.now() - started;
        }
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
