import type { World } from "../../../engine";
import { type KernelState, kernel, kernelState, kernelViewKey, kernelViewsCurrent } from "./kernel";

type ViewOwner = { ensureViews(): void };
type ViewFields = { viewOwner: ViewOwner; viewData: Record<string, unknown> };

/** Accessors resolve kernel views through their live owner. */
export function guardViews(target: object, owner: ViewOwner): void {
    const fields = target as ViewFields;
    fields.viewOwner = owner;
    fields.viewData = {};
    for (const name of Object.keys(target)) {
        const value = Object.getOwnPropertyDescriptor(target, name)!.value;
        if (
            !ArrayBuffer.isView(value) &&
            !(Array.isArray(value) && value.length > 0 && value.every(ArrayBuffer.isView))
        )
            continue;
        fields.viewData[name] = value;
        Object.defineProperty(target, name, {
            enumerable: true,
            configurable: true,
            get: function (this: ViewFields) {
                this.viewOwner.ensureViews();
                return this.viewData[name];
            },
            set: function (this: ViewFields, value: unknown) {
                this.viewData[name] = value;
            },
        });
    }
}

export interface CheckpointStore {
    captureCheckpoint(): unknown;
    restoreCheckpoint(state: unknown): void;
}

/** Stores expose current views, not arrays retained across a kernel allocation or restore. */
export abstract class KernelViews {
    readonly ecsState: World | undefined;
    private _viewKey = -1;
    private _refreshing = false;
    private readonly _kernel: KernelState;

    constructor(ecsState: World | undefined) {
        this.ecsState = ecsState;
        this._kernel = kernelState(ecsState);
        // Resolve guarded calls once per owner, rather than inherited lookups across store kinds.
        this.ensureViews = this.ensureViews.bind(this);
        this.refreshViews = this.refreshViews.bind(this);
        this.deriveViews = this.deriveViews.bind(this);
    }

    abstract captureCheckpoint(): unknown;
    abstract restoreCheckpoint(state: unknown): void;

    protected guardViews(): void {
        guardViews(this, this);
    }
    refreshIfStale(): void {
        this.ensureViews();
    }

    get stale(): boolean {
        return this._viewKey !== kernelViewKey(this.ecsState);
    }

    ensureViews(): void {
        if (
            this._viewKey === this._kernel.viewRevision &&
            kernelViewsCurrent(this.ecsState, this._kernel)
        )
            return;
        if (!this._refreshing && this._viewKey !== kernelViewKey(this.ecsState, this._kernel))
            this.refreshViews();
    }

    refreshViews(): void {
        if (this._refreshing) return;
        const k = kernel(this.ecsState);
        const selected = k.activeWorld();
        this._refreshing = true;
        try {
            this.deriveViews();
            this._viewKey = kernelViewKey(this.ecsState);
        } finally {
            // A view read inside a query callback must not switch the outer traversal's World.
            k.bodySetActiveWorld(selected);
            this._refreshing = false;
        }
    }

    protected abstract deriveViews(): void;
}
