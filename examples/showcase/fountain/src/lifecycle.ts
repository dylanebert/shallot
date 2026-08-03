type LifecycleState = { readonly disposed: boolean; onDispose: (cleanup: () => void) => void };

/** Owns each throwaway as soon as it is created, including when a later allocation throws. */
export function withOwnedFountainBuffers<Buffer, Result>(
    prepare: (own: (buffer: Buffer) => Buffer) => Result,
    destroy: (buffer: Buffer) => void,
): Result {
    const buffers: Buffer[] = [];
    const own = (buffer: Buffer): Buffer => {
        buffers.push(buffer);
        return buffer;
    };
    try {
        return prepare(own);
    } finally {
        for (let i = buffers.length - 1; i >= 0; i--) destroy(buffers[i]);
    }
}

/** The warm-generation state machine, separated from GPU operations so its failure paths stay unit-testable. */
export function createFountainLifecycle<
    StateKey extends LifecycleState,
    Draft,
    Owner extends object,
>(ops: {
    current: () => Owner | null;
    owned: (state: StateKey) => Owner | undefined;
    draft: (state: StateKey) => Draft;
    prepare: (draft: Draft) => Owner;
    activate: (state: StateKey, owner: Owner) => void;
    cleanupDraft: (draft: Draft) => void;
    cleanup: (owner: Owner) => void;
    precompile: (owner: Owner, force: () => unknown) => Promise<void>;
    force: (owner: Owner) => unknown;
    publish: (owner: Owner) => void;
}) {
    const warm = async (state: StateKey): Promise<void> => {
        if (state.disposed) return;
        const prior = ops.current();
        if (prior) ops.cleanup(prior);

        const draft = ops.draft(state);
        let owner: Owner | undefined;
        try {
            owner = ops.prepare(draft);
            ops.activate(state, owner);
            state.onDispose(() => ops.cleanup(owner!));
            await ops.precompile(owner, () => ops.force(owner!));
            if (ops.current() !== owner) return;
            ops.publish(owner);
        } catch (cause) {
            if (owner) ops.cleanup(owner);
            else ops.cleanupDraft(draft);
            throw cause;
        }
    };
    const dispose = (state: StateKey): void => {
        const owner = ops.owned(state);
        if (owner) ops.cleanup(owner);
    };
    return { warm, dispose };
}

const cleanedGenerations = new WeakSet<object>();

/** Exact-identity generation cleanup; an old State can never remove a later owner's publications. */
export function cleanupFountainGeneration<
    Buffer,
    Typed,
    DrawEntry extends { name: string },
    Gate,
    Owner extends {
        readonly buffers: readonly Buffer[];
        readonly particlesRaw: Buffer;
        readonly particles: Typed;
        readonly draw: DrawEntry;
        readonly gate: Gate;
    },
>(
    owner: Owner,
    ops: {
        raw: () => Buffer | undefined;
        typed: () => Typed | undefined;
        draw: (name: string) => DrawEntry | undefined;
        gate: () => Gate | undefined;
        deleteRaw: () => void;
        deleteTyped: () => void;
        deleteDraw: (name: string) => void;
        deleteGate: () => void;
        destroy: (buffer: Buffer) => void;
        active: () => Owner | null;
        clearActive: () => void;
    },
): void {
    if (cleanedGenerations.has(owner)) return;
    cleanedGenerations.add(owner);
    if (ops.raw() === owner.particlesRaw) ops.deleteRaw();
    if (ops.typed() === owner.particles) ops.deleteTyped();
    if (ops.draw(owner.draw.name) === owner.draw) ops.deleteDraw(owner.draw.name);
    if (ops.gate() === owner.gate) ops.deleteGate();
    for (const buffer of owner.buffers) ops.destroy(buffer);
    if (ops.active() === owner) ops.clearActive();
}
