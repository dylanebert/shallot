/** Paired simulation ownership. Images are local, reusable and independent of subsequent writes. */
export interface Recovery<S = unknown> {
    snapshot(): S;
    restore(state: S): void;
}

type Participant = { owner: object; snapshot?: () => () => void };
const stateless: Recovery<undefined> = { snapshot: () => undefined, restore() {} };

/** Opaque simulation image, valid only in its capturing world and compatible composition. */
export class WorldSnapshot {
    #owner: SnapshotComposition;
    #revision: string;
    #states: { owner: object; restore(): void }[];

    /** @internal */
    constructor(
        owner: SnapshotComposition,
        revision: string,
        states: { owner: object; restore(): void }[],
    ) {
        this.#owner = owner;
        this.#revision = revision;
        this.#states = states;
    }

    /** @internal */
    restore(
        owner: SnapshotComposition,
        revision: string,
        participants: readonly Participant[],
    ): void {
        if (owner !== this.#owner)
            throw new Error("World.restore: snapshot belongs to another world");
        if (revision !== this.#revision)
            throw new Error("World.restore: snapshot has a different component registry");
        if (
            participants.length !== this.#states.length ||
            participants.some((p, i) => p.owner !== this.#states[i].owner)
        )
            throw new Error("World.restore: snapshot has a different recovery composition");
        for (const state of this.#states) state.restore();
    }
}

/** @internal Ordered simulation owners; validation precedes all capture and restore work. */
export class SnapshotComposition {
    readonly #participants = new Map<string | symbol, Participant>();
    readonly #boundary: () => boolean;
    readonly #revision: () => string;
    readonly #prepare: () => void;
    constructor(boundary: () => boolean, revision: () => string, prepare: () => void) {
        this.#boundary = boundary;
        this.#revision = revision;
        this.#prepare = prepare;
    }

    clear(): void {
        this.#participants.clear();
    }

    require(name: string): void {
        if (!this.#participants.has(name)) this.register(name, undefined);
    }

    register<S>(name: string | symbol, recovery: Recovery<S> | "stateless" | undefined): void {
        if (recovery === undefined) {
            this.#participants.set(name, { owner: {} });
            return;
        }
        if (recovery === "stateless") {
            this.register(name, stateless);
            return;
        }
        this.#participants.set(name, {
            owner: recovery,
            snapshot: () => {
                const state = recovery.snapshot();
                return () => recovery.restore(state);
            },
        });
    }

    snapshot(): WorldSnapshot {
        if (this.#boundary()) throw new Error("World.snapshot: refuses inside a step or tick");
        for (const [name, participant] of this.#participants)
            if (!participant.snapshot)
                throw new Error(
                    `World.snapshot: plugin ${String(name)} has fixed systems but declares no recovery`,
                );
        this.#prepare();
        return new WorldSnapshot(
            this,
            this.#revision(),
            [...this.#participants.values()].map((p) => ({
                owner: p.owner,
                restore: p.snapshot!(),
            })),
        );
    }

    restore(snapshot: WorldSnapshot): void {
        if (this.#boundary()) throw new Error("World.restore: refuses inside a step or tick");
        if (!(snapshot instanceof WorldSnapshot))
            throw new Error("World.restore: invalid snapshot");
        snapshot.restore(this, this.#revision(), [...this.#participants.values()]);
    }
}
