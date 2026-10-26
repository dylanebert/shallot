import type { Resource, World } from "./world";

/** Paired simulation ownership. Images are local, reusable and independent of subsequent writes. */
export interface Recovery<S = unknown> {
    snapshot(): S;
    restore(state: S): void;
}

/** Recovery for a plain-data resource: snapshot and restore copy it with structuredClone and keep
 * its identity, removing keys the image lacks. Functions and host objects throw at the first
 * snapshot; class instances restore without their prototype. Write a Recovery for those, for
 * partial state or for restore-time work. */
export function recoverResource<T extends object>(
    resource: Resource<T>,
): (world: World) => Recovery<T> {
    return (world) => ({
        snapshot: () => structuredClone(world.resource(resource)),
        restore(state) {
            const live = world.resource(resource) as Record<string, unknown>;
            for (const key of Object.keys(live)) if (!(key in state)) delete live[key];
            Object.assign(live, structuredClone(state));
        },
    });
}

type Participant = { owner: object; snapshot?: () => () => void; refusal?: string };
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
        let failure: { error: unknown } | undefined;
        for (const state of this.#states) {
            try {
                state.restore();
            } catch (error) {
                failure ??= { error };
            }
        }
        if (failure) throw failure.error;
    }
}

/** @internal Ordered simulation owners; validation precedes all capture and restore work. */
export class SnapshotComposition {
    readonly #participants = new Map<string | symbol | object, Participant>();
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

    requireSystem(system: { name?: string }): void {
        this.#participants.set(system, {
            owner: system,
            refusal: `unattributed fixed system ${system.name ?? "(unnamed)"} declares no recovery owner`,
        });
    }

    removeSystem(system: object): void {
        this.#participants.delete(system);
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
                    `World.snapshot: ${participant.refusal ?? `plugin ${String(name)} has fixed systems but declares no recovery`}`,
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
