import { Body, BodyType } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import {
    component,
    type EntityRef,
    type Plugin,
    type System,
    u8,
    vec3,
    type World,
    type WorldSnapshot,
} from "../../engine";
import {
    CharacterPlugin,
    getAngularVelocity,
    isAwake,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    setAngularVelocity,
    setAwake,
    setBodyType,
    setLinearVelocity,
    setTransform,
} from "../../standard/physics";
import { ReplicationReceiver, type ReplicationSchema } from ".";
import type { AcceptedIntent, IntentRecord } from "./intent";
import type { Connection } from "./link";

/** Local display selection. ClientPrediction adds it to the local avatar; other entities opt in. */
export const PredictionTarget = component(
    "PredictionTarget",
    { enabled: u8 },
    { defaults: () => ({ enabled: 1 }) },
);
/** Live solver state required by a correction; Body's velocity and awake defaults are spawn-only. */
export const PredictionMotion = component("PredictionMotion", { angularVelocity: vec3, awake: u8 });

const SampleMotionSystem: System = {
    name: "sample-prediction-motion",
    group: "fixed",
    after: [StepPhysicsSystem, ...(CharacterPlugin.systems ?? [])],
    update(world) {
        for (const eid of world.query([Body])) {
            const angular = getAngularVelocity(world, eid);
            if (!angular) continue;
            if (!world.has(eid, PredictionMotion)) world.add(eid, PredictionMotion);
            world
                .storage(PredictionMotion)
                .angularVelocity.set(eid, angular.x, angular.y, angular.z);
            world.storage(PredictionMotion).awake.set(eid, Number(isAwake(world, eid)));
        }
    },
};
export const PredictionPlugin: Plugin = {
    name: "Prediction",
    dependencies: [StandardPhysicsPlugin],
    components: [PredictionTarget, PredictionMotion],
    recovery: "stateless",
    systems: [SampleMotionSystem],
};

/** Presentation deliveries outlive World recovery. The game supplies a stable key within a tick. */
export class FixedOutputs {
    private _delivered = new Map<number, Set<string>>();
    deliver(tick: number, key: string, emit: () => void): void {
        let keys = this._delivered.get(tick);
        if (!keys) this._delivered.set(tick, (keys = new Set()));
        if (keys.has(key)) return;
        keys.add(key);
        emit();
    }
    /** Forget only ticks that can no longer be replayed. */
    discardBefore(tick: number): void {
        for (const old of this._delivered.keys()) if (old < tick) this._delivered.delete(old);
    }
}

interface History {
    snapshot: WorldSnapshot;
    entities: Map<number, number>;
}

/** Whole-World rollback over the same ordered confirmations as replication. Intent values and permissions belong to the game. */
export class ClientPrediction {
    readonly receiver: ReplicationReceiver;
    readonly records = new Map<number, IntentRecord | AcceptedIntent>();
    readonly outputs = new FixedOutputs();
    private _history = new Map<number, History>();
    private _switches: { tick: number; entity: EntityRef; enabled: boolean }[] = [];
    private _remote = new Map<
        EntityRef,
        {
            translation: readonly [number, number, number];
            rotation: readonly [number, number, number, number];
        }
    >();
    private _types = new Map<number, number>();
    private _replaying = false;
    readonly world: World;
    /** Selects the local avatar's client eid after confirmed rows are applied. Ownership is the game's data. */
    readonly avatar: (world: World, entities: ReadonlyMap<number, number>) => number | undefined;
    /** Runs again on replay. Receives the cumulative log through tick, ordered by tick then ID; apply discrete operations only at their tick, or derive cumulative counters from the log. */
    readonly applyIntent: (world: World, tick: number, records: readonly IntentRecord[]) => void;
    readonly observe: (world: World, outputs: FixedOutputs, confirmed: boolean) => void;
    constructor(
        world: World,
        schema: ReplicationSchema,
        connection: Connection,
        avatar: (world: World, entities: ReadonlyMap<number, number>) => number | undefined,
        applyIntent: (world: World, tick: number, records: readonly IntentRecord[]) => void,
        observe: (world: World, outputs: FixedOutputs, confirmed: boolean) => void = () => {},
    ) {
        if (
            schema.includes(Body) &&
            (!schema.includes(GlobalTransform) || !schema.includes(PredictionMotion))
        )
            throw new Error(
                "Body prediction requires GlobalTransform and PredictionMotion in the replication schema",
            );
        this.world = world;
        this.avatar = avatar;
        this.applyIntent = applyIntent;
        this.observe = observe;
        this.receiver = new ReplicationReceiver(
            world,
            schema,
            connection,
            () => {},
            (tick, _rows, intents, apply) => this._confirm(tick, intents, apply),
        );
        for (const eid of world.query([Body]))
            this._types.set(eid, world.storage(Body).type.get(eid));
        this._save();
    }
    accept(records: Iterable<IntentRecord | AcceptedIntent>): void {
        for (const record of records) {
            const previous = this.records.get(record.id);
            if (previous && "lateness" in previous && !("lateness" in record)) continue;
            this.records.set(record.id, record);
        }
    }
    private _save(): void {
        this._history.set(this.world.time.fixedTick, {
            snapshot: this.world.snapshot(),
            entities: new Map(this.receiver.entities),
        });
    }
    private _prepare(): void {
        const tick = this.world.time.fixedTick + 1;
        for (const change of this._switches) {
            if (change.tick > tick) continue;
            const eid = this.world.resolve(change.entity);
            if (!eid) continue;
            if (!this.world.has(eid, PredictionTarget)) this.world.add(eid, PredictionTarget);
            this.world.storage(PredictionTarget).enabled.set(eid, Number(change.enabled));
        }
        for (const eid of this.world.query([Body])) {
            const type = this.world.storage(Body).type.get(eid);
            if (type === BodyType.Static) continue;
            const predicted = this.predicted(eid);
            const desired = predicted ? (this._types.get(eid) ?? type) : BodyType.Kinematic;
            if (type !== desired) this._mode(eid, desired);
        }
        this.applyIntent(
            this.world,
            tick,
            [...this.records.values()]
                .filter((r) => r.tick <= tick)
                .sort((a, b) => a.tick - b.tick || a.id - b.id),
        );
    }
    private _finish(): void {
        for (const [entity, confirmed] of this._remote) {
            const eid = this.world.resolve(entity);
            if (!eid || !this.world.has(eid, Body) || this.predicted(eid)) continue;
            const pose = this.world.get(eid, GlobalTransform)!;
            if (
                !same(pose.translation, confirmed.translation) ||
                !same(pose.rotation, confirmed.rotation)
            )
                setTransform(this.world, eid, confirmed.translation, confirmed.rotation);
        }
        this.observe(this.world, this.outputs, false);
        this._save();
    }
    /** Advances one prediction tick. Observation repeats on replay; FixedOutputs delivery does not. */
    advance(): void {
        this._prepare();
        this.world.tick();
        this._finish();
    }
    /** A mode change preserves pose and velocities. The selection is replayed at the same boundary. */
    switchBody(eid: number, enabled: boolean): void {
        this._switches.push({
            tick: this.world.time.fixedTick + 1,
            entity: this.world.ref(eid),
            enabled,
        });
        if (!this.world.has(eid, PredictionTarget)) this.world.add(eid, PredictionTarget);
        this.world.storage(PredictionTarget).enabled.set(eid, Number(enabled));
        this._mode(eid, enabled ? (this._types.get(eid) ?? BodyType.Dynamic) : BodyType.Kinematic);
    }
    predicted(eid: number): boolean {
        return (
            this.world.has(eid, PredictionTarget) &&
            this.world.storage(PredictionTarget).enabled.get(eid) !== 0
        );
    }
    private _mode(eid: number, type: number): void {
        if (this.world.storage(Body).type.get(eid) === type) return;
        const pose = this.world.get(eid, GlobalTransform);
        const angular = getAngularVelocity(this.world, eid);
        setBodyType(this.world, eid, type);
        if (pose)
            setLinearVelocity(this.world, eid, {
                x: pose.linearVelocity[0]!,
                y: pose.linearVelocity[1]!,
                z: pose.linearVelocity[2]!,
            });
        if (angular) setAngularVelocity(this.world, eid, angular);
    }
    private _confirm(tick: number, intents: readonly AcceptedIntent[], apply: () => void): void {
        if (this._replaying) throw new Error("Prediction cannot receive during replay");
        this.accept(intents);
        const head = Math.max(tick, this.world.time.fixedTick);
        const history = this._history.get(tick);
        if (history) {
            const entities =
                this.receiver.confirmed < 0 ? new Map(this.receiver.entities) : history.entities;
            this.world.restore(history.snapshot);
            this.receiver.entities.clear();
            for (const [id, eid] of entities) this.receiver.entities.set(id, eid);
        } else if (this.receiver.confirmed >= 0 || this.world.time.fixedTick > tick)
            throw new Error(`Missing prediction tick ${tick}`);
        else while (this.world.time.fixedTick < tick) this.world.tick();
        const before = new Map(
            [...this.world.query([Body, GlobalTransform, PredictionMotion])].map((eid) => [
                eid,
                {
                    pose: this.world.get(eid, GlobalTransform)!,
                    motion: this.world.get(eid, PredictionMotion)!,
                    type: this.world.storage(Body).type.get(eid),
                },
            ]),
        );
        apply();
        const avatar = this.avatar(this.world, this.receiver.entities);
        if (avatar !== undefined && !this.world.has(avatar, PredictionTarget))
            this.world.add(avatar, PredictionTarget, { enabled: 1 });
        this._remote.clear();
        for (const eid of this.world.query([Body, GlobalTransform, PredictionMotion])) {
            const pose = this.world.get(eid, GlobalTransform)!;
            const motion = this.world.get(eid, PredictionMotion)!;
            const body = this.world.storage(Body);
            const type = body.type.get(eid);
            if (!getAngularVelocity(this.world, eid)) {
                body.linearVelocity.set(eid, ...pose.linearVelocity);
                body.angularVelocity.set(eid, ...motion.angularVelocity);
                body.isAwake.set(eid, motion.awake);
            }
            this._types.set(eid, type);
            const predicted = this.predicted(eid);
            if (!predicted)
                this._remote.set(this.world.ref(eid), {
                    translation: pose.translation,
                    rotation: pose.rotation,
                });
            const previous = before.get(eid);
            const desired = type === BodyType.Static || predicted ? type : BodyType.Kinematic;
            if (previous?.type !== desired) setBodyType(this.world, eid, desired);
            else this.world.storage(Body).type.set(eid, desired);
            if (
                !same(previous?.pose.translation, pose.translation) ||
                !same(previous?.pose.rotation, pose.rotation)
            )
                setTransform(this.world, eid, pose.translation, pose.rotation);
            if (!same(previous?.pose.linearVelocity, pose.linearVelocity))
                setLinearVelocity(this.world, eid, {
                    x: pose.linearVelocity[0]!,
                    y: pose.linearVelocity[1]!,
                    z: pose.linearVelocity[2]!,
                });
            if (!same(previous?.motion.angularVelocity, motion.angularVelocity))
                setAngularVelocity(this.world, eid, {
                    x: motion.angularVelocity[0]!,
                    y: motion.angularVelocity[1]!,
                    z: motion.angularVelocity[2]!,
                });
            if (previous?.motion.awake !== motion.awake)
                setAwake(this.world, eid, motion.awake !== 0);
        }
        this.observe(this.world, this.outputs, true);
        this._save();
        this._replaying = true;
        try {
            while (this.world.time.fixedTick < head) this.advance();
        } finally {
            this._replaying = false;
        }
        for (const old of this._history.keys()) if (old < tick) this._history.delete(old);
        this.outputs.discardBefore(tick);
        this._switches = this._switches.filter((change) => change.tick > tick);
    }
}

/** Contact identities include childIndex so compound-shape outputs remain distinct. */
export function contactOutputKey(
    kind: string,
    event: { shapeA: EntityRef; shapeB: EntityRef; childIndex: number },
): string {
    return `${kind}:${event.shapeA}:${event.shapeB}:${event.childIndex}`;
}

function same(a: readonly number[] | undefined, b: readonly number[]): boolean {
    return a?.length === b.length && a.every((v, i) => v === b[i]);
}
