import {
    PrepareViewsSystem,
    RenderTransformHistoryPlugin,
    readRenderGlobalTransform,
    VisualCorrection,
} from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import { component, type EntityRef, type Plugin, Time, u8, type World } from "../../engine";

/** Local display selection beside PredictionTarget. Translation and rotation interpolate;
 * scale always uses the latest confirmed value. ClientPrediction selects non-predicted,
 * non-owned entities by default. Set enabled to zero to opt out. */
export const InterpolationTarget = component(
    "InterpolationTarget",
    { enabled: u8 },
    { presentation: true, defaults: () => ({ enabled: 1 }) },
);

const Capacity = 128;
const Stride = 8; // tick, translation xyz, rotation xyzw
interface History {
    ref: EntityRef;
    samples: Float64Array;
    start: number;
    count: number;
    seen: number;
}
const Presentation = { create: (): { client?: ClientInterpolation } => ({}) };
const corrections = [VisualCorrection];

/** Confirmed-state display and prediction-error decay, with no simulation writes. */
export const InterpolationPlugin: Plugin = {
    name: "Interpolation",
    dependencies: [RenderTransformHistoryPlugin],
    components: [InterpolationTarget, VisualCorrection],
    recovery: "stateless",
    systems: [
        {
            name: "interpolate-confirmations",
            group: "draw",
            first: true,
            before: [PrepareViewsSystem],
            update(world) {
                const client = world.resource(Presentation).client;
                const correction = world.storage(VisualCorrection);
                const remaining = Math.exp(-20 * world.time.deltaTime);
                for (const eid of world.query(corrections)) {
                    if (client?.has(eid)) continue;
                    const x = correction.translation.x.get(eid) * remaining;
                    const y = correction.translation.y.get(eid) * remaining;
                    const z = correction.translation.z.get(eid) * remaining;
                    const flip = correction.rotation.w.get(eid) < 0 ? -1 : 1;
                    const qx = correction.rotation.x.get(eid) * remaining * flip;
                    const qy = correction.rotation.y.get(eid) * remaining * flip;
                    const qz = correction.rotation.z.get(eid) * remaining * flip;
                    const qw = 1 - remaining + correction.rotation.w.get(eid) * remaining * flip;
                    const lengthSquared = qx * qx + qy * qy + qz * qz + qw * qw;
                    if (
                        x * x + y * y + z * z < 1e-10 &&
                        (qx * qx + qy * qy + qz * qz) / lengthSquared < 1e-10
                    ) {
                        world.remove(eid, VisualCorrection);
                        continue;
                    }
                    const length = Math.sqrt(lengthSquared);
                    correction.translation.set(eid, x, y, z);
                    correction.rotation.set(
                        eid,
                        qx / length,
                        qy / length,
                        qz / length,
                        qw / length,
                    );
                }
                client?.draw();
            },
        },
    ],
};

/** One confirmed-state timeline per World. Install InterpolationPlugin. confirm borrows
 * client eids after replication applies, before prediction replays. It samples only enabled
 * InterpolationTargets; exclude also filters current predicted and owned entities in draw,
 * so a display-mode switch takes effect before another confirmation arrives.
 * Delay is one measured confirmation interval plus twice its arrival jitter, in seconds,
 * capped at 126 fixed ticks. Each entity retains at most 128 samples, pruning behind the
 * delayed display tick. Missing brackets hold the nearest confirmed pose, never extrapolate.
 * History and offsets are presentation state and remain current across prediction recovery. */
export class ClientInterpolation {
    private _samples = new Map<number, History>();
    private _tick = -1;
    private _arrival = 0;
    private _interval = Time.FIXED_DT;
    private _jitter = 0;
    private _pose = new Float32Array(12);
    private _rotation = [0, 0, 0, 1];
    private _inverse = [0, 0, 0, 1];
    readonly world: World;
    readonly exclude: (eid: number) => boolean;
    constructor(world: World, exclude: (eid: number) => boolean = () => false) {
        this.world = world;
        this.exclude = exclude;
        const presentation = world.resource(Presentation);
        if (presentation.client) throw new Error("Only one ClientInterpolation may drive a World");
        presentation.client = this;
    }
    get delay(): number {
        return Math.min((Capacity - 2) * Time.FIXED_DT, this._interval + 2 * this._jitter);
    }
    /** @internal Whether this timeline owns an entity's visual offset. */
    has(eid: number): boolean {
        const history = this._samples.get(eid);
        return history !== undefined && this.world.resolve(history.ref) === eid;
    }
    confirm(tick: number, entities: Iterable<number>): void {
        const now = this.world.time.elapsed;
        if (tick <= this._tick) throw new Error("Interpolation confirmations must advance");
        if (this._tick >= 0) {
            const interval = (tick - this._tick) * Time.FIXED_DT;
            this._interval += (interval - this._interval) * 0.125;
            this._jitter += (Math.abs(now - this._arrival - interval) - this._jitter) * 0.125;
        }
        this._tick = tick;
        this._arrival = now;
        const target = tick - this.delay / Time.FIXED_DT;
        const pose = this.world.storage(GlobalTransform);
        for (const eid of entities) {
            if (
                this.exclude(eid) ||
                !this.world.has(eid, InterpolationTarget) ||
                !this.world.storage(InterpolationTarget).enabled.get(eid) ||
                !this.world.has(eid, GlobalTransform)
            )
                continue;
            let history = this._samples.get(eid);
            if (!history || this.world.resolve(history.ref) !== eid) {
                history = {
                    ref: this.world.ref(eid),
                    samples: new Float64Array(Capacity * Stride),
                    start: 0,
                    count: 0,
                    seen: tick,
                };
                this._samples.set(eid, history);
            }
            history.seen = tick;
            prune(history, target);
            if (history.count === Capacity) {
                history.start = (history.start + 1) % Capacity;
                history.count--;
            }
            const offset = ((history.start + history.count++) % Capacity) * Stride;
            const samples = history.samples;
            samples[offset] = tick;
            samples[offset + 1] = pose.translation.x.get(eid);
            samples[offset + 2] = pose.translation.y.get(eid);
            samples[offset + 3] = pose.translation.z.get(eid);
            samples[offset + 4] = pose.rotation.x.get(eid);
            samples[offset + 5] = pose.rotation.y.get(eid);
            samples[offset + 6] = pose.rotation.z.get(eid);
            samples[offset + 7] = pose.rotation.w.get(eid);
        }
        for (const history of this._samples.values())
            if (history.seen !== tick) {
                const eid = this.world.resolve(history.ref);
                if (eid) this.world.remove(eid, VisualCorrection);
                this._samples.delete(history.ref >>> 0);
            }
    }
    /** @internal Runs before rendering reads placement. */
    draw(): void {
        const target =
            this._tick + (this.world.time.elapsed - this._arrival - this.delay) / Time.FIXED_DT;
        for (const history of this._samples.values()) {
            const eid = this.world.resolve(history.ref);
            if (!eid) {
                this._samples.delete(history.ref >>> 0);
                continue;
            }
            if (
                this.exclude(eid) ||
                !this.world.has(eid, InterpolationTarget) ||
                !this.world.storage(InterpolationTarget).enabled.get(eid)
            ) {
                this.world.remove(eid, VisualCorrection);
                this._samples.delete(eid);
                continue;
            }
            prune(history, target);
            const samples = history.samples;
            const a = history.start * Stride;
            const b = history.count > 1 ? ((history.start + 1) % Capacity) * Stride : a;
            const alpha =
                b === a
                    ? 0
                    : Math.max(
                          0,
                          Math.min(1, (target - samples[a]!) / (samples[b]! - samples[a]!)),
                      );
            // Read the same fixed-sample blend as rendering, without our previous offset.
            if (!this.world.has(eid, VisualCorrection)) this.world.add(eid, VisualCorrection);
            const correction = this.world.storage(VisualCorrection);
            correction.translation.set(eid, 0, 0, 0);
            correction.rotation.set(eid, 0, 0, 0, 1);
            readRenderGlobalTransform(this.world, eid, this._pose);
            correction.translation.set(
                eid,
                samples[a + 1]! + (samples[b + 1]! - samples[a + 1]!) * alpha - this._pose[0]!,
                samples[a + 2]! + (samples[b + 2]! - samples[a + 2]!) * alpha - this._pose[1]!,
                samples[a + 3]! + (samples[b + 3]! - samples[a + 3]!) * alpha - this._pose[2]!,
            );
            // Normalized linear interpolation matches rendering's fixed-sample blend.
            let dot = 0;
            for (let i = 0; i < 4; i++) dot += samples[a + 4 + i]! * samples[b + 4 + i]!;
            const flip = dot < 0 ? -1 : 1;
            let length = 0;
            for (let i = 0; i < 4; i++) {
                const value =
                    samples[a + 4 + i]! * (1 - alpha) + samples[b + 4 + i]! * flip * alpha;
                this._rotation[i] = value;
                length += value * value;
            }
            const inverseLength = 1 / Math.sqrt(length);
            for (let i = 0; i < 4; i++) {
                this._rotation[i] = this._rotation[i]! * inverseLength;
                this._inverse[i] = this._pose[4 + i]! * (i === 3 ? 1 : -1);
            }
            const rotation = multiply(this._rotation, this._inverse, this._rotation);
            correction.rotation.set(eid, rotation[0]!, rotation[1]!, rotation[2]!, rotation[3]!);
        }
    }
}

function prune(history: History, target: number): void {
    while (
        history.count > 2 &&
        history.samples[((history.start + 1) % Capacity) * Stride]! <= target
    ) {
        history.start = (history.start + 1) % Capacity;
        history.count--;
    }
}

export function multiply(a: readonly number[], b: readonly number[], out: number[]): number[] {
    const x = a[0]!,
        y = a[1]!,
        z = a[2]!,
        w = a[3]!;
    const u = b[0]!,
        v = b[1]!,
        s = b[2]!,
        t = b[3]!;
    out[0] = w * u + x * t + y * s - z * v;
    out[1] = w * v - x * s + y * t + z * u;
    out[2] = w * s + x * v - y * u + z * t;
    out[3] = w * t - x * u - y * v - z * s;
    return out;
}
