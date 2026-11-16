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

type Sample = { tick: number; translation: readonly number[]; rotation: readonly number[] };
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
                const correction = world.storage(VisualCorrection);
                const remaining = Math.exp(-20 * world.time.deltaTime);
                for (const eid of world.query(corrections)) {
                    const x = correction.translation.x.get(eid) * remaining;
                    const y = correction.translation.y.get(eid) * remaining;
                    const z = correction.translation.z.get(eid) * remaining;
                    const flip = correction.rotation.w.get(eid) < 0 ? -1 : 1;
                    const qx = correction.rotation.x.get(eid) * remaining * flip;
                    const qy = correction.rotation.y.get(eid) * remaining * flip;
                    const qz = correction.rotation.z.get(eid) * remaining * flip;
                    const qw = 1 - remaining + correction.rotation.w.get(eid) * remaining * flip;
                    const length = Math.hypot(qx, qy, qz, qw);
                    if (Math.hypot(x, y, z) < 1e-5 && Math.hypot(qx, qy, qz) / length < 1e-5) {
                        world.remove(eid, VisualCorrection);
                        continue;
                    }
                    correction.translation.set(eid, x, y, z);
                    correction.rotation.set(
                        eid,
                        qx / length,
                        qy / length,
                        qz / length,
                        qw / length,
                    );
                }
                world.resource(Presentation).client?.draw();
            },
        },
    ],
};

/** One confirmed-state timeline per World. Install InterpolationPlugin. confirm borrows
 * client eids after replication applies, before prediction replays. It samples only enabled
 * InterpolationTargets; exclude also filters current predicted and owned entities in draw,
 * so a display-mode switch takes effect before another confirmation arrives.
 * Delay is one measured confirmation interval plus twice its arrival jitter, in seconds.
 * Missing brackets hold the nearest confirmed pose, never extrapolate. History and offsets
 * are presentation state and remain current across prediction recovery. */
export class ClientInterpolation {
    private _samples = new Map<EntityRef, Sample[]>();
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
        return this._interval + 2 * this._jitter;
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
        const live = new Set<EntityRef>();
        for (const eid of entities) {
            if (
                this.exclude(eid) ||
                !this.world.has(eid, InterpolationTarget) ||
                !this.world.storage(InterpolationTarget).enabled.get(eid) ||
                !this.world.has(eid, GlobalTransform)
            )
                continue;
            const ref = this.world.ref(eid);
            live.add(ref);
            let samples = this._samples.get(ref);
            if (!samples) this._samples.set(ref, (samples = []));
            const pose = this.world.get(eid, GlobalTransform)!;
            samples.push({ tick, translation: pose.translation, rotation: pose.rotation });
        }
        for (const ref of this._samples.keys())
            if (!live.has(ref)) {
                const eid = this.world.resolve(ref);
                if (eid) this.world.remove(eid, VisualCorrection);
                this._samples.delete(ref);
            }
    }
    /** @internal Runs before rendering reads placement. */
    draw(): void {
        const target =
            this._tick + (this.world.time.elapsed - this._arrival - this.delay) / Time.FIXED_DT;
        for (const [ref, samples] of this._samples) {
            const eid = this.world.resolve(ref);
            if (!eid) {
                this._samples.delete(ref);
                continue;
            }
            if (
                this.exclude(eid) ||
                !this.world.has(eid, InterpolationTarget) ||
                !this.world.storage(InterpolationTarget).enabled.get(eid)
            ) {
                this.world.remove(eid, VisualCorrection);
                this._samples.delete(ref);
                continue;
            }
            while (samples.length > 2 && samples[1]!.tick <= target) samples.shift();
            const a = samples[0]!;
            const b = samples[1] ?? a;
            const alpha =
                b.tick === a.tick
                    ? 0
                    : Math.max(0, Math.min(1, (target - a.tick) / (b.tick - a.tick)));
            // Read the same fixed-sample blend as rendering, without our previous offset.
            if (!this.world.has(eid, VisualCorrection)) this.world.add(eid, VisualCorrection);
            const correction = this.world.storage(VisualCorrection);
            correction.translation.set(eid, 0, 0, 0);
            correction.rotation.set(eid, 0, 0, 0, 1);
            readRenderGlobalTransform(this.world, eid, this._pose);
            correction.translation.set(
                eid,
                a.translation[0]! +
                    (b.translation[0]! - a.translation[0]!) * alpha -
                    this._pose[0]!,
                a.translation[1]! +
                    (b.translation[1]! - a.translation[1]!) * alpha -
                    this._pose[1]!,
                a.translation[2]! +
                    (b.translation[2]! - a.translation[2]!) * alpha -
                    this._pose[2]!,
            );
            blend(a.rotation, b.rotation, alpha, this._rotation);
            for (let i = 0; i < 4; i++) this._inverse[i] = this._pose[4 + i]! * (i === 3 ? 1 : -1);
            const rotation = multiply(this._rotation, this._inverse, this._rotation);
            correction.rotation.set(eid, rotation[0]!, rotation[1]!, rotation[2]!, rotation[3]!);
        }
    }
}

function blend(a: readonly number[], b: readonly number[], alpha: number, out: number[]): void {
    let dot = 0;
    for (let i = 0; i < 4; i++) dot += a[i]! * b[i]!;
    const flip = dot < 0 ? -1 : 1;
    let length = 0;
    for (let i = 0; i < 4; i++) {
        out[i] = a[i]! * (1 - alpha) + b[i]! * flip * alpha;
        length += out[i]! * out[i]!;
    }
    for (let i = 0; i < 4; i++) out[i] = out[i]! / Math.sqrt(length);
}

export function multiply(a: readonly number[], b: readonly number[], out: number[] = []): number[] {
    const [x, y, z, w] = a as [number, number, number, number];
    const [u, v, s, t] = b as [number, number, number, number];
    out[0] = w * u + x * t + y * s - z * v;
    out[1] = w * v - x * s + y * t + z * u;
    out[2] = w * s + x * v - y * u + z * t;
    out[3] = w * t - x * u - y * v - z * s;
    return out;
}
