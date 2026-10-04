// Field-accessor workloads for the allocation sampler: a bare World, 64 entities, one row per input.
import { component, f32, u32, vec4, World } from "../../src/engine";

export const ROWS = ["raw", "single-set", "mixed-set", "get-set", "five-field"] as const;
export type Row = (typeof ROWS)[number];

const Probe = component("FieldAccessorProbe", { v: vec4, s: f32, k: u32 });
const N = 64;
let sink = 0;

/** The sampler's control: one known literal per call. */
export function control(): unknown {
    return { sink };
}

export default async function create(input: string) {
    const row = input.trim() as Row;
    const world = new World();
    const eids: number[] = [];
    for (let i = 0; i < N; i++) {
        const eid = world.create();
        world.add(eid, Probe);
        eids.push(eid);
    }
    const p = world.storage(Probe);
    let t = 0;
    const rows: Record<Row, () => void> = {
        raw: () => {
            t++;
            for (let i = 0; i < N; i++) {
                const eid = eids[i];
                const o = eid * 4;
                const c = p.v.column;
                c[o] = c[o] + 1;
                c[o + 1] = c[o + 1] + 0.5;
                c[o + 2] = t;
                c[o + 3] = 0;
                p.v.markChanged(eid);
                p.s.column[eid] = p.s.column[eid] + 0.25;
                p.s.markChanged(eid);
                sink += p.k.column[eid];
            }
            world.clearChanges();
        },
        "single-set": () => {
            t++;
            const d = t * 0.5 + 0.1;
            for (let i = 0; i < N; i++) p.v.set(eids[i], d, d, d, 0);
            world.clearChanges();
        },
        "mixed-set": () => {
            t++;
            const d = t * 0.5 + 0.1;
            for (let i = 0; i < N; i++) {
                p.v.set(eids[i], d, d, d, 0);
                p.s.set(eids[i], d);
            }
            world.clearChanges();
        },
        "get-set": () => {
            t++;
            for (let i = 0; i < N; i++) {
                const eid = eids[i];
                p.v.set(eid, p.v.x.get(eid) + 1, p.v.y.get(eid) + 0.5, t, 0);
                p.s.set(eid, p.s.get(eid) + 0.25);
            }
            world.clearChanges();
        },
        "five-field": () => {
            t++;
            for (let i = 0; i < N; i++) {
                const eid = eids[i];
                p.v.set(eid, p.v.x.get(eid) + 1, p.v.y.get(eid) + 0.5, t, 0);
                p.s.set(eid, p.s.get(eid) + 0.25);
                sink += p.k.get(eid);
            }
            world.clearChanges();
        },
    };
    const step = rows[row];
    if (!step) throw new Error(`field-accessor workload: unknown row "${row}"`);
    return { step, dispose: () => world.dispose() };
}

/** Entities each step touches, for per-entity timing. */
export const ENTITIES = N;
