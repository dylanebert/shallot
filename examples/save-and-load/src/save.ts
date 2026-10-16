import { component, f32, Transform, type World } from "@dylanebert/shallot";
import { fields, type ScalarField, type Vector4Field } from "@dylanebert/shallot/ecs";

export const Coin = component("Coin", { height: f32, speed: f32 });
export const persisted = { Coin, Transform };
type Values = Record<string, number | number[]>;
export type Save = { version: 2; entities: Record<string, Values>[] };
export type PreviousSave = { version: 1; count: number };
export const SAVE_KEY = "shallot-save-and-load";

export const COIN_CAP = 104;
export const COIN_DIAMETER = 0.13;
export const COIN_THICKNESS = 0.035;

export function sampleSpots(seed: number): [number, number][] {
    const random = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 2 ** 32;
    };
    const spacing = 0.15;
    const spots: [number, number][] = [[-1.8 + random() * 3.6, 0.65 + random() * 1.15]];
    const active = [0];
    // Bridson: grow the sample from random active points, retiring one after 30 misses.
    while (active.length) {
        const slot = Math.floor(random() * active.length);
        const origin = spots[active[slot]!]!;
        let found = false;
        for (let attempt = 0; attempt < 30; attempt++) {
            const angle = random() * Math.PI * 2;
            const radius = spacing * Math.sqrt(1 + random() * 3);
            const x = origin[0] + Math.cos(angle) * radius;
            const z = origin[1] + Math.sin(angle) * radius;
            if (x < -1.8 || x > 1.8 || z < 0.65 || z > 1.8) continue;
            if (spots.some((p) => Math.hypot(x - p[0], z - p[1]) < spacing)) continue;
            active.push(spots.length);
            spots.push([x, z]);
            found = true;
            break;
        }
        if (!found) active.splice(slot, 1);
    }
    // Sampling grows locally; shuffle so the first few coins span the whole lit area too.
    for (let i = spots.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [spots[i], spots[j]] = [spots[j]!, spots[i]!];
    }
    return spots;
}
const spots = sampleSpots(0x5a1107);
if (spots.length < COIN_CAP) throw new Error("Coin scatter cannot fill the cap");

export function placement(index: number): [number, number, number] {
    const [x, z] = spots[index]!;
    return [x, -0.5 + COIN_THICKNESS / 2, z];
}

export function encode(world: World): Save {
    return {
        version: 2,
        entities: [...world.query([Coin])].map((eid) => {
            const record: Record<string, Values> = {};
            for (const [key, component] of Object.entries(persisted)) {
                const storage = world.storage(component) as Record<
                    string,
                    ScalarField | Vector4Field
                >;
                const values: Values = {};
                for (const { name, field } of fields(component)) {
                    const handle = storage[name]!;
                    values[name] =
                        field.lanes === 1
                            ? (handle as ScalarField).get(eid)
                            : Array.from(
                                  handle.column.slice(eid * field.lanes, (eid + 1) * field.lanes),
                              );
                }
                record[key] = values;
            }
            return record;
        }),
    };
}

export function migrate(record: Save | PreviousSave): Save {
    if (record.version === 2) return record;
    if (record.version !== 1) throw new Error("Unsupported save version");
    return {
        version: 2,
        entities: Array.from({ length: Math.min(COIN_CAP, record.count) }, (_, index) => {
            const translation = placement(index);
            return {
                Coin: { height: translation[1], speed: 0 },
                Transform: {
                    translation: [...translation, 0],
                    rotation: [0, 0, 0, 1],
                    scale: [COIN_DIAMETER, COIN_THICKNESS, COIN_DIAMETER, 0],
                },
            };
        }),
    };
}

export function decode(world: World, record: Save | PreviousSave): number[] {
    const save = migrate(record);
    for (const eid of world.query([Coin])) world.destroy(eid);
    return save.entities.map((entry) => {
        const eid = world.create();
        for (const [key, component] of Object.entries(persisted)) {
            world.add(eid, component);
            const storage = world.storage(component) as Record<string, ScalarField | Vector4Field>;
            for (const { name, field } of fields(component)) {
                const handle = storage[name]!;
                const value = entry[key]![name]!;
                if (field.lanes === 1) (handle as ScalarField).set(eid, value as number);
                else {
                    handle.column.set(value as number[], eid * field.lanes);
                    handle.markChanged(eid);
                }
            }
        }
        return eid;
    });
}

export function readSave(storage: Pick<Storage, "getItem">): Save | null {
    const value = storage.getItem(SAVE_KEY);
    return value === null ? null : migrate(JSON.parse(value));
}
