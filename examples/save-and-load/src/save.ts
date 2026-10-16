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

export function placement(index: number): [number, number, number] {
    // A seeded permutation visits each cell once; jitter leaves a gap between discs.
    const spot = (index * 37 + 19) % COIN_CAP;
    const jitter = (seed: number) => ((Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
    return [
        -1.68 + (spot % 13) * 0.28 + (jitter(spot + 7) - 0.5) * 0.02,
        -0.5 + COIN_THICKNESS / 2,
        0.65 + Math.floor(spot / 13) * 0.16 + (jitter(spot + 91) - 0.5) * 0.02,
    ];
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
