import { component, f32, Transform, type World } from "@dylanebert/shallot";
import { fields, type ScalarField, type Vector4Field } from "@dylanebert/shallot/ecs";

export const Coin = component("Coin", { height: f32, speed: f32 });
export const persisted = { Coin, Transform };
type Values = Record<string, number | number[]>;
export type Save = { version: 2; entities: Record<string, Values>[] };
export type PreviousSave = { version: 1; count: number };
export const SAVE_KEY = "shallot-save-and-load";

export function placement(index: number): [number, number, number] {
    const column = Math.floor(index / 5) % 7;
    const angle = (column * Math.PI * 2) / 6;
    const layer = index % 5;
    return [
        1.15 + (column === 6 ? 0 : Math.cos(angle) * 0.35) - (layer % 2) * 0.075,
        -0.46 + (Math.floor(index / 35) * 5 + (index % 5)) * 0.085,
        (column === 6 ? 0 : Math.sin(angle) * 0.35) + layer * 0.015,
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
        entities: Array.from({ length: record.count }, (_, index) => {
            const translation = placement(index);
            return {
                Coin: { height: translation[1], speed: 0 },
                Transform: {
                    translation: [...translation, 0],
                    rotation: [0, 0, 0, 1],
                    scale: [0.28, 0.08, 0.28, 0],
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
