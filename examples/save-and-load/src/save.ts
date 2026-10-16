import { component, entity, Transform, u8, u32, type World } from "@dylanebert/shallot";
import { fields, type ScalarField, type Vector4Field } from "@dylanebert/shallot/ecs";
import { Player } from "./player";

export const Kind = { Camera: 0, Player: 1, Coin: 2, Crate: 3 } as const;
export const Saved = component("Saved", { kind: u8 });
export const Wallet = component("Wallet", { money: u32, carried: entity });
export const persisted = { Saved, Transform, Player, Wallet };
type Values = Record<string, number | number[]>;
export type Save = { version: 2; entities: Record<string, Values>[] };
export type PreviousSave = { version: 1; entities: Record<string, Values>[] };
export const SAVE_KEY = "shallot-save-and-load";

export function encode(world: World): Save {
    const eids = [...world.query([Saved])];
    const ids = new Map(eids.map((eid, index) => [eid, index]));
    return {
        version: 2,
        entities: eids.map((eid) => {
            const record: Record<string, Values> = {};
            for (const [key, component] of Object.entries(persisted)) {
                if (!world.has(eid, component)) continue;
                const storage = world.storage(component) as Record<
                    string,
                    ScalarField | Vector4Field
                >;
                const values: Values = {};
                for (const { name, field } of fields(component)) {
                    const handle = storage[name] as ScalarField | Vector4Field;
                    if (field === entity) {
                        const target = (handle as ScalarField).get(eid);
                        values[name] = target ? (ids.get(target) ?? -1) : -1;
                    } else if (field.lanes === 1) values[name] = (handle as ScalarField).get(eid);
                    else
                        values[name] = Array.from(
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
        entities: record.entities.map((entry) => {
            if (!entry.Wallet) return entry;
            const { coins, ...wallet } = entry.Wallet;
            return { ...entry, Wallet: { ...wallet, money: coins } };
        }),
    };
}

export function decode(world: World, record: Save | PreviousSave): number[] {
    const save = migrate(record);
    const eids = save.entities.map(() => world.create());
    save.entities.forEach((entry, index) => {
        const eid = eids[index];
        for (const [key, component] of Object.entries(persisted)) {
            const values = entry[key];
            if (!values) continue;
            world.add(eid, component);
            const storage = world.storage(component) as Record<string, ScalarField | Vector4Field>;
            for (const { name, field } of fields(component)) {
                const handle = storage[name] as ScalarField | Vector4Field;
                const value = values[name];
                if (field === entity) (handle as ScalarField).set(eid, eids[value as number] ?? 0);
                else if (field.lanes === 1) (handle as ScalarField).set(eid, value as number);
                else {
                    handle.column.set(value as number[], eid * field.lanes);
                    handle.markChanged(eid);
                }
            }
        }
    });
    return eids;
}

export function readSave(storage: Pick<Storage, "getItem">): Save | null {
    const value = storage.getItem(SAVE_KEY);
    return value === null ? null : migrate(JSON.parse(value));
}
