import tgpu, { type TgpuBindGroupLayout } from "typegpu";
import type { AnyWgslStruct } from "typegpu/data";
import * as d from "typegpu/data";
import type { Component, GpuTable, World } from "../../engine";
import { laneLayout } from "../../engine/ecs";

const Presence = d.struct({ present: d.u32 });

/** CPU component fields consumed by materials. Declare matching field names and GPU types;
 * list the declaration in a material's instanceData before warm, or in materialLayout's
 * options (extensionLayout for extensions) when supplying an explicit layout. All declarations in a world
 * share one row-following record and one storage binding. read takes the material's own fallback;
 * present tests membership. GPU producer instances are absent, regardless of their row or eid.
 * Materials use these accessors, never the record's offsets or buffer directly.
 * Omit fields to declare presence alone. */
export function instanceData<P extends AnyWgslStruct = typeof Presence>(
    component: Component,
    fields?: P,
) {
    for (const [name, field] of Object.entries(fields?.propTypes ?? {})) {
        if (!laneLayout(field as d.BaseData))
            throw new Error(
                `Instance data field "${name}" requires an f32, i32 or u32 scalar or vector`,
            );
    }
    const schema = fields ?? (Presence as unknown as P);
    const readFn = tgpu.fn(
        [d.u32, schema],
        schema,
    )((_row, fallback) => {
        "use gpu";
        return fallback;
    });
    const presentFn = tgpu.fn(
        [d.u32],
        d.bool,
    )((_row) => {
        "use gpu";
        return false;
    });
    const read = tgpu.slot<typeof readFn>();
    const present = tgpu.slot<typeof presentFn>();
    return {
        component,
        fields,
        /** @internal World-specific shader substitutions. */
        readSlot: read,
        /** @internal */
        presentSlot: present,
        read: (ctx: { instanceRow: number }, fallback: d.InferGPU<P>): d.InferGPU<P> => {
            "use gpu";
            return read.$(ctx.instanceRow, fallback);
        },
        present: (ctx: { instanceRow: number }): boolean => {
            "use gpu";
            return present.$(ctx.instanceRow);
        },
    };
}

/** A component declaration shared by material definitions; storage belongs to each world. */
export type InstanceData = ReturnType<typeof instanceData<any>>;

/** @internal Fill declaration slots with world-local record offsets during pipeline preparation. */
export function instanceDataRoot(
    world: World,
    root: ReturnType<World["gpu"]["root"]["with"]>,
    layout: TgpuBindGroupLayout<any>,
    declarations: readonly InstanceData[] = [],
    rowSource: GpuTable,
) {
    if (!declarations.length) return root;
    const table = instanceDataTable(world, rowSource);
    const state = world.resource(stateKey);
    const record = table.record;
    const stride = table.rowBytes / 4;
    const offset = (name: string) => d.memoryLayoutOf(record, (row: any) => row[name]).offset / 4;
    for (const declaration of declarations) {
        const index = state.declarations.indexOf(declaration);
        const presence = tgpu
            .fn(
                [d.u32],
                d.bool,
            )(`(row: u32) -> bool {
            if (row == 0xFFFFFFFFu) { return false; }
            return bound.instanceData[row * ${stride}u + ${offset(`p${index}`)}u] != 0u;
        }`)
            .$uses({ bound: layout.$ });
        root = root.with(declaration.presentSlot, presence);
        if (!declaration.fields) continue;
        const values = Object.entries(declaration.fields.propTypes).map(([name, field]) => {
            const { format, lanes } = laneLayout(field as d.BaseData)!;
            const word = offset(`c${index}_${name}`);
            const scalar = `${format}32`;
            const parts = Array.from({ length: lanes }, (_, lane) => {
                const value = `bound.instanceData[row * ${stride}u + ${word + lane}u]`;
                return scalar === "u32" ? value : `bitcast<${scalar}>(${value})`;
            });
            return lanes === 1 ? parts[0] : `vec${lanes}<${scalar}>(${parts.join(", ")})`;
        });
        const read = tgpu
            .fn(
                [d.u32, declaration.fields],
                declaration.fields,
            )(`(row: u32, fallback: Fields) -> Fields {
            if (!present(row)) { return fallback; }
            return Fields(${values.join(", ")});
        }`)
            .$uses({ bound: layout.$, Fields: declaration.fields, present: presence });
        root = root.with(declaration.readSlot, read);
    }
    return root;
}

const stateKey = {
    create: () => ({
        declarations: [] as InstanceData[],
        table: undefined as GpuTable | undefined,
    }),
};

/** @internal */
export function registerInstanceData(world: World, declarations: readonly InstanceData[] = []) {
    const state = world.resource(stateKey);
    for (const declaration of declarations) {
        if (state.declarations.includes(declaration)) continue;
        if (state.table) throw new Error("Instance data declarations must register before warm");
        state.declarations.push(declaration);
    }
}

/** @internal */
export function instanceDataTable(world: World, rowSource: GpuTable): GpuTable {
    const state = world.resource(stateKey);
    if (state.table) return state.table;
    const fields: Record<string, d.AnyWgslData> = {};
    for (const [index, declaration] of state.declarations.entries()) {
        fields[`p${index}`] = d.u32;
        for (const [name, field] of Object.entries(declaration.fields?.propTypes ?? {}))
            fields[`c${index}_${name}`] = field as d.AnyWgslData;
    }
    const schema = d.struct(fields);
    const table = world.table("material-instance-data", schema, {
        rowSource,
    });
    for (const [index, declaration] of state.declarations.entries()) {
        table.bindPresence(declaration.component, `p${index}`);
        table.bindFields(
            declaration.component,
            Object.fromEntries(
                Object.keys(declaration.fields?.propTypes ?? {}).map((name) => [
                    `c${index}_${name}`,
                    name,
                ]),
            ),
        );
    }
    state.table = table;
    return table;
}
