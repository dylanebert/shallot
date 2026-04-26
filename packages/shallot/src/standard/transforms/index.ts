import {
    hierarchy,
    not,
    Wildcard,
    ChildOf,
    traits,
    type Derived,
    type State,
    type System,
    type Plugin,
} from "../../engine";
import { eulerToQuaternion, quaternionToEuler } from "../../engine/utils";
import * as wasm from "./wasm";

interface RotProxy extends Array<number> {
    get(eid: number): number;
    set(eid: number, value: number): void;
}

function rotProxy(axis: "x" | "y" | "z"): RotProxy {
    function getValue(eid: number): number {
        const e = quaternionToEuler(
            Transform.quatX[eid],
            Transform.quatY[eid],
            Transform.quatZ[eid],
            Transform.quatW[eid],
        );
        return e[axis];
    }

    function setValue(eid: number, value: number): void {
        const e = quaternionToEuler(
            Transform.quatX[eid],
            Transform.quatY[eid],
            Transform.quatZ[eid],
            Transform.quatW[eid],
        );
        e[axis] = value;
        const q = eulerToQuaternion(e.x, e.y, e.z);
        Transform.quatX[eid] = q.x;
        Transform.quatY[eid] = q.y;
        Transform.quatZ[eid] = q.z;
        Transform.quatW[eid] = q.w;
    }

    return new Proxy([] as unknown as RotProxy, {
        get(_, prop) {
            if (prop === "get") return getValue;
            if (prop === "set") return setValue;
            const eid = Number(prop);
            if (Number.isNaN(eid)) return undefined;
            return getValue(eid);
        },
        set(_, prop, value) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return false;
            setValue(eid, value);
            return true;
        },
    });
}

export const Transform: {
    readonly posX: Float32Array;
    readonly posY: Float32Array;
    readonly posZ: Float32Array;
    readonly quatX: Float32Array;
    readonly quatY: Float32Array;
    readonly quatZ: Float32Array;
    readonly quatW: Float32Array;
    readonly scaleX: Float32Array;
    readonly scaleY: Float32Array;
    readonly scaleZ: Float32Array;
    readonly rotX: RotProxy;
    readonly rotY: RotProxy;
    readonly rotZ: RotProxy;
} = {
    get posX() {
        wasm.sync();
        return wasm.posX;
    },
    get posY() {
        wasm.sync();
        return wasm.posY;
    },
    get posZ() {
        wasm.sync();
        return wasm.posZ;
    },
    get quatX() {
        wasm.sync();
        return wasm.quatX;
    },
    get quatY() {
        wasm.sync();
        return wasm.quatY;
    },
    get quatZ() {
        wasm.sync();
        return wasm.quatZ;
    },
    get quatW() {
        wasm.sync();
        return wasm.quatW;
    },
    get scaleX() {
        wasm.sync();
        return wasm.scaleX;
    },
    get scaleY() {
        wasm.sync();
        return wasm.scaleY;
    },
    get scaleZ() {
        wasm.sync();
        return wasm.scaleZ;
    },
    rotX: rotProxy("x"),
    rotY: rotProxy("y"),
    rotZ: rotProxy("z"),
};

export const WorldTransform: { readonly data: Float32Array } = {
    get data() {
        wasm.sync();
        return wasm.matrices;
    },
};

function rotDerived(axis: "x" | "y" | "z"): Derived {
    return {
        get(parsed) {
            return quaternionToEuler(
                parsed.quatX ?? 0,
                parsed.quatY ?? 0,
                parsed.quatZ ?? 0,
                parsed.quatW ?? 1,
            )[axis];
        },
        set(value, parsed) {
            const e = quaternionToEuler(
                parsed.quatX ?? 0,
                parsed.quatY ?? 0,
                parsed.quatZ ?? 0,
                parsed.quatW ?? 1,
            );
            e[axis] = value;
            const q = eulerToQuaternion(e.x, e.y, e.z);
            return { quatX: q.x, quatY: q.y, quatZ: q.z, quatW: q.w };
        },
    };
}

traits(Transform, {
    defaults: () => ({
        posX: 0,
        posY: 0,
        posZ: 0,
        quatX: 0,
        quatY: 0,
        quatZ: 0,
        quatW: 1,
        scaleX: 1,
        scaleY: 1,
        scaleZ: 1,
    }),
    annotations: {
        derived: { rotX: rotDerived("x"), rotY: rotDerived("y"), rotZ: rotDerived("z") },
    },
});

const TransformSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },
    last: true,

    update(state: State) {
        wasm.sync();

        for (const eid of state.query([Transform, not(WorldTransform)])) {
            state.addComponent(eid, WorldTransform);
        }

        for (const eid of state.query([not(Transform), WorldTransform])) {
            state.removeComponent(eid, WorldTransform);
        }

        let count = 0;

        for (const eid of state.query([Transform, not(ChildOf.relation(Wildcard))])) {
            wasm.indices[count] = eid;
            wasm.parents[count] = wasm.NoParent;
            count++;
        }

        for (const eid of state.query([
            Transform,
            ChildOf.relation(Wildcard),
            hierarchy(ChildOf.relation),
        ])) {
            wasm.indices[count] = eid;
            wasm.parents[count] = state.getRelationTargets(eid, ChildOf)[0];
            count++;
        }

        wasm.compute(count);
    },
};

export const TransformsPlugin: Plugin = {
    name: "Transforms",
    systems: [TransformSystem],
    components: { Transform },
    async initialize(_state, onProgress) {
        await wasm.init();
        onProgress?.(1);
    },
};
