import type { Pair, Quad, Single, Type, TypedArray } from "./component";

/**
 * Map-backed component field — the CPU storage primitive. Memory is
 * O(live entities), demand-allocated on first write, so a component held by a
 * handful of entities (cameras, players, orbits, singletons) costs nothing for
 * the empty slots. CPU-only — pair it with `slab(type)` for a GPU-mirrored
 * field: same {@link Type} descriptor, same {@link Single}/{@link Pair}/{@link Quad}
 * surface, so consumers, scene parse, and traits don't see the difference.
 * Reads of unset entities return the type's zero value.
 *
 * Per-access cost is a `Map.get` plus (for vectors) one allocated stride-N
 * TypedArray per live entity. That's fine for the CPU side of the engine — no
 * CPU path iterates enough entities per frame to feel it. Per-entity data a GPU
 * pass reads every frame belongs on `slab(type)` instead, where the bytes live
 * in a contiguous buffer and the iteration runs on the GPU.
 *
 * @example
 * const Orbit = {
 *     yaw: sparse(f32),
 *     pan: sparse(vec4),
 * };
 */
export function sparse(type: Type & { readonly lanes: 1 }): Single;
export function sparse(type: Type & { readonly lanes: 2 }): Pair;
export function sparse(type: Type & { readonly lanes: 4 }): Quad;
export function sparse(type: Type): Single | Pair | Quad {
    const { encode, decode } = type;
    const stride = type.lanes;
    const enc = encode ?? identity;
    const dec = decode ?? identity;
    const zero = dec(0);

    if (stride === 1) {
        const map = new Map<number, number>();
        const out: Single = {
            set: (eid: number, v: number) => {
                map.set(eid, enc(v));
            },
            get: (eid: number) => {
                const raw = map.get(eid);
                return raw === undefined ? zero : dec(raw);
            },
            type,
            gpu: null,
        };
        return out;
    }

    const map = new Map<number, TypedArray>();
    const ensure = (eid: number): TypedArray => {
        let arr = map.get(eid);
        if (!arr) {
            arr = new type.ctor(stride);
            map.set(eid, arr);
        }
        return arr;
    };

    const lane = (offset: number): Single => ({
        set: (eid: number, v: number) => {
            ensure(eid)[offset] = enc(v);
        },
        get: (eid: number) => {
            const arr = map.get(eid);
            return arr === undefined ? zero : dec(arr[offset]);
        },
        type,
        gpu: null,
    });

    if (stride === 2) {
        const out: Pair = {
            set: (eid: number, x: number, y: number) => {
                const arr = ensure(eid);
                arr[0] = enc(x);
                arr[1] = enc(y);
            },
            read: (eid: number, dst: Float32Array) => {
                const arr = map.get(eid);
                if (!arr) {
                    dst[0] = zero;
                    dst[1] = zero;
                    return dst;
                }
                dst[0] = dec(arr[0]);
                dst[1] = dec(arr[1]);
                return dst;
            },
            x: lane(0),
            y: lane(1),
            type,
            gpu: null,
        };
        return out;
    }

    const out: Quad = {
        set: (eid: number, x: number, y: number, z: number, w: number) => {
            const arr = ensure(eid);
            arr[0] = enc(x);
            arr[1] = enc(y);
            arr[2] = enc(z);
            arr[3] = enc(w);
        },
        read: (eid: number, dst: Float32Array) => {
            const arr = map.get(eid);
            if (!arr) {
                dst[0] = zero;
                dst[1] = zero;
                dst[2] = zero;
                dst[3] = zero;
                return dst;
            }
            dst[0] = dec(arr[0]);
            dst[1] = dec(arr[1]);
            dst[2] = dec(arr[2]);
            dst[3] = dec(arr[3]);
            return dst;
        },
        x: lane(0),
        y: lane(1),
        z: lane(2),
        w: lane(3),
        type,
        gpu: null,
    };
    return out;
}

function identity(v: number): number {
    return v;
}
