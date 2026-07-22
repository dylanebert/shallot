// Open-addressing hash set of packed shape-pair keys, ported from Box3D's table.c.
//
// The set is a pure membership oracle: the engine only ever asks contains / add / remove, and
// nothing iterates it, so slot and probe order are unobservable. The key layout, fmix hash, linear
// probe, growth threshold, and backward-shift deletion are still ported op-for-op — bit-exact by
// construction is the cheapest thing to verify, and it keeps this file a readable mirror of table.c
// for upstream syncs.
//
// The u64 key is stored as two u32 halves (`keyHi`/`keyLo`) instead of a BigUint64Array, whose every
// read mints a fresh BigInt — one per probe iteration. Hashes live in a parallel Uint32Array; hash 0
// is the empty-slot sentinel, matching the C (which also leaves the collision unguarded).

import { roundUpPowerOf2 } from "./bits";

const SHAPE_MASK = (1 << 22) - 1;
const CHILD_MASK = (1 << 20) - 1;

/**
 * The kernel-resident set arrays' backing. When set, `keyHi`/`keyLo`/`hashes` are views over the
 * kernel's broad-phase region (broadcolumns.ts); `growSet` reserves the region + rewrites the views,
 * and `refreshIfStale` re-derives them if a grow/relocation moved the region. Absent (standalone sets,
 * e.g. tests) → the set owns private Uint32Arrays.
 */
export type SetBacking = { growSet(setCap: number): void; refreshIfStale(): void };

export type HashSet = {
    keyHi: Uint32Array;
    keyLo: Uint32Array;
    hashes: Uint32Array;
    capacity: number;
    count: number;
    // Kernel-resident backing (broadcolumns.ts), or null for a standalone private set. Set → the arrays
    // view the kernel region and the region is reserved lazily on first use (sized to initCapacity).
    store: SetBacking | null;
    // The capacity the first resident reservation sizes to; `capacity` starts 0 until then.
    initCapacity: number;
};

/**
 * High 32 bits of the symmetric shape-pair key. The u64 layout is
 * `[63:42] = min(s1,s2)`, `[41:20] = max(s1,s2)`, `[19:0] = child`, so the larger shape index is
 * the only field straddling bit 32.
 *
 * @example
 * const hi = pairKeyHi(3, 7);
 */
export function pairKeyHi(s1: number, s2: number): number {
    const lo = s1 < s2 ? s1 : s2;
    const hi = s1 < s2 ? s2 : s1;
    return (((lo & SHAPE_MASK) << 10) | ((hi & SHAPE_MASK) >>> 12)) >>> 0;
}

/** Low 32 bits of the symmetric shape-pair key: the larger shape index's low 12 bits, then child. */
export function pairKeyLo(s1: number, s2: number, c: number): number {
    const hi = s1 < s2 ? s2 : s1;
    return (((hi & 0xfff) << 20) | (c & CHILD_MASK)) >>> 0;
}

// Box3D's Murmur3 fmix64 constants. Pair keys are built from increasing integers, so a weak hash
// collides badly.
const K1_HI = 0xff51afd7;
const K1_LO = 0xed558ccd;
const K2_HI = 0xc4ceb9fe;
const K2_LO = 0x1a85ec53;

/**
 * Murmur3's fmix64 over a split key, truncated to the low 32 bits (b3KeyHash). `h ^= h >> 33` on a
 * hi/lo pair is structurally free: it is `lo ^= hi >>> 1` with `hi` unchanged. Each `h *= k` is a
 * 64x64 -> low-64 wrapping multiply over 16-bit limbs, unrolled inline so the accumulator stays in
 * locals — the function is pure and allocates nothing.
 *
 * @example
 * const hash = keyHash(pairKeyHi(3, 7), pairKeyLo(3, 7, 0));
 */
export function keyHash(kHi: number, kLo: number): number {
    let hHi = kHi;
    let hLo = (kLo ^ (kHi >>> 1)) >>> 0;

    let a0 = hLo & 0xffff;
    let a1 = hLo >>> 16;
    let p00 = a0 * (K1_LO & 0xffff);
    let p01 = a0 * (K1_LO >>> 16);
    let p10 = a1 * (K1_LO & 0xffff);
    let p11 = a1 * (K1_LO >>> 16);
    let mid = (p00 >>> 16) + (p01 & 0xffff) + (p10 & 0xffff);
    let outLo = (((mid & 0xffff) << 16) | (p00 & 0xffff)) >>> 0;
    let outHi =
        (p11 +
            (p01 >>> 16) +
            (p10 >>> 16) +
            (mid >>> 16) +
            Math.imul(hLo, K1_HI) +
            Math.imul(hHi, K1_LO)) >>>
        0;

    hHi = outHi;
    hLo = (outLo ^ (outHi >>> 1)) >>> 0;

    a0 = hLo & 0xffff;
    a1 = hLo >>> 16;
    p00 = a0 * (K2_LO & 0xffff);
    p01 = a0 * (K2_LO >>> 16);
    p10 = a1 * (K2_LO & 0xffff);
    p11 = a1 * (K2_LO >>> 16);
    mid = (p00 >>> 16) + (p01 & 0xffff) + (p10 & 0xffff);
    outLo = (((mid & 0xffff) << 16) | (p00 & 0xffff)) >>> 0;
    outHi =
        (p11 +
            (p01 >>> 16) +
            (p10 >>> 16) +
            (mid >>> 16) +
            Math.imul(hLo, K2_HI) +
            Math.imul(hHi, K2_LO)) >>>
        0;

    return (outLo ^ (outHi >>> 1)) >>> 0;
}

export function createSet(capacity: number, store: SetBacking | null = null): HashSet {
    const cap = capacity > 16 ? roundUpPowerOf2(capacity) : 16;
    // Resident set: the arrays live in the kernel's broad-phase region, reserved lazily on first use
    // (which sizes it to initCapacity + zeroes the window so a singleton reused across worlds carries no
    // phantom membership). Starts with empty views + capacity 0.
    if (store !== null) {
        return {
            keyHi: new Uint32Array(0),
            keyLo: new Uint32Array(0),
            hashes: new Uint32Array(0),
            capacity: 0,
            count: 0,
            store,
            initCapacity: cap,
        };
    }
    return {
        keyHi: new Uint32Array(cap),
        keyLo: new Uint32Array(cap),
        hashes: new Uint32Array(cap),
        capacity: cap,
        count: 0,
        store: null,
        initCapacity: cap,
    };
}

// Reserve + zero the resident set's arrays on first use. Idempotent no-op once capacity is set — the
// membership state persists across steps, so this never re-clears a live set. Zeroing the window is what
// makes a fresh world safe over the singleton region (a stale non-zero hash would be phantom membership).
export function ensureResident(set: HashSet): void {
    if (set.store === null) return;
    if (set.capacity === 0) {
        set.capacity = set.initCapacity;
        set.store.growSet(set.capacity); // reserves the region + rewrites keyHi/keyLo/hashes views
        set.keyHi.fill(0);
        set.keyLo.fill(0);
        set.hashes.fill(0);
        return;
    }
    // Already reserved — re-derive the views if a grow/relocation moved the region since the last op.
    set.store.refreshIfStale();
}

export function clearSet(set: HashSet): void {
    ensureResident(set);
    set.count = 0;
    set.keyHi.fill(0);
    set.keyLo.fill(0);
    set.hashes.fill(0);
}

function findSlot(set: HashSet, kHi: number, kLo: number, hash: number): number {
    const mask = set.capacity - 1;
    const hashes = set.hashes;
    const keyHi = set.keyHi;
    const keyLo = set.keyLo;

    let index = hash & mask;
    while (hashes[index] !== 0 && (keyHi[index] !== kHi || keyLo[index] !== kLo)) {
        index = (index + 1) & mask;
    }
    return index;
}

function addKeyHaveCapacity(set: HashSet, kHi: number, kLo: number, hash: number): void {
    const index = findSlot(set, kHi, kLo, hash);
    set.keyHi[index] = kHi;
    set.keyLo[index] = kLo;
    set.hashes[index] = hash;
    set.count += 1;
}

function growTable(set: HashSet): void {
    const oldCapacity = set.capacity;

    if (set.store !== null) {
        // Resident: copy the old contents out to JS scratch *before* reserving (the reserve relocates
        // the arrays and rewrites the views), then reserve the larger region, zero the new window, and
        // re-insert. Membership is preserved; slot/probe order is unobservable, so the rehash is free of
        // bit-exact concern (table.c grows and re-probes identically).
        const oldKeyHi = set.keyHi.slice(0, oldCapacity);
        const oldKeyLo = set.keyLo.slice(0, oldCapacity);
        const oldHashes = set.hashes.slice(0, oldCapacity);

        set.count = 0;
        set.capacity = 2 * oldCapacity;
        set.store.growSet(set.capacity); // reserves the region + rewrites keyHi/keyLo/hashes views
        set.keyHi.fill(0);
        set.keyLo.fill(0);
        set.hashes.fill(0);

        for (let i = 0; i < oldCapacity; ++i) {
            if (oldHashes[i] === 0) {
                continue;
            }
            addKeyHaveCapacity(set, oldKeyHi[i], oldKeyLo[i], oldHashes[i]);
        }
        return;
    }

    const oldKeyHi = set.keyHi;
    const oldKeyLo = set.keyLo;
    const oldHashes = set.hashes;

    set.count = 0;
    set.capacity = 2 * oldCapacity;
    set.keyHi = new Uint32Array(set.capacity);
    set.keyLo = new Uint32Array(set.capacity);
    set.hashes = new Uint32Array(set.capacity);

    for (let i = 0; i < oldCapacity; ++i) {
        if (oldHashes[i] === 0) {
            continue;
        }
        addKeyHaveCapacity(set, oldKeyHi[i], oldKeyLo[i], oldHashes[i]);
    }
}

/** Whether the shape pair is in the set (b3ContainsKey). Symmetric in `s1`/`s2`. */
export function containsKey(set: HashSet, s1: number, s2: number, c: number): boolean {
    ensureResident(set);
    const kHi = pairKeyHi(s1, s2);
    const kLo = pairKeyLo(s1, s2, c);
    const index = findSlot(set, kHi, kLo, keyHash(kHi, kLo));
    return set.keyHi[index] === kHi && set.keyLo[index] === kLo;
}

/** Insert a shape pair. Returns true if it was already present (b3AddKey). */
export function addKey(set: HashSet, s1: number, s2: number, c: number): boolean {
    ensureResident(set);
    const kHi = pairKeyHi(s1, s2);
    const kLo = pairKeyLo(s1, s2, c);
    const hash = keyHash(kHi, kLo);

    const index = findSlot(set, kHi, kLo, hash);
    if (set.hashes[index] !== 0) {
        return true;
    }

    // The C grows only after the miss is confirmed, then re-probes the new table — the pre-grow
    // slot index is stale.
    if (2 * set.count >= set.capacity) {
        growTable(set);
    }
    addKeyHaveCapacity(set, kHi, kLo, hash);
    return false;
}

/** Remove a shape pair. Returns true if it was present. Backward-shifts the probe run to stay dense. */
export function removeKey(set: HashSet, s1: number, s2: number, c: number): boolean {
    ensureResident(set);
    const kHi = pairKeyHi(s1, s2);
    const kLo = pairKeyLo(s1, s2, c);

    let i = findSlot(set, kHi, kLo, keyHash(kHi, kLo));
    if (set.hashes[i] === 0) {
        return false;
    }

    set.keyHi[i] = 0;
    set.keyLo[i] = 0;
    set.hashes[i] = 0;
    set.count -= 1;

    const mask = set.capacity - 1;
    let j = i;
    for (;;) {
        j = (j + 1) & mask;
        if (set.hashes[j] === 0) {
            break;
        }

        // k is the ideal slot for the item at j. Keep it if k lies cyclically in (i, j].
        const k = set.hashes[j] & mask;
        if (i <= j) {
            if (i < k && k <= j) {
                continue;
            }
        } else {
            if (i < k || k <= j) {
                continue;
            }
        }

        set.keyHi[i] = set.keyHi[j];
        set.keyLo[i] = set.keyLo[j];
        set.hashes[i] = set.hashes[j];
        set.keyHi[j] = 0;
        set.keyLo[j] = 0;
        set.hashes[j] = 0;
        i = j;
    }

    return true;
}
