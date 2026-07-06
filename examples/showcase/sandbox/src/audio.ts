import { type Mirror, play, Sound, type State, type System } from "@dylanebert/shallot";
import { type InstrumentDef, instrument } from "@dylanebert/shallot/audio/core";
import { type PhysicsStep, qRotate } from "@dylanebert/shallot/physics/core";
import { Brick } from "./spawn";

// The synthetic SFX — three modal instruments (no sample assets) + the contact-driven impact system.
// The instruments port from the legacy sandbox: van den Doel & Pai modal synthesis, frequencies in the
// measured clay-brick flexural range, decays ∝ 1/f. Impacts are detected on the CPU by Mirror-scanning
// the solver's persistent contact store for FRESH records (a (a,b,feature) key absent last tick) whose
// solved normal force clears a threshold — the rebuilt physics keeps contacts GPU-side, and Mirror is
// the sanctioned readback (small capacity makes the whole-buffer copy cheap).

interface Mode {
    freq: number;
    amp: number;
    decay: number;
}

// Linear mix-chain: each mix outputs (a + b) / 2, so osc volumes are boosted by inverse chain depth.
function modal(
    modes: Mode[],
    attack: number,
    volume: number,
    pitched: boolean,
    name: string,
): void {
    const nodes: InstrumentDef["nodes"] = {};
    const values: Record<string, number> = {};
    const pitchParams: string[] = [];

    // mode i enters the chain at mix i (mode 0 at mix 1) and halves once per mix it passes through
    const scaling: number[] = new Array(modes.length);
    scaling[modes.length - 1] = 1 / 2;
    for (let i = 1; i < modes.length - 1; i++) scaling[i] = 1 / 2 ** (modes.length - i);
    scaling[0] = 1 / 2 ** (modes.length - 1);
    for (let i = 0; i < modes.length; i++) {
        const m = modes[i];
        nodes[`osc${i}`] = { type: "oscillator" };
        nodes[`env${i}`] = { type: "envelope", input: `osc${i}` };
        values[`osc${i}.frequency`] = m.freq;
        values[`osc${i}.waveform`] = 0;
        values[`osc${i}.volume`] = m.amp / scaling[i];
        values[`env${i}.attack`] = attack;
        values[`env${i}.decay`] = m.decay;
        values[`env${i}.sustain`] = 0;
        values[`env${i}.release`] = 0.02;
        if (pitched) pitchParams.push(`osc${i}.frequency`);
    }

    let prev = "env0";
    for (let i = 1; i < modes.length; i++) {
        nodes[`mix${i}`] = { type: "mix", input: prev, inputB: `env${i}` };
        values[`mix${i}.mix`] = 0.5;
        prev = `mix${i}`;
    }
    nodes.vol = { type: "gain", input: prev };
    values["vol.level"] = volume;

    instrument(
        {
            nodes,
            output: "vol",
            volumeParam: "vol.level",
            pitchParams: pitched ? pitchParams : undefined,
            values,
        },
        name,
    );
}

export function registerInstruments(): void {
    // brick strike — flexural modes ~400-3000 Hz, d ∝ 1/f (η ≈ 0.04). Modal-only reads as a
    // generic inanimate thunk (~50% material recognition without a noise transient) — accepted.
    modal(
        [
            { freq: 330, amp: 1.0, decay: 0.1 },
            { freq: 540, amp: 0.6, decay: 0.05 },
            { freq: 905, amp: 0.35, decay: 0.028 },
            { freq: 1450, amp: 0.18, decay: 0.016 },
            { freq: 2280, amp: 0.08, decay: 0.009 },
        ],
        0.001,
        0.5,
        true,
        "impact",
    );
    // grab — inharmonic ~1 : 1.57 : 2.69, reads physical rather than synth
    modal(
        [
            { freq: 175, amp: 1.0, decay: 0.22 },
            { freq: 275, amp: 0.55, decay: 0.14 },
            { freq: 470, amp: 0.25, decay: 0.08 },
        ],
        0.015,
        0.6,
        false,
        "grab",
    );
    // launch — kick-drum shape, low thump + high tick, all sub-100 ms
    modal(
        [
            { freq: 130, amp: 1.0, decay: 0.09 },
            { freq: 210, amp: 0.5, decay: 0.055 },
            { freq: 380, amp: 0.25, decay: 0.03 },
            { freq: 1600, amp: 0.35, decay: 0.02 },
        ],
        0.0015,
        0.85,
        false,
        "launch",
    );
}

// ── impact detection over the Mirror'd contact store ──

const FORCE_MIN = 15; // N of solved normal force below which a fresh contact is silent (resting ≈ mg/4 ≈ 5)
const FORCE_MAX = 60; // full-volume / full-pitch force
const MAX_EMIT = 4; // impacts per tick
const PAIR_COOLDOWN = 0.15; // s a pair stays silent after clacking
const BODY_COOLDOWN = 0.08; // s a body stays silent after clacking
const PITCH_MIN = -12; // semitones at the softest audible impact
const PITCH_MAX = 0;

interface Wiring {
    step: PhysicsStep;
    contacts: Mirror;
    bodies: Mirror;
}

let wiring: Wiring | null = null;
let lastTick = -1;
let prevKeys = new Set<string>();
const pairCooldown = new Map<string, number>();
const bodyCooldown = new Map<number, number>();

/** arm the impact system with the step + its two Mirrors (born in build); disarm with null on dispose. */
export function armImpacts(w: Wiring | null): void {
    wiring = w;
    lastTick = -1;
    prevKeys = new Set();
    pairCooldown.clear();
    bodyCooldown.clear();
}

interface Hit {
    a: number;
    b: number;
    force: number;
    pos: [number, number, number];
}

export const ImpactSystem: System = {
    name: "impacts",
    group: "simulation",
    update(state: State) {
        if (!wiring) return;
        const snap = wiring.contacts.snapshot;
        const bodySnap = wiring.bodies.snapshot;
        if (!snap || !bodySnap || snap.fixedTick === lastTick) return;
        lastTick = snap.fixedTick;

        const rc = wiring.step.recordCap;
        const f = new Float32Array(snap.bytes);
        const u = new Uint32Array(snap.bytes);
        const bf = new Float32Array(bodySnap.bytes);
        const bcap = wiring.step.eidCap;
        const now = state.time.elapsed;

        // scan the active records: build this tick's key set, collect fresh hard hits per pair
        const keys = new Set<string>();
        const hits = new Map<string, Hit>();
        for (let rec = 0; rec < rc; rec++) {
            const mo = rec * 4; // col 0 (meta)
            if (u[mo] !== 1) continue; // CONSTRAINT_CONTACT, 0 = inactive
            const a = u[mo + 1];
            const b = u[mo + 2];
            const key = `${a},${b},${u[mo + 3]}`;
            keys.add(key);
            if (prevKeys.has(key)) continue;
            const force = -f[(6 * rc + rec) * 4]; // λ normal — ≤ 0 pushing
            if (force < FORCE_MIN) continue;
            if (!state.has(a, Brick) && !state.has(b, Brick)) continue;
            const pair = `${a},${b}`;
            const prev = hits.get(pair);
            if (prev && prev.force >= force) continue;
            // contact point = posA + qRotate(quatA, rA); bricks are boxes (radius 0), so no normal offset
            const po = a * 4; // col 0
            const qo = (1 * bcap + a) * 4;
            const ro = (2 * rc + rec) * 4;
            const [rx, ry, rz] = qRotate(
                bf[qo],
                bf[qo + 1],
                bf[qo + 2],
                bf[qo + 3],
                f[ro],
                f[ro + 1],
                f[ro + 2],
            );
            hits.set(pair, {
                a,
                b,
                force,
                pos: [bf[po] + rx, bf[po + 1] + ry, bf[po + 2] + rz],
            });
        }
        prevKeys = keys;
        if (hits.size === 0) return;

        const sorted = [...hits.values()].sort((x, y) => y.force - x.force);
        let emitted = 0;
        for (const hit of sorted) {
            if (emitted >= MAX_EMIT) break;
            const pair = `${hit.a},${hit.b}`;
            if ((pairCooldown.get(pair) ?? 0) > now) continue;
            if ((bodyCooldown.get(hit.a) ?? 0) > now) continue;
            if ((bodyCooldown.get(hit.b) ?? 0) > now) continue;
            pairCooldown.set(pair, now + PAIR_COOLDOWN);
            bodyCooldown.set(hit.a, now + BODY_COOLDOWN);
            bodyCooldown.set(hit.b, now + BODY_COOLDOWN);

            const t = Math.min(1, (hit.force - FORCE_MIN) / (FORCE_MAX - FORCE_MIN));
            const eid = play(state, "impact", { pos: hit.pos, volume: t });
            if (eid >= 0) Sound.pitch.set(eid, PITCH_MIN + t * (PITCH_MAX - PITCH_MIN));
            emitted++;
        }

        // the cooldown maps only grow while bodies clack — sweep expired entries occasionally
        if (pairCooldown.size > 256) {
            for (const [k, v] of pairCooldown) if (v <= now) pairCooldown.delete(k);
            for (const [k, v] of bodyCooldown) if (v <= now) bodyCooldown.delete(k);
        }
    },
};
