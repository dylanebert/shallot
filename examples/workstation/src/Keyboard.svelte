<script lang="ts">
    import { midiFreq, noteFreq } from "@dylanebert/shallot";
    import { instrumentRegistry, alloc, assign, gate, setParam, onIdle, free, type AudioState } from "@dylanebert/shallot/audio/core";

    let {
        audio,
        instrumentId,
        octave,
        semitone,
        fine,
        volume,
        values,
    }: {
        audio: AudioState;
        instrumentId: number;
        octave: number;
        semitone: number;
        fine: number;
        volume: number;
        values: Record<string, number>;
    } = $props();

    function findParams(suffix: string): string[] {
        const inst = instrumentRegistry.get(instrumentId);
        if (!inst) return [];
        const result: string[] = [];
        for (const key of inst.paramLayout.keys()) {
            if (key.endsWith(suffix)) result.push(key);
        }
        return result;
    }

    const KEYS = [
        { note: 0, name: "C", bind: "KeyZ" },
        { note: 1, name: "C#", bind: "KeyS" },
        { note: 2, name: "D", bind: "KeyX" },
        { note: 3, name: "D#", bind: "KeyD" },
        { note: 4, name: "E", bind: "KeyC" },
        { note: 5, name: "F", bind: "KeyV" },
        { note: 6, name: "F#", bind: "KeyG" },
        { note: 7, name: "G", bind: "KeyB" },
        { note: 8, name: "G#", bind: "KeyH" },
        { note: 9, name: "A", bind: "KeyN" },
        { note: 10, name: "A#", bind: "KeyJ" },
        { note: 11, name: "B", bind: "KeyM" },
    ];
    const OCTAVE_START = 60;
    const BIND_LABELS: Record<string, string> = {
        KeyZ: "Z", KeyS: "S", KeyX: "X", KeyD: "D",
        KeyC: "C", KeyV: "V", KeyG: "G", KeyB: "B",
        KeyH: "H", KeyN: "N", KeyJ: "J", KeyM: "M",
    };

    let held = $state(new Map<number, number>());

    function isBlack(i: number): boolean {
        return [1, 3, 6, 8, 10].includes(i);
    }

    function scaledVolume(count: number): number {
        const gain = volume * volume;
        return count <= 1 ? gain : gain / Math.sqrt(count);
    }

    function updateAllVolumes() {
        const vps = findParams(".volume");
        if (vps.length === 0) return;
        const vol = scaledVolume(held.size);
        for (const [, slot] of held) {
            for (const vp of vps) {
                setParam(audio, slot, vp, vol, instrumentId);
            }
        }
    }

    function setFreqs(slot: number, noteIdx: number, fps: string[]) {
        const base = midiFreq(OCTAVE_START + noteIdx);
        for (const fp of fps) {
            const node = fp.slice(0, fp.indexOf("."));
            const oOct = values[`${node}.octave`] ?? 0;
            const oSemi = values[`${node}.semitone`] ?? 0;
            const oFine = values[`${node}.fine`] ?? 0;
            const freq = noteFreq(base, octave + oOct, semitone + oSemi, fine + oFine);
            setParam(audio, slot, fp, freq, instrumentId);
        }
    }

    function press(i: number) {
        if (held.has(i)) return;
        const slot = alloc(audio);
        if (slot < 0) return;
        assign(audio, slot, instrumentId);
        const fps = findParams(".frequency");
        const vps = findParams(".volume");
        setFreqs(slot, i, fps);
        held.set(i, slot);
        held = new Map(held);
        for (const vp of vps) setParam(audio, slot, vp, scaledVolume(held.size), instrumentId);
        gate(audio, slot, 1);
        updateAllVolumes();
    }

    function release(i: number) {
        const slot = held.get(i);
        if (slot === undefined) return;
        gate(audio, slot, 0);
        onIdle(audio, slot, () => free(audio, slot));
        held.delete(i);
        held = new Map(held);
        updateAllVolumes();
    }

    $effect(() => {
        const fps = findParams(".frequency");
        if (fps.length === 0) return;
        for (const [i, slot] of held) {
            setFreqs(slot, i, fps);
        }
        updateAllVolumes();
    });

    export function onkeydown(e: KeyboardEvent) {
        if (e.repeat) return;
        const key = KEYS.find((k) => k.bind === e.code);
        if (key) {
            e.preventDefault();
            press(key.note);
        }
    }

    export function onkeyup(e: KeyboardEvent) {
        const key = KEYS.find((k) => k.bind === e.code);
        if (key) {
            e.preventDefault();
            release(key.note);
        }
    }
</script>

<div class="keyboard-section">
    <span class="section-label">KEYBOARD</span>
    <div class="keys">
        {#each KEYS as key}
            {@const black = isBlack(key.note)}
            <button
                class="key"
                class:black
                class:active={held.has(key.note)}
                onpointerdown={() => press(key.note)}
                onpointerup={() => release(key.note)}
                onpointerleave={() => release(key.note)}
            >
                <span class="key-label">{key.name}</span>
                <span class="key-bind">{BIND_LABELS[key.bind]}</span>
            </button>
        {/each}
    </div>
</div>

<style>
    .keyboard-section {
        display: flex;
        flex-direction: column;
        gap: 6px;
        height: 100%;
        padding: 10px 16px 12px;
        max-width: 560px;
        width: 100%;
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .keys {
        display: flex;
        gap: 2px;
        flex: 1;
        min-height: 56px;
        max-height: 80px;
    }

    .key {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        gap: 2px;
        padding-bottom: 4px;
        border: 1px solid var(--border);
        border-radius: 0 0 4px 4px;
        cursor: pointer;
        user-select: none;
        touch-action: none;
        background: var(--surface-2);
        color: var(--text-muted);
        font-size: 9px;
        font-family: "JetBrains Mono", monospace;
        transition: background-color 80ms var(--ease-out);
    }

    .key.black {
        background: var(--bg);
        flex: 0.7;
        height: 60%;
    }

    .key:hover {
        background: var(--surface-3);
    }

    .key.black:hover {
        background: var(--surface-2);
    }

    .key.active {
        background: var(--accent);
        border-color: var(--accent);
        color: var(--text);
    }

    .key-label, .key-bind {
        pointer-events: none;
    }

    .key-bind {
        font-size: 8px;
        opacity: 0.5;
    }
</style>
