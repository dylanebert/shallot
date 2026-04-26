import { registry } from "../../engine";

export interface Note {
    beat: number;
    note: number;
    velocity: number;
    duration: number;
}

export interface PatternTrack {
    instrument: string;
    notes: Note[];
}

export interface Pattern {
    tracks: Record<string, PatternTrack>;
    length: number;
}

export interface PatternInfo {
    def: Pattern;
    trackOrder: string[];
    version: number;
}

const MAX_PATTERNS = 256;

export const patternRegistry = registry<PatternInfo>(MAX_PATTERNS);

function compilePattern(def: Pattern, version: number): PatternInfo {
    const trackOrder = Object.keys(def.tracks).sort();
    const tracks: Record<string, PatternTrack> = {};
    for (const name of trackOrder) {
        tracks[name] = {
            instrument: def.tracks[name].instrument,
            notes: [...def.tracks[name].notes].sort((a, b) => a.beat - b.beat),
        };
    }
    return { def: { tracks, length: def.length }, trackOrder, version };
}

export function pattern(def: Pattern, name?: string): number {
    const compiled = compilePattern(def, patternRegistry.version + 1);
    const existing = name ? patternRegistry.getByName(name) : undefined;
    if (existing !== undefined) {
        patternRegistry.set(existing, compiled);
        return existing;
    }
    return patternRegistry.add(compiled, name);
}

export function clearPatterns(): void {
    patternRegistry.clear();
}

export function notes(track: PatternTrack, fromBeat: number, toBeat: number): Note[] {
    const arr = track.notes;
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (arr[mid].beat < fromBeat) lo = mid + 1;
        else hi = mid;
    }
    const result: Note[] = [];
    for (let i = lo; i < arr.length; i++) {
        if (arr[i].beat >= toBeat) break;
        result.push(arr[i]);
    }
    return result;
}

export function midiFreq(note: number): number {
    return 440 * 2 ** ((note - 69) / 12);
}

const BASE_FREQ = 523.2511;

export function noteFreq(base: number, octave = 0, semitone = 0, fine = 0): number {
    const freq = base > 0 ? base : BASE_FREQ;
    return freq * 2 ** (octave + semitone / 12 + fine / 1200);
}
