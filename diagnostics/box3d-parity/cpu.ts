// The main thread's CPU profile over a stretch of steps, summarized per step phase (scenes.ts CPU=<step>).
// Node only: V8's sampling profiler through node:inspector. A sample is charged to the innermost step
// phase on its stack, and its self time is split into kernel (wasm) and TypeScript; each phase lists the inclusive
// time of the functions it calls, five levels deep.
import { Session } from "node:inspector";
import { KERNEL_SHARED_WASM_BASE64 } from "../../src/standard/physics/kernel/kernel.shared.wasm";
import { KERNEL_WASM_BASE64 } from "../../src/standard/physics/kernel/kernel.wasm";

type Frame = { functionName: string; url: string };
type ProfileNode = { id: number; callFrame: Frame; children?: number[] };
type Profile = { nodes: ProfileNode[]; samples: number[]; timeDeltas: number[] };

// The step's phase functions (solver/step.ts); `solve` holds Box3D's constraints, transforms and sleep.
const PHASES = new Map([
    ["updateBroadPhasePairs", "pairs"],
    ["collide", "collide"],
    ["solve", "solve"],
    ["overlapSensors", "sensors"],
]);
const DEPTH = 5;

const session = new Session();

// V8 names a wasm function by its index in the module's function space, as the export section does.
function exportNames(base64: string): Map<number, string> {
    const b = Buffer.from(base64, "base64");
    let at = 8;
    const leb = () => {
        let v = 0,
            shift = 0,
            byte: number;
        do {
            byte = b[at++];
            v += (byte & 0x7f) * 2 ** shift;
            shift += 7;
        } while (byte & 0x80);
        return v;
    };
    const str = () => {
        const n = leb();
        at += n;
        return b.toString("utf8", at - n, at);
    };
    const names = new Map<number, string>();
    while (at < b.length) {
        const id = b[at++],
            size = leb(),
            end = at + size;
        if (id === 7) {
            for (let n = leb(); n > 0; --n) {
                const name = str(),
                    kind = b[at++],
                    index = leb();
                if (kind === 0) names.set(index, name);
            }
        }
        at = end;
    }
    return names;
}

function post<T>(method: string, params?: object): T {
    let out: T | undefined;
    let error: Error | null = null;
    session.post(method, params, (e, result) => {
        error = e;
        out = result as T;
    });
    if (error) throw error;
    if (out === undefined) throw new Error(`${method} answered asynchronously`);
    return out;
}

/** Start sampling every `intervalUs` microseconds. */
export function startCpu(intervalUs = 100): void {
    session.connect();
    post("Profiler.enable");
    post("Profiler.setSamplingInterval", { interval: intervalUs });
    post("Profiler.start");
}

/** Stop sampling and return the summary lines, milliseconds per step over `steps` steps. */
export function stopCpu(steps: number, shared: boolean): string[] {
    const exports = exportNames(shared ? KERNEL_SHARED_WASM_BASE64 : KERNEL_WASM_BASE64);
    const { profile } = post<{ profile: Profile }>("Profiler.stop");
    session.disconnect();
    const byId = new Map(profile.nodes.map((n) => [n.id, n]));
    const parent = new Map<number, number>();
    for (const n of profile.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
    const time = new Map<number, number>();
    for (let i = 0; i < profile.samples.length; ++i)
        time.set(profile.samples[i], (time.get(profile.samples[i]) ?? 0) + profile.timeDeltas[i]);

    type Phase = {
        total: number;
        wasm: number;
        ts: number;
        calls: Map<string, number>;
    };
    const phases = new Map<string, Phase>();
    const name = (id: number) => {
        const fn = byId.get(id)?.callFrame.functionName || "(anonymous)";
        const index = /^wasm-function\[(\d+)\]$/.exec(fn);
        return index ? (exports.get(Number(index[1])) ?? fn) : fn;
    };
    for (const [id, us] of time) {
        const stack: number[] = [];
        for (let n: number | undefined = id; n !== undefined; n = parent.get(n)) stack.push(n);
        const at = stack.findIndex((n) => PHASES.has(name(n)));
        if (at < 0) continue;
        const key = PHASES.get(name(stack[at])) as string;
        let phase = phases.get(key);
        if (!phase) {
            phase = { total: 0, wasm: 0, ts: 0, calls: new Map() };
            phases.set(key, phase);
        }
        const ms = us / 1000 / steps;
        phase.total += ms;
        const self = byId.get(id)?.callFrame as Frame;
        if (self.url.startsWith("wasm") || self.functionName.startsWith("js-to-wasm"))
            phase.wasm += ms;
        else phase.ts += ms;
        // A js-to-wasm wrapper frame names nothing; the export it enters does.
        const below = stack
            .slice(0, at)
            .reverse()
            .map(name)
            .filter((n) => !n.startsWith("js-to-wasm"));
        const path = below.slice(0, DEPTH);
        for (let d = 1; d <= path.length; ++d) {
            const k = path.slice(0, d).join(" > ");
            phase.calls.set(k, (phase.calls.get(k) ?? 0) + ms);
        }
    }
    const lines: string[] = [];
    for (const [key, p] of phases) {
        lines.push(
            `J ${key} total ${p.total.toFixed(3)} wasm ${p.wasm.toFixed(3)} ts ${p.ts.toFixed(3)}`,
        );
        // Depth first, heaviest first, dropping callees under 0.5% of the phase.
        const visit = (prefix: string) => {
            const depth = prefix === "" ? 1 : prefix.split(" > ").length + 1;
            const children = [...p.calls]
                .filter(([k, ms]) => {
                    const parts = k.split(" > ");
                    return (
                        parts.length === depth &&
                        (prefix === "" || k.startsWith(`${prefix} > `)) &&
                        ms >= 0.005 * p.total
                    );
                })
                .sort((a, b) => b[1] - a[1]);
            for (const [k, ms] of children) {
                lines.push(`JC ${key} ${ms.toFixed(3)} ${k}`);
                visit(k);
            }
        };
        visit("");
    }
    return lines;
}
