import { readFileSync } from "node:fs";

// Reads a `sample` call graph and charges each sample to what a thread was doing: for the GPU process's
// WebGPU work queue, the IPC message it was handling and the WebGPU::Metal frame that called into Metal;
// for the web content main thread, the synchronous IPC it was waiting on, or JavaScript.
// Usage: bun diagnostics/first-frame/stacks.ts <file.sample.txt> ...
type Node = { name: string; count: number; children: Node[] };

function parse(file: string): { thread: string; root: Node }[] {
    const lines = readFileSync(file, "utf8").split("\n");
    const start = lines.findIndex((line) => line.startsWith("Call graph:"));
    const end = lines.findIndex((line, i) => i > start && line.startsWith("Total number in stack"));
    const threads: { thread: string; root: Node }[] = [];
    const stack: { depth: number; node: Node }[] = [];
    for (const line of lines.slice(start + 1, end < 0 ? undefined : end)) {
        const match = /^([ +!:|]*)(\d+) (.*)$/.exec(line);
        if (!match) continue;
        const depth = match[1]!.length;
        const node: Node = {
            name: match[3]!.replace(/ +\[0x[0-9a-f]+\]$/, "").replace(/ +\(in [^)]+\) \+ \d+$/, ""),
            count: Number(match[2]),
            children: [],
        };
        while (stack.length && stack[stack.length - 1]!.depth >= depth) stack.pop();
        if (stack.length) stack[stack.length - 1]!.node.children.push(node);
        else threads.push({ thread: node.name, root: node });
        stack.push({ depth, node });
    }
    return threads;
}

/** Each root-to-leaf path with its self count. */
function* paths(node: Node, above: string[] = []): Generator<[string[], number]> {
    const path = [...above, node.name];
    const inner = node.children.reduce((sum, child) => sum + child.count, 0);
    if (node.count > inner) yield [path, node.count - inner];
    for (const child of node.children) yield* paths(child, path);
}

const COMPILE = /newLibraryWithSource|newRenderPipelineState|newComputePipelineState|compileFunctionRequest|MTLCompilerScheduler/;
const IDLE = /semaphore_wait_trap|__psynch_cvwait|mach_msg2_trap|kevent|__workq_kernreturn/;

function gpuCause(path: string[]): string {
    // Deduplicated symbols hide some message handlers; the first WebKit::Remote or wgpu frame names them.
    const message =
        path.map((f) => /Messages::(\w+::\w+)/.exec(f)?.[1]).find(Boolean) ??
        path
            .map((f) => /WebKit::(Remote\w+::\w+)\(/.exec(f)?.[1])
            .find((name) => name && !name.endsWith("::didReceiveStreamMessage")) ??
        path.map((f) => /WebKit::(Remote\w+::\w+)\(/.exec(f)?.[1]).find(Boolean) ??
        path.map((f) => /^(wgpu\w+)\(/.exec(f)?.[1]).find(Boolean);
    // Safari's WebKit has no Metal namespace in its WebGPU frames.
    const metal = path.map((f) => /^WebGPU::(?:Metal::)?([\w:]+)\(/.exec(f)?.[1]).filter(Boolean);
    const compile = path.some((f) => COMPILE.test(f)) ? " [compile]" : "";
    if (!message) return IDLE.test(path[path.length - 1]!) ? "idle" : "other";
    return `${message}${metal.length ? ` > ${metal[metal.length - 1]}` : ""}${compile}`;
}

function webCause(path: string[]): string {
    // Trunk WebKit names the message in sendSync; Safari's names only the proxy method that waits.
    const sync = path.some((f) => f.includes("waitForSyncReply"))
        ? (path.map((f) => /sendSync<Messages::(\w+::\w+)/.exec(f)?.[1]).find(Boolean) ??
          path.map((f) => /WebKit::WebGPU::(Remote\w+Proxy::\w+)\(/.exec(f)?.[1]).filter(Boolean).pop() ??
          "sync IPC")
        : undefined;
    const raf = path.some((f) => f.includes("JSRequestAnimationFrameCallback::invoke"));
    const where = raf ? "rAF callback" : path.some((f) => f.includes("updateRendering")) ? "rendering update" : "other";
    if (sync) return `${where}: wait ${sync}`;
    if (IDLE.test(path[path.length - 1]!)) return `${where}: idle`;
    return `${where}: running`;
}

for (const file of process.argv.slice(2)) {
    const threads = parse(file);
    const gpu = file.includes(".gpu-");
    const pick = threads.filter((t) =>
        gpu ? t.thread.includes("WebGPU work queue") : t.thread.includes("com.apple.main-thread"),
    );
    const totals = new Map<string, number>();
    for (const { root } of pick)
        for (const [path, count] of paths(root)) {
            const cause = gpu ? gpuCause(path) : webCause(path);
            totals.set(cause, (totals.get(cause) ?? 0) + count);
        }
    const sum = [...totals.values()].reduce((a, b) => a + b, 0);
    console.log(`\n${file} (${gpu ? "WebGPU work queue" : "main thread"}, ${sum} samples)`);
    for (const [cause, count] of [...totals].sort((a, b) => b[1] - a[1]).filter(([, c]) => c >= 3))
        console.log(`${String(count).padStart(6)}  ${cause}`);
}
