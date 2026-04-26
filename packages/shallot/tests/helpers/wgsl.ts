export interface DeadCodeResult {
    dead: string[];
    reachable: Set<string>;
    defined: Set<string>;
}

export function findDeadFunctions(wgsl: string, entryPoints?: string[]): DeadCodeResult {
    const fnDefs = new Map<string, string>();
    for (const m of wgsl.matchAll(/^fn (\w+)\s*\(/gm)) {
        const name = m[1];
        const start = m.index!;
        let pos = wgsl.indexOf("{", start);
        if (pos === -1) continue;
        let braceDepth = 1;
        pos++;
        while (braceDepth > 0 && pos < wgsl.length) {
            if (wgsl[pos] === "{") braceDepth++;
            else if (wgsl[pos] === "}") braceDepth--;
            pos++;
        }
        fnDefs.set(name, wgsl.slice(start, pos));
    }

    const entries =
        entryPoints ??
        [...wgsl.matchAll(/@(?:compute|vertex|fragment)[\s\S]*?fn (\w+)\s*\(/g)].map((m) => m[1]);

    const reachable = new Set<string>();
    const queue = [...entries];

    while (queue.length > 0) {
        const name = queue.pop()!;
        if (reachable.has(name)) continue;
        reachable.add(name);

        const body = fnDefs.get(name);
        if (!body) continue;

        for (const [calledName] of fnDefs) {
            if (calledName === name) continue;
            if (reachable.has(calledName)) continue;
            const pattern = new RegExp(`\\b${calledName}\\s*\\(`, "g");
            if (pattern.test(body)) {
                queue.push(calledName);
            }
        }
    }

    const dead = [...fnDefs.keys()].filter((name) => !reachable.has(name));
    return { dead, reachable, defined: new Set(fnDefs.keys()) };
}
