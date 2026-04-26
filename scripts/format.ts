import {
    State,
    parse,
    serialize,
    normalizeAttr,
    TransformsPlugin,
    RenderPlugin,
    RasterPlugin,
    RaytracingPlugin,
    TweenPlugin,
    SkylabPlugin,
    OrbitPlugin,
    PlayerPlugin,
    LinesPlugin,
    ArrowsPlugin,
    TextPlugin,
    type Node,
    type Plugin,
} from "../packages/shallot/src";
import { Glob } from "bun";

const PLUGINS: Plugin[] = [
    TransformsPlugin,
    RenderPlugin,
    RasterPlugin,
    RaytracingPlugin,
    TweenPlugin,
    SkylabPlugin,
    OrbitPlugin,
    PlayerPlugin,
    LinesPlugin,
    ArrowsPlugin,
    TextPlugin,
];

const state = new State();
for (const plugin of PLUGINS) state.register(plugin);

function normalizeNodes(nodes: Node[]) {
    for (const node of nodes) {
        for (const attr of node.attrs) {
            if (!attr.value) continue;
            const normalized = normalizeAttr(attr.name, attr.value);
            if (normalized !== null) {
                attr.value = normalized;
            }
        }
        normalizeNodes(node.children);
    }
}

const glob = new Glob("**/*.scene");
const ignore = ["node_modules", "dist"];

let formatted = 0;
let unchanged = 0;
let errors = 0;

for await (const path of glob.scan({ cwd: process.cwd() })) {
    if (ignore.some((dir) => path.includes(dir))) continue;

    try {
        const content = await Bun.file(path).text();
        const nodes = parse(content);
        normalizeNodes(nodes);
        const output = serialize(nodes) + "\n";

        if (content !== output) {
            await Bun.write(path, output);
            console.log(`formatted: ${path}`);
            formatted++;
        } else {
            unchanged++;
        }
    } catch (e) {
        console.error(`error: ${path}: ${(e as Error).message}`);
        errors++;
    }
}

console.log(`\n${formatted} formatted, ${unchanged} unchanged, ${errors} errors`);
process.exit(errors > 0 ? 1 : 0);
