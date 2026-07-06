import type { Plugin, System } from "@dylanebert/shallot";
import { box, LinesPlugin, Part, Transform } from "@dylanebert/shallot";
import { start } from "./boot";

// The debug-overlay shape the BVH wireframe viz will lean on: the scene authors a grid of lit cubes; the
// script reads each cube's Transform and draws a per-frame box() wireframe around it, depth-tested against
// the cubes. A cycling "selected" box is brighter + thicker — the immediate API reading the retained scene
// (the cube layout) and drawing over it. Half a cube is half its scale.
const FeedSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },
    update(state) {
        const cubes = [...state.query([Part, Transform])];
        const sel = Math.floor(performance.now() * 0.0008) % cubes.length;
        cubes.forEach((eid, i) => {
            const x = Transform.pos.x.get(eid);
            const y = Transform.pos.y.get(eid);
            const z = Transform.pos.z.get(eid);
            const hx = Transform.scale.x.get(eid) / 2;
            const hy = Transform.scale.y.get(eid) / 2;
            const hz = Transform.scale.z.get(eid) / 2;
            const min: [number, number, number] = [x - hx, y - hy, z - hz];
            const max: [number, number, number] = [x + hx, y + hy, z + hz];
            if (i === sel) box(min, max, 0xffdd44, 3);
            else box(min, max, 0x4488bb, 1.5);
        });
    },
};

const FeedPlugin: Plugin = {
    name: "WireframeFeed",
    systems: [FeedSystem],
    dependencies: [LinesPlugin],
};

await start([FeedPlugin], "../scenes/wireframe.scene");
