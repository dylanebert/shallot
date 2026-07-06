import { beforeEach, describe, expect, test } from "bun:test";
import type { GltfImport } from "@dylanebert/shallot";
import { PartPlugin, TransformsPlugin } from "@dylanebert/shallot";
import { clear, register } from "@dylanebert/shallot/ecs/core";
import type { Node } from "@dylanebert/shallot/editor";
import { groupModels, type ImportFile, mintNodes } from "./import";

const enc = (s: string): ArrayBuffer => new TextEncoder().encode(s).buffer as ArrayBuffer;
const bin = new ArrayBuffer(4);

function file(path: string, bytes: ArrayBuffer = bin): ImportFile {
    return { path, bytes };
}

describe("groupModels", () => {
    test("a .glb stands alone; unreferenced extras are ignored", () => {
        const groups = groupModels([file("model.glb"), file("readme.txt")]);
        expect(groups).toHaveLength(1);
        expect(groups[0].root.path).toBe("model.glb");
        expect(groups[0].sidecars).toEqual([]);
        expect(groups[0].missing).toEqual([]);
    });

    test("a .gltf claims its referenced sidecars from the drop", () => {
        const gltf = enc(
            JSON.stringify({
                buffers: [{ uri: "scene.bin" }],
                images: [{ uri: "textures/wall.png" }, { uri: "data:image/png;base64,AAA" }],
            }),
        );
        const groups = groupModels([
            file("scan/scene.gltf", gltf),
            file("scan/scene.bin"),
            file("scan/textures/wall.png"),
        ]);
        expect(groups).toHaveLength(1);
        expect(groups[0].sidecars.map((s) => s.path)).toEqual([
            "scan/scene.bin",
            "scan/textures/wall.png",
        ]);
        expect(groups[0].missing).toEqual([]);
    });

    test("a referenced file the drop lacks lands in missing, named", () => {
        const gltf = enc(JSON.stringify({ buffers: [{ uri: "scene.bin" }] }));
        const groups = groupModels([file("scene.gltf", gltf)]);
        expect(groups[0].missing).toEqual(["scene.bin"]);
    });

    test("percent-encoded uris match their decoded file names", () => {
        const gltf = enc(JSON.stringify({ buffers: [{ uri: "Box%20With%20Spaces.bin" }] }));
        const groups = groupModels([file("scene.gltf", gltf), file("Box With Spaces.bin")]);
        expect(groups[0].sidecars.map((s) => s.path)).toEqual(["Box With Spaces.bin"]);
        expect(groups[0].missing).toEqual([]);
    });

    test("a drop with no model file yields no groups", () => {
        expect(groupModels([file("readme.txt"), file("wall.png")])).toEqual([]);
    });
});

describe("mintNodes", () => {
    beforeEach(() => {
        clear();
        for (const p of [PartPlugin, TransformsPlugin]) {
            for (const [n, c] of Object.entries(p.components ?? {})) {
                register(n, c, p.traits?.[n]);
            }
        }
    });

    function imp(): GltfImport {
        const mesh = (i: number, color: [number, number, number, number]) => ({
            name: `m.glb#${i}`,
            mesh: i + 3,
            surface: 1,
            material: i,
            color,
            skinned: false,
            textured: true,
            duration: 0,
        });
        return {
            meshes: [mesh(0, [1, 1, 1, 1]), mesh(1, [0.5, 0.25, 0, 1])],
            instances: [
                { handle: 0, pos: [0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1] },
                { handle: 1, pos: [1, 2, 3], rot: [0, 1, 0, 0], scale: [2, 2, 2] },
            ],
        };
    }

    test("one node per placement: named mesh, baked TRS + drop offset, baked color", () => {
        const doc = { nodes: [] as Node[] };
        const nodes = mintNodes(imp(), "m.glb", [5, 0, -1], doc);
        expect(nodes).toHaveLength(2);

        const [a, b] = nodes;
        expect(a.attrs).toEqual([
            { name: "part", value: "mesh: m.glb#0" },
            { name: "transform", value: "pos: 5 0 -1" },
            { name: "color", value: "" }, // baseColorFactor 1,1,1,1 sits at the trait default
        ]);
        expect(b.attrs[0].value).toBe("mesh: m.glb#1");
        expect(b.attrs[1].value).toBe("pos: 6 2 2; rot: 0 1 0 0; scale: 2 2 2");
        expect(b.attrs[2].value).toBe("rgba: 0.5 0.25 0"); // trailing default alpha elides
    });

    test("ids mint from the file stem, deduped against the document and each other", () => {
        const doc = { nodes: [{ id: "m", attrs: [], children: [] }] as Node[] };
        const nodes = mintNodes(imp(), "scan/m.glb", [0, 0, 0], doc);
        expect(nodes.map((n) => n.id)).toEqual(["m-2", "m-3"]);
    });

    test("an asset with no placements gets one node per primitive at the drop point", () => {
        const empty = { ...imp(), instances: [] };
        const nodes = mintNodes(empty, "m.glb", [1, 0, 1], { nodes: [] });
        expect(nodes).toHaveLength(2);
        expect(nodes[0].attrs[1].value).toBe("pos: 1 0 1");
    });
});
