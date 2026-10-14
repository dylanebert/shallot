import { expect, test } from "bun:test";
import gold from "../../../../crates/physics/tests/nonconvex_build.gold.json";
import { createGrid, createHeightField, createWave } from "./heightfield";
import {
    createBoxMesh,
    createGridMesh,
    createHollowBoxMesh,
    createMesh,
    createTorusMesh,
    createWaveMesh,
} from "./mesh";
import { drawHeightField, drawMesh } from "./readgeometry";

for (const c of gold) {
    test(`nonconvex authoring equals native Box3D: ${c.name}`, () => {
        const a = Array.from(new Float32Array(new Uint32Array(c.args).buffer));
        const v = (i: number) => ({ x: a[i], y: a[i + 1], z: a[i + 2] });
        const f = (words: number[]) => Array.from(new Float32Array(new Uint32Array(words).buffer));
        let actual: unknown;
        switch (c.operation) {
            case 0: {
                const p = f(c.vertices!);
                actual = createMesh({
                    vertices: Array.from({ length: p.length / 3 }, (_, i) => ({
                        x: p[3 * i],
                        y: p[3 * i + 1],
                        z: p[3 * i + 2],
                    })),
                    indices: c.indices!,
                    materialIndices: c.materials!.length ? c.materials : null,
                    weldTolerance: a[0],
                    weldVertices: !!a[1],
                    useMedianSplit: !!a[2],
                    identifyEdges: !!a[3],
                });
                break;
            }
            case 1:
                actual = createBoxMesh(v(0), v(3), !!a[6]);
                break;
            case 2:
                actual = createHollowBoxMesh(v(0), v(3));
                break;
            case 3:
                actual = createGridMesh(a[0], a[1], a[2], a[3], !!a[4]);
                break;
            case 4:
                actual = createWaveMesh(a[0], a[1], a[2], a[3], a[4], a[5]);
                break;
            case 5:
                actual = createTorusMesh(a[0], a[1], a[2], a[3]);
                break;
            case 10:
                actual = createHeightField({
                    countX: a[0],
                    countZ: a[1],
                    scale: v(2),
                    globalMinimumHeight: a[5],
                    globalMaximumHeight: a[6],
                    clockwiseWinding: !!a[7],
                    heights: f(c.heights!),
                    materialIndices: c.materials!.length ? c.materials! : null,
                });
                break;
            case 11:
                actual = createGrid(a[0], a[1], v(2), !!a[5]);
                break;
            case 12:
                actual = createWave(a[0], a[1], v(2), a[5], a[6], !!a[7]);
                break;
            default:
                throw new Error(`unknown authoring operation ${c.operation}`);
        }
        if (c.words === null) {
            expect(actual).toBeNull();
            return;
        }
        const image = new Uint32Array(c.words.length + 3);
        image.set(c.words, 3);
        const view = new DataView(image.buffer);
        const expected = c.operation < 10 ? drawMesh(view, 0).data : drawHeightField(view, 12);
        expect(actual).toEqual(expected);
    });
}
