import { expect, test } from "bun:test";
import {
    createPacker,
    MAX_POINT_CASTERS,
    type Packer,
    type PointShadowFrame,
    packCasters,
} from "./shadows";

// Each case's `[x, y, size]` tile in pixels per caster and face, pinned as literals: a changed layout moves
// shadow tiles between casters, which neither the frame nor the allocation checks detect. The cases run in
// order through one packer and rect buffer, as updatePointShadows does frame to frame, so a caster set reuses
// storage the previous set wrote.
const cases: {
    name: string;
    side: number;
    casters: [score: number, spot: boolean][];
    count: number;
    tiles: [number, number, number][][];
}[] = [
    {
        name: "one point caster",
        side: 1024,
        casters: [[1, false]],
        count: 1,
        tiles: [
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
        ],
    },
    {
        name: "tied casters keep caster order",
        side: 1024,
        casters: [
            [1, false],
            [1, false],
            [1, false],
        ],
        count: 3,
        tiles: [
            [
                [0, 0, 128],
                [128, 128, 128],
                [0, 128, 128],
                [128, 0, 128],
                [256, 256, 128],
                [384, 384, 128],
            ],
            [
                [256, 384, 128],
                [384, 256, 128],
                [0, 256, 128],
                [128, 384, 128],
                [0, 384, 128],
                [128, 256, 128],
            ],
            [
                [256, 0, 128],
                [384, 128, 128],
                [256, 128, 128],
                [384, 0, 128],
                [512, 512, 128],
                [640, 640, 128],
            ],
        ],
    },
    {
        name: "importance sizes tiles",
        side: 1024,
        casters: [
            [1, false],
            [16, false],
            [4, true],
        ],
        count: 3,
        tiles: [
            [
                [640, 896, 64],
                [704, 960, 64],
                [640, 960, 64],
                [704, 896, 64],
                [512, 896, 64],
                [576, 960, 64],
            ],
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
            [[512, 768, 128]],
        ],
    },
    {
        name: "the same casters reordered",
        side: 1024,
        casters: [
            [4, true],
            [1, false],
            [16, false],
        ],
        count: 3,
        tiles: [
            [[512, 768, 128]],
            [
                [640, 896, 64],
                [704, 960, 64],
                [640, 960, 64],
                [704, 896, 64],
                [512, 896, 64],
                [576, 960, 64],
            ],
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
        ],
    },
    {
        name: "a caster leaves",
        side: 1024,
        casters: [
            [16, false],
            [4, true],
        ],
        count: 2,
        tiles: [
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
            [[512, 768, 128]],
        ],
    },
    {
        name: "casters join",
        side: 1024,
        casters: [
            [16, false],
            [4, true],
            [64, false],
            [0.25, true],
        ],
        count: 4,
        tiles: [
            [
                [512, 768, 128],
                [640, 896, 128],
                [512, 896, 128],
                [640, 768, 128],
                [768, 512, 128],
                [896, 640, 128],
            ],
            [[768, 640, 64]],
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
            [[832, 704, 64]],
        ],
    },
    {
        name: "the smallest atlas",
        side: 256,
        casters: [
            [1, false],
            [1, true],
        ],
        count: 2,
        tiles: [
            [
                [0, 0, 64],
                [64, 64, 64],
                [0, 64, 64],
                [64, 0, 64],
                [128, 128, 64],
                [192, 192, 64],
            ],
            [[128, 192, 64]],
        ],
    },
    {
        name: "a hero and three tied spots",
        side: 512,
        casters: [
            [100, false],
            [1, false],
            [1, true],
            [1, true],
        ],
        count: 4,
        tiles: [
            [
                [0, 0, 128],
                [128, 128, 128],
                [0, 128, 128],
                [128, 0, 128],
                [256, 256, 128],
                [384, 384, 128],
            ],
            [
                [256, 384, 64],
                [320, 448, 64],
                [256, 448, 64],
                [320, 384, 64],
                [384, 256, 64],
                [448, 320, 64],
            ],
            [[384, 320, 64]],
            [[448, 256, 64]],
        ],
    },
    {
        name: "eight mixed casters at 2048",
        side: 2048,
        casters: [
            [9, false],
            [3, true],
            [27, false],
            [1, false],
            [81, true],
            [2, false],
            [2, true],
            [5, false],
        ],
        count: 8,
        tiles: [
            [
                [0, 1536, 256],
                [256, 1792, 256],
                [0, 1792, 256],
                [256, 1536, 256],
                [512, 1024, 256],
                [768, 1280, 256],
            ],
            [[512, 1280, 256]],
            [
                [1024, 1024, 512],
                [1536, 1536, 512],
                [1024, 1536, 512],
                [1536, 1024, 512],
                [0, 1024, 512],
                [512, 1536, 512],
            ],
            [
                [1792, 768, 128],
                [1920, 896, 128],
                [1792, 896, 128],
                [1920, 768, 128],
                [1536, 768, 128],
                [1664, 896, 128],
            ],
            [[0, 0, 1024]],
            [
                [1536, 896, 128],
                [1664, 768, 128],
                [1792, 512, 128],
                [1920, 640, 128],
                [1792, 640, 128],
                [1920, 512, 128],
            ],
            [[1024, 512, 128]],
            [
                [768, 1024, 256],
                [1024, 0, 256],
                [1280, 256, 256],
                [1024, 256, 256],
                [1280, 0, 256],
                [1536, 512, 256],
            ],
        ],
    },
    {
        name: "eight mixed casters at 4096",
        side: 4096,
        casters: [
            [1, true],
            [1000, false],
            [10, false],
            [10, true],
            [100, false],
            [0.5, false],
            [3, true],
            [30, false],
        ],
        count: 8,
        tiles: [
            [[2432, 3968, 64]],
            [
                [0, 0, 1024],
                [1024, 1024, 1024],
                [0, 1024, 1024],
                [1024, 0, 1024],
                [2048, 2048, 1024],
                [3072, 3072, 1024],
            ],
            [
                [2560, 3840, 128],
                [2688, 3968, 128],
                [2560, 3968, 128],
                [2688, 3840, 128],
                [2816, 3584, 128],
                [2944, 3712, 128],
            ],
            [[2816, 3712, 128]],
            [
                [2048, 3072, 256],
                [2304, 3328, 256],
                [2048, 3328, 256],
                [2304, 3072, 256],
                [2560, 3584, 256],
                [2816, 3840, 256],
            ],
            [
                [2496, 4032, 64],
                [2432, 4032, 64],
                [2496, 3968, 64],
                [2304, 3968, 64],
                [2368, 4032, 64],
                [2304, 4032, 64],
            ],
            [[2368, 3968, 64]],
            [
                [2944, 3584, 128],
                [2048, 3584, 128],
                [2176, 3712, 128],
                [2048, 3712, 128],
                [2176, 3584, 128],
                [2304, 3840, 128],
            ],
        ],
    },
    {
        name: "an overfull atlas drops the tail",
        side: 256,
        casters: [
            [1, false],
            [1, false],
            [1, false],
            [1, false],
            [1, false],
            [1, false],
            [1, false],
            [1, false],
        ],
        count: 2,
        tiles: [
            [
                [0, 0, 64],
                [64, 64, 64],
                [0, 64, 64],
                [64, 0, 64],
                [128, 128, 64],
                [192, 192, 64],
            ],
            [
                [128, 192, 64],
                [192, 128, 64],
                [0, 128, 64],
                [64, 192, 64],
                [0, 192, 64],
                [64, 128, 64],
            ],
        ],
    },
    {
        name: "the atlas grows back",
        side: 1024,
        casters: [
            [16, false],
            [4, true],
            [64, false],
            [0.25, true],
        ],
        count: 4,
        tiles: [
            [
                [512, 768, 128],
                [640, 896, 128],
                [512, 896, 128],
                [640, 768, 128],
                [768, 512, 128],
                [896, 640, 128],
            ],
            [[768, 640, 64]],
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
            [[832, 704, 64]],
        ],
    },
    {
        name: "a score range past the drop clamp",
        side: 1024,
        casters: [
            [1000000000.0, false],
            [1e-9, true],
            [1e-9, false],
        ],
        count: 3,
        tiles: [
            [
                [0, 0, 256],
                [256, 256, 256],
                [0, 256, 256],
                [256, 0, 256],
                [512, 512, 256],
                [768, 768, 256],
            ],
            [[512, 768, 64]],
            [
                [576, 832, 64],
                [512, 832, 64],
                [576, 768, 64],
                [640, 896, 64],
                [704, 960, 64],
                [640, 960, 64],
            ],
        ],
    },
];

test("packCasters places each caster's tiles by importance, through churn, atlas sizes and overflow drops", () => {
    let packer: Packer | undefined;
    const sizes = new Int32Array(MAX_POINT_CASTERS);
    const rects = new Float64Array(MAX_POINT_CASTERS * 6 * 4);
    for (const c of cases) {
        if (packer?.side !== c.side) packer = createPacker(c.side, 64);
        const frames = c.casters.map(([score, spot]) => ({ score, spot }) as PointShadowFrame);
        let count = frames.length;
        // the caller's retry: drop the least important caster until the set packs
        while (count > 0 && !packCasters(frames, count, c.side, packer, sizes, rects)) count--;
        const tiles = frames.slice(0, count).map((frame, i) =>
            Array.from({ length: frame.spot ? 1 : 6 }, (_, face) => {
                const at = (i * 6 + face) * 4;
                expect(rects[at + 3]).toBe(rects[at + 2]);
                return [rects[at] * c.side, rects[at + 1] * c.side, rects[at + 2] * c.side];
            }),
        );
        expect({ name: c.name, count, tiles }).toEqual({
            name: c.name,
            count: c.count,
            tiles: c.tiles,
        });
    }
});
