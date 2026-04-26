import { describe, test, expect } from "bun:test";
import {
    SCENE_STRUCT_WGSL,
    SKY_STRUCT_WGSL,
    DATA_STRUCT_WGSL,
    SHADOW_STRUCT_WGSL,
} from "../src/standard/render/surface/structs";
import { SCENE_UNIFORM_SIZE, SKY_UNIFORM_SIZE } from "../src/standard/render/scene";
import {
    BVH_NODE_STRUCT_WGSL,
    TREE_NODE_STRUCT_WGSL,
    BLAS_NODE_STRUCT_WGSL,
    BLAS_TRIANGLE_STRUCT_WGSL,
    BVH_NODE_SIZE,
    TREE_NODE_SIZE,
    BLAS_TRIANGLE_SIZE,
    TREE_NODE_STRIDE,
} from "../src/extras/raytracing/bvh/structs";
import { SHADOW_BUFFER_SIZE } from "../src/standard/raster/shadow";
interface WgslField {
    name: string;
    type: string;
}

function splitFields(body: string): string[] {
    const parts: string[] = [];
    let current = "";
    let depth = 0;

    for (const ch of body) {
        if (ch === "<") depth++;
        else if (ch === ">") depth--;

        if ((ch === "," || ch === "\n") && depth === 0) {
            if (current.trim()) parts.push(current.trim());
            current = "";
        } else {
            current += ch;
        }
    }
    if (current.trim()) parts.push(current.trim());

    return parts;
}

function parseWgslStruct(src: string): WgslField[] {
    const match = src.match(/struct\s+\w+\s*\{([^}]*)\}/s);
    if (!match) throw new Error("No struct found in WGSL source");

    const parts = splitFields(match[1]);
    const fields: WgslField[] = [];

    for (const part of parts) {
        const stripped = part.replace(/@\w+(\([^)]*\))?\s*/g, "").trim();
        const fieldMatch = stripped.match(/^(\w+)\s*:\s*(.+)$/);
        if (fieldMatch) {
            fields.push({ name: fieldMatch[1], type: fieldMatch[2].trim() });
        }
    }

    return fields;
}

interface TypeInfo {
    size: number;
    align: number;
}

function wgslTypeInfo(type: string): TypeInfo {
    const t = type
        .replace(/vec2f/g, "vec2<f32>")
        .replace(/vec3f/g, "vec3<f32>")
        .replace(/vec4f/g, "vec4<f32>");

    if (t === "f32" || t === "u32" || t === "i32" || t === "bool") {
        return { size: 4, align: 4 };
    }
    if (t === "vec2<f32>" || t === "vec2<u32>" || t === "vec2<i32>") {
        return { size: 8, align: 8 };
    }
    if (t === "vec3<f32>" || t === "vec3<u32>" || t === "vec3<i32>") {
        return { size: 12, align: 16 };
    }
    if (t === "vec4<f32>" || t === "vec4<u32>" || t === "vec4<i32>") {
        return { size: 16, align: 16 };
    }
    if (t === "mat4x4<f32>") {
        return { size: 64, align: 16 };
    }
    if (t.startsWith("atomic<")) {
        return { size: 4, align: 4 };
    }

    const arrayMatch = t.match(/^array<(.+),\s*(\d+)>$/);
    if (arrayMatch) {
        const elemInfo = wgslTypeInfo(arrayMatch[1].trim());
        const n = parseInt(arrayMatch[2]);
        const stride = Math.ceil(elemInfo.size / elemInfo.align) * elemInfo.align;
        return { size: stride * n, align: elemInfo.align };
    }

    throw new Error(`Unknown WGSL type: ${type}`);
}

interface FieldLayout {
    name: string;
    offset: number;
    size: number;
    align: number;
}

interface StructLayout {
    fields: FieldLayout[];
    totalSize: number;
}

function wgslLayout(fields: WgslField[]): StructLayout {
    let offset = 0;
    let structAlign = 4;
    const result: FieldLayout[] = [];

    for (const field of fields) {
        const info = wgslTypeInfo(field.type);
        structAlign = Math.max(structAlign, info.align);
        offset = Math.ceil(offset / info.align) * info.align;

        result.push({
            name: field.name,
            offset,
            size: info.size,
            align: info.align,
        });

        offset += info.size;
    }

    const totalSize = Math.ceil(offset / structAlign) * structAlign;
    return { fields: result, totalSize };
}

function fieldOffset(layout: StructLayout, name: string): number {
    const field = layout.fields.find((f) => f.name === name);
    if (!field) throw new Error(`Field "${name}" not found in layout`);
    return field.offset;
}

describe("CPU-GPU struct contracts", () => {
    describe("WGSL parser", () => {
        test("parses simple struct", () => {
            const fields = parseWgslStruct(`struct Foo { x: f32, y: u32, }`);
            expect(fields).toEqual([
                { name: "x", type: "f32" },
                { name: "y", type: "u32" },
            ]);
        });

        test("parses multiline struct", () => {
            const fields = parseWgslStruct(
                `struct Bar {\n    a: vec4<f32>,\n    b: mat4x4<f32>,\n}`,
            );
            expect(fields).toEqual([
                { name: "a", type: "vec4<f32>" },
                { name: "b", type: "mat4x4<f32>" },
            ]);
        });

        test("parses comma-separated fields on one line", () => {
            const fields = parseWgslStruct(
                `struct N { minX: f32, minY: f32, minZ: f32, child: u32, }`,
            );
            expect(fields.length).toBe(4);
            expect(fields[0]).toEqual({ name: "minX", type: "f32" });
            expect(fields[3]).toEqual({ name: "child", type: "u32" });
        });

        test("strips annotations", () => {
            const fields = parseWgslStruct(
                `struct V { @location(0) position: vec3<f32>, @builtin(instance_index) instance: u32, }`,
            );
            expect(fields).toEqual([
                { name: "position", type: "vec3<f32>" },
                { name: "instance", type: "u32" },
            ]);
        });

        test("handles array types with inner commas", () => {
            const fields = parseWgslStruct(`struct S { planes: array<vec4<f32>, 6>, }`);
            expect(fields).toEqual([{ name: "planes", type: "array<vec4<f32>, 6>" }]);
        });
    });

    describe("WGSL layout calculator", () => {
        test("f32 fields pack tightly", () => {
            const layout = wgslLayout([
                { name: "a", type: "f32" },
                { name: "b", type: "f32" },
            ]);
            expect(layout.totalSize).toBe(8);
            expect(layout.fields[0].offset).toBe(0);
            expect(layout.fields[1].offset).toBe(4);
        });

        test("vec3 alignment adds padding", () => {
            const layout = wgslLayout([
                { name: "a", type: "f32" },
                { name: "b", type: "vec3<f32>" },
            ]);
            expect(layout.fields[1].offset).toBe(16);
            expect(layout.totalSize).toBe(32);
        });

        test("mat4x4 size and alignment", () => {
            const layout = wgslLayout([{ name: "m", type: "mat4x4<f32>" }]);
            expect(layout.fields[0].size).toBe(64);
            expect(layout.fields[0].align).toBe(16);
            expect(layout.totalSize).toBe(64);
        });

        test("array stride rounds to element alignment", () => {
            const layout = wgslLayout([{ name: "a", type: "array<vec4<f32>, 6>" }]);
            expect(layout.fields[0].size).toBe(96);
            expect(layout.totalSize).toBe(96);
        });
    });

    describe("Scene uniform", () => {
        const fields = parseWgslStruct(SCENE_STRUCT_WGSL);
        const layout = wgslLayout(fields);

        test("total size matches SCENE_UNIFORM_SIZE", () => {
            expect(layout.totalSize).toBe(SCENE_UNIFORM_SIZE);
        });

        test("viewProj at f32 index 0", () => {
            expect(fieldOffset(layout, "viewProj")).toBe(0 * 4);
        });

        test("invViewProj at f32 index 16", () => {
            expect(fieldOffset(layout, "invViewProj")).toBe(16 * 4);
        });

        test("cameraWorld at f32 index 32", () => {
            expect(fieldOffset(layout, "cameraWorld")).toBe(32 * 4);
        });

        test("ambientColor at byte 192 (light packing)", () => {
            expect(fieldOffset(layout, "ambientColor")).toBe(192);
        });

        test("sunDirection at byte 208 (light packing)", () => {
            expect(fieldOffset(layout, "sunDirection")).toBe(208);
        });

        test("sunColor at byte 224 (light packing)", () => {
            expect(fieldOffset(layout, "sunColor")).toBe(224);
        });

        test("clearColor at f32 index 60", () => {
            expect(fieldOffset(layout, "clearColor")).toBe(60 * 4);
        });

        test("cameraMode at f32 index 64", () => {
            expect(fieldOffset(layout, "cameraMode")).toBe(64 * 4);
        });

        test("cameraSize at f32 index 65", () => {
            expect(fieldOffset(layout, "cameraSize")).toBe(65 * 4);
        });

        test("viewport at f32 index 66", () => {
            expect(fieldOffset(layout, "viewport")).toBe(66 * 4);
        });

        test("fov at f32 index 68", () => {
            expect(fieldOffset(layout, "fov")).toBe(68 * 4);
        });

        test("near at f32 index 69", () => {
            expect(fieldOffset(layout, "near")).toBe(69 * 4);
        });

        test("far at f32 index 70", () => {
            expect(fieldOffset(layout, "far")).toBe(70 * 4);
        });

        test("shadowSoftness at f32 index 71", () => {
            expect(fieldOffset(layout, "shadowSoftness")).toBe(71 * 4);
        });

        test("shadowSamples at u32 index 72", () => {
            expect(fieldOffset(layout, "shadowSamples")).toBe(72 * 4);
        });

        test("reflectionEnabled at u32 index 73", () => {
            expect(fieldOffset(layout, "reflectionEnabled")).toBe(73 * 4);
        });

        test("_reserved0 at u32 index 74", () => {
            expect(fieldOffset(layout, "_reserved0")).toBe(74 * 4);
        });

        test("instanceCount at u32 index 75", () => {
            expect(fieldOffset(layout, "instanceCount")).toBe(75 * 4);
        });

        test("time at byte 304", () => {
            expect(fieldOffset(layout, "time")).toBe(304);
        });

        test("exposure at f32 index 80", () => {
            expect(fieldOffset(layout, "exposure")).toBe(80 * 4);
        });

        test("vignetteStrength at f32 index 81", () => {
            expect(fieldOffset(layout, "vignetteStrength")).toBe(81 * 4);
        });

        test("posterizeBands at f32 index 84", () => {
            expect(fieldOffset(layout, "posterizeBands")).toBe(84 * 4);
        });

        test("ditherStrength at f32 index 85", () => {
            expect(fieldOffset(layout, "ditherStrength")).toBe(85 * 4);
        });

        test("tonemapMode at u32 index 86", () => {
            expect(fieldOffset(layout, "tonemapMode")).toBe(86 * 4);
        });

        test("fxaaEnabled at u32 index 87", () => {
            expect(fieldOffset(layout, "fxaaEnabled")).toBe(87 * 4);
        });
    });

    describe("Sky uniform", () => {
        const fields = parseWgslStruct(SKY_STRUCT_WGSL);
        const layout = wgslLayout(fields);

        test("total size matches SKY_UNIFORM_SIZE", () => {
            expect(layout.totalSize).toBe(SKY_UNIFORM_SIZE);
        });

        test("hazeDensity at f32 index 0", () => {
            expect(fieldOffset(layout, "hazeDensity")).toBe(0);
        });

        test("horizonBand at f32 index 1", () => {
            expect(fieldOffset(layout, "horizonBand")).toBe(4);
        });

        test("hazeColor at f32 index 4", () => {
            expect(fieldOffset(layout, "hazeColor")).toBe(16);
        });

        test("skyZenith at f32 index 8", () => {
            expect(fieldOffset(layout, "skyZenith")).toBe(32);
        });

        test("skyHorizon at f32 index 12", () => {
            expect(fieldOffset(layout, "skyHorizon")).toBe(48);
        });

        test("moonParams at f32 index 16", () => {
            expect(fieldOffset(layout, "moonParams")).toBe(64);
        });

        test("moonDirection at f32 index 20", () => {
            expect(fieldOffset(layout, "moonDirection")).toBe(80);
        });

        test("starParams at f32 index 24", () => {
            expect(fieldOffset(layout, "starParams")).toBe(96);
        });

        test("cloudParams at f32 index 28", () => {
            expect(fieldOffset(layout, "cloudParams")).toBe(112);
        });

        test("cloudColor at f32 index 32", () => {
            expect(fieldOffset(layout, "cloudColor")).toBe(128);
        });

        test("sunParams at f32 index 36", () => {
            expect(fieldOffset(layout, "sunParams")).toBe(144);
        });

        test("sunVisualColor at f32 index 40", () => {
            expect(fieldOffset(layout, "sunVisualColor")).toBe(160);
        });
    });

    describe("Data struct", () => {
        const fields = parseWgslStruct(DATA_STRUCT_WGSL);
        const layout = wgslLayout(fields);

        test("total size is 64 bytes per entity", () => {
            expect(layout.totalSize).toBe(64);
        });
    });

    describe("Shadow struct", () => {
        const fields = parseWgslStruct(SHADOW_STRUCT_WGSL);
        const layout = wgslLayout(fields);

        test("total size matches SHADOW_BUFFER_SIZE", () => {
            expect(layout.totalSize).toBe(SHADOW_BUFFER_SIZE);
        });

        test("cascade viewProj offsets match i * 16 packing", () => {
            expect(fieldOffset(layout, "cascade0ViewProj")).toBe(0 * 16 * 4);
            expect(fieldOffset(layout, "cascade1ViewProj")).toBe(1 * 16 * 4);
            expect(fieldOffset(layout, "cascade2ViewProj")).toBe(2 * 16 * 4);
            expect(fieldOffset(layout, "cascade3ViewProj")).toBe(3 * 16 * 4);
        });

        test("cascadeSplits at f32 index 64", () => {
            expect(fieldOffset(layout, "cascadeSplits")).toBe(64 * 4);
        });

        test("cascadeTexelSizes at f32 index 68", () => {
            expect(fieldOffset(layout, "cascadeTexelSizes")).toBe(68 * 4);
        });
    });

    describe("BVH structs", () => {
        test("BVH_NODE size matches BVH_NODE_SIZE", () => {
            const fields = parseWgslStruct(BVH_NODE_STRUCT_WGSL);
            const layout = wgslLayout(fields);
            expect(layout.totalSize).toBe(BVH_NODE_SIZE);
        });

        test("TREE_NODE size matches TREE_NODE_SIZE", () => {
            const fields = parseWgslStruct(TREE_NODE_STRUCT_WGSL);
            const layout = wgslLayout(fields);
            expect(layout.totalSize).toBe(TREE_NODE_SIZE);
        });

        test("TREE_NODE field count matches TREE_NODE_STRIDE", () => {
            const fields = parseWgslStruct(TREE_NODE_STRUCT_WGSL);
            expect(fields.length).toBe(TREE_NODE_STRIDE);
        });

        test("BLAS_NODE size is 32", () => {
            const fields = parseWgslStruct(BLAS_NODE_STRUCT_WGSL);
            const layout = wgslLayout(fields);
            expect(layout.totalSize).toBe(32);
        });

        test("BLAS_TRIANGLE size matches BLAS_TRIANGLE_SIZE", () => {
            const fields = parseWgslStruct(BLAS_TRIANGLE_STRUCT_WGSL);
            const layout = wgslLayout(fields);
            expect(layout.totalSize).toBe(BLAS_TRIANGLE_SIZE);
        });
    });

    describe("Batch slots", () => {
        test("INDIRECT_STRIDE is 5", () => {
            const { INDIRECT_STRIDE } = require("../src/standard/render/batch");
            expect(INDIRECT_STRIDE).toBe(5);
        });
    });
});
