import { readFileSync } from "node:fs";
import type { World } from "../../engine";
import type { gradingDefaults } from "./color-grading";

// Bevy b3bdd75be. Imports resolve to these verbatim modules; specializations below
// select exactly the flags TonemappingPipeline::specialize supplies.
function source(name: string): string {
    return readFileSync(new URL(`./bevy/${name}.wesl`, import.meta.url), "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/[^\n]*/g, "")
        .replace(/^import [\s\S]*?;\s*/gm, "");
}

// Resolve the two named color_operations imports, not its unrelated color spaces.
function colorOperations(): string {
    const code = source("color_operations");
    return ["rgb_to_hsv", "hsv_to_rgb"]
        .map((name) => {
            const start = code.indexOf(`fn ${name}(`);
            const body = code.indexOf("{", start);
            let end = body + 1;
            let depth = 1;
            while (depth) {
                if (code[end] === "{") depth++;
                if (code[end] === "}") depth--;
                end++;
            }
            return code.slice(start, end);
        })
        .join("\n");
}

function specialize(code: string, flags: Set<string>): string {
    // sample_current_lut uses WESL's unbraced conditional statements.
    code = code
        .replace(/@(if|elif)\(([^)]+)\)\s*(return[^;]+;)/g, "@$1($2) {$3}")
        .replace(/@else\s*(return[^;]+;)/g, "@else {$1}");
    while (code.includes("@if(")) {
        const start = code.indexOf("@if(");
        let cursor = start;
        let selected = "";
        let taken = false;
        for (;;) {
            const match = /^@(if|elif)\(([^)]+)\)\s*\{|^@(else)\s*\{/.exec(code.slice(cursor));
            if (!match)
                throw new Error(`Unsupported WESL conditional: ${code.slice(cursor, cursor + 80)}`);
            const enabled = match[3] === "else" || flags.has(match[2]!);
            const bodyStart = cursor + match[0].length;
            let end = bodyStart;
            let depth = 1;
            while (depth) {
                const char = code[end++];
                if (char === undefined) throw new Error("Unclosed WESL conditional");
                if (char === "{") depth++;
                if (char === "}") depth--;
            }
            if (enabled && !taken) {
                selected = code.slice(bodyStart, end - 1);
                taken = true;
            }
            cursor = end;
            const whitespace = /^\s*/.exec(code.slice(cursor))![0].length;
            if (!/^@(elif|else)/.test(code.slice(cursor + whitespace))) break;
            cursor += whitespace;
        }
        code = code.slice(0, start) + selected + code.slice(cursor);
    }
    if (/@(?:if|elif|else)/.test(code)) throw new Error("Unresolved WESL conditional");
    return code;
}

export const methods = [
    ["TonyMcMapface", 0, "TONY_MC_MAPFACE"],
    ["None", 1, "LINEAR"],
    ["Linear", 8, "LINEAR"],
    ["AcesFitted", 2, "ACES_FITTED"],
    ["Reinhard", 3, "REINHARD"],
    ["ReinhardLuminance", 4, "REINHARD_LUMINANCE"],
    ["AgX", 5, "AGX"],
    ["SomewhatBoringDisplayTransform", 6, "SOMEWHAT_BORING_DISPLAY_TRANSFORM"],
    ["KhronosPbrNeutral", 7, "PBR_NEUTRAL"],
] as const;
export type Method = (typeof methods)[number];
export type Grade = typeof gradingDefaults;

// Blender v3.6.0's original classification block, translated mechanically from
// C++ declarations/macros/member access to WGSL. No grading operations are taken.
function blenderWeights(): string {
    const code = readFileSync(
        new URL("./blender/COM_ColorCorrectionOperation.cc", import.meta.url),
        "utf8",
    );
    const start = code.indexOf("  float level_shadows =");
    const end = code.indexOf("#undef MARGIN_DIV", start);
    if (start < 0 || end < 0) throw new Error("Missing Blender classification block");
    const weights = code
        .slice(start, end)
        .replace(/#define (\w+) ([^\n]+)/g, "const $1: f32 = $2;")
        .replace(/#undef[^\n]*/g, "")
        .replace(/float (\w+) =/g, "var $1: f32 =")
        .replace(/\b(\d+(?:\.\d+)?)f\b/g, "$1")
        .replaceAll("data_->startmidtones", "midtone_range.x")
        .replaceAll("data_->endmidtones", "midtone_range.y");
    return `let midtone_range = (*color_grading).midtone_range;\n${weights}\nlet levels = vec3(level_shadows, level_midtones, level_highlights);\n`;
}

export async function bevyPipeline(
    world: World,
    method: Method,
    dither: boolean,
    grading: boolean,
) {
    const flags = new Set([`TONEMAP_METHOD_${method[2]}`]);
    if (dither && method[1] !== 1) flags.add("DEBAND_DITHER");
    if (grading && method[1] !== 1)
        for (const flag of ["HUE_ROTATE", "WHITE_BALANCE", "SECTIONAL_COLOR_GRADING"])
            flags.add(flag);
    let tonemapping = source("tonemapping");
    if (grading) {
        // The Render referent grounds the lower crossover in Blender, not Bevy.
        tonemapping = tonemapping.replace(
            /var levels = vec3\(0.0\);[\s\S]*?(?=\s*let contrast =)/,
            blenderWeights(),
        );
    }
    let fragment = source("tonemapping_frag");
    // Bevy's None skips this pass. Its source still reaches display encoding.
    if (method[1] === 1)
        fragment = fragment.replace(
            "tone_mapping(hdr_color, view.color_grading).rgb",
            "hdr_color.rgb",
        );
    const code = specialize(
        [
            source("maths"),
            colorOperations(),
            source("view").match(/struct ColorGrading[\s\S]*?struct View[\s\S]*?};/)![0],
            source("fullscreen_vertex_shader"),
            tonemapping,
            fragment,
        ].join("\n"),
        flags,
    );
    const module = world.gpu.device.createShaderModule({ code });
    const errors = (await module.getCompilationInfo()).messages.filter(
        (message) => message.type === "error",
    );
    if (errors.length)
        throw new Error(
            errors.map((message) => `${message.lineNum}: ${message.message}`).join("\n"),
        );
    const device = world.gpu.device;
    const layout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "unfilterable-float" },
            },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "non-filtering" } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "3d" } },
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        ],
    });
    return device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: "fullscreen_vertex_shader" },
        fragment: { module, entryPoint: "fragment", targets: [{ format: "rgba8unorm-srgb" }] },
    });
}

export function bevyLut(world: World, method: Method): GPUTexture {
    const tony = method[1] === 0;
    const file = readFileSync(
        new URL(
            `./bevy/${tony ? "tony_mc_mapface" : "AgX-default_contrast"}.ktx2`,
            import.meta.url,
        ),
    );
    const dimension = tony ? 48 : 32;
    if (file.readUInt32LE(12) !== (tony ? 123 : 97) || file.readUInt32LE(44) !== 2)
        throw new Error("Unexpected Bevy LUT format");
    const offset = Number(file.readBigUInt64LE(80));
    const length = Number(file.readBigUInt64LE(88));
    const data = Bun.zstdDecompressSync(file.subarray(offset, offset + length));
    if (data.length !== dimension ** 3 * (tony ? 4 : 8))
        throw new Error("Unexpected Bevy LUT length");
    const texture = world.gpu.device.createTexture({
        size: [dimension, dimension, dimension],
        dimension: "3d",
        format: tony ? "rgb9e5ufloat" : "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(texture);
    world.gpu.device.queue.writeTexture(
        { texture },
        data,
        { bytesPerRow: dimension * (tony ? 4 : 8), rowsPerImage: dimension },
        [dimension, dimension, dimension],
    );
    return texture;
}

export function bevyUniform(world: World, g: Grade) {
    // View and ColorGrading from view.wesl: grading starts at byte 608,
    // has 160 bytes, and is followed by mip_bias and frame_count (total 784).
    const bytes = new Float32Array(196);
    const at = 608 / 4;
    const x = 0.31272 - g.temperature;
    const y = 0.32903 + g.tint;
    const white = [
        0.701634 + (-0.051461 + 0.452749 * x) / y,
        1.15856 + (0.045854 - 0.296122 * x) / y,
        -0.904175 + (0.953127 - 0.955206 * x) / y,
    ];
    const adjust = [0.975538, 1.01648, 1.08475].map((v, i) => v / white[i]!);
    // ColorGradingUniform::from: LMS_TO_RGB * diagonal(D65_LMS / white) * RGB_TO_LMS.
    const toLms = [
        [0.311692, 0.0905138, 0.00764433],
        [0.652085, 0.901341, 0.0486554],
        [0.0362225, 0.00814478, 0.9437],
    ];
    const toRgb = [
        [4.06305, -0.40791, -0.0118812],
        [-2.93241, 1.40437, -0.0486532],
        [-0.130646, 0.0035363, 1.0605344],
    ];
    for (let col = 0; col < 3; col++)
        for (let row = 0; row < 3; row++)
            bytes[at + col * 4 + row] = [0, 1, 2].reduce(
                (sum, k) => sum + toRgb[k]![row]! * adjust[k]! * toLms[col]![k]!,
                0,
            );
    for (const [i, field] of (
        ["saturation", "contrast", "gamma", "gain", "lift"] as const
    ).entries())
        bytes.set(g[field], at + 12 + i * 4);
    bytes.set([...g.midtonesRange, g.exposure, g.hue, g.postSaturation], at + 32);
    const buffer = world.gpu.device.createBuffer({
        size: bytes.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    world.gpu.device.queue.writeBuffer(buffer, 0, bytes);
    world.own(buffer);
    return buffer;
}
