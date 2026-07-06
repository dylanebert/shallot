// the six sprite surface variants: (screen | y | world) billboard × (clip | alpha) blend,
// compile-time variants rather than runtime branches. The VS bodies are the WGSL port of
// billboard.ts (the executable spec); sprite.test.ts checks the generated shape through sear's
// `surfaceCode`.

const VARIANTS = ["screen", "y", "world"] as const;

export const surfaceName = (bucket: number): string =>
    `sprite-${VARIANTS[bucket >> 1]}${bucket & 1 ? "-alpha" : ""}`;

// struct + the sRGB→linear the packed tint needs before the linear target (texture rgb is already
// linear — the array is rgba8unorm-srgb, hardware-decoded on sample)
const SPRITE_WGSL = /* wgsl */ `
struct SpriteData {
    offset: vec2<f32>,
    size: vec2<f32>,
    eid: u32,
    layer: u32,
    color: u32,
    fill: u32,
}

fn spriteSrgbToLinear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// fill mask from the packed (unorm16 amount | mode << 16) word: 1 inside the leading fraction of
// the image, 0 past it. Radial sweeps clockwise from 12 o'clock; vertical fills bottom-up (uv.y
// is image-down); horizontal left-to-right. Mode 0 is unfilled (always 1)
fn spriteFillMask(fill: u32, uv: vec2<f32>) -> f32 {
    let mode = fill >> 16u;
    if (mode == 0u) { return 1.0; }
    let amount = f32(fill & 0xffffu) / 65535.0;
    var t = uv.x;
    if (mode == 1u) {
        let d = uv - vec2<f32>(0.5, 0.5);
        t = fract(atan2(d.x, -d.y) / 6.283185307179586);
    } else if (mode == 2u) {
        t = 1.0 - uv.y;
    }
    return select(0.0, 1.0, t <= amount);
}
`;

const VS_HEAD = /* wgsl */ `
let s = spriteData[iid];
let lp = s.offset + localPos.xy * s.size;
let t = xformMat(transforms[s.eid]);
eid = s.eid;
uv = vec2<f32>(localPos.x, 1.0 - localPos.y);
slayer = s.layer;
scolor = s.color;
sfill = s.fill;
`;

// per-billboard world position, mirroring billboard.ts (worldCorner / screenCorner / yLockedCorner).
// Screen + YLocked substitute the camera basis (view.right/up) for the model rotation, re-applying
// the per-axis scale from the model columns; in the shadow pass the View is the light camera's, so
// billboards face the light (Godot-consistent)
const VS_BODY: Record<(typeof VARIANTS)[number], string> = {
    screen: /* wgsl */ `
let bx = lp.x * length(t[0].xyz);
let by = lp.y * length(t[1].xyz);
world = vec4<f32>(t[3].xyz + view.right.xyz * bx + view.up.xyz * by, 1.0);
`,
    y: /* wgsl */ `
let toViewer = cross(view.right.xyz, view.up.xyz);
var facing = vec2<f32>(toViewer.x, toViewer.z);
if (dot(facing, facing) < 1e-8) { facing = vec2<f32>(view.up.x, view.up.z); }
facing = normalize(facing);
let bx = lp.x * length(t[0].xyz);
let by = lp.y * length(t[1].xyz);
world = vec4<f32>(t[3].xyz + vec3<f32>(facing.y, 0.0, -facing.x) * bx + vec3<f32>(0.0, by, 0.0), 1.0);
`,
    world: /* wgsl */ `
world = t * vec4<f32>(lp, 0.0, 1.0);
`,
};

// unlit icon shading: texture × sRGB-decoded tint. Clip discards below the 0.5 cutoff (opacity
// shrinks the cutout) and authors `tag = eid` so sprites are hover/pick targets — a surface
// without the `eids` binding defaults to TAG_NONE; alpha has no tag lane (no single owner)
function spriteFs(alpha: boolean): string {
    return /* wgsl */ `
let tex = textureSample(spriteAtlas, spriteSamp, uv, i32(slayer));
let unp = unpack4x8unorm(scolor);
let mask = spriteFillMask(sfill, uv);
${
    alpha
        ? /* wgsl */ `col = vec4<f32>(tex.rgb * spriteSrgbToLinear(unp.rgb), tex.a * unp.a * mask);`
        : /* wgsl */ `if (tex.a * unp.a * mask < 0.5) { discard; }
col = vec4<f32>(tex.rgb * spriteSrgbToLinear(unp.rgb), 1.0);
tag = eid;`
}`;
}

export function spriteSurface(bucket: number) {
    return {
        name: surfaceName(bucket),
        blend: bucket & 1 ? ("alpha" as const) : ("clip" as const),
        bindings: {
            spriteData: { type: "storage" as const, element: "SpriteData" },
            transforms: { type: "storage" as const, element: "Xform" },
            spriteAtlas: { type: "texture-2d-array" as const },
            spriteSamp: { type: "sampler" as const },
        },
        interpolators: { slayer: "u32", scolor: "u32", sfill: "u32" },
        preamble: SPRITE_WGSL,
        vs: VS_HEAD + VS_BODY[VARIANTS[bucket >> 1]],
        fs: spriteFs((bucket & 1) === 1),
    };
}
