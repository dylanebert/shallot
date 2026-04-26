---
paths:
    - "packages/shallot/src/standard/render/**/*.ts"
    - "packages/shallot/src/standard/render/**/*.wgsl"
    - "packages/shallot/src/standard/raster/**/*.ts"
    - "packages/shallot/src/extras/raytracing/**/*.ts"
---

# Render

Reference: `docs/standard/render.md` for mesh format, surfaces, and pipeline extension.

## Gamma pipeline

End-to-end gamma-correct. Three boundaries do conversion; everywhere else is linear:

1. **Hex decode** (`unpackColor`, `createColorProxy.set`, `hexColorProxy.set`): sRGB byte → linear float. All scene-supplied hex values land in linear-space SoA buffers.
2. **Hex encode** (`createColorProxy.get`, `hexColorProxy.get`): linear → sRGB byte. Round-trips so `Part.color = 0xff8080` reads back as `0xff8080`.
3. **Display encode** (`present.ts` shader): `linearToSrgb(saturate(color))` once, mid-shader. Forward/tonemap/FXAA are linear; dither/posterize/vignette are display-space.

**Do not** add `linearToSrgb` at the textureStore — it's already encoded by then. **Do not** apply it before tonemap or FXAA.

## Effect ordering in present

Required order:
```
FXAA (linear) → tonemap (linear) → linearToSrgb(saturate) → dither → posterize → vignette → saturate(output)
```

Dither/posterize/vignette **must** run in display (sRGB) space. They were calibrated against display values:
- Dither amplitude `0.04` in linear at dark values (`~0.02`) blows up to `>0.3` perceptual after gamma encode — ugly noise everywhere.
- Posterize uses OKLab L bands; band positions only land at familiar visual locations when input is sRGB.
- Vignette is multiplicative darkening; in linear it produces non-physical compression after gamma encode.

The trailing `saturate` after vignette clamps dither overflow before write.

## Direct color SoA writes

`AmbientLight.color` / `DirectionalLight.color` are raw hex arrays. Direct writes (e.g. skylab driving lights from a gradient) must encode linear values to sRGB byte before storing — otherwise `unpackColor` in the lighting math will decode the byte as sRGB and produce a different linear value than intended. Skylab's `packColor` (extras/skylab) is the reference: applies `linearToSrgb` per channel before packing, so its `ColorRGB` literals round-trip correctly.

## Content tuning

When changing engine defaults or scene values:
- Gamma is non-linear; no global intensity multiplier matches every albedo. Tune for dominant scene tones; accept drift on saturated-bright accents.
- Reflectivity for dielectrics belongs in `0.02–0.05` (concrete to glass). Higher values are artifacts of pre-gamma tuning where crushed diffuse required inflated specular to compensate.
- Bright accent hex values (e.g. `0xd49560`) saturate aggressively under intense lighting (linear * intensity often exceeds 1.0 in all channels, collapsing hue to white). Use darker variants (e.g. `0x8b6040`) for accent objects in scenes with strong key lights, OR lower the intensity.
