import { Camera, Glaze, type Plugin, type State, Tonemap } from "@dylanebert/shallot";

// #doc:intro page:standard/rendering
// Give the scene a `Camera` with the `sear` renderer and a light, spawn a `Part` — it draws. This page
// composes the default renderer: PBR materials, sun and point lights, shadows, and the postfx that
// tonemaps the HDR scene to the screen.

// #doc:code page:standard/rendering source:kitchen/public/scenes/kitchen.scene
// ## The scene
//
// The camera carries `sear` (the renderer), `glaze` (postfx), and `orbit`. `directional-light` is the sun,
// `point-light` a warm local fill. Each `part` selects its geometry and shades under sear's default PBR
// surface, tinted by `color` and finished by its `material` lanes — metallic dielectric→metal, roughness
// sharp→soft highlight, emissive an HDR glow.

// #doc:code page:standard/rendering
// ## Shadows
//
// The sun carries `shadow`, so every sphere casts onto the ground — sampled inline in the fragment shader,
// no separate pass. Omit it for the fully-lit path. `Shadow` on a `PointLight` casts the same way, budgeted
// by `PointShadows`.

// #doc:code page:standard/rendering
// ## Postfx
//
// The renderer draws into an HDR buffer; `glaze` composites it to the screen, rolling the brights off through a
// tonemap. `Glaze.tonemap` picks the operator — a scene attribute, or set it on the camera in code:
// #region look
const Look = {
    name: "Look",
    warm(state: State) {
        for (const cam of state.query([Camera, Glaze])) {
            Glaze.tonemap.set(cam, Tonemap.Neutral); // the zero-config default display transform
        }
    },
} satisfies Plugin;
// #endregion

export default Look;
