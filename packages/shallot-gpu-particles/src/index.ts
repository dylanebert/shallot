// The package's whole surface: the plugin a project's manifest enables, and the live-state readers a
// consumer's own verification needs. One module so the manifest entry can be the bare package name
// (`"Particles": "shallot-gpu-particles"`) — the project host passes a bare specifier through to the
// project root's own resolver, so an installed copy loads the same way a local `./src/...` plugin does.
//
// The default export is the plugin, which is what a manifest entry resolves to. `particleState` and
// `particlesStepped` are the producer's observable output: the buffer the compute pass writes and
// whether a dispatch from an earlier frame has been submitted. A consumer that only draws the particles
// never calls them; the recipe's smoke does.

export {
    integrateKernel,
    PARTICLE_BYTES,
    PARTICLE_COUNT,
    PARTICLE_WORKGROUP,
    Particle,
    ParticleArray,
    type ParticleBuffer,
    particleLayout,
    SPAWN_Y,
} from "./kernel";
export {
    ParticlesPlugin,
    ParticlesPlugin as default,
    particleState,
    particlesStepped,
} from "./particles";
