// The plugin the recipe's manifest enables by path, plus the live-state readers its smoke needs:
// `particleState` is the buffer the compute pass writes, `particlesStepped` whether an earlier frame
// submitted a dispatch. A project that only draws the particles never calls them.

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
