# striped-material — expected artifacts (withheld)

Problem shape from `examples/recipes/custom-material`: authoring a procedural surface (a WGSL shading
chunk registered with `Surfaces`, selected per entity by `Part.surface`) — something a flat entity
`Color` cannot produce.

What a correct project does:
- registers a custom surface whose fragment chunk computes a banded pattern from `uv` (a `floor`/`step`
  stripe or checker), animated by a time uniform so the bands shift each frame
- a `Part` cube wearing that surface, a static camera framing it

Gate observation: from one settled frame, a brightness profile is sampled across the cube's face along
each axis; a real striped surface oscillates (peak-to-peak > 30 with ≥ 2 light/dark band crossings),
while a flat cube face is near-constant and even a spinning flat cube shows at most one face-to-face
step (≤ 1 crossing). Three frames sampled over ~1.1s at two different gaps, camera untouched, must
differ (max pairwise mean-abs > 0.5) — a settled static scene is bit-static (delta 0.00), so any change
is the pattern animating; the two gaps keep a pattern whose period aliases one interval from reading as
static. The crossing count is the discriminator against a flat or rigid-spinning cube; the peak-to-peak
floor keeps antialiasing wobble from counting.

Common failure modes this catches: flat entity colour with no custom surface (fails band crossings);
a static pattern that never moves (fails the frame-delta check); a plain spinning cube mistaken for
"alive" (flat face → fails band crossings even though it moves).
