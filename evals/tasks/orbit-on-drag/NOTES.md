# orbit-on-drag — expected artifacts (withheld)

Problem shape from `examples/recipes/orbit-camera` + `respond-to-input`: input wiring with a
controlled, causal observation.

What a correct project does:
- an `Orbit` camera framing an object at the origin, no `simulation` system auto-rotating anything

Gate observation: an idle 0.7s interval on a settled scene is bit-static (measured delta exactly
0.00; threshold 0.5 = floor + epsilon); a synthetic left-to-right mouse drag produces a visible
change (measured 1.77 mean-abs full-frame; threshold > 1 and > 3× idle). The idle baseline is what
rules out a scene that merely animates — the change must be *caused* by the drag.

Common failure modes this catches: auto-spinning cube (fails "holds still"); orbit not wired / drag
does nothing (fails "dragging orbits"); a static image with no camera (fails "dragging orbits").
