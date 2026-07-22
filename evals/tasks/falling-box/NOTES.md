# falling-box — expected artifacts (withheld)

Problem shape from `examples/recipes/physics-playground`: a verifiable physics behavior.

What a correct project does:
- enables the physics plugin, a dynamic rigidbody box (blue) starting above a static floor collider
- gravity does the rest; the box falls and rests on the floor

Gate observation: the fall completes ~1.05s after scene build (measured 2026-07-13), before a
settle-stable boot returns — so the gate boots (proving serve + settle), then reloads and samples the
blue-pixel centroid continuously from the fresh load. Descent = the highest observed centroid sits
>0.03 above the resting one; settled = the last two samples agree within 0.01 (settled render is
bit-static). Real GPU only — physics needs a real device, so this gate never runs under the software
adapter (grader is display-gated).

Common failure modes this catches: static box that never falls (fails "fell"); box tunnels through /
keeps accelerating off-screen (fails "settled" or "on screen"); no physics wired at all (fails "fell").
