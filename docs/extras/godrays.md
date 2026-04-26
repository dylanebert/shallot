---
title: God Rays
description: volumetric light scattering from the sun
source: extras/godrays
icon: cloud-sun
---

# God Rays

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Screen-space volumetric scattering. Marches from each pixel toward the sun's screen position, accumulating brightness from samples that pass the depth test (i.e. unoccluded sky). Requires a depth-writing pipeline (raster forward); silently no-ops under raytracing. Per-camera component drives intensity, sample count, decay, and density.

<!-- tab: Reference -->

<!-- API:extras/godrays -->

<!-- /tabs -->
