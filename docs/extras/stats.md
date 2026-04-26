---
title: Stats
description: performance overlay
source: extras/stats
icon: activity
---

# Stats

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Add `StatsPlugin` to display a live performance overlay with FPS, GPU timing, CPU timing, memory, and shader compile stats.

```typescript
import { StatsPlugin } from "@dylanebert/shallot/extras";

const config = {
    plugins: [StatsPlugin],
};
```

Collapsible sections for GPU, CPU, memory, and startup. Updates throttled to 250ms.

<!-- tab: Reference -->

<!-- API:extras/stats -->

<!-- /tabs -->
