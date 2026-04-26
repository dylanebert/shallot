---
title: Orbit
description: orbit camera
source: extras/orbit
icon: orbit
---

# Orbit

<!-- tabs -->
<!-- tab: UI -->

Third-person camera with orbit, pan, zoom, and fly controls.

### Controls

| Input | Action |
|-------|--------|
| Left click drag | Orbit around target |
| Right click drag | Pan (shift view laterally) |
| Scroll wheel | Zoom in/out |
| WASD | Fly forward/back/left/right |
| Q / E | Fly down / up |

Fly mode activates while movement keys are held. When released, the camera returns to orbit mode at its new position.

### Scene File

```xml
<a id="camera"
   camera
   orbit="distance: 20; min-distance: 5; max-distance: 100"
   transform
   viewport />
```

Set an orbit target:

```xml
<a id="target" transform="pos: 5 0 0" />
<a id="camera" camera orbit target="@target" transform viewport />
```

<!-- tab: Code -->

### Setup

```typescript
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";

const config = {
    plugins: [OrbitPlugin],
};
```

### Controls

| Input | Action |
|-------|--------|
| Left click drag | Orbit (configurable via `orbitButton`) |
| Right click drag | Pan (configurable via `panButton`) |
| Scroll | Zoom |
| WASD / QE | Fly |

### Configuration

```typescript
Orbit.distance[cam] = 20;
Orbit.sensitivity[cam] = 0.005;
Orbit.flySpeed[cam] = 10;

// swap orbit to middle click
Orbit.orbitButton[cam] = 1;
```

### Pan Offset

Pan accumulates an offset (`panX`, `panY`, `panZ`) added to the target position. Reset with:

```typescript
Orbit.panX[cam] = 0;
Orbit.panY[cam] = 0;
Orbit.panZ[cam] = 0;
```

### Fly Mode

WASD/QE movement activates automatically while keys are held. Movement uses the current orbit view direction. Orbiting while flying changes the flight direction. On release, the orbit center is reprojected to the camera's new position.


Third-person orbiting camera with pan, zoom, and keyboard fly mode. Web-native control scheme: left click orbits, right click pans, scroll zooms, WASD/QE flies.

### Component Fields

| Field | Default | Description |
|-------|---------|-------------|
| `yaw` | π/6 | Horizontal angle (radians) |
| `pitch` | π/9 | Vertical angle (radians) |
| `distance` | 10 | Distance from target |
| `size` | 5 | Orthographic viewport size |
| `minPitch` / `maxPitch` | ±π/2 - 0.01 | Pitch clamp |
| `minDistance` / `maxDistance` | 1 / 30 | Zoom clamp |
| `minSize` / `maxSize` | 0.5 / 50 | Ortho size clamp |
| `smoothness` | 0.3 | Interpolation factor (0 = instant, 1 = slow) |
| `sensitivity` | 0.005 | Orbit mouse sensitivity |
| `zoomSpeed` | 0.025 | Scroll zoom speed |
| `orbitButton` | 0 (Left) | Mouse button for orbit |
| `panButton` | 2 (Right) | Mouse button for pan |
| `panX` / `panY` / `panZ` | 0 | Pan offset (added to target) |
| `flySpeed` | 5 | WASD/QE movement speed |
| `flyActive` | 0 | Runtime flag (1 while flying) |
| `suppress` | 0 | When non-zero, orbit rotation is suppressed |

### Fly Exit

When all movement keys are released, the orbit center (pan offset) is reprojected so the camera stays at its current position. Yaw, pitch, and distance are preserved. Orbit input (rotation, zoom) continues to work during fly, allowing direction changes mid-flight.

### Dependencies

`InputPlugin` (required). Optional `Target` relation for orbiting around a specific entity.

<!-- tab: Reference -->

<!-- API:extras/orbit -->

<!-- /tabs -->
