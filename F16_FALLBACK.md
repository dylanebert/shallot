# F16 Fallback Support

## Overview

Shallot now supports GPUs that don't have the `shader-f16` extension, such as older AMD cards (RX 550, etc). The engine automatically detects GPU capabilities and falls back to f32 precision when needed.

## What's New

### GPU Capability Detection

When initializing the Compute plugin, Shallot now detects:
- **F16 Support**: `shader-f16` WebGPU extension
- **Timestamp Queries**: For performance profiling
- **BGRA8 Storage**: For storage texture formats

### GPUCapabilities Interface

```typescript
export interface GPUCapabilities {
    supportsF16: boolean;
    supportsTimestampQuery: boolean;
    supportsBGRA8UnormStorage: boolean;
}
```

Access capabilities from the Compute resource:

```typescript
const compute = Compute.from(state);
if (compute.capabilities.supportsF16) {
    // GPU supports f16 - use optimized shaders
} else {
    // GPU doesn't support f16 - use f32 shaders
}
```

### Precision Helper

Use `getPrecisionType()` when generating shaders:

```typescript
import { Compute, getPrecisionType } from "@dylanebert/shallot/compute/core";

function compileMyShader(state: State): string {
    const compute = Compute.from(state);
    const precision = getPrecisionType(compute.capabilities);
    
    // Use precision in shader generation
    return /* wgsl */ `
        var data: array<${precision}>
    `;
}
```

## Console Output

When the engine initializes, it logs GPU capabilities:

```
GPU Capabilities: {
  f16: "supported",
  timestampQuery: true,
  bgra8UnormStorage: false
}
```

## Performance Impact

- **With F16 support**: Optimal performance, reduced memory bandwidth
- **Without F16 support (F32 fallback)**: Slightly higher memory usage but full functionality

The performance difference is typically negligible for most use cases, as the bottleneck is usually elsewhere (geometry processing, texture sampling, etc).

## Supported GPUs

### F16 Supported
- NVIDIA RTX series (Ampere and newer)
- NVIDIA GeForce RTX 30/40 series
- Intel Arc GPUs
- AMD RDNA 2/3 series (RX 6000 series)

### F16 Not Supported (F32 Fallback)
- AMD GCN-based GPUs (RX 550, RX 580, Radeon Vega)
- Older NVIDIA cards (pre-Ampere)
- Some integrated GPUs

## Implementation Notes

The fallback works at the device initialization level:
1. Adapter queries f16 availability
2. F16 is added to `requiredFeatures` only if supported
3. Device is created with optimal feature set
4. `GPUCapabilities` is stored and accessible throughout the app

No shader code changes are required unless you explicitly want to use f16 for optimization.

## Usage Example

```typescript
import { App, Render } from "@dylanebert/shallot";

const app = new App();
await app.initialize();

// Engine automatically handles f16 detection and fallback
// Your code works the same regardless of GPU capabilities

const cube = app.add({ Mesh: "box" });
```

## Debugging

To check GPU capabilities programmatically:

```typescript
const compute = Compute.from(state);
console.log("GPU Capabilities:", compute.capabilities);
```

For browser developer tools:
```javascript
// Check in Chrome/Edge DevTools console
await navigator.gpu.requestAdapter().then(a => 
    console.log("Supported features:", Array.from(a.features))
);
```
