# AI Agent Development Guide

## F16 Fallback Support

When working with Shallot shaders, remember:

1. **GPU capabilities are available** via `Compute.capabilities`
2. **Use `getPrecisionType()`** helper function for conditional compilation
3. **F32 is always safe** - it's the default fallback
4. **F16 requires opt-in** - only use if you've verified support

### Example: Shader with Precision Support

```typescript
import { Compute, getPrecisionType } from "../compute";

export function compileMyShader(state: State): string {
    const compute = Compute.from(state);
    const precision = getPrecisionType(compute.capabilities);
    
    return /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> data: array<${precision}>;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            let value: ${precision} = ${precision}(3.14159);
            data[idx] = value;
        }
    `;
}
```

### GPU Feature Detection

```typescript
const compute = Compute.from(state)!;

if (compute.capabilities.supportsF16) {
    // Use optimized shaders with f16
    useOptimizedPath();
} else {
    // Use standard f32 path
    useStandardPath();
}
```

### Device Limits & Features

The device is created with appropriate feature limits:
- `maxTextureDimension2D`: Adapter's max
- `maxStorageBuffersPerShaderStage`: 10
- Optional features: `f16`, `timestamp-query`, `bgra8unorm-storage`
