# Changelog

## 0.10.0

Fog now uses a per-camera frustum grid for opaque surfaces, with closed-form integration and trilinear application before tonemapping. `Fog.viewDistance` sets its far view-space depth in world units (default 64); surfaces beyond it read the last slice. `Fog.stepCount` and `Fog.jitter` are removed; grid dimensions are internal constants.

`Fog.color` is the medium's own ambient light, scattered at the new `Fog.brightness` (cd/m², scaled by exposure and scattering albedo; the default matches the old fade at EV100 9.7). Ambient in-scatter now scatters uniform ambient radiance at its own radiance, 4π brighter than before.

`lookAtRotationInto(aim, out)` writes a quaternion into caller-owned `Float64Array` storage. Like `lookAtPacked`, `aim` holds eye at lane 0, target at 3 and up at 6. `lookAtRotation` keeps its signature and fresh tuple result. Sky packing now uses the existing `unpackColorInto(packed, out, offset?)`; `unpackColor` keeps its signature and fresh RGB result.

Entity ids reserve `0xFFFFFFFF` as the invalid id. Allocation refuses when the u32 id range is exhausted; restoring an allocator image refuses a next id beyond that value or a live invalid id.

`registerProducerTable` in `/standard/rendering` (also at the root) registers World-owned GPU placement and mesh-instance columns with a GPU-written count. Standard culls and draws those rows beside entities, including shadows, without readback. Materials are opaque, masked or alpha-to-coverage; blended materials are refused because transparent sorting is on the CPU. Material shaders read the entity id in `input.eid` and the producer-table row index in the new `input.row`, in both vertex and fragment functions. A producer row's `eid` is `0xFFFFFFFF` (the invalid entity id); an entity's `row` is the same sentinel. CPU picking, interpolation and snapshots do not include producer rows.

`openGpuApp` from `@dylanebert/shallot/testing/gpu` installs WebGPU globals and opens a check's app while owning the adapter notice. It refuses a missing adapter or Dawn's `null-backend` before creating a device, rather than accepting blank GPU results.

0.10 is a hard reset. It holds everything Shallot includes to a higher standard of verification, moves experimental features to their own packages, and arranges the engine in layers, from the engine at the center to optional extras at the edge, so a game can replace or drop the outer ones. [`CONTRIBUTING.md`](CONTRIBUTING.md) describes the layers.

Fonts are identified by URL, without registration aliases. Text font 0 always selects the bundled default face, including in worlds with registered fonts.

`registerImage(world, source, { sampler })` accepts `addressModeU`, `addressModeV` (`ImageAddressMode.ClampToEdge`, `Repeat`, `MirrorRepeat`), and `magFilter`, `minFilter` and `mipmapFilter` (`ImageFilterMode.Nearest`, `Linear`). StandardMaterial and Sprite use each image's sampler. Omitted settings take Bevy's `ImageSamplerDescriptor::linear()`: clamp-to-edge with linear texel filtering and linear filtering between mip levels, so a receding surface no longer steps between levels; pass `mipmapFilter: ImageFilterMode.Nearest` for one level per pixel. Level of detail stays isotropic (anisotropy 1), so surfaces at grazing angles still blur along their receding axis. Image URLs with the same effective sampler share an id; different samplers return distinct ids. Each Blob registration still receives its own id.

`pointer.buttonEvents` lists every pointer-button press and release of a frame in order, each at its position, so presses and releases between two frames are all seen. A one-finger touch, from `touchPoint` or the browser, moves the pointer and presses `left`; a second finger cancels that press, so a pinch or two-finger pan presents no press and release.

`Arrow` and `drawArrow` heads are filled world-space triangles perpendicular to a supplied normal, world up by default. The normal must be nonzero and perpendicular to the shaft. Head length is in world units (`tipLength`), defaulting to a tenth of the shaft length.

`OrbitPick` is removed. Orbit runs on every drag; games decide click selection using pointer events and their own travel threshold.

To port a 0.9.5 app, see [`MIGRATION.md`](MIGRATION.md). Changes before 0.10 are at the [`archive/changelog-pre-0.10`](https://github.com/dylanebert/shallot/tree/archive/changelog-pre-0.10) tag and in the git history.
