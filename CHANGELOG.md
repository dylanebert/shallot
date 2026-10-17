# Changelog

## 0.10.0

`openGpuApp` from `@dylanebert/shallot/testing/gpu` installs WebGPU globals and opens a check's app while owning the adapter notice.

0.10 is a hard reset. It holds everything Shallot includes to a higher standard of verification, moves experimental features to their own packages, and arranges the engine in layers, from the engine at the center to optional extras at the edge, so a game can replace or drop the outer ones. [`CONTRIBUTING.md`](CONTRIBUTING.md) describes the layers.

Fonts and images are identified by URL, without registration aliases; each Blob image registration receives its own id. Text font 0 always selects the bundled default face, including in worlds with registered fonts.

`pointer.buttonEvents` lists every pointer-button press and release of a frame in order, each at its position, so presses and releases between two frames are all seen. A one-finger touch, from `touchPoint` or the browser, moves the pointer and presses `left`; a second finger cancels that press, so a pinch or two-finger pan presents no press and release.

`Arrow` and `drawArrow` heads are filled world-space triangles perpendicular to a supplied normal, world up by default. The normal must be nonzero and perpendicular to the shaft. Head length is in world units (`tipLength`), defaulting to a tenth of the shaft length.

`OrbitPick` is removed. Orbit runs on every drag; games decide click selection using pointer events and their own travel threshold.

To port a 0.9.5 app, see [`MIGRATION.md`](MIGRATION.md). Changes before 0.10 are at the [`archive/changelog-pre-0.10`](https://github.com/dylanebert/shallot/tree/archive/changelog-pre-0.10) tag and in the git history.
