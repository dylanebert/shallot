# Migrating from 0.10 to 0.11

Start with the compiler errors in your app. Search this guide for the old symbol or behavior; each entry goes directly from the published 0.10 package to 0.11. For the engine's layers and extension rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

Code fragments assume an owning `world`, an existing entity `eid`, and the imports named in the surrounding text.

## A second camera and the first canvas

In 0.10 a camera without a target auto-bound the document's first canvas only while no other camera held it; a second such camera stayed unbound and drew nothing. Now every camera without a target joins the first canvas, and cameras sharing a canvas or texture draw bottom to top by the new `Camera.order`, the bottom one clearing. `attachCanvas` on a canvas another camera of the World holds joins it the same way, where 0.10 let both cameras present over each other. A camera that drew nothing in 0.10 now draws on top of the first: give it its own target, or remove its `Camera`.

```ts
// 0.10: a second camera without a target drew nothing
world.add(eid, Camera);
```

```ts
// 0.11: give it its own target
world.add(eid, Camera);
attachTexture(world, eid, { width: 320, height: 180 });
// or stack it over the first camera on purpose
world.add(eid, Camera, { order: 1 });
```
