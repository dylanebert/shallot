import { benchmarkParity } from "./benchmark-parity";

for (const threads of [2, 4]) benchmarkParity("manyPyramids", threads);
