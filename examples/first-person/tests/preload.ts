import { setDefaultTimeout } from "bun:test";
import { shallot } from "@dylanebert/shallot/bun";
import { CEILING } from "@dylanebert/shallot/testing";
import { plugin } from "bun";

setDefaultTimeout(CEILING.cheap);
plugin(shallot({ root: import.meta.dir }));
