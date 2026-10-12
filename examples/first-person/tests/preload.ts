import { shallot } from "@dylanebert/shallot/bun";
import { plugin } from "bun";

plugin(shallot({ root: import.meta.dir }));
