import { KernelSpec } from "./kernel";

export const registry: { [name: string]: KernelSpec } = {};

import { kernels as kernels_opgen } from "./kernels_opgen";
for (const name in kernels_opgen) {
    registry[name] = kernels_opgen[name];
}

import { kernels as kernels_artisanal } from "./kernels_artisanal";
for (const name in kernels_artisanal) {
    registry[name] = kernels_artisanal[name];
}
