import { KernelSpec } from "./kernel";
import { opSpecToKernelSpecs } from "./opgen";
import { registry as opRegistry } from "./op_table";
import { OpSpec } from "./op_spec";

export const kernels: {[name: string]: KernelSpec} = {};
export const opKernelSpecs: [OpSpec, KernelSpec][] = [];
for (const [name, spec] of Object.entries(opRegistry)) {
    const ks = opSpecToKernelSpecs(spec);
    for (const kernel of ks) {
        kernels[kernel.name] = kernel;
        opKernelSpecs.push([spec, kernel]);
    }
}
