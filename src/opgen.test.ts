import { opSpecToKernelSpecs } from "./opgen";
import { registry } from "./op_table";

export const opSpecs = registry.reduce((map, spec) => {
    map[spec.name] = spec;
    return map;
}, {} as {[name: string]: any});

test("binary sub op generate kernel spec", () => {
    const spec = opSpecs["sub"];
    const kernels = opSpecToKernelSpecs(spec);
    expect(kernels.length).toBeGreaterThan(0);
    const kernel = kernels[0];
    expect(kernel.name).toBe("sub");
    expect(kernel.config).toEqual([
        {
            name: "dtype",
        },
    ]);
});

test("can generate all kernel specs", () => {
    for (const spec of registry) {
        const kernels = opSpecToKernelSpecs(spec);
        expect(kernels.length).toBeGreaterThan(0);
    }
});
