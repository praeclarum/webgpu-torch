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
    expect(kernel.shader).toBe(`
        if (global_id.x >= parameters.size) {
            return;
        }
        output[global_id.x] = (input[global_id.x] - other[global_id.x]);`);
});

test("atan2 forward shader", () => {
    const spec = opSpecs["atan2"];
    const kernels = opSpecToKernelSpecs(spec);
    expect(kernels[0].shader).toBe(`
        if (global_id.x >= parameters.size) {
            return;
        }
        output[global_id.x] = atan2(input[global_id.x], other[global_id.x]);`);
});


test("can generate all kernel specs", () => {
    for (const spec of registry) {
        const kernels = opSpecToKernelSpecs(spec);
        expect(kernels.length).toBeGreaterThan(0);
    }
});
