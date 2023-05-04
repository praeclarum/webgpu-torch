import { getKernelSpecs } from "./op_spec";
import {registry} from "./op_table";

const opSpecs = registry.reduce((map, spec) => {
    map[spec.name] = spec;
    return map;
}, {} as {[name: string]: any});


test("binary sub op generate kernel spec", () => {
    const spec = opSpecs["sub"];
    const kernels = getKernelSpecs(spec);
    expect(kernels.length).toBe(1);
    const kernel = kernels[0];
    expect(kernel.name).toBe("Sub");
    expect(kernel.config).toEqual([
        {
            name: "dtype",
        },
    ]);
    expect(kernel.shader).toBe(`
        if (global_id.x >= parameters.size) {
            return;
        }
        out[global_id.x] = (input[global_id.x] - other[global_id.x]);`);
});

test("binary atan2 op kernel spec shader", () => {
    const spec = opSpecs["atan2"];
    const kernels = getKernelSpecs(spec);
    expect(kernels[0].shader).toBe(`
        if (global_id.x >= parameters.size) {
            return;
        }
        out[global_id.x] = atan2(input[global_id.x], other[global_id.x]);`);
});

test("can generate all kernel specs", () => {
    for (const spec of registry) {
        const kernels = getKernelSpecs(spec);
        expect(kernels.length).toBeGreaterThan(0);
    }
});

