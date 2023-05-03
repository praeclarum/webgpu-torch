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

