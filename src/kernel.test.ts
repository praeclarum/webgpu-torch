import { Kernel, getKernelConfig, getKernelShaderCode } from "./kernel";
import { registry } from "./kernels";

test("mm shader code", () => {
    const spec = registry["MM"];
    const config = getKernelConfig(spec, { resultDtype: "f32" });
    const shaderCode = getKernelShaderCode(spec, config);
    expect(shaderCode).toEqual(`// MM kernel
struct MMParameters {
    resultRows: u32,
    resultCols: u32,
    innerDim: u32,
    alpha: f32,
}
@group(0) @binding(0) var<storage, read> firstMatrix: array<f32>;
@group(0) @binding(1) var<storage, read> secondMatrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultMatrix: array<f32>;
@group(0) @binding(3) var<storage, read> parameters: MMParameters;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3u) {
    if (global_id.x >= parameters.resultRows || global_id.y >= u32(parameters.resultCols)) {
        return;
    }
    var result = 0.0;
    for (var i = 0u; i < parameters.innerDim; i = i + 1u) {
        let a = global_id.x * parameters.innerDim + i;
        let b = i * parameters.resultCols + global_id.y;
        result = result + firstMatrix[a] * secondMatrix[b];
    }
    let index = global_id.y + global_id.x * parameters.resultCols;
    resultMatrix[index] = result;
}`);
});
