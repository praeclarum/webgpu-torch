import { discoverWebGPUDevicesAsync } from "./devices";

export {
    FunctionInput,
    GradientContext,
    GradientFunction,
    GradientFunctionOutput,
} from "./autograd";
export { ATypedArray, Dtype } from "./dtype";
export { Device, Deviceish, DeviceId, DeviceType } from "./device";
export { CompiledExpr, EvalEnv, ExprCode } from "./expr";
export * from "./factories";
// export * as functions from "./functions";
export {
    Kernel,
    KernelSpec,
    KernelConfig,
    KernelConfigInput,
    KernelConfigSpec,
    KernelConfigValue,
    KernelInputSpec,
    KernelKey,
    KernelOutputSpec,
    KernelParamSpec,
    KernelParamType,
    KernelParamValue,
    KernelParamsInput,
    ShaderType,
} from "./kernel";
export * from "./ops";
export * from "./shape";
export { TensorArrayData, UntypedStorage } from "./storage";
export * from "./tensor";
export { hasWebGPU } from "./webgpu";

export async function initWebGPUAsync() {
    await discoverWebGPUDevicesAsync();
}
