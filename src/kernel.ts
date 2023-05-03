import { ExprCode, evalCode, compileCode, CompiledExpr, EvalEnv } from "./expr";

export type KernelParamType = "u32" | "f32";
export type KernelParam = number;

export type ShaderType =
    | KernelParamType
    | "u8"
    | "array<u8>"
    | "i32"
    | "array<i32>"
    | "array<u32>"
    | "array<f32>";

export interface KernelSpec {
    name: string;
    config: KernelConfigSpec[];
    workgroupSize: [ExprCode, ExprCode, ExprCode];
    parameters: KernelParamSpec[];
    workgroupCount: [ExprCode, ExprCode, ExprCode];
    inputs: KernelInputSpec[];
    outputs: KernelOutputSpec[];
    shader: string;
}

export interface KernelInputSpec {
    name: string;
    shaderType: ShaderType;
}

export interface KernelOutputSpec {
    name: string;
    shaderType: ShaderType;
    size: ExprCode;
}

export interface KernelParamSpec {
    name: string;
    shaderType: KernelParamType;
}

export interface KernelConfigSpec {
    name: string;
}

export type KernelConfigValue = string | number;
export type KernelConfigInput = { [key: KernelKey]: KernelConfigValue };
export type KernelConfig = KernelConfigValue[];
export type KernelKey = string;

export class Kernel {
    private _key: KernelKey;
    private _spec: KernelSpec;
    private _config: KernelConfig;
    private _shaderCode: string;
    private _device: GPUDevice;
    private _bindGroupLayout: GPUBindGroupLayout;
    private _computePipeline: GPUComputePipeline;
    private _workgroupCountXFunc: CompiledExpr;
    private _workgroupCountYFunc: CompiledExpr;
    private _workgroupCountZFunc: CompiledExpr;
    private _outputSizeFuncs: CompiledExpr[];
    get key(): KernelKey {
        return this._key;
    }
    get spec(): KernelSpec {
        return this._spec;
    }
    constructor(spec: KernelSpec, config: KernelConfig, device: GPUDevice) {
        this._key = getKernelKey(spec, config);
        this._device = device;
        this._spec = spec;
        this._config = config;
        let bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
        let bindingIndex = 0;
        for (let i = 0; i < spec.inputs.length; i++, bindingIndex++) {
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage" as GPUBufferBindingType,
                },
            });
        }
        for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage" as GPUBufferBindingType,
                },
            });
        }
        bindGroupLayoutEntries.push({
            binding: bindingIndex,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "read-only-storage" as GPUBufferBindingType,
            },
        });
        this._bindGroupLayout = device.createBindGroupLayout({
            entries: bindGroupLayoutEntries,
        });
        this._shaderCode = getKernelShaderCode(spec, config);
        const shaderModule = device.createShaderModule({
            code: this._shaderCode,
        });
        this._computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this._bindGroupLayout],
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });
        this._workgroupCountXFunc = compileCode(spec.workgroupCount[0]);
        this._workgroupCountYFunc = compileCode(spec.workgroupCount[1]);
        this._workgroupCountZFunc = compileCode(spec.workgroupCount[2]);
        this._outputSizeFuncs = [];
        for (let i = 0; i < this._spec.outputs.length; i++, bindingIndex++) {
            const outputSpec = this._spec.outputs[i];
            const outputElementCount = compileCode(outputSpec.size);
            this._outputSizeFuncs.push(outputElementCount);
        }
    }
    run(
        inputs: GPUBuffer[],
        parameters: { [name: string]: KernelParam },
        outputs?: GPUBuffer[]
        ): GPUBuffer[] {
        console.log("run kernel", this._key);

        // Build the parameter environment
        const env: EvalEnv = {};
        let paramsBufferSize = 0;
        for (let i = 0; i < this._spec.parameters.length; i++) {
            const param = this._spec.parameters[i];
            const paramValue = parameters[param.name];
            if (paramValue === undefined) {
                throw new Error(`Missing parameter ${param.name}`);
            }
            env[param.name] = paramValue;
            paramsBufferSize += getShaderTypeElementByteSize(param.shaderType);
        }

        // Get input buffers with storage usage
        const storageInputs = this.spec.inputs.map((input, i) => this.getStorageInputBuffer(input, inputs[i], i, env));

        // Get output buffers with storage usage
        const storageOutputs = this.spec.outputs.map((output, i) => this.getStorageOutputBuffer(output, outputs ? outputs[i] : null, i, env));

        // Build the params buffer
        const paramsBuffer = this._device.createBuffer({
            mappedAtCreation: true,
            size: paramsBufferSize,
            usage: GPUBufferUsage.STORAGE,
        });
        const paramsArrayBuffer = paramsBuffer.getMappedRange();
        for (let paramDtype of ["u32", "f32"]) {
            let paramsArray = new (
                paramDtype === "u32" ? Uint32Array : Float32Array
            )(paramsArrayBuffer);
            for (let i = 0; i < this._spec.parameters.length; i++) {
                const param = this._spec.parameters[i];
                if (param.shaderType === paramDtype) {
                    paramsArray[i] = env[param.name];
                }
            }
        }
        paramsBuffer.unmap();

        // Bind the buffers
        const bindGroup = this.createBindGroup(storageInputs, paramsBuffer, storageOutputs);

        // Get the workgroup counts
        const workgroupCountX = Math.ceil(this._workgroupCountXFunc(env));
        const workgroupCountY = Math.ceil(this._workgroupCountYFunc(env));
        const workgroupCountZ = Math.ceil(this._workgroupCountZFunc(env));

        // Start a new command encoder
        const commandEncoder = this._device.createCommandEncoder();

        // Encode the kernel using pass encoder
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this._computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            workgroupCountX,
            workgroupCountY,
            workgroupCountZ
        );
        passEncoder.end();

        // Submit GPU commands
        const gpuCommands = commandEncoder.finish();
        this._device.queue.submit([gpuCommands]);

        // Return the storage output buffers
        return storageOutputs;
    }
    private getStorageInputBuffer(inputSpec: KernelInputSpec, providedInput: GPUBuffer, inputIndex: number, env: EvalEnv): GPUBuffer {
        if (providedInput.usage & GPUBufferUsage.STORAGE) {
            providedInput.unmap();
            return providedInput;
        }
        else {
            throw new Error("Provided input buffer is not a storage buffer");
        }
    }
    private getStorageOutputBuffer(outputSpec: KernelOutputSpec, providedOutput: GPUBuffer | null, outputIndex: number, env: EvalEnv): GPUBuffer {
        if (providedOutput !== null) {
            if (providedOutput.usage & GPUBufferUsage.STORAGE) {
                providedOutput.unmap();
                return providedOutput;
            }
            else {
                throw new Error("Provided output buffer is not a storage buffer");
            }
        }
        else {
            const outputElementByteSize = getShaderTypeElementByteSize(
                outputSpec.shaderType
            );
            const outputElementCount = Math.ceil(this._outputSizeFuncs[outputIndex](env));
            // console.log("output size", outputElementCount, outputElementByteSize);
            const outputBufferSize = outputElementByteSize * outputElementCount;
            const outputBuffer = this._device.createBuffer({
                mappedAtCreation: false,
                size: outputBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            return outputBuffer;
        }
    }
    private createBindGroup(
        inputBuffers: GPUBuffer[],
        paramsBuffer: GPUBuffer,
        outputBuffers: GPUBuffer[]
    ): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];
        let bindingIndex = 0;
        for (let i = 0; i < inputBuffers.length; i++, bindingIndex++) {
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: inputBuffers[i],
                },
            });
        }
        for (let i = 0; i < this._spec.outputs.length; i++, bindingIndex++) {
            const outputBuffer = outputBuffers[i];
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: outputBuffer,
                },
            });
            outputBuffers.push(outputBuffer);
        }
        entries.push({
            binding: bindingIndex,
            resource: {
                buffer: paramsBuffer,
            },
        });
        const bindGroup = this._device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: entries,
        });
        return bindGroup;
    }
}

export function getKernelConfig(
    spec: KernelSpec,
    config: KernelConfigInput
): KernelConfig {
    let configValues: KernelConfig = [];
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[configSpec.name];
        if (configValue === undefined) {
            throw new Error(
                `Missing config value for ${configSpec.name} in kernel ${spec.name}`
            );
        }
        configValues.push(configValue);
    }
    return configValues;
}

export function getKernelKey(
    spec: KernelSpec,
    config: KernelConfig
): KernelKey {
    let keyParts: string[] = [spec.name];
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        keyParts.push(`${configSpec.name}=${configValue}`);
    }
    return keyParts.join(",");
}

export function getKernelShaderCode(
    spec: KernelSpec,
    config: KernelConfig
): string {
    let shaderCodeParts: string[] = ["// " + spec.name + " kernel"];
    shaderCodeParts.push(`struct ${spec.name}Parameters {`);
    for (let i = 0; i < spec.parameters.length; i++) {
        let parameter = spec.parameters[i];
        shaderCodeParts.push(`    ${parameter.name}: ${parameter.shaderType},`);
    }
    shaderCodeParts.push(`}`);
    let bindingIndex = 0;
    for (let i = 0; i < spec.inputs.length; i++, bindingIndex++) {
        let input = spec.inputs[i];
        shaderCodeParts.push(
            `@group(0) @binding(${bindingIndex}) var<storage, read> ${input.name}: ${input.shaderType};`
        );
    }
    for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
        let output = spec.outputs[i];
        shaderCodeParts.push(
            `@group(0) @binding(${bindingIndex}) var<storage, read_write> ${output.name}: ${output.shaderType};`
        );
    }
    shaderCodeParts.push(
        `@group(0) @binding(${bindingIndex}) var<storage, read> parameters: ${spec.name}Parameters;`
    );
    const env: { [name: string]: any } = {};
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        env[configSpec.name] = configValue;
    }
    const workgroupSizeX = Math.ceil(evalCode(spec.workgroupSize[0], env));
    const workgroupSizeY = Math.ceil(evalCode(spec.workgroupSize[1], env));
    const workgroupSizeZ = Math.ceil(evalCode(spec.workgroupSize[2], env));
    shaderCodeParts.push(
        `@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})`
    );
    shaderCodeParts.push(
        `fn main(@builtin(global_invocation_id) global_id : vec3u) {`
    );
    shaderCodeParts.push("    " + spec.shader.trim());
    shaderCodeParts.push("}");
    return shaderCodeParts.join("\n");
}

function getShaderTypeElementByteSize(shaderType: ShaderType): number {
    switch (shaderType) {
        case "f32":
        case "i32":
        case "u32":
        case "array<f32>":
        case "array<i32>":
        case "array<u32>":
            return 4;
        case "u8":
        case "array<u8>":
            return 1;
        default:
            throw new Error(`Unknown shader type ${shaderType}`);
    }
}
