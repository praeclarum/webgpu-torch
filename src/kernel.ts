import { FunctionInput } from "./tensor";

export type ShaderType = "u8" | "i32" | "u32" | "f32";

export interface KernelSpec {
    name: string;
    config: KernelConfigSpec[];
    parameters: KernelParamSpec[];
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
}

export interface KernelParamSpec {
    name: string;
    shaderType: ShaderType;
}

export interface KernelConfigSpec {
    name: string;
}

type ShaderValue = FunctionInput;
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
            let input = spec.inputs[i];
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage" as GPUBufferBindingType,
                },
            });
        }
        for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
            let output = spec.outputs[i];
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage" as GPUBufferBindingType,
                },
            });
        }
        this._bindGroupLayout = device.createBindGroupLayout({
            entries: bindGroupLayoutEntries,
        });
        const shaderCodeParts: string[] = [];
        shaderCodeParts.push(spec.shader);
        this._shaderCode = shaderCodeParts.join("\n");
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
    }
    run(inputs: GPUBuffer[], parameters: ShaderValue[]) {
        const commandEncoder = this._device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this._computePipeline);
        const bindGroup = this.createBindGroup(inputs);
        passEncoder.setBindGroup(0, bindGroup.bindGroup);
        const workgroupCountX = 1;// TODO: Compute workgroup size Math.ceil(resultRows / 8);
        const workgroupCountY = 1;// TODO: Math.ceil(resultCols / 8);
        const workgroupCountZ = 1;
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
        passEncoder.end();
    }
    private createBindGroup(inputBuffers: GPUBuffer[]): {bindGroup:GPUBindGroup} {
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
        const outputBuffers: GPUBuffer[] = [];
        const readBuffers: GPUBuffer[] = [];
        for (let i = 0; i < this._spec.outputs.length; i++, bindingIndex++) {
            const outputSpec = this._spec.outputs[i];
            const outputBufferSize = 4;// TODO: Compute output buffer size
            const outputBuffer = this._device.createBuffer({
                mappedAtCreation: false,
                size: outputBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            const readBuffer = this._device.createBuffer({
                mappedAtCreation: false,
                size: outputBufferSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: outputBuffer,
                },
            });
            outputBuffers.push(outputBuffer);
            readBuffers.push(readBuffer);
        }
        const bindGroup = this._device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: entries,
        });
        return { bindGroup };
    }    
}

export function getKernelConfig(spec: KernelSpec, config: KernelConfigInput): KernelConfig {
    let configValues: KernelConfig = [];
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[configSpec.name];
        if (configValue === undefined) {
            throw new Error(`Missing config value for ${configSpec.name} in kernel ${spec.name}`);
        }
        configValues.push(configValue);
    }
    return configValues;
}

export function getKernelKey(spec: KernelSpec, config: KernelConfig): KernelKey {
    let keyParts: string[] = [spec.name];
    for (let i = 0; i < spec.config.length; i++) {
        let configSpec = spec.config[i];
        let configValue = config[i];
        keyParts.push(`${configSpec.name}=${configValue}`);
    }
    return keyParts.join(",");
}
