import { FunctionInput } from "./tensor";

export type ShaderType = "u8" | "i32" | "u32" | "f32";

export type ShaderValue = FunctionInput

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
    shaderType: string;
}

export class Kernel {
    private _device: GPUDevice;
    private _spec: KernelSpec;
    private _shaderCode: string;
    private _bindGroupLayout: GPUBindGroupLayout;
    private _computePipeline: GPUComputePipeline;
    get spec(): KernelSpec {
        return this._spec;
    }
    constructor(device: GPUDevice, spec: KernelSpec, params: ShaderValue[]) {
        this._device = device;
        this._spec = spec;
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
        passEncoder.setBindGroup(0, this.createBindGroup(inputs));
        const workgroupCountX = 1;// TODO: Compute workgroup size Math.ceil(resultRows / 8);
        const workgroupCountY = 1;// TODO: Math.ceil(resultCols / 8);
        const workgroupCountZ = 1;
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
        passEncoder.end();
    }
    private createBindGroup(inputBuffers: GPUBuffer[]): GPUBindGroup {
        let entries: GPUBindGroupEntry[] = [];
        for (let i = 0; i < inputBuffers.length; i++) {
            entries.push({
                binding: i,
                resource: {
                    buffer: inputBuffers[i],
                },
            });
        }
        return this._device.createBindGroup({
            layout: this._bindGroupLayout,
            entries: entries,
        });
    }    
}
