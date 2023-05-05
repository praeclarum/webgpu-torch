import { Device } from "./device";
import { ATypedArray } from "./dtype";
import { EvalEnv } from "./expr";
import {
    Kernel,
    KernelConfig,
    KernelInputSpec,
    KernelOutputSpec,
    KernelParamsInput,
    KernelSpec,
    getKernelShaderCode,
    getShaderTypeElementByteSize,
    shaderTypeToDtype,
} from "./kernel";

export class KernelCPU extends Kernel {
    private _shaderCode: string;

    constructor(spec: KernelSpec, config: KernelConfig, device: Device) {
        super(spec, config, device);
        this._shaderCode = getKernelShaderCode(spec, config);
    }

    run(
        inputs: ATypedArray[],
        parameters: KernelParamsInput,
        outputs?: ATypedArray[]
    ): ATypedArray[] {
        console.log("run cpu kernel", this.key);

        // Build the parameter environment
        const env = this.getRunEnv(parameters);

        // Get the workgroup counts
        const [workgroupCountX, workgroupCountY, workgroupCountZ] =
            this.getWorkgroupCounts(env);

        // Get input buffers with storage usage
        const storageInputs = this.spec.inputs.map((input, i) =>
            this.getStorageInputBuffer(
                input,
                inputs[i] ? inputs[i] : null,
                i,
                env
            )
        );

        // Get output buffers with storage usage
        const storageOutputs = this.spec.outputs.map((output, i) =>
            this.getStorageOutputBuffer(
                output,
                outputs ? outputs[i] : null,
                i,
                env
            )
        );

        throw new Error("CPU kernel runs are not implemented");
    }

    private getStorageInputBuffer(
        inputSpec: KernelInputSpec,
        providedInput: ATypedArray | null,
        inputIndex: number,
        env: EvalEnv
    ): ATypedArray {
        if (providedInput === null) {
            throw new Error(
                `Missing input buffer #${inputIndex} (out of ${this.spec.inputs.length}) named "${inputSpec.name}" in kernel "${this.key}"`
            );
        }
        return providedInput;
    }

    private getStorageOutputBuffer(
        outputSpec: KernelOutputSpec,
        providedOutput: ATypedArray | null,
        outputIndex: number,
        env: EvalEnv
    ): ATypedArray {
        if (providedOutput !== null) {
            return providedOutput;
        } else {
            const outputElementByteSize = getShaderTypeElementByteSize(
                outputSpec.shaderType
            );
            const outputElementCount = Math.ceil(
                this._outputSizeFuncs[outputIndex](env)
            );
            // console.log("output size", outputElementCount, outputElementByteSize);
            const outputBufferSize = outputElementByteSize * outputElementCount;
            const outputBuffer = this.device.alloc(outputBufferSize);
            return outputBuffer.getTypedArray(
                shaderTypeToDtype(outputSpec.shaderType)
            );
        }
    }
}
