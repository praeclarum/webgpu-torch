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
    getKernelJavaScriptCode,
    getShaderTypeElementByteSize,
    shaderTypeToDtype,
} from "./kernel";

export class KernelCPU extends Kernel {
    private _javaScriptCode: string;
    private _javaScriptFunction: Function;

    constructor(spec: KernelSpec, config: KernelConfig, device: Device) {
        super(spec, config, device);
        this._javaScriptCode = getKernelJavaScriptCode(spec, config);
        this._javaScriptFunction = eval(this._javaScriptCode);
    }

    run(
        inputs: ATypedArray[],
        parameters: KernelParamsInput,
        outputs?: ATypedArray[]
    ): ATypedArray[] {
        // console.log("run cpu kernel", this.key);

        // Build the parameter environment
        const env = this.getRunEnv(parameters);

        // Build up the args
        const args: any[] = [];
        const outputsToReturn: ATypedArray[] = [];

        // Get input buffers with storage usage
        this.spec.inputs.forEach((input, i) =>
            args.push(
                this.getStorageInputBuffer(
                    input,
                    inputs[i] ? inputs[i] : null,
                    i,
                    env
                )
            )
        );

        // Get output buffers with storage usage
        this.spec.outputs.forEach((output, i) => {
            const o = this.getStorageOutputBuffer(
                output,
                outputs ? outputs[i] : null,
                i,
                env
            );
            args.push(o);
            outputsToReturn.push(o);
        });

        // Add the parameters
        args.push(parameters);

        // Finally the workgroupCount
        const [workgroupCountX, workgroupCountY, workgroupCountZ] =
            this.getWorkgroupCounts(env);
        args.push(workgroupCountX);
        args.push(workgroupCountY);
        args.push(workgroupCountZ);

        // Run the kernel
        this._javaScriptFunction.apply(null, args);

        return outputsToReturn;
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
            const outputBufferSize = outputElementByteSize * outputElementCount;
            // console.log("output", outputSpec.name, "size:", outputElementCount, "* byte size:", outputElementByteSize, "= buffer size:", outputBufferSize);
            const outputBuffer = this.device.alloc(outputBufferSize);
            return outputBuffer.getTypedArray(
                shaderTypeToDtype(outputSpec.shaderType)
            );
        }
    }
}
