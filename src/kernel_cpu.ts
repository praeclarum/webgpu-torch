import { Device } from "./device";
import { ATypedArray } from "./dtype";
import { EvalEnv } from "./expr";
import {
    Kernel,
    KernelConfig,
    KernelInputSpec,
    KernelOutputCompiledSpec,
    KernelOutputSpec,
    KernelParamsInput,
    KernelSpec,
    getKernelJavaScriptCode,
    getShaderTypeElementByteSize,
    shaderTypeToDtype,
} from "./kernel";
import { ArrayBufferStorage, UntypedStorage } from "./storage";

export class KernelCPU extends Kernel {
    private _javaScriptCode: string;
    private _javaScriptFunction: Function;

    constructor(spec: KernelSpec, config: KernelConfig, device: Device) {
        super(spec, config, device, jsMathEnv);
        this._javaScriptCode = getKernelJavaScriptCode(spec, config);
        this._javaScriptFunction = eval(this._javaScriptCode);
    }

    run(
        inputs: UntypedStorage[],
        parameters: KernelParamsInput,
        outputs?: UntypedStorage[]
    ): UntypedStorage[] {
        // console.log("run cpu kernel", this.key);
        if (inputs.length !== this.spec.inputs.length) {
            throw new Error(
                `Expected ${this.spec.inputs.length} inputs for kernel \"${this.spec.name}\", got ${inputs.length}`
            );
        }

        // Build the parameter environment
        const [env, paramValues] = this.getRunEnv(parameters);

        // Build up the args
        const args: any[] = [];
        const outputsToReturn: UntypedStorage[] = [];

        // Get input buffers with storage usage
        this.spec.inputs.forEach((input, i) => {
            const o = this.getStorageInputBuffer(
                input,
                inputs[i],
                i,
                env
            );
            const arg = o.getTypedArray(
                shaderTypeToDtype(input.shaderType)
            );
            args.push(arg);
        });

        // Get output buffers with storage usage
        this.spec.outputs.forEach((output, i) => {
            const o = this.getStorageOutputBuffer(
                output,
                outputs ? outputs[i] : null,
                i,
                env
            );
            const arg = o.getTypedArray(
                shaderTypeToDtype(output.shaderType)
            );
            args.push(arg);
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
        providedInput: UntypedStorage,
        inputIndex: number,
        env: EvalEnv
    ): ArrayBufferStorage {
        if (providedInput instanceof ArrayBufferStorage) {
            return providedInput;
        }
        throw new Error(
            `Input buffer #${inputIndex} (out of ${this.spec.inputs.length}) named "${inputSpec.name}" in kernel "${this.key}" is not an ArrayBufferStorage`
        );
    }

    private getStorageOutputBuffer(
        outputSpec: KernelOutputCompiledSpec,
        providedOutput: UntypedStorage | null,
        outputIndex: number,
        env: EvalEnv
    ): ArrayBufferStorage {
        if (providedOutput !== null) {
            if (providedOutput instanceof ArrayBufferStorage) {
                return providedOutput;
            }
            throw new Error(
                `Output buffer #${outputIndex} (out of ${this.spec.outputs.length}) named "${outputSpec.name}" in kernel "${this.key}" is not an ArrayBufferStorage. It's a ${providedOutput.constructor.name}`
            );
        } else {
            const outputElementByteSize = getShaderTypeElementByteSize(
                outputSpec.shaderType
            );
            const outputElementCount = Math.ceil(
                this._spec.outputs[outputIndex].size(env)
            );
            const outputBufferSize = outputElementByteSize * outputElementCount;
            // console.log("output", outputSpec.name, "size:", outputElementCount, "* byte size:", outputElementByteSize, "= buffer size:", outputBufferSize);
            const outputHeapBuffer = this.device.heapAlloc(outputBufferSize);
            return outputHeapBuffer as ArrayBufferStorage;
            // const outputBuffer = this.device.alloc(outputBufferSize);
            // return outputBuffer;
        }
    }
}

const jsMathEnv: EvalEnv = {};
for (const name of Object.getOwnPropertyNames(Math)) {
    const f = (Math as any)[name];
    if (typeof f === "function") {
        jsMathEnv[name] = f;
    }
}
