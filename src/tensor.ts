import { ITensor, TensorImpl } from "./tensor_if";
import { Device, DeviceType } from "./device";
import { getDevice } from "./devices";
import { UntypedStorage } from "./storage";
import { Shape, Strides } from "./shape";
import { ones } from "./factories";

export type Dtype = "float32" | "int32" | "boolean" | "string";

export type FunctionInput = Tensor | number | boolean | string;
export type GradientFunctionOutput = Tensor | null;

export class GradientFunctionContext {
    needsInputGradient: boolean[];
    inputsWithGradient: (Tensor | null)[];
    savedTensors: Tensor[] = [];
    constructor(inputs: FunctionInput[]) {
        this.needsInputGradient = inputs.map(
            (input) => input instanceof Tensor && input.requiresGrad
        );
        this.inputsWithGradient = inputs.map((input) =>
            input instanceof Tensor && input.requiresGrad ? input : null
        );
    }
    saveForBackward(...tensors: Tensor[]) {
        this.savedTensors = tensors;
    }
}

export type GradientFunction = (
    ctx: GradientFunctionContext,
    output: Tensor
) => (Tensor | null)[];

export class Tensor implements ITensor {
    private _device: Device;
    private _untypedStorage: UntypedStorage;
    private _dtype: Dtype;
    private _requiresGrad: boolean = false;
    private _gradFunc: GradientFunction | null;
    private _gradCtx: GradientFunctionContext | null;
    private _grad: Tensor | null = null;

    get untypedStorage(): UntypedStorage {
        return this._untypedStorage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get requiresGrad(): boolean {
        return this._requiresGrad;
    }
    set requiresGrad(value: boolean) {
        this._requiresGrad = value;
    }
    get gradFunc(): GradientFunction | null {
        return this._gradFunc;
    }
    get grad(): Tensor | null {
        return this._grad;
    }

    constructor(
        data: Array<any> | TensorImpl,
        dtype: Dtype,
        requiresGrad: boolean = false,
        device: string | Device | DeviceType | null = null
    ) {
        this._device = getDevice(device);
        this._untypedStorage = new UntypedStorage();
        this._dtype = dtype;
        this._requiresGrad = requiresGrad;
        this._gradFunc = null;
        this._gradCtx = null;
        this._grad = null;
    }
    detach(): Tensor {
        return new Tensor(this._untypedStorage, this._dtype);
    }
    setGradientFunction(
        ctx: GradientFunctionContext,
        gradFunc: GradientFunction
    ): void {
        this._gradFunc = gradFunc;
        this._gradCtx = ctx;
        this._requiresGrad = true;
    }

    backward(gradient?: Tensor): void {
        const grad = gradient || ones(1);

        if (this._grad) {
            this._grad.add_(grad);
        } else {
            this._grad = grad;
        }
        if (!this._gradFunc || !this._gradCtx) {
            return;
        }
        // console.log("GRADIENT OF " + this + " IS " + grad + "")
        const grads = this._gradFunc(this._gradCtx, grad);
        // console.log(grads)
        const inputs = this._gradCtx.inputsWithGradient;
        // console.log(inputs);
        for (let i = 0; i < inputs.length; i++) {
            const input = inputs[i];
            if (input === null) {
                continue;
            }
            const grad = grads[i];
            if (grad) {
                input.backward(grad);
            } else {
                throw new Error(
                    "Gradient function did not return a gradient for input " + i
                );
            }
        }
    }

    add_(tensor: Tensor): Tensor {
        return this;
    }
    mm(tensor: Tensor): Tensor {
        return new Tensor(this._untypedStorage, this._dtype);
    }
    sum(arg0: number) {
        return new Tensor(this._untypedStorage, this._dtype);
    }
    t(): Tensor {
        return new Tensor(this._untypedStorage, this._dtype);
    }
}
