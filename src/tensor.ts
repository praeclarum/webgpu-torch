import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";
import { Device, DeviceType } from "./device";
import { getDevice } from "./devices";
import { Shape } from "./shape";
import { ones } from "./factories";
import { Dtype } from "./dtype";
import { IDevice } from "./device_if";
import { add_, mm, sum, t } from "./ops";

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
    private _impl: TensorImpl;
    private _requiresGrad: boolean = false;
    private _gradFunc: GradientFunction | null;
    private _gradCtx: GradientFunctionContext | null;
    private _grad: Tensor | null = null;

    get impl(): TensorImpl {
        return this._impl;
    }

    get dtype(): Dtype {
        return this._impl.dtype;
    }
    get shape(): Shape {
        return this._impl.shape;
    }
    get device(): IDevice {
        return this._impl.device;
    }

    get(...indices: number[]): number | ITensor {
        return this._impl.get(...indices);
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
        data: TensorArrayData | TensorImpl,
        dtype: Dtype = "float32",
        requiresGrad: boolean = false,
        device: string | Device | DeviceType | null = null
    ) {
        if (data instanceof TensorImpl) {
            this._impl = data;
        } else {
            this._impl = getDevice(device).tensor(data, dtype);
        }
        this._requiresGrad = requiresGrad;
        this._gradFunc = null;
        this._gradCtx = null;
        this._grad = null;
    }

    detach(): Tensor {
        return new Tensor(this._impl, this.dtype, false);
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

    add_(other: Tensor): Tensor {
        return add_(this, other);
    }
    mm(other: Tensor): Tensor {
        return mm(this, other);
    }
    sum(axis: number | null = null): Tensor {
        return sum(this, axis);
    }
    t(): Tensor {
        return t(this);
    }
}
