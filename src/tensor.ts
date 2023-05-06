import { ITensor, TensorArrayData, TensorJsonData } from "./tensor_if";
import { Deviceish } from "./device";
import { getDevice } from "./devices";
import { Shape } from "./shape";
import { ones } from "./factories";
import { Dtype } from "./dtype";
import { IDevice } from "./device_if";
import { UntypedStorage } from "./storage";
import { GradientFunction, GradientContext } from "./autograd";
import { TensorImpl } from "./tensor_impl";
import * as ops from "./ops";
import { KernelConfigInput, KernelParamsInput } from "./kernel";

export class Tensor implements ITensor {
    private _impl: TensorImpl;
    private _requiresGrad: boolean = false;
    private _gradFunc: GradientFunction | null;
    private _gradCtx: GradientContext | null;
    private _grad: Tensor | null = null;

    get impl(): TensorImpl {
        return this._impl;
    }

    get storage(): UntypedStorage {
        return this._impl.storage;
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
    get isContiguous(): boolean {
        return this._impl.isContiguous;
    }
    get isScalar(): boolean {
        return this._impl.isScalar;
    }

    get(...indices: number[]): number | ITensor {
        return this._impl.get(...indices);
    }

    get requiresGrad(): boolean {
        return this._requiresGrad;
    }
    set requiresGrad(value: boolean) {
        if (this._gradFunc) {
            throw new Error(
                "You can only change requiresGrad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use valueNoGrad = value.detach()."
            );
        }
        this._requiresGrad = value;
    }
    get gradFunc(): GradientFunction | null {
        return this._gradFunc;
    }
    get grad(): Tensor | null {
        return this._grad;
    }

    constructor(
        data: TensorArrayData | TensorJsonData | TensorImpl | null = null,
        dtype: Dtype = "float32",
        device: Deviceish | null = null,
        requiresGrad: boolean = false
    ) {
        if (data instanceof TensorImpl) {
            this._impl = data;
        } else if (data === null) {
            this._impl = getDevice(device).tensor(data, dtype);
        } else if (data instanceof Array) {
            this._impl = getDevice(device).tensor(data, dtype);
        } else if (data.hasOwnProperty("data")) {
            const jdata = data as TensorJsonData;
            if (jdata.data instanceof TensorImpl) {
                this._impl = jdata.data;
            } else {
                dtype = dtype || jdata.dtype;
                device = device || jdata.device || null;
                requiresGrad = requiresGrad || jdata.requiresGrad || false;
                this._impl = getDevice(device).tensor(jdata.data, dtype);
            }
        } else {
            throw new Error(
                "Invalid data type for Tensor constructor. Expected an array of values or a json object with a 'data' property."
            );
        }
        this._requiresGrad = requiresGrad;
        this._gradFunc = null;
        this._gradCtx = null;
        this._grad = null;
    }

    get [Symbol.toStringTag]() {
        return "Tensor";
    }
    toString(options?: {}): string {
        let rg = this.requiresGrad ? ", requiresGrad=true" : "";
        if (this._gradFunc) {
            rg = ", gradFunc";
        }
        return `tensor([${this.shape}], ${this.dtype}${rg})`;
    }
    async toArrayAsync(): Promise<TensorArrayData> {
        await this._impl.storage.mapReadAsync();
        const data = this._impl.getTypedArray();
        const shape = this._impl.shape;
        const strides = this._impl.strides;
        if (shape.length == 0 || (shape.length == 1 && shape[0] == 1)) {
            return [data[0]];
        }
        const index: number[] = [];
        return readArray(index);
        function readArray(index: number[]): TensorArrayData {
            const dim = index.length;
            // console.log("Read array: ", index, "dim=", dim);
            if (dim == shape.length - 1) {
                const offset = index.reduce(
                    (acc, cur, i) => acc + cur * strides[i],
                    0
                );
                // console.log("offset=", offset);
                const length = shape[dim];
                // console.log("length=", length);
                const subarray = data.subarray(offset, offset + length);
                // console.log("subarray=", subarray);
                const ar = Array.from(subarray);
                // console.log("ar=", ar);
                return ar;
            } else {
                const result: TensorArrayData = [];
                for (let i = 0; i < shape[dim]; i++) {
                    index.push(i);
                    result.push(readArray(index));
                    index.pop();
                }
                return result;
            }
        }
    }

    detach(): Tensor {
        if (this._requiresGrad || this._gradFunc) {
            return new Tensor({
                data: this._impl,
                dtype: this.dtype,
                requiresGrad: false,
            });
        }
        return this;
    }

    setGradientFunction(
        ctx: GradientContext,
        gradFunc: GradientFunction
    ): void {
        this._gradFunc = gradFunc;
        this._gradCtx = ctx;
        this._requiresGrad = true;
    }

    backward(gradient?: Tensor): void {
        let grad: Tensor;
        if (gradient) {
            grad = gradient;
        }
        else {
            if (!this.isScalar) {
                throw new Error(`Gradient can only be implicitly created for scalar outputs`);
            }
            grad = ones(1);
        }
        if (this._grad) {
            this._grad.add_(grad);
        } else {
            this._grad = grad;
        }
        if (!this._gradFunc || !this._gradCtx) {
            return;
        }
        const grads = this._gradFunc(this._gradCtx, grad);
        const inputs = this._gradCtx.inputsWithGradient;
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
                    `Gradient function did not return a gradient for input #${i} (out of ${inputs.length}). ${grads.length} gradients were returned.`
                );
            }
        }
    }

    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputShapes: Shape[],
        ...additionalInputs: Tensor[]
    ): Tensor[] {
        const impls = this.impl.runKernel(
            name,
            config,
            params,
            outputShapes,
            ...additionalInputs.map((t) => t.impl)
        );
        return impls.map((impl) => new Tensor(impl));
    }

    /** Returns a new view of this tensor with singleton dimensions expanded to a larger size.
    Passing -1 as the size for a dimension means not changing the size of that dimension.
    Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.
    Expanding a tensor does not allocate new memory, but only creates a new view on the existing tensor where a dimension of size one is expanded to a larger size by setting the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory. */
    expand(shape: Shape): Tensor {
        return new Tensor(this.impl.expand(shape));
    }
    mm(other: Tensor): Tensor {
        return ops.mm(this, other);
    }
    sum(axis: number | null = null): Tensor {
        return ops.sum(this, axis);
    }
    t(): Tensor {
        return ops.t(this);
    }

    // Codegen marker
    abs(): Tensor {
        return ops.abs(this);
    }
    absolute(): Tensor {
        return ops.abs(this);
    }
    abs_(): Tensor {
        this._impl.abs_();
        return this;
    }
    acos(): Tensor {
        return ops.acos(this);
    }
    arccos(): Tensor {
        return ops.acos(this);
    }
    acos_(): Tensor {
        this._impl.acos_();
        return this;
    }
    acosh(): Tensor {
        return ops.acosh(this);
    }
    arccosh(): Tensor {
        return ops.acosh(this);
    }
    acosh_(): Tensor {
        this._impl.acosh_();
        return this;
    }
    add(other: Tensor, alpha?: number): Tensor {
        return ops.add(this, other, alpha);
    }
    add_(other: Tensor, alpha?: number): Tensor {
        this._impl.add_(other._impl, alpha);
        return this;
    }
    asin(): Tensor {
        return ops.asin(this);
    }
    arcsin(): Tensor {
        return ops.asin(this);
    }
    asin_(): Tensor {
        this._impl.asin_();
        return this;
    }
    asinh(): Tensor {
        return ops.asinh(this);
    }
    arcsinh(): Tensor {
        return ops.asinh(this);
    }
    asinh_(): Tensor {
        this._impl.asinh_();
        return this;
    }
    atan(): Tensor {
        return ops.atan(this);
    }
    arctan(): Tensor {
        return ops.atan(this);
    }
    atan_(): Tensor {
        this._impl.atan_();
        return this;
    }
    atan2(other: Tensor): Tensor {
        return ops.atan2(this, other);
    }
    arctan2(other: Tensor): Tensor {
        return ops.atan2(this, other);
    }
    atan2_(other: Tensor): Tensor {
        this._impl.atan2_(other._impl);
        return this;
    }
    ceil(): Tensor {
        return ops.ceil(this);
    }
    ceil_(): Tensor {
        this._impl.ceil_();
        return this;
    }
    copysign(other: Tensor): Tensor {
        return ops.copysign(this, other);
    }
    copysign_(other: Tensor): Tensor {
        this._impl.copysign_(other._impl);
        return this;
    }
    cos(): Tensor {
        return ops.cos(this);
    }
    cos_(): Tensor {
        this._impl.cos_();
        return this;
    }
    cosh(): Tensor {
        return ops.cosh(this);
    }
    cosh_(): Tensor {
        this._impl.cosh_();
        return this;
    }
    deg2rad(): Tensor {
        return ops.deg2rad(this);
    }
    deg2rad_(): Tensor {
        this._impl.deg2rad_();
        return this;
    }
    div(other: Tensor, alpha?: number): Tensor {
        return ops.div(this, other, alpha);
    }
    divide(other: Tensor, alpha?: number): Tensor {
        return ops.div(this, other, alpha);
    }
    div_(other: Tensor, alpha?: number): Tensor {
        this._impl.div_(other._impl, alpha);
        return this;
    }
    exp(): Tensor {
        return ops.exp(this);
    }
    exp_(): Tensor {
        this._impl.exp_();
        return this;
    }
    exp2(): Tensor {
        return ops.exp2(this);
    }
    exp2_(): Tensor {
        this._impl.exp2_();
        return this;
    }
    expm1(): Tensor {
        return ops.expm1(this);
    }
    expm1_(): Tensor {
        this._impl.expm1_();
        return this;
    }
    floor(): Tensor {
        return ops.floor(this);
    }
    floor_(): Tensor {
        this._impl.floor_();
        return this;
    }
    floor_divide(other: Tensor): Tensor {
        return ops.floor_divide(this, other);
    }
    floor_divide_(other: Tensor): Tensor {
        this._impl.floor_divide_(other._impl);
        return this;
    }
    frac(): Tensor {
        return ops.frac(this);
    }
    frac_(): Tensor {
        this._impl.frac_();
        return this;
    }
    hypot(other: Tensor): Tensor {
        return ops.hypot(this, other);
    }
    hypot_(other: Tensor): Tensor {
        this._impl.hypot_(other._impl);
        return this;
    }
    ldexp(other: Tensor): Tensor {
        return ops.ldexp(this, other);
    }
    ldexp_(other: Tensor): Tensor {
        this._impl.ldexp_(other._impl);
        return this;
    }
    log(): Tensor {
        return ops.log(this);
    }
    log_(): Tensor {
        this._impl.log_();
        return this;
    }
    log10(): Tensor {
        return ops.log10(this);
    }
    log10_(): Tensor {
        this._impl.log10_();
        return this;
    }
    log1p(): Tensor {
        return ops.log1p(this);
    }
    log1p_(): Tensor {
        this._impl.log1p_();
        return this;
    }
    log2(): Tensor {
        return ops.log2(this);
    }
    log2_(): Tensor {
        this._impl.log2_();
        return this;
    }
    logaddexp(other: Tensor): Tensor {
        return ops.logaddexp(this, other);
    }
    logaddexp_(other: Tensor): Tensor {
        this._impl.logaddexp_(other._impl);
        return this;
    }
    logaddexp2(other: Tensor): Tensor {
        return ops.logaddexp2(this, other);
    }
    logaddexp2_(other: Tensor): Tensor {
        this._impl.logaddexp2_(other._impl);
        return this;
    }
    mul(other: Tensor, alpha?: number): Tensor {
        return ops.mul(this, other, alpha);
    }
    multiply(other: Tensor, alpha?: number): Tensor {
        return ops.mul(this, other, alpha);
    }
    mul_(other: Tensor, alpha?: number): Tensor {
        this._impl.mul_(other._impl, alpha);
        return this;
    }
    neg(): Tensor {
        return ops.neg(this);
    }
    negative(): Tensor {
        return ops.neg(this);
    }
    neg_(): Tensor {
        this._impl.neg_();
        return this;
    }
    positive(): Tensor {
        return ops.positive(this);
    }
    positive_(): Tensor {
        this._impl.positive_();
        return this;
    }
    pow(other: Tensor): Tensor {
        return ops.pow(this, other);
    }
    pow_(other: Tensor): Tensor {
        this._impl.pow_(other._impl);
        return this;
    }
    rad2deg(): Tensor {
        return ops.rad2deg(this);
    }
    rad2deg_(): Tensor {
        this._impl.rad2deg_();
        return this;
    }
    reciprocal(): Tensor {
        return ops.reciprocal(this);
    }
    reciprocal_(): Tensor {
        this._impl.reciprocal_();
        return this;
    }
    round(): Tensor {
        return ops.round(this);
    }
    round_(): Tensor {
        this._impl.round_();
        return this;
    }
    rsqrt(): Tensor {
        return ops.rsqrt(this);
    }
    rsqrt_(): Tensor {
        this._impl.rsqrt_();
        return this;
    }
    sigmoid(): Tensor {
        return ops.sigmoid(this);
    }
    sigmoid_(): Tensor {
        this._impl.sigmoid_();
        return this;
    }
    sign(): Tensor {
        return ops.sign(this);
    }
    sign_(): Tensor {
        this._impl.sign_();
        return this;
    }
    sin(): Tensor {
        return ops.sin(this);
    }
    sin_(): Tensor {
        this._impl.sin_();
        return this;
    }
    sinc(): Tensor {
        return ops.sinc(this);
    }
    sinc_(): Tensor {
        this._impl.sinc_();
        return this;
    }
    sinh(): Tensor {
        return ops.sinh(this);
    }
    sinh_(): Tensor {
        this._impl.sinh_();
        return this;
    }
    sqrt(): Tensor {
        return ops.sqrt(this);
    }
    sqrt_(): Tensor {
        this._impl.sqrt_();
        return this;
    }
    square(): Tensor {
        return ops.square(this);
    }
    square_(): Tensor {
        this._impl.square_();
        return this;
    }
    sub(other: Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    subtract(other: Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    sub_(other: Tensor, alpha?: number): Tensor {
        this._impl.sub_(other._impl, alpha);
        return this;
    }
    tan(): Tensor {
        return ops.tan(this);
    }
    tan_(): Tensor {
        this._impl.tan_();
        return this;
    }
    tanh(): Tensor {
        return ops.tanh(this);
    }
    tanh_(): Tensor {
        this._impl.tanh_();
        return this;
    }
    trunc(): Tensor {
        return ops.trunc(this);
    }
    fix(): Tensor {
        return ops.trunc(this);
    }
    trunc_(): Tensor {
        this._impl.trunc_();
        return this;
    }
    xlogy(other: Tensor): Tensor {
        return ops.xlogy(this, other);
    }
    xlogy_(other: Tensor): Tensor {
        this._impl.xlogy_(other._impl);
        return this;
    }
    // End codegen marker
}
