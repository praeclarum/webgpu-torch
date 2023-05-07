import { Device, Deviceish } from "./device";
import { getDevice } from "./devices";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { ones } from "./factories";
import { Dtype } from "./dtype";
import { UntypedStorage } from "./storage";
import { GradientFunction, GradientContext } from "./autograd";
import * as ops from "./ops";
import { KernelConfigInput, KernelParamsInput } from "./kernel";

export type TensorArrayData = Array<number | TensorArrayData>;

export type TensorJsonData = {
    data: TensorArrayData | UntypedStorage;
    dtype?: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
    shape?: Shape;
    strides?: Strides;
};

export class Tensor {
    private _device: Device;

    private _storage: UntypedStorage;
    private _dtype: Dtype;
    private _shape: Shape;
    private _strides: Strides;
    // private _offset: number;

    private _requiresGrad: boolean = false;
    private _gradFunc: GradientFunction | null;
    private _gradCtx: GradientContext | null;
    private _grad: Tensor | null = null;

    get storage(): UntypedStorage {
        return this._storage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get shape(): Shape {
        return this._shape;
    }
    get strides(): Strides {
        return this._strides;
    }
    get device(): Device {
        return this._device;
    }
    get isContiguous(): boolean {
        let strides = this.strides;
        let shape = this.shape;
        let offset = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== offset) {
                return false;
            }
            offset *= shape[i];
        }
        return true;
    }
    private get isScalar(): boolean {
        return this.shape.length === 0 || (this.shape.length === 1 && this.shape[0] === 1);
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
        data: TensorArrayData | TensorJsonData | null = null,
        dtype: Dtype = "float32",
        device: Deviceish | null = null,
        requiresGrad: boolean = false
    ) {
        const d = getDevice(device);
        this._device = d;
        if (data === null) {
            // this._impl = d.tensor(data, dtype);
        } else if (data instanceof Array) {
            // this._impl = d.tensor(data, dtype);
        } else if (data.hasOwnProperty("data")) {
            const jdata = data as TensorJsonData;
            if (jdata.data instanceof UntypedStorage) {
                throw new Error("Untyped Not implemented");
            } else {
                dtype = dtype || jdata.dtype;
                device = device || jdata.device || null;
                requiresGrad = requiresGrad || jdata.requiresGrad || false;
                // this._impl = d.tensor(jdata.data, dtype);
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
    withShape(shape: Shape, strides: Strides): Tensor {
        if (shapeSize(shape) != shapeSize(this.shape)) {
            throw new Error(
                `Cannot reshape tensor of size ${this.shape} to ${shape}`
            );
        }
        return new Tensor({
            data: this.storage,
            dtype: this.dtype,
            device: this.device,
            requiresGrad: this.requiresGrad,
            shape,
            strides,
        });
    }

    get [Symbol.toStringTag]() {
        return "Tensor";
    }
    toString(): string {
        let rg = this.requiresGrad ? ", requiresGrad=true" : "";
        if (this._gradFunc) {
            rg = ", gradFunc";
        }
        return `tensor([${this.shape}], ${this.dtype}${rg})`;
    }
    async toArrayAsync(): Promise<TensorArrayData> {
        await this.storage.mapReadAsync();
        const data = this.storage.getTypedArray(this.dtype);
        const shape = this.shape;
        const strides = this.strides;
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
                data: this.storage,
                dtype: this.dtype,
                requiresGrad: false,
                shape: this.shape,
                strides: this.strides,
                device: this.device,
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
        } else {
            if (!this.isScalar) {
                throw new Error(
                    `Gradient can only be implicitly created for scalar outputs`
                );
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

    runKernelInplace(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        ...additionalInputs: Tensor[]
    ): Tensor {
        const d = this.device;
        const kernel = this.device.getKernel(name, config);
        const inputBuffers = additionalInputs.map((t) =>
            d.unwrapKernelStorage(t.storage)
        );
        const outputBuffers = [d.unwrapKernelStorage(this.storage)];
        kernel.run(inputBuffers, params, outputBuffers);
        return this;
    }
    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputShapes: Shape[],
        ...additionalInputs: Tensor[]
    ): Tensor[] {
        const d = this.device;
        const kernel = d.getKernel(name, config);
        const inputBuffers = [
            d.unwrapKernelStorage(this.storage),
            ...additionalInputs.map((t) => d.unwrapKernelStorage(t.storage)),
        ];
        const outputBuffers = kernel.run(inputBuffers, params) as GPUBuffer[];
        if (outputBuffers.length !== outputShapes.length) {
            throw new Error(
                `Expected ${outputShapes.length} output buffers (given the provided outputShapes to runKernel), but got ${outputBuffers.length} output buffers when running the kernel "${name}".`
            );
        }
        return outputBuffers.map(
            (outputBuffer, i) =>
                new Tensor({
                    data: this.device.wrapKernelStorage(outputBuffer),
                    dtype: this.dtype,
                    shape: outputShapes[i],
                    strides: defaultStrides(outputShapes[i]),
                    device: this.device,
                })
        );
    }

    expand(shape: Shape): Tensor {
        const newShape = shape.slice();
        const newStrides = Array(newShape.length).fill(0);
        // Update newStrides based on the current strides
        // so that the expansion happens
        // in the correct direction
        let j = newShape.length - 1;
        let thisShape = this.shape;
        let thisStrides = this.strides;
        for (let i = thisShape.length - 1; i >= 0; i--) {
            if (thisShape[i] === 1) {
                newStrides[j] = 0;
            } else {
                newStrides[j] = thisStrides[i];
                j--;
            }
            if (newShape[j] === -1) {
                newShape[j] = thisShape[i];
            }
        }
        // console.log("EXPAND", this.shape, this.strides, shape, newShape, newStrides);
        return this.withShape(newShape, newStrides);
    }
    mm(other: Tensor): Tensor {
        return ops.mm(this, other);
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
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("abs_", { dtype: this.dtype }, params);
    }
    acos(): Tensor {
        return ops.acos(this);
    }
    arccos(): Tensor {
        return ops.acos(this);
    }
    acos_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acos_", { dtype: this.dtype }, params);
    }
    acosh(): Tensor {
        return ops.acosh(this);
    }
    arccosh(): Tensor {
        return ops.acosh(this);
    }
    acosh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acosh_", { dtype: this.dtype }, params);
    }
    add(other: Tensor, alpha?: number): Tensor {
        return ops.add(this, other, alpha);
    }
    add_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace(
            "add_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    asin(): Tensor {
        return ops.asin(this);
    }
    arcsin(): Tensor {
        return ops.asin(this);
    }
    asin_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asin_", { dtype: this.dtype }, params);
    }
    asinh(): Tensor {
        return ops.asinh(this);
    }
    arcsinh(): Tensor {
        return ops.asinh(this);
    }
    asinh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asinh_", { dtype: this.dtype }, params);
    }
    atan(): Tensor {
        return ops.atan(this);
    }
    arctan(): Tensor {
        return ops.atan(this);
    }
    atan_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan_", { dtype: this.dtype }, params);
    }
    atan2(other: Tensor): Tensor {
        return ops.atan2(this, other);
    }
    arctan2(other: Tensor): Tensor {
        return ops.atan2(this, other);
    }
    atan2_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "atan2_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    ceil(): Tensor {
        return ops.ceil(this);
    }
    ceil_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ceil_", { dtype: this.dtype }, params);
    }
    copysign(other: Tensor): Tensor {
        return ops.copysign(this, other);
    }
    copysign_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "copysign_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    cos(): Tensor {
        return ops.cos(this);
    }
    cos_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cos_", { dtype: this.dtype }, params);
    }
    cosh(): Tensor {
        return ops.cosh(this);
    }
    cosh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cosh_", { dtype: this.dtype }, params);
    }
    deg2rad(): Tensor {
        return ops.deg2rad(this);
    }
    deg2rad_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("deg2rad_", { dtype: this.dtype }, params);
    }
    div(other: Tensor, alpha?: number): Tensor {
        return ops.div(this, other, alpha);
    }
    divide(other: Tensor, alpha?: number): Tensor {
        return ops.div(this, other, alpha);
    }
    div_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace(
            "div_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    exp(): Tensor {
        return ops.exp(this);
    }
    exp_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp_", { dtype: this.dtype }, params);
    }
    exp2(): Tensor {
        return ops.exp2(this);
    }
    exp2_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp2_", { dtype: this.dtype }, params);
    }
    expm1(): Tensor {
        return ops.expm1(this);
    }
    expm1_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("expm1_", { dtype: this.dtype }, params);
    }
    floor(): Tensor {
        return ops.floor(this);
    }
    floor_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_", { dtype: this.dtype }, params);
    }
    floor_divide(other: Tensor): Tensor {
        return ops.floor_divide(this, other);
    }
    floor_divide_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "floor_divide_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    frac(): Tensor {
        return ops.frac(this);
    }
    frac_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("frac_", { dtype: this.dtype }, params);
    }
    hypot(other: Tensor): Tensor {
        return ops.hypot(this, other);
    }
    hypot_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "hypot_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    ldexp(other: Tensor): Tensor {
        return ops.ldexp(this, other);
    }
    ldexp_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "ldexp_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    log(): Tensor {
        return ops.log(this);
    }
    log_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log_", { dtype: this.dtype }, params);
    }
    log10(): Tensor {
        return ops.log10(this);
    }
    log10_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log10_", { dtype: this.dtype }, params);
    }
    log1p(): Tensor {
        return ops.log1p(this);
    }
    log1p_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log1p_", { dtype: this.dtype }, params);
    }
    log2(): Tensor {
        return ops.log2(this);
    }
    log2_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log2_", { dtype: this.dtype }, params);
    }
    logaddexp(other: Tensor): Tensor {
        return ops.logaddexp(this, other);
    }
    logaddexp_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "logaddexp_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    logaddexp2(other: Tensor): Tensor {
        return ops.logaddexp2(this, other);
    }
    logaddexp2_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "logaddexp2_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    mul(other: Tensor, alpha?: number): Tensor {
        return ops.mul(this, other, alpha);
    }
    multiply(other: Tensor, alpha?: number): Tensor {
        return ops.mul(this, other, alpha);
    }
    mul_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace(
            "mul_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    neg(): Tensor {
        return ops.neg(this);
    }
    negative(): Tensor {
        return ops.neg(this);
    }
    neg_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("neg_", { dtype: this.dtype }, params);
    }
    positive(): Tensor {
        return ops.positive(this);
    }
    positive_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "positive_",
            { dtype: this.dtype },
            params
        );
    }
    pow(other: Tensor): Tensor {
        return ops.pow(this, other);
    }
    pow_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "pow_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    rad2deg(): Tensor {
        return ops.rad2deg(this);
    }
    rad2deg_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rad2deg_", { dtype: this.dtype }, params);
    }
    reciprocal(): Tensor {
        return ops.reciprocal(this);
    }
    reciprocal_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "reciprocal_",
            { dtype: this.dtype },
            params
        );
    }
    round(): Tensor {
        return ops.round(this);
    }
    round_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("round_", { dtype: this.dtype }, params);
    }
    rsqrt(): Tensor {
        return ops.rsqrt(this);
    }
    rsqrt_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rsqrt_", { dtype: this.dtype }, params);
    }
    sigmoid(): Tensor {
        return ops.sigmoid(this);
    }
    sigmoid_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sigmoid_", { dtype: this.dtype }, params);
    }
    sign(): Tensor {
        return ops.sign(this);
    }
    sign_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sign_", { dtype: this.dtype }, params);
    }
    sin(): Tensor {
        return ops.sin(this);
    }
    sin_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sin_", { dtype: this.dtype }, params);
    }
    sinc(): Tensor {
        return ops.sinc(this);
    }
    sinc_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinc_", { dtype: this.dtype }, params);
    }
    sinh(): Tensor {
        return ops.sinh(this);
    }
    sinh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinh_", { dtype: this.dtype }, params);
    }
    sqrt(): Tensor {
        return ops.sqrt(this);
    }
    sqrt_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sqrt_", { dtype: this.dtype }, params);
    }
    square(): Tensor {
        return ops.square(this);
    }
    square_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("square_", { dtype: this.dtype }, params);
    }
    sub(other: Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    subtract(other: Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    sub_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace(
            "sub_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    tan(): Tensor {
        return ops.tan(this);
    }
    tan_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tan_", { dtype: this.dtype }, params);
    }
    tanh(): Tensor {
        return ops.tanh(this);
    }
    tanh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tanh_", { dtype: this.dtype }, params);
    }
    trunc(): Tensor {
        return ops.trunc(this);
    }
    fix(): Tensor {
        return ops.trunc(this);
    }
    trunc_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("trunc_", { dtype: this.dtype }, params);
    }
    xlogy(other: Tensor): Tensor {
        return ops.xlogy(this, other);
    }
    xlogy_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "xlogy_",
            { dtype: this.dtype },
            params,
            other
        );
    }
    all(dim?: number, keepdim?: boolean): Tensor {
        return ops.all(this);
    }
    all_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("all_", { dtype: this.dtype }, params);
    }
    any(dim?: number, keepdim?: boolean): Tensor {
        return ops.any(this);
    }
    any_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("any_", { dtype: this.dtype }, params);
    }
    mean(dim?: number, keepdim?: boolean): Tensor {
        return ops.mean(this);
    }
    mean_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("mean_", { dtype: this.dtype }, params);
    }
    norm(dim?: number, keepdim?: boolean): Tensor {
        return ops.norm(this);
    }
    norm_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("norm_", { dtype: this.dtype }, params);
    }
    prod(dim?: number, keepdim?: boolean): Tensor {
        return ops.prod(this);
    }
    prod_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("prod_", { dtype: this.dtype }, params);
    }
    sum(dim?: number, keepdim?: boolean): Tensor {
        return ops.sum(this);
    }
    sum_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sum_", { dtype: this.dtype }, params);
    }
    countNonzero(dim?: number, keepdim?: boolean): Tensor {
        return ops.countNonzero(this);
    }
    countNonzero_(dim?: number, keepdim?: boolean): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace(
            "countNonzero_",
            { dtype: this.dtype },
            params
        );
    }
    // End codegen marker
}
