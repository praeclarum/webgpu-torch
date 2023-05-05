import { Device } from "./device";
import { ATypedArray, Dtype } from "./dtype";
import { KernelConfigInput, KernelParamsInput } from "./kernel";
import {
    Shape,
    Strides,
    defaultStrides,
    shapeGetAxis,
    shapeSize,
} from "./shape";
import { ArrayBufferStorage } from "./storage";
import { TensorImpl } from "./tensor_impl";

export class TensorCPU extends TensorImpl {
    private _storage: ArrayBufferStorage;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

    get storage(): ArrayBufferStorage {
        return this._storage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get shape(): number[] {
        return this._shape;
    }
    get strides(): number[] {
        return this._strides;
    }
    get device(): Device {
        return this._device;
    }

    constructor(
        data: ArrayBufferStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides,
        device: Device
    ) {
        super();
        this._storage = data;
        this._dtype = dtype;
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }

    withShape(shape: Shape, strides: Strides): TensorImpl {
        return new TensorCPU(
            this._storage,
            this._dtype,
            shape,
            strides,
            this._device
        );
    }

    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputShapes: Shape[],
        ...additionalInputs: TensorCPU[]
    ): TensorCPU[] {
        const kernel = this.device.getKernel(name, config);
        const inputBuffers = [
            this.getTypedArray(),
            ...additionalInputs.map((t) => t.getTypedArray()),
        ];
        const outputBuffers = kernel.run(inputBuffers, params) as ATypedArray[];
        if (outputBuffers.length !== outputShapes.length) {
            throw new Error(
                `Expected ${outputShapes.length} output buffers (given the provided outputShapes to runKernel), but got ${outputBuffers.length} output buffers when running the kernel "${name}".`
            );
        }
        return outputBuffers.map(
            (outputBuffer, i) =>
                new TensorCPU(
                    new ArrayBufferStorage(outputBuffer.buffer),
                    this.dtype,
                    outputShapes[i],
                    defaultStrides(outputShapes[i]),
                    this._device
                )
        );
    }

    add_(other: TensorImpl, alpha?: number): TensorImpl {
        if (!(other instanceof TensorCPU)) {
            throw new Error("Only CPU tensors can be added to CPU tensors");
        }
        const d = this.getTypedArray();
        const od = other.getTypedArray();
        if (alpha === undefined) {
            for (let i = 0; i < d.length; i++) {
                d[i] += od[i];
            }
        } else {
            for (let i = 0; i < d.length; i++) {
                d[i] += alpha * od[i];
            }
        }
        return this;
    }
    mm(other: TensorImpl): TensorImpl {
        // Matrix multiply
        const newShape = [this._shape[0], other.shape[1]];
        const [newStorage, newData] = this.device.allocTypedArray(
            newShape[0] * newShape[1],
            this.dtype
        );
        const newStrides = defaultStrides(newShape);
        for (let i = 0; i < newShape[0]; i++) {
            for (let j = 0; j < newShape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < this._shape[1]; k++) {
                    sum +=
                        (this.get(i, k) as number) *
                        (other.get(k, j) as number);
                }
                newData[i * newStrides[0] + j] = sum;
            }
        }
        return new TensorCPU(
            newStorage as ArrayBufferStorage,
            this._dtype,
            newShape,
            newStrides,
            this._device
        );
    }
    sum(axis: number | null): TensorImpl {
        const d = this.getTypedArray();
        if (axis === null) {
            let sum = 0;
            for (let i = 0; i < d.length; i++) {
                sum += d[i];
            }
            const [newStorage, newData] = this.device.allocTypedArray(
                1,
                this.dtype
            );
            return new TensorCPU(
                newStorage as ArrayBufferStorage,
                this._dtype,
                [],
                [],
                this._device
            );
        } else {
            axis = shapeGetAxis(this._shape, axis);
            const newShape = this._shape.slice();
            newShape.splice(axis, 1);
            const [newStorage, newData] = this.device.allocTypedArray(
                shapeSize(newShape),
                this.dtype
            );
            const newStrides = defaultStrides(newShape);
            const axisStride = this._strides[axis];
            for (let i = 0; i < newData.length; i++) {
                let sum = 0;
                for (let j = 0; j < axisStride; j++) {
                    sum += d[i * newStrides[0] + j];
                }
                newData[i] = sum;
            }
            return new TensorCPU(
                newStorage as ArrayBufferStorage,
                this._dtype,
                newShape,
                newStrides,
                this._device
            );
        }
    }
    // Codegen marker
    abs(): TensorCPU {
        throw new Error("CPU abs not supported");
    }
    abs_(): TensorCPU {
        throw new Error("CPU abs_ not supported");
    }
    acos(): TensorCPU {
        throw new Error("CPU acos not supported");
    }
    acos_(): TensorCPU {
        throw new Error("CPU acos_ not supported");
    }
    acosh(): TensorCPU {
        throw new Error("CPU acosh not supported");
    }
    acosh_(): TensorCPU {
        throw new Error("CPU acosh_ not supported");
    }
    add(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU add not supported");
    }
    asin(): TensorCPU {
        throw new Error("CPU asin not supported");
    }
    asin_(): TensorCPU {
        throw new Error("CPU asin_ not supported");
    }
    asinh(): TensorCPU {
        throw new Error("CPU asinh not supported");
    }
    asinh_(): TensorCPU {
        throw new Error("CPU asinh_ not supported");
    }
    atan(): TensorCPU {
        throw new Error("CPU atan not supported");
    }
    atan_(): TensorCPU {
        throw new Error("CPU atan_ not supported");
    }
    atan2(other: TensorCPU): TensorCPU {
        throw new Error("CPU atan2 not supported");
    }
    atan2_(other: TensorCPU): TensorCPU {
        throw new Error("CPU atan2_ not supported");
    }
    ceil(): TensorCPU {
        throw new Error("CPU ceil not supported");
    }
    ceil_(): TensorCPU {
        throw new Error("CPU ceil_ not supported");
    }
    copysign(other: TensorCPU): TensorCPU {
        throw new Error("CPU copysign not supported");
    }
    copysign_(other: TensorCPU): TensorCPU {
        throw new Error("CPU copysign_ not supported");
    }
    cos(): TensorCPU {
        throw new Error("CPU cos not supported");
    }
    cos_(): TensorCPU {
        throw new Error("CPU cos_ not supported");
    }
    cosh(): TensorCPU {
        throw new Error("CPU cosh not supported");
    }
    cosh_(): TensorCPU {
        throw new Error("CPU cosh_ not supported");
    }
    deg2rad(): TensorCPU {
        throw new Error("CPU deg2rad not supported");
    }
    deg2rad_(): TensorCPU {
        throw new Error("CPU deg2rad_ not supported");
    }
    div(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU div not supported");
    }
    div_(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU div_ not supported");
    }
    exp(): TensorCPU {
        throw new Error("CPU exp not supported");
    }
    exp_(): TensorCPU {
        throw new Error("CPU exp_ not supported");
    }
    exp2(): TensorCPU {
        throw new Error("CPU exp2 not supported");
    }
    exp2_(): TensorCPU {
        throw new Error("CPU exp2_ not supported");
    }
    expm1(): TensorCPU {
        throw new Error("CPU expm1 not supported");
    }
    expm1_(): TensorCPU {
        throw new Error("CPU expm1_ not supported");
    }
    floor(): TensorCPU {
        throw new Error("CPU floor not supported");
    }
    floor_(): TensorCPU {
        throw new Error("CPU floor_ not supported");
    }
    floor_divide(other: TensorCPU): TensorCPU {
        throw new Error("CPU floor_divide not supported");
    }
    floor_divide_(other: TensorCPU): TensorCPU {
        throw new Error("CPU floor_divide_ not supported");
    }
    frac(): TensorCPU {
        throw new Error("CPU frac not supported");
    }
    frac_(): TensorCPU {
        throw new Error("CPU frac_ not supported");
    }
    hypot(other: TensorCPU): TensorCPU {
        throw new Error("CPU hypot not supported");
    }
    hypot_(other: TensorCPU): TensorCPU {
        throw new Error("CPU hypot_ not supported");
    }
    ldexp(other: TensorCPU): TensorCPU {
        throw new Error("CPU ldexp not supported");
    }
    ldexp_(other: TensorCPU): TensorCPU {
        throw new Error("CPU ldexp_ not supported");
    }
    log(): TensorCPU {
        throw new Error("CPU log not supported");
    }
    log_(): TensorCPU {
        throw new Error("CPU log_ not supported");
    }
    log10(): TensorCPU {
        throw new Error("CPU log10 not supported");
    }
    log10_(): TensorCPU {
        throw new Error("CPU log10_ not supported");
    }
    log1p(): TensorCPU {
        throw new Error("CPU log1p not supported");
    }
    log1p_(): TensorCPU {
        throw new Error("CPU log1p_ not supported");
    }
    log2(): TensorCPU {
        throw new Error("CPU log2 not supported");
    }
    log2_(): TensorCPU {
        throw new Error("CPU log2_ not supported");
    }
    logaddexp(other: TensorCPU): TensorCPU {
        throw new Error("CPU logaddexp not supported");
    }
    logaddexp_(other: TensorCPU): TensorCPU {
        throw new Error("CPU logaddexp_ not supported");
    }
    logaddexp2(other: TensorCPU): TensorCPU {
        throw new Error("CPU logaddexp2 not supported");
    }
    logaddexp2_(other: TensorCPU): TensorCPU {
        throw new Error("CPU logaddexp2_ not supported");
    }
    mul(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU mul not supported");
    }
    mul_(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU mul_ not supported");
    }
    neg(): TensorCPU {
        throw new Error("CPU neg not supported");
    }
    neg_(): TensorCPU {
        throw new Error("CPU neg_ not supported");
    }
    positive(): TensorCPU {
        throw new Error("CPU positive not supported");
    }
    positive_(): TensorCPU {
        throw new Error("CPU positive_ not supported");
    }
    pow(other: TensorCPU): TensorCPU {
        throw new Error("CPU pow not supported");
    }
    pow_(other: TensorCPU): TensorCPU {
        throw new Error("CPU pow_ not supported");
    }
    rad2deg(): TensorCPU {
        throw new Error("CPU rad2deg not supported");
    }
    rad2deg_(): TensorCPU {
        throw new Error("CPU rad2deg_ not supported");
    }
    reciprocal(): TensorCPU {
        throw new Error("CPU reciprocal not supported");
    }
    reciprocal_(): TensorCPU {
        throw new Error("CPU reciprocal_ not supported");
    }
    round(): TensorCPU {
        throw new Error("CPU round not supported");
    }
    round_(): TensorCPU {
        throw new Error("CPU round_ not supported");
    }
    rsqrt(): TensorCPU {
        throw new Error("CPU rsqrt not supported");
    }
    rsqrt_(): TensorCPU {
        throw new Error("CPU rsqrt_ not supported");
    }
    sigmoid(): TensorCPU {
        throw new Error("CPU sigmoid not supported");
    }
    sigmoid_(): TensorCPU {
        throw new Error("CPU sigmoid_ not supported");
    }
    sign(): TensorCPU {
        throw new Error("CPU sign not supported");
    }
    sign_(): TensorCPU {
        throw new Error("CPU sign_ not supported");
    }
    sin(): TensorCPU {
        throw new Error("CPU sin not supported");
    }
    sin_(): TensorCPU {
        throw new Error("CPU sin_ not supported");
    }
    sinc(): TensorCPU {
        throw new Error("CPU sinc not supported");
    }
    sinc_(): TensorCPU {
        throw new Error("CPU sinc_ not supported");
    }
    sinh(): TensorCPU {
        throw new Error("CPU sinh not supported");
    }
    sinh_(): TensorCPU {
        throw new Error("CPU sinh_ not supported");
    }
    sqrt(): TensorCPU {
        throw new Error("CPU sqrt not supported");
    }
    sqrt_(): TensorCPU {
        throw new Error("CPU sqrt_ not supported");
    }
    square(): TensorCPU {
        throw new Error("CPU square not supported");
    }
    square_(): TensorCPU {
        throw new Error("CPU square_ not supported");
    }
    sub(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU sub not supported");
    }
    sub_(other: TensorCPU, alpha?: number): TensorCPU {
        throw new Error("CPU sub_ not supported");
    }
    tan(): TensorCPU {
        throw new Error("CPU tan not supported");
    }
    tan_(): TensorCPU {
        throw new Error("CPU tan_ not supported");
    }
    tanh(): TensorCPU {
        throw new Error("CPU tanh not supported");
    }
    tanh_(): TensorCPU {
        throw new Error("CPU tanh_ not supported");
    }
    trunc(): TensorCPU {
        throw new Error("CPU trunc not supported");
    }
    trunc_(): TensorCPU {
        throw new Error("CPU trunc_ not supported");
    }
    xlogy(other: TensorCPU): TensorCPU {
        throw new Error("CPU xlogy not supported");
    }
    xlogy_(other: TensorCPU): TensorCPU {
        throw new Error("CPU xlogy_ not supported");
    }
    // End codegen marker
}
