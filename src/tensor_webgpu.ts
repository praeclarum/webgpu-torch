import { Device } from "./device";
import { DeviceWebGPU } from "./device_webgpu";
import { Dtype, dtypeByteSize } from "./dtype";
import { KernelConfigInput, KernelParamsInput } from "./kernel";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { GPUBufferStorage } from "./storage";
import { TensorImpl } from "./tensor_impl";

export class TensorWebGPU extends TensorImpl {
    private _storage: GPUBufferStorage;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: DeviceWebGPU;

    get gpuBuffer(): GPUBuffer {
        return this._storage.gpuBuffer;
    }
    get gpuDevice(): GPUDevice {
        return this._device.gpuDevice;
    }

    get storage(): GPUBufferStorage {
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
        storage: GPUBufferStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides,
        device: DeviceWebGPU
    ) {
        super();
        this._storage = storage;
        this._dtype = dtype;
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }

    withShape(shape: Shape, strides: Strides): TensorWebGPU {
        return new TensorWebGPU(
            this._storage,
            this._dtype,
            shape,
            strides,
            this._device
        );
    }

    mm(other: TensorWebGPU): TensorWebGPU {
        const kernel = this._device.getKernel("mm", { resultDtype: "f32" });
        const params = {
            resultRows: this.shape[0],
            resultCols: other.shape[1],
            innerDim: this.shape[1],
            alpha: 1.0,
        };
        this.gpuBuffer.unmap();
        other.gpuBuffer.unmap();
        const outputs = kernel.run([this.gpuBuffer, other.gpuBuffer], params) as GPUBuffer[];
        const readBuffer = outputs[0];
        const readStorage = new GPUBufferStorage(readBuffer, this.gpuDevice);
        const resultShape = [params.resultRows, params.resultCols];
        const readTensor = new TensorWebGPU(
            readStorage,
            this.dtype,
            resultShape,
            defaultStrides(resultShape),
            this._device
        );
        return readTensor;
    }
    sum(axis: number | null): TensorImpl {
        throw new Error("Sum not implemented.");
    }

    runKernelInplace(name: string, config: KernelConfigInput, params: KernelParamsInput, ...additionalInputs: TensorWebGPU[]): TensorWebGPU {
        const kernel = this._device.getKernel(name, config);
        const inputBuffers = additionalInputs.map(t => t.gpuBuffer);
        const outputBuffers = [this.gpuBuffer];
        kernel.run(inputBuffers, params, outputBuffers);
        return this;
    }
    runKernel(name: string, config: KernelConfigInput, params: KernelParamsInput, outputShapes: Shape[], ...additionalInputs: TensorWebGPU[]): TensorWebGPU[] {
        const kernel = this._device.getKernel(name, config);
        const inputBuffers = [this.gpuBuffer, ...additionalInputs.map(t => t.gpuBuffer)];
        const outputBuffers = kernel.run(inputBuffers, params) as GPUBuffer[];
        if (outputBuffers.length !== outputShapes.length) {
            throw new Error(`Expected ${outputShapes.length} output buffers (given the provided outputShapes to runKernel), but got ${outputBuffers.length} output buffers when running the kernel "${name}".`);
        }
        return outputBuffers.map((outputBuffer, i) => new TensorWebGPU(
            new GPUBufferStorage(outputBuffer, this.gpuDevice),
            this.dtype,
            outputShapes[i],
            defaultStrides(outputShapes[i]),
            this._device)
        );
    }

    // Codegen marker
    abs(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("abs", { dtype: this.dtype }, params, [this.shape])[0];
    }
    abs_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("abs_", { dtype: this.dtype }, params);
    }
    acos(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("acos", { dtype: this.dtype }, params, [this.shape])[0];
    }
    acos_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acos_", { dtype: this.dtype }, params);
    }
    acosh(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("acosh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    acosh_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acosh_", { dtype: this.dtype }, params);
    }
    add(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("add", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    add_(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("add_", { dtype: this.dtype }, params, other);
    }
    asin(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("asin", { dtype: this.dtype }, params, [this.shape])[0];
    }
    asin_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asin_", { dtype: this.dtype }, params);
    }
    asinh(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("asinh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    asinh_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asinh_", { dtype: this.dtype }, params);
    }
    atan(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("atan", { dtype: this.dtype }, params, [this.shape])[0];
    }
    atan_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan_", { dtype: this.dtype }, params);
    }
    atan2(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("atan2", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    atan2_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan2_", { dtype: this.dtype }, params, other);
    }
    ceil(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("ceil", { dtype: this.dtype }, params, [this.shape])[0];
    }
    ceil_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ceil_", { dtype: this.dtype }, params);
    }
    copysign(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("copysign", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    copysign_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("copysign_", { dtype: this.dtype }, params, other);
    }
    cos(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("cos", { dtype: this.dtype }, params, [this.shape])[0];
    }
    cos_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cos_", { dtype: this.dtype }, params);
    }
    cosh(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("cosh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    cosh_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cosh_", { dtype: this.dtype }, params);
    }
    deg2rad(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("deg2rad", { dtype: this.dtype }, params, [this.shape])[0];
    }
    deg2rad_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("deg2rad_", { dtype: this.dtype }, params);
    }
    div(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("div", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    div_(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("div_", { dtype: this.dtype }, params, other);
    }
    exp(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("exp", { dtype: this.dtype }, params, [this.shape])[0];
    }
    exp_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp_", { dtype: this.dtype }, params);
    }
    exp2(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("exp2", { dtype: this.dtype }, params, [this.shape])[0];
    }
    exp2_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp2_", { dtype: this.dtype }, params);
    }
    expm1(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("expm1", { dtype: this.dtype }, params, [this.shape])[0];
    }
    expm1_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("expm1_", { dtype: this.dtype }, params);
    }
    floor(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("floor", { dtype: this.dtype }, params, [this.shape])[0];
    }
    floor_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_", { dtype: this.dtype }, params);
    }
    floor_divide(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("floor_divide", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    floor_divide_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_divide_", { dtype: this.dtype }, params, other);
    }
    frac(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("frac", { dtype: this.dtype }, params, [this.shape])[0];
    }
    frac_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("frac_", { dtype: this.dtype }, params);
    }
    hypot(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("hypot", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    hypot_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("hypot_", { dtype: this.dtype }, params, other);
    }
    ldexp(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("ldexp", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    ldexp_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ldexp_", { dtype: this.dtype }, params, other);
    }
    log(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log_", { dtype: this.dtype }, params);
    }
    log10(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log10", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log10_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log10_", { dtype: this.dtype }, params);
    }
    log1p(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log1p", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log1p_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log1p_", { dtype: this.dtype }, params);
    }
    log2(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("log2", { dtype: this.dtype }, params, [this.shape])[0];
    }
    log2_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log2_", { dtype: this.dtype }, params);
    }
    logaddexp(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("logaddexp", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    logaddexp_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("logaddexp_", { dtype: this.dtype }, params, other);
    }
    logaddexp2(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("logaddexp2", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    logaddexp2_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("logaddexp2_", { dtype: this.dtype }, params, other);
    }
    mul(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("mul", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    mul_(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("mul_", { dtype: this.dtype }, params, other);
    }
    neg(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("neg", { dtype: this.dtype }, params, [this.shape])[0];
    }
    neg_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("neg_", { dtype: this.dtype }, params);
    }
    positive(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("positive", { dtype: this.dtype }, params, [this.shape])[0];
    }
    positive_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("positive_", { dtype: this.dtype }, params);
    }
    pow(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("pow", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    pow_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("pow_", { dtype: this.dtype }, params, other);
    }
    rad2deg(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("rad2deg", { dtype: this.dtype }, params, [this.shape])[0];
    }
    rad2deg_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rad2deg_", { dtype: this.dtype }, params);
    }
    reciprocal(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("reciprocal", { dtype: this.dtype }, params, [this.shape])[0];
    }
    reciprocal_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("reciprocal_", { dtype: this.dtype }, params);
    }
    round(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("round", { dtype: this.dtype }, params, [this.shape])[0];
    }
    round_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("round_", { dtype: this.dtype }, params);
    }
    rsqrt(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("rsqrt", { dtype: this.dtype }, params, [this.shape])[0];
    }
    rsqrt_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rsqrt_", { dtype: this.dtype }, params);
    }
    sigmoid(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sigmoid", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sigmoid_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sigmoid_", { dtype: this.dtype }, params);
    }
    sign(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sign", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sign_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sign_", { dtype: this.dtype }, params);
    }
    sin(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sin", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sin_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sin_", { dtype: this.dtype }, params);
    }
    sinc(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sinc", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sinc_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinc_", { dtype: this.dtype }, params);
    }
    sinh(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sinh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sinh_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinh_", { dtype: this.dtype }, params);
    }
    sqrt(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sqrt", { dtype: this.dtype }, params, [this.shape])[0];
    }
    sqrt_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sqrt_", { dtype: this.dtype }, params);
    }
    square(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("square", { dtype: this.dtype }, params, [this.shape])[0];
    }
    square_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("square_", { dtype: this.dtype }, params);
    }
    sub(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernel("sub", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    sub_(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("sub_", { dtype: this.dtype }, params, other);
    }
    tan(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("tan", { dtype: this.dtype }, params, [this.shape])[0];
    }
    tan_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tan_", { dtype: this.dtype }, params);
    }
    tanh(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("tanh", { dtype: this.dtype }, params, [this.shape])[0];
    }
    tanh_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tanh_", { dtype: this.dtype }, params);
    }
    trunc(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("trunc", { dtype: this.dtype }, params, [this.shape])[0];
    }
    trunc_(): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("trunc_", { dtype: this.dtype }, params);
    }
    xlogy(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("xlogy", { dtype: this.dtype }, params, [this.shape], other)[0];
    }
    xlogy_(other: TensorWebGPU): TensorWebGPU {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("xlogy_", { dtype: this.dtype }, params, other);
    }
    // End codegen marker
}
