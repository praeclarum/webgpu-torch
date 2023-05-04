import { Device } from "./device";
import { DeviceWebGPU } from "./device_webgpu";
import { Dtype, dtypeByteSize } from "./dtype";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { GPUBufferStorage } from "./storage";
import { TensorImpl } from "./tensor_if";

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
        const outputs = kernel.run([this.gpuBuffer, other.gpuBuffer], params);
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

    // Codegen marker
    add(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const kernel = this._device.getKernel("add", { dtype: "f32" });
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        const outputBuffer = kernel.run([this.gpuBuffer, other.gpuBuffer], params)[0];
        return new TensorWebGPU(
            new GPUBufferStorage(outputBuffer, this.gpuDevice),
            this.dtype,
            this.shape,
            defaultStrides(this.shape),
            this._device
        );
        return this;
    }
    add_(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const kernel = this._device.getKernel("add_", { dtype: "f32" });
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        kernel.run([other.gpuBuffer], params, [this.gpuBuffer]);
        return this;
    }
    atan2(other: TensorWebGPU): TensorWebGPU {
        const kernel = this._device.getKernel("atan2", { dtype: "f32" });
        const params = {
            size: shapeSize(this.shape),
        };
        const outputBuffer = kernel.run([this.gpuBuffer, other.gpuBuffer], params)[0];
        return new TensorWebGPU(
            new GPUBufferStorage(outputBuffer, this.gpuDevice),
            this.dtype,
            this.shape,
            defaultStrides(this.shape),
            this._device
        );
        return this;
    }
    atan2_(other: TensorWebGPU): TensorWebGPU {
        const kernel = this._device.getKernel("atan2_", { dtype: "f32" });
        const params = {
            size: shapeSize(this.shape),
        };
        kernel.run([other.gpuBuffer], params, [this.gpuBuffer]);
        return this;
    }
    sub(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const kernel = this._device.getKernel("sub", { dtype: "f32" });
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        const outputBuffer = kernel.run([this.gpuBuffer, other.gpuBuffer], params)[0];
        return new TensorWebGPU(
            new GPUBufferStorage(outputBuffer, this.gpuDevice),
            this.dtype,
            this.shape,
            defaultStrides(this.shape),
            this._device
        );
        return this;
    }
    sub_(other: TensorWebGPU, alpha?: number): TensorWebGPU {
        const kernel = this._device.getKernel("sub_", { dtype: "f32" });
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        kernel.run([other.gpuBuffer], params, [this.gpuBuffer]);
        return this;
    }
    // End codegen marker
}
