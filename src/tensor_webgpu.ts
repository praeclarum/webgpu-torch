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
        const outputBuffers = kernel.run(inputBuffers, params);
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
    // End codegen marker
}
