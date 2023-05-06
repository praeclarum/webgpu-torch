import { Device } from "./device";
import { DeviceWebGPU } from "./device_webgpu";
import { Dtype } from "./dtype";
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

    runKernelInplace(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        ...additionalInputs: TensorWebGPU[]
    ): TensorWebGPU {
        const kernel = this._device.getKernel(name, config);
        const inputBuffers = additionalInputs.map((t) => t.gpuBuffer);
        const outputBuffers = [this.gpuBuffer];
        kernel.run(inputBuffers, params, outputBuffers);
        return this;
    }
    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputShapes: Shape[],
        ...additionalInputs: TensorWebGPU[]
    ): TensorWebGPU[] {
        const kernel = this._device.getKernel(name, config);
        const inputBuffers = [
            this.gpuBuffer,
            ...additionalInputs.map((t) => t.gpuBuffer),
        ];
        const outputBuffers = kernel.run(inputBuffers, params) as GPUBuffer[];
        if (outputBuffers.length !== outputShapes.length) {
            throw new Error(
                `Expected ${outputShapes.length} output buffers (given the provided outputShapes to runKernel), but got ${outputBuffers.length} output buffers when running the kernel "${name}".`
            );
        }
        return outputBuffers.map(
            (outputBuffer, i) =>
                new TensorWebGPU(
                    new GPUBufferStorage(outputBuffer, this.gpuDevice),
                    this.dtype,
                    outputShapes[i],
                    defaultStrides(outputShapes[i]),
                    this._device
                )
        );
    }

    sum(axis: number | null): TensorImpl {
        const config = {
            dtype: this.dtype,
            workgroupSize: 64,
        };
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernel("sum", config, params, [[config.workgroupSize]])[0];
    }
}
