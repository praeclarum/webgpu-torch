import { Device } from "./device";
import { Dtype } from "./dtype";
import { Shape, defaultStrides } from "./shape";
import { TensorArrayData } from "./tensor_if";
import { TensorWebGPU } from "./tensor_webgpu";
import { GPUBufferStorage, newTypedArrayFromArray } from "./storage";
import {
    Kernel,
    KernelConfig,
    KernelConfigInput,
    KernelKey,
    KernelSpec,
    getKernelConfig,
    getKernelKey,
} from "./kernel";
import { KernelWebGPU } from "./kernel_webgpu";

export class DeviceWebGPU extends Device {
    private _device: GPUDevice;
    get gpuDevice(): GPUDevice {
        return this._device;
    }
    constructor(id: string, adapter: GPUAdapter, device: GPUDevice) {
        super(id, "webgpu");
        this._device = device;
    }
    alloc(byteSize: number): GPUBufferStorage {
        return new GPUBufferStorage(
            byteSize,
            this._device,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        );
    }
    createKernel(spec: KernelSpec, config: KernelConfig): Kernel {
        return new KernelWebGPU(spec, config, this);
    }
    ones(shape: Shape, dtype: Dtype): TensorWebGPU {
        const storage = this.allocFor(shape, dtype) as GPUBufferStorage;
        const array = storage.getTypedArray(dtype);
        array.fill(1);
        return new TensorWebGPU(
            storage,
            dtype,
            shape,
            defaultStrides(shape),
            this
        );
    }
    tensor(data: TensorArrayData | null, dtype: Dtype): TensorWebGPU {
        const info = newTypedArrayFromArray(data, dtype, (shape) =>
            this.allocFor(shape, dtype)
        );
        return new TensorWebGPU(
            info.storage as GPUBufferStorage,
            dtype,
            info.shape,
            info.strides,
            this
        );
    }
    zeros(shape: Shape, dtype: Dtype): TensorWebGPU {
        const storage = this.allocFor(shape, dtype) as GPUBufferStorage;
        return new TensorWebGPU(
            storage,
            dtype,
            shape,
            defaultStrides(shape),
            this
        );
    }
}
