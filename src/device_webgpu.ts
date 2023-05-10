import { Device } from "./device";
import type { ATypedArray, Dtype } from "./dtype";
import { GPUBufferStorage, UntypedStorage } from "./storage";
import type { Kernel, KernelConfig, KernelSpec } from "./kernel";
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
    getStorageFromKernel(storage: ATypedArray | GPUBuffer): UntypedStorage {
        if (storage instanceof GPUBuffer) {
            return new GPUBufferStorage(storage, this.gpuDevice);
        }
        throw new Error(
            `Cannot wrap buffer of type ${storage.constructor.name} to get GPU storage`
        );
    }
    getBufferForKernel(
        storage: UntypedStorage,
        dtype: Dtype
    ): ATypedArray | GPUBuffer {
        if (storage instanceof GPUBufferStorage) {
            return storage.gpuBuffer;
        }
        throw new Error(
            `Cannot unwrap storage of type ${storage.constructor.name} to get GPU buffer`
        );
    }
}
