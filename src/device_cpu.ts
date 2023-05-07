import { Device } from "./device";
import { ATypedArray, Dtype } from "./dtype";
import { ArrayBufferStorage, UntypedStorage } from "./storage";
import { Kernel, KernelConfig, KernelSpec } from "./kernel";
import { KernelCPU } from "./kernel_cpu";

export class DeviceCPU extends Device {
    constructor() {
        super("cpu", "cpu");
    }
    alloc(byteSize: number): UntypedStorage {
        return new ArrayBufferStorage(byteSize);
    }
    createKernel(spec: KernelSpec, config: KernelConfig): Kernel {
        return new KernelCPU(spec, config, this);
    }
    getStorageFromKernel(storage: ATypedArray | GPUBuffer): UntypedStorage {
        if (storage instanceof ArrayBuffer) {
            return new ArrayBufferStorage(storage);
        }
        throw new Error(
            `Cannot wrap buffer of type ${storage.constructor.name} to get CPU storage`
        );
    }
    getBufferForKernel(
        storage: UntypedStorage,
        dtype: Dtype
    ): ATypedArray | GPUBuffer {
        if (storage instanceof ArrayBufferStorage) {
            return storage.getTypedArray(dtype);
        }
        throw new Error(
            `Cannot unwrap buffer of type ${storage.constructor.name} to get CPU buffer`
        );
    }
}
