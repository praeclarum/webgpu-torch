import { Device } from "./device";
import type { ATypedArray, Dtype } from "./dtype";
import { ArrayBufferStorage, UntypedStorage } from "./storage";
import type { Kernel, KernelConfig, KernelSpec } from "./kernel";
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
        if (
            storage instanceof Uint8Array ||
            storage instanceof Uint32Array ||
            storage instanceof Int32Array ||
            storage instanceof Float32Array
        ) {
            return new ArrayBufferStorage(storage.buffer);
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
