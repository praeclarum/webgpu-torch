import { Device } from "./device";
import { Dtype } from "./dtype";
import { Shape, defaultStrides } from "./shape";
import { TensorArrayData } from "./tensor_if";
import { TensorImpl } from "./tensor_impl";
import { TensorCPU } from "./tensor_cpu";
import {
    ArrayBufferStorage,
    UntypedStorage,
    newTypedArrayFromArray,
} from "./storage";
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
    ones(shape: Shape, dtype: Dtype): TensorImpl {
        const storage = this.allocFor(shape, dtype) as ArrayBufferStorage;
        const array = storage.getTypedArray(dtype);
        array.fill(1);
        return new TensorCPU(
            storage,
            dtype,
            shape,
            defaultStrides(shape),
            this
        );
    }
    tensor(data: TensorArrayData | null, dtype: Dtype): TensorImpl {
        const info = newTypedArrayFromArray(data, dtype, (shape) =>
            this.allocFor(shape, dtype)
        );
        return new TensorCPU(
            info.storage as ArrayBufferStorage,
            dtype,
            info.shape,
            info.strides,
            this
        );
    }
    zeros(shape: Shape, dtype: Dtype): TensorImpl {
        const storage = this.allocFor(shape, dtype) as ArrayBufferStorage;
        return new TensorCPU(
            storage,
            dtype,
            shape,
            defaultStrides(shape),
            this
        );
    }
}
