import { Device  } from "./device";
import { Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData, TensorImpl } from "./tensor_if";
import { TensorCPU } from "./tensor_cpu";
import { TensorWebGPU } from "./tensor_webgpu";
import { GPUBufferStorage, UntypedStorage, newTypedArrayFromArray } from "./storage";
import { Kernel, KernelConfigInput, KernelKey, getKernelConfig, getKernelKey } from "./kernel";
import { registry as kernelRegistry } from "./kernels";

export class DeviceWebGPU extends Device {
    device: GPUDevice;
    private _kernels: { [key: KernelKey]: Kernel } = {};
    constructor(id: string, adapter: GPUAdapter, device: GPUDevice) {
        super(id, "webgpu");
        this.device = device;
    }
    alloc(byteSize: number): GPUBufferStorage {
        return new GPUBufferStorage(byteSize, GPUBufferUsage.STORAGE, this.device);
    }
    getKernel(name: string, config: KernelConfigInput): Kernel {
        const spec = kernelRegistry[name];
        if (spec === undefined) {
            throw new Error(`Kernel ${name} not found`);
        }
        const kconfig = getKernelConfig(spec, config);
        const key = getKernelKey(spec, kconfig);
        let kernel = this._kernels[key];
        if (kernel === undefined) {
            kernel = new Kernel(spec, kconfig, this.device);
            this._kernels[key] = kernel;
        }
        return kernel;
    }
    ones(shape: Shape, dtype: Dtype): TensorWebGPU {
        const storage = this.allocFor(shape, dtype) as GPUBufferStorage;
        const array = storage.getTypedArray(dtype);
        array.fill(1);
        return new TensorWebGPU(storage, dtype, shape, defaultStrides(shape), this);
    }
    tensor(data: TensorArrayData | null, dtype: Dtype): TensorWebGPU {
        const info = newTypedArrayFromArray(data, dtype, shape => this.allocFor(shape, dtype));
        return new TensorWebGPU(info.storage as GPUBufferStorage, dtype, info.shape, info.strides, this);
    }
    zeros(shape: Shape, dtype: Dtype): TensorWebGPU {
        const storage = this.allocFor(shape, dtype) as GPUBufferStorage;
        return new TensorWebGPU(storage, dtype, shape, defaultStrides(shape), this);
    }
}
