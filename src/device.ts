import { Shape, Shapeish, shapeSize } from "./shape";
import { ATypedArray, Dtype, dtypeByteSize } from "./dtype";
import { TensorArrayData } from "./tensor_if";
import { IDevice } from "./device_if";
import { UntypedStorage } from "./storage";
import { TensorImpl } from "./tensor_impl";
import { Kernel, KernelConfig, KernelConfigInput, KernelKey, KernelSpec, getKernelConfig, getKernelKey } from "./kernel";
import { registry as kernelRegistry } from "./kernels";

export type DeviceType = "cpu" | "webgpu";
export type DeviceId = string;

export type Deviceish = DeviceType | Device | DeviceId;

export abstract class Device implements IDevice {
    private _id: DeviceId;
    private _type: DeviceType;
    private _kernels: { [key: KernelKey]: Kernel } = {};
    get id(): DeviceId {
        return this._id;
    }
    get type(): DeviceType {
        return this._type;
    }
    constructor(id: DeviceId, type: DeviceType) {
        this._id = id;
        this._type = type;
    }
    abstract alloc(byteSize: number): UntypedStorage;
    allocFor(shape: Shape, dtype: Dtype): UntypedStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        return this.alloc(byteSize);
    }
    allocTypedArray(size: number, dtype: Dtype): [UntypedStorage, ATypedArray] {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = size * elementByteSize;
        const storage = this.alloc(byteSize);
        return [storage, storage.getTypedArray(dtype)];
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
            kernel = this.createKernel(spec, kconfig);
            this._kernels[key] = kernel;
        }
        return kernel;
    }
    abstract createKernel(spec: KernelSpec, config: KernelConfig): Kernel;
    abstract ones(shape: Shape, dtype: Dtype): TensorImpl;
    abstract tensor(data: TensorArrayData | null, dtype: Dtype): TensorImpl;
    abstract zeros(shape: Shape, dtype: Dtype): TensorImpl;
}
