import { Shape, Shapeish, shapeSize } from "./shape";
import { Dtype, dtypeByteSize } from "./dtype";
import { TensorArrayData, TensorImpl } from "./tensor_if";
import { IDevice } from "./device_if";
import { UntypedStorage } from "./storage";

export type DeviceType = "cpu" | "webgpu";
export type DeviceId = string;

export type Deviceish = DeviceType | Device | DeviceId;

export abstract class Device implements IDevice {
    private _id: DeviceId;
    private _type: DeviceType;
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
    abstract alloc(byteSize: number, alignment: number): UntypedStorage;
    allocFor(shape: Shape, dtype: Dtype): UntypedStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        return this.alloc(byteSize, elementByteSize);
    }
    abstract ones(shape: Shape, dtype: Dtype): TensorImpl;
    abstract tensor(data: TensorArrayData | null, dtype: Dtype): TensorImpl;
    abstract zeros(shape: Shape, dtype: Dtype): TensorImpl;
}
