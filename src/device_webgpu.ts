import { Device  } from "./device";
import { Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData, TensorImpl } from "./tensor_if";
import { TensorCPU } from "./tensor_cpu";
import { TensorWebGPU } from "./tensor_webgpu";
import { newTypedArrayFromArray } from "./storage";

export class DeviceWebGPU extends Device {
    constructor(id: string, adapter: any, device: any) {
        super(id, "webgpu");
    }
    ones(shape: Shape, dtype: Dtype): TensorImpl {
        throw new Error("Method not implemented.");
    }
    tensor(data: TensorArrayData | null, dtype: Dtype): TensorImpl {
        const info = newTypedArrayFromArray(data, dtype);
        return new TensorWebGPU(info.data, info.shape, info.strides, this);
    }
    zeros(shape: Shape, dtype: Dtype): TensorImpl {
        throw new Error("Method not implemented.");
    }
}
