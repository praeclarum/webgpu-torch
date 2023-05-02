import { Device  } from "./device";
import { Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData, TensorImpl } from "./tensor_if";
import { TensorCPU } from "./tensor_cpu";
import { ArrayBufferStorage, UntypedStorage, newTypedArrayForDtype, newTypedArrayFromArray } from "./storage";

export class DeviceCPU extends Device {
    constructor() {
        super("cpu", "cpu");
    }
    alloc(byteSize: number): UntypedStorage {
        return new ArrayBufferStorage(byteSize);
    }
    ones(shape: Shape, dtype: Dtype): TensorImpl {
        const storage = this.allocFor(shape, dtype);
        const data = newTypedArrayForDtype(shapeSize(shape), dtype);
        data.fill(1);
        return new TensorCPU(data, shape, defaultStrides(shape), this);
    }
    tensor(data: TensorArrayData | null, dtype: Dtype): TensorImpl {
        const info = newTypedArrayFromArray(data, dtype, shape => this.allocFor(shape, dtype));
        return new TensorCPU(info.data as any, info.shape, info.strides, this);
    }
    zeros(shape: Shape, dtype: Dtype): TensorImpl {
        const data = newTypedArrayForDtype(shapeSize(shape), dtype);
        return new TensorCPU(data, shape, defaultStrides(shape), this);
    }
}

