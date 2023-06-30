import { Shapeish, defaultStrides, getShape } from "./shape";
import { Deviceish } from "./device";
import { getDevice } from "./devices";
import { ATypedArray, Dtype, getDtype } from "./dtype";
import { Tensor } from "./tensor";

function initTensor(
    shape: Shapeish,
    dtype: Dtype | undefined,
    device: Deviceish | undefined,
    init: (array: ATypedArray) => void
): Tensor {
    const d = getDevice(device);
    const s = getShape(shape);
    const dt = getDtype(dtype);
    const storage = d.initStorage(s, dt, init);
    return new Tensor({
        data: storage,
        dtype: dt,
        shape: s,
        strides: defaultStrides(s),
        device: d,
    });
}

/** Create a tensor with uninitialized elements */
export function empty(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    return initTensor(shape, dtype, device, (array) => {
        // Empty tensors are not initialized
    });
}

/** Create a tensor with all elements set to 1 */
export function ones(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    return initTensor(shape, dtype, device, (array) => {
        array.fill(1);
    });
}

/** Create a tensor with all elements set to 0 */
export function zeros(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    return initTensor(shape, dtype, device, (array) => {
        array.fill(0);
    });
}
