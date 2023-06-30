import { Shapeish, defaultStrides, getShape } from "./shape";
import { Deviceish } from "./device";
import { getDevice } from "./devices";
import { Dtype, getDtype } from "./dtype";
import { Tensor } from "./tensor";

export function ones(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    const d = getDevice(device);
    const s = getShape(shape);
    const dt = getDtype(dtype);
    const storage = d.initStorage(s, dt, (array) => {
        array.fill(1);
    });
    return new Tensor({
        data: storage,
        dtype: dt,
        shape: s,
        strides: defaultStrides(s),
        device: d,
    });
}

export function zeros(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    const d = getDevice(device);
    const s = getShape(shape);
    const dt = getDtype(dtype);
    const storage = d.initStorage(s, dt, (array) => {
        array.fill(0);
    });
    return new Tensor({
        data: storage,
        dtype: dt,
        shape: s,
        strides: defaultStrides(s),
        device: d,
    });
}
