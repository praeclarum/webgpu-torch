import type { Deviceish } from "./device";
import type { Shape, Strides } from "./shape";
import type { ATypedArray, Dtype } from "./dtype";
import type { TensorArrayData, UntypedStorage } from "./storage";

export type TensorData = TensorArrayData | ATypedArray | UntypedStorage;

export type TensorSpec = {
    data: TensorData;
    dtype?: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
    shape?: Shape;
    strides?: Strides;
};

/**
 * This is a little implementation detail that helps us avoid circular imports.
 */
export class TensorBase {}
