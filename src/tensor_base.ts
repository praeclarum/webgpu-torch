import { Deviceish } from "./device";
import { Shape, Strides } from "./shape";
import { ATypedArray, Dtype } from "./dtype";
import {
    TensorArrayData,
    UntypedStorage,
} from "./storage";

export type TensorData = TensorArrayData | ATypedArray | UntypedStorage;

export type TensorSpec = {
    data: TensorData;
    dtype?: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
    shape?: Shape;
    strides?: Strides;
};

export class TensorBase {

}
