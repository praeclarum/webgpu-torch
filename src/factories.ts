import { Shape, Shapeish, shapeishToShape } from "./shape";
import { Tensor } from "./tensor";
import { UntypedStorage } from "./storage";

export function zeros(shape: Shapeish): Tensor {
    const shape_ = shapeishToShape(shape);
    return new Tensor(new UntypedStorage(), "float32");
}

export function ones(shape: Shapeish): Tensor {
    const shape_ = shapeishToShape(shape);
    return new Tensor(new UntypedStorage(), "float32");
}
