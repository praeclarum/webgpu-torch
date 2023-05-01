import { Dtype } from "./dtype";
import { ITensor, TensorImpl } from "./tensor_if";

export class TensorWebGPU implements TensorImpl {
    get dtype(): Dtype {
        throw new Error("Method not implemented.");
    }
    add_(other: ITensor): ITensor {
        throw new Error("Method not implemented.");
    }
    mm(other: ITensor): ITensor {
        throw new Error("Method not implemented.");
    }
    sum(arg0: number): ITensor {
        throw new Error("Method not implemented.");
    }
    t(): ITensor {
        throw new Error("Method not implemented.");
    }
}
