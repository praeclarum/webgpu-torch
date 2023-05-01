import { Dtype } from "./dtype";
import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";

export class TensorCPU extends TensorImpl {
    private _data: TensorArrayData;
    private _dtype: Dtype;
    get dtype(): Dtype {
        return this._dtype;
    }
    constructor(data: TensorArrayData, dtype: Dtype) {
        super();
        this._data = data;
        this._dtype = dtype;
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
