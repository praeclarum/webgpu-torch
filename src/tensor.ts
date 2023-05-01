import { UntypedStorage } from './storage';
import { Shape, Strides } from './shape';

export type Dtype = "float32" | "int32" | "boolean" | "string";

export class Tensor {
    private _untypedStorage: UntypedStorage;
    private _dtype: Dtype;
    private _requiresGrad: boolean = false;

    get untypedStorage(): UntypedStorage { 
        return this._untypedStorage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get requiresGrad(): boolean {
        return this._requiresGrad;
    }
    set requiresGrad(value: boolean) {
        this._requiresGrad = value;
    }

    constructor(data: UntypedStorage|Array<any>, dtype: Dtype) {
        this._untypedStorage = new UntypedStorage();
        this._dtype = dtype;
    }

    add_(tensor: Tensor): Tensor {
        return this;
    }
    mm(tensor: Tensor): Tensor {
        return new Tensor(this._untypedStorage, this._dtype);
    }
    sum(arg0: number) {
        return new Tensor(this._untypedStorage, this._dtype);
    }
    t(): Tensor {
        return new Tensor(this._untypedStorage, this._dtype);
    }
}

