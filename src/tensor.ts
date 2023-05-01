import { UntypedStorage } from './storage';

export class Tensor {
    private _untypedStorage: UntypedStorage;
    private _dtype: string;
    private _requiresGrad: boolean = false;

    get untypedStorage(): UntypedStorage { 
        return this._untypedStorage;
    }
    get dtype(): string {
        return this._dtype;
    }
    get requiresGrad(): boolean {
        return this._requiresGrad;
    }
    set requiresGrad(value: boolean) {
        this._requiresGrad = value;
    }

    constructor(untypedStorage: UntypedStorage, dtype: string) {
        this._untypedStorage = untypedStorage;
        this._dtype = dtype;
    }
}

