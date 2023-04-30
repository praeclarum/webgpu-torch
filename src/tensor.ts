import { UntypedStorage } from './storage';

export class Tensor {
    private _untypedStorage: UntypedStorage;
    private _dtype: string;

    constructor(untypedStorage: UntypedStorage, dtype: string) {
        this._untypedStorage = untypedStorage;
        this._dtype = dtype;
    }

    get untypedStorage(): UntypedStorage {
        return this._untypedStorage;
    }
}

