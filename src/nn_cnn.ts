import { Module } from "./nn_module";

export class Conv2d extends Module {
    private _inChannels: number;
    private _outChannels: number;
    get inChannels(): number {
        return this._inChannels;
    }
    get outChannels(): number {
        return this._outChannels;
    }
    constructor(inChannels: number, outChannels: number) {
        super();
        // Set properties directly on 'this'
        this._inChannels = inChannels;
        this._outChannels = outChannels;
    }
}

export class ConvTranspose2d extends Module {
}

export class AvgPooling2d extends Module {
}
