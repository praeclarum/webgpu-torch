import { Module } from "./nn_module";

export class AvgPooling2d extends Module {}

export class Conv2d extends Module {
    inChannels: number;
    outChannels: number;
    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | [number, number],
        stride: number | [number, number],
        padding: number | [number, number] | "valid" | "same",
        dtype: string
    ) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
    }
}

export class ConvTranspose2d extends Module {}

export class GroupNorm extends Module {
    numGroups: number;
    numChannels: number;
    constructor(numGroups: number, numChannels: number) {
        super();
        this.numGroups = numGroups;
        this.numChannels = numChannels;
    }
}

export class Linear extends Module {
    inChannels: number;
    outChannels: number;
    constructor(inChannels: number, outChannels: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
    }
}
