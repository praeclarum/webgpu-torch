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

