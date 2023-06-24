import { Module } from "./nn_module";
import { Tensor } from "./tensor";
// import { linear } from "./ops_artisanal";

export class Linear extends Module {
    inChannels: number;
    outChannels: number;
    constructor(inChannels: number, outChannels: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
    }
    forward(input: Tensor): Tensor {
        // linear(input, this.weight, this.bias);
        throw new Error("Method not implemented.");
    }
}
