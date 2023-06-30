import { ones, zeros } from "./factories";
import { Module, Parameter } from "./nn_module";
import { Tensor } from "./tensor";
import { linear } from "./ops_artisanal";

export class Linear extends Module {
    readonly inChannels: number;
    readonly outChannels: number;
    readonly weight: Parameter;
    readonly bias: Parameter;
    constructor(inChannels: number, outChannels: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.weight = new Parameter(ones([outChannels, inChannels]));
        this.bias = new Parameter(zeros([outChannels]));
    }
    forward(input: Tensor): Tensor {
        return linear(input, this.weight, this.bias);
    }
}
