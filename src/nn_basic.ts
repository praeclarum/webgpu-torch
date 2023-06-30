import { empty, ones, zeros } from "./factories";
import { Module, Parameter } from "./nn_module";
import { Tensor } from "./tensor";
import { linear } from "./ops_artisanal";
import { Deviceish } from "./device";
import { Dtype } from "./dtype";
import { uniform_, kaimingUniform_, calculateFanInAndFanOut } from "./init";

/** Applies a linear transformation to the incoming data: `y = xA^T + b` */
export class Linear extends Module {
    readonly inFeatures: number;
    readonly outFeatures: number;
    readonly weight: Parameter;
    readonly bias: Parameter | undefined;
    constructor(inFeatures: number, outFeatures: number, bias: boolean = true, device?: Deviceish, dtype?: Dtype) {
        super();
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = new Parameter(empty([outFeatures, inFeatures], dtype, device));
        if (bias) {
            this.bias = new Parameter(empty([outFeatures], dtype, device));
        }
        else {
            this.registerBuffer("bias", null);
        }
        this.resetParameters();
    }
    private resetParameters(): void {
        kaimingUniform_(this.weight, Math.sqrt(5));
        if (this.bias) {
            const [fanIn, fanOut] = calculateFanInAndFanOut(this.weight);
            const bound = fanIn > 0 ? 1.0 / Math.sqrt(fanIn) : 0.0;
            uniform_(this.bias, -bound, bound);
        }
    }
    forward(input: Tensor): Tensor {
        return linear(input, this.weight, this.bias);
    }
}
