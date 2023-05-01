import { Tensor } from "./tensor";

export type FunctionInput = Tensor | number | boolean | string;

export class AutoFunction {
    constructor() {
    }
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Do not call forward on AutoFunction directly.");
    }
    static backward() {
    }
    static apply(...inputs: FunctionInput[]): Tensor {
        return this.forward(...inputs);
    }
}

export class LinearFunction extends AutoFunction {
    static forward(...inputs: FunctionInput[]) {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor];
        const output = input.mm(weight.t());
        if (bias) {
            output.add_(bias);
        }
        return output;
    }
}
