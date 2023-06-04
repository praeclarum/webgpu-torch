import { Tensor } from "./tensor";
import { Module } from "./nn_module";
/**
* ![Plot of relu and its gradient](../../plots/relu.svg)
*
* Calculates:
* ```js
* output = max(input, 0.0)
* ```
*
* Gradient:
* ```js
* inputGrad = input > 0.0 ? outputGrad : 0.0
* ```
*
*/
export class ReLU extends Module {
    forward(input: Tensor): Tensor {
        return input.relu();
    }
}
/**
* ![Plot of silu and its gradient](../../plots/silu.svg)
*
* Calculates:
* ```js
* output = input / (1.0 + exp(-input))
* ```
*
* Gradient:
* ```js
* var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
* ```
*
*/
export class SiLU extends Module {
    forward(input: Tensor): Tensor {
        return input.silu();
    }
}
