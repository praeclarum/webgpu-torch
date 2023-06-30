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
* ![Plot of sigmoid and its gradient](../../plots/sigmoid.svg)
*
* Calculates:
* ```js
* output = 1.0 / (1.0 + exp(-input))
* ```
*
* Gradient:
* ```js
* var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)
* ```
*
*/
export class Sigmoid extends Module {
    forward(input: Tensor): Tensor {
        return input.sigmoid();
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
/**
* ![Plot of tanh and its gradient](../../plots/tanh.svg)
*
* Calculates:
* ```js
* output = tanh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))
* ```
*
*/
export class Tanh extends Module {
    forward(input: Tensor): Tensor {
        return input.tanh();
    }
}
