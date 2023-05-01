import { Tensor } from "./tensor";
import { AutoFunction, LinearFunction } from "./autograd";

test("auto function forward fails", () => {
    expect(() => AutoFunction.forward()).toThrow();
});

test("linear function apply", () => {
    const input = new Tensor([[1, 2, 3], [4, 5, 6]], "float32");
    const weight = new Tensor([[1, 2], [3, 4], [5, 6]], "float32");
    const bias = new Tensor([1, 2], "float32");
    const output = LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
});
