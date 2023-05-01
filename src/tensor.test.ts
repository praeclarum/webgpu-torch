import { UntypedStorage } from "./storage";
import { Tensor } from "./index";
import { LinearFunction } from "./autograd";

test("create tensor with storage and dtype", () => {
    const tensor = new Tensor([10, 20, 30], "float32");
    expect(tensor).toBeInstanceOf(Tensor);
    expect(tensor.dtype).toBe("float32");
});

test("can toggle requiresGrad", () => {
    const tensor = new Tensor([10, 20, 30], "float32");
    expect(tensor.requiresGrad).toBe(false);
    tensor.requiresGrad = true;
    expect(tensor.requiresGrad).toBe(true);
});

test("linear makes grad func and req gradient", () => {
    const input = new Tensor([10, 20, 30], "float32", true);
    const weight = new Tensor([10, 20, 30], "float32", true);
    const bias = new Tensor([1000], "float32", true);
    const output = LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.gradFunc).not.toBeNull();
    expect(output.requiresGrad).toBe(true);
});

test("linear backwards", () => {
    const input = new Tensor([10, 20, 30], "float32", true);
    const weight = new Tensor([10, 20, 30], "float32", true);
    const bias = new Tensor([1000], "float32", true);
    const output = LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.gradFunc).not.toBeNull();
    output.backward();
    expect(input.grad).not.toBeNull();
    expect(weight.grad).not.toBeNull();
    expect(bias.grad).not.toBeNull();
    expect(output.grad).not.toBeNull();
});
