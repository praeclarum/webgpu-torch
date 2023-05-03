import { UntypedStorage } from "./storage";
import { Tensor, tensor, ones } from "./index";
import { LinearFunction } from "./autograd";
import { TensorCPU } from "./tensor_cpu";

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

test("linear forward", () => {
    const input = new Tensor([[3]], "float32");
    const weight = new Tensor([[10], [11]], "float32");
    const bias = new Tensor([[1000, 10000]], "float32");
    const output = LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.shape).toEqual([1, 2]);
});

test("complete sum over 2D tensor", () => {
    const input = new Tensor([[1, 2, 3], [4, 5, 6]], "float32");
    const sum = input.sum();
    expect(sum).toBeInstanceOf(Tensor);
    expect(sum.shape).toEqual([]);
});

test("tensor from json", () => {
    const x = tensor({data: [[1, 2, 3], [4, 5, 6]], dtype:"float32"});
    expect(x.shape).toEqual([2, 3]);
    expect(x.dtype).toEqual("float32");
});

test("ones are all ones", async () => {
    const x = ones([2, 3]);
    expect(await x.toArrayAsync()).toEqual([[1, 1, 1], [1, 1, 1]]);
});

test("tensor init", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    expect(await a.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("matrix multiply", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = tensor([[7, 8], [9, 10], [11, 12]]);
    const c = a.mm(b);
    expect(await c.toArrayAsync()).toEqual([[58, 64], [139, 154]]);
});
