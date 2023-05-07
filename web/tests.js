"use strict";
/*
These are tests that do not work in the normal node test environment
because they require WebGPU.
*/
const tests = [];
function test(description, callback) { tests.push({ description, callback }); }
const tensor = torch.tensor;
const Tensor = torch.Tensor;
const functions = torch.functions;


test("webgpu is supported", () => {
    expect(torch.hasWebGPU()).toBe(true);
});

test("tensor is webgpu", async () => {
    const x = torch.tensor([[1, 2, 3], [4, 5, 6]]);
    expect(x.requiresGrad).toBe(false);
    expect(x.device.type).toBe("webgpu");
    expect(await x.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("add_ vectors", async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([100, 200, 300]);
    const aa = a.add_(b);
    expect(await a.toArrayAsync()).toEqual([101, 202, 303]);
});

test("abs value", async () => {
    const x = tensor([[-1, 2, -3], [4, -5, 6]]);
    const y = x.abs();
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("abs value with grad", async () => {
    const x = tensor({data:[[-1, 2, -3], [4, -5, 6]], requiresGrad:true});
    const y = x.abs();
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(y.requiresGrad).toBe(true);
});

test("matrix multiply", async () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = tensor([[7, 8], [9, 10], [11, 12]]);
    const c = a.mm(b);
    expect(await c.toArrayAsync()).toEqual([[58, 64], [139, 154]]);
});

test("linear forward", async () => {
    const input = tensor([[3]]);
    const weight = tensor([[10], [11]]);
    const bias = tensor([[1000, 10000]]);
    const output = functions.LinearFunction.apply(input, weight, bias);
    const expected = [[3 * 10 + 1000, 3 * 11 + 10000]];
    expect(output.shape).toEqual([1, 2]);
    expect(await output.toArrayAsync()).toEqual(expected);
});

test("abs backwards", async () => {
    const input = new Tensor({data:[-3], requiresGrad:true});
    const output = input.abs();
    output.backward();
    expect(input.grad).not.toBeNull();
    expect(output.grad).not.toBeNull();
    expect(await input.grad.toArrayAsync()).toEqual([-1]);
});

test("linear backwards", () => {
    const input = new Tensor([[3]], "float32", null, true);
    const weight = new Tensor([[10], [11]], "float32", null, true);
    const bias = new Tensor([[1000, 10000]], "float32", null, true);
    const output = functions.LinearFunction.apply(input, weight, bias);
    expect(output).toBeInstanceOf(Tensor);
    expect(output.shape).toEqual([1, 2]);
    // const loss = output.sum();
    // expect(loss.gradFunc).not.toBeNull();
    // expect(loss.requiresGrad).toBe(true);
    // loss.backward();
    // expect(input.grad).not.toBeNull();
    // expect(weight.grad).not.toBeNull();
    // expect(bias.grad).not.toBeNull();
    // expect(output.grad).not.toBeNull();
});

test("sum(x, n=2)", async () => {
    const n = 2;
    const xar = [];
    for (let i = 0; i < n; i++) {
        xar.push(i);
    }
    const x = tensor(xar);
    const sum = x.sum();
    expect(sum.shape).toEqual([]);
    expect(await sum.toArrayAsync()).toEqual([n * (n - 1) / 2]);
});

test("sum(x, n=3)", async () => {
    const n = 3;
    const xar = [];
    for (let i = 0; i < n; i++) {
        xar.push(i);
    }
    const x = tensor(xar);
    const sum = x.sum();
    expect(sum.shape).toEqual([]);
    expect(await sum.toArrayAsync()).toEqual([n * (n - 1) / 2]);
});

test("sum(x, n=911)", async () => {
    const n = 911;
    const xar = [];
    for (let i = 0; i < n; i++) {
        xar.push(i);
    }
    const x = tensor(xar);
    const sum = x.sum();
    expect(sum.shape).toEqual([]);
    expect(await sum.toArrayAsync()).toEqual([n * (n - 1) / 2]);
});

// test("norm", async () => {
//     const x = tensor([3, 4]);
//     const norm = x.norm();
//     expect(norm.shape).toEqual([]);
//     expect(await norm.toArrayAsync()).toEqual([5]);
// });
