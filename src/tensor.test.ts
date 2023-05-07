import { ones } from "./factories";
import { LinearFunction } from "./functions";
import { tensor, conv2d } from "./ops_artisanal";
import { Tensor } from "./tensor";

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

test("linear forward", async () => {
    const input = tensor([[3]]);
    const weight = tensor([[10], [11]]);
    const bias = tensor([[1000, 10000]]);
    const output = LinearFunction.apply(input, weight, bias);
    const expected = [[3 * 10 + 1000, 3 * 11 + 10000]];
    expect(output.shape).toEqual([1, 2]);
    expect(await output.toArrayAsync()).toEqual(expected);
});

test("abs backwards", async () => {
    const input = new Tensor({ data: [-3], requiresGrad: true });
    const output = input.abs();
    output.backward();
    expect(input.grad).not.toBeNull();
    expect(output.grad).not.toBeNull();
    expect(await input.grad!.toArrayAsync()).toEqual([-1]);
});

test("complete sum over 2D tensor", () => {
    const input = new Tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        "float32"
    );
    const sum = input.sum();
    expect(sum).toBeInstanceOf(Tensor);
    expect(sum.shape).toEqual([]);
});

test("tensor from json", () => {
    const x = tensor({
        data: [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype: "float32",
    });
    expect(x.shape).toEqual([2, 3]);
    expect(x.dtype).toEqual("float32");
});

test("ones are all ones", async () => {
    const x = ones([2, 3]);
    expect(await x.toArrayAsync()).toEqual([
        [1, 1, 1],
        [1, 1, 1],
    ]);
});

test("tensor init", async () => {
    const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
    ]);
    expect(await a.toArrayAsync()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
    ]);
});

test("add_ vectors", async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([100, 200, 300]);
    a.add_(b);
    expect(await a.toArrayAsync()).toEqual([101, 202, 303]);
});

// test("add_ vectors with alpha", async () => {
//     const a = tensor([1, 2, 3]);
//     const b = tensor([100, 200, 300]);
//     a.add_(b, 5);
//     expect(await a.toArrayAsync()).toEqual([501, 1002, 1503]);
// });

test("matrix multiply", async () => {
    const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
    ]);
    const b = tensor([
        [7, 8],
        [9, 10],
        [11, 12],
    ]);
    const c = a.mm(b);
    expect(await c.toArrayAsync()).toEqual([
        [58, 64],
        [139, 154],
    ]);
});

test("sum(x, n=3)", async () => {
    const n = 3;
    const xar: number[] = [];
    for (let i = 0; i < n; i++) {
        xar.push(i);
    }
    const x = tensor(xar);
    const sum = x.sum();
    expect(sum.shape).toEqual([]);
    expect(await sum.toArrayAsync()).toEqual([(n * (n - 1)) / 2]);
});

test("sum(x, n=911)", async () => {
    const n = 911;
    const xar: number[] = [];
    for (let i = 0; i < n; i++) {
        xar.push(i);
    }
    const x = tensor(xar);
    const sum = x.sum();
    expect(sum.shape).toEqual([]);
    expect(await sum.toArrayAsync()).toEqual([(n * (n - 1)) / 2]);
});

test("norm", async () => {
    const x = tensor([3, 4]);
    const norm = x.norm();
    expect(norm.shape).toEqual([]);
    expect(await norm.toArrayAsync()).toEqual([5]);
});

test("conv2d", async () => {
    const image = tensor([
        [
            [
                [10, 0, 26, 64],
                [61, 90, 62, 54],
                [7, 85, 95, 86],
                [46, 0, 76, 37],
                [9, 31, 56, 64],
            ],
        ],
    ]);
    expect(image.shape).toEqual([1, 1, 5, 4]);
    const kernel = tensor([
        [
            [
                [15, 42, 14],
                [17, 31, 32],
                [4, 12, 0],
            ],
        ],
        [
            [
                [26, 10, 26],
                [25, 31, 48],
                [21, 42, 13],
            ],
        ],
    ]);
    expect(kernel.shape).toEqual([2, 1, 3, 3]);
    const conv = conv2d(image, kernel);
    expect(conv.shape).toEqual([1, 2, 3, 2]);
    expect(await conv.toArrayAsync()).toEqual([
        [
            [
                [7373, 8648],
                [11541, 12764],
                [8627, 10805],
            ],
            [
                [13179, 15581],
                [13422, 17235],
                [10519, 13363],
            ],
        ],
    ]);
});

test("batched conv2d", async () => {
    const image = tensor([
        [
            [
                [10, 0, 26, 64],
                [61, 90, 62, 54],
                [7, 85, 95, 86],
                [46, 0, 76, 37],
                [9, 31, 56, 64],
            ],
        ],
        [
            [
                [10, 0, 26, 64],
                [61, 90, 62, 54],
                [7, 85, 95, 86],
                [46, 0, 76, 37],
                [9, 31, 56, 64],
            ],
        ],
    ]);
    expect(image.shape).toEqual([2, 1, 5, 4]);
    const kernel = tensor([
        [
            [
                [15, 42, 14],
                [17, 31, 32],
                [4, 12, 0],
            ],
        ],
        [
            [
                [26, 10, 26],
                [25, 31, 48],
                [21, 42, 13],
            ],
        ],
    ]);
    expect(kernel.shape).toEqual([2, 1, 3, 3]);
    const conv = conv2d(image, kernel);
    expect(conv.shape).toEqual([2, 2, 3, 2]);
    expect(await conv.toArrayAsync()).toEqual([
        [
            [
                [7373, 8648],
                [11541, 12764],
                [8627, 10805],
            ],
            [
                [13179, 15581],
                [13422, 17235],
                [10519, 13363],
            ],
        ],
        [
            [
                [7373, 8648],
                [11541, 12764],
                [8627, 10805],
            ],
            [
                [13179, 15581],
                [13422, 17235],
                [10519, 13363],
            ],
        ],
    ]);
});
