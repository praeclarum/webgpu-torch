import { ones } from "./factories";
import { tensor } from "./ops_artisanal";

test("sigmoid(1).size == 307", async () => {
    // prime number > 300
    const n = 307;
    const x = ones(n);
    const y = x.sigmoid();
    expect(y.shape[0]).toBe(n);
    const a = await y.toArrayAsync() as number[];
    expect(a.length).toBe(n);
    for (let i = 0; i < n; i++) {
        expect(a[i]).toBeCloseTo(0.7310585786300049);
    }
});

test("add with alpha", async () => {
    const x = ones(3);
    const y = x.add(ones(3), 5);
    const a = await y.toArrayAsync();
    expect(a).toEqual([6, 6, 6]);
});

test("sub with alpha", async () => {
    const x = ones(3);
    const y = x.sub(ones(3), 7);
    const a = await y.toArrayAsync();
    expect(a).toEqual([-6, -6, -6]);
});

test("sum vector", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.sum();
    const a = await y.toArrayAsync();
    expect(a).toEqual(6);
});

test("sum vector grad", async () => {
    const x = tensor({data:[1, 2, 3, 4, 5, 6], requiresGrad: true});
    const y = x.sum();
    y.backward();
    const a = await x.grad!.toArrayAsync();
    expect(a).toEqual([1, 1, 1, 1, 1, 1]);
});

test("mean vector", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.mean();
    expect(await y.toArrayAsync()).toBeCloseTo(2);
});

test("mean vector grad", async () => {
    const x = tensor({data:[1, 2, 3, 4], requiresGrad: true});
    const y = x.mean();
    y.backward();
    expect(await x.grad!.toArrayAsync()).toEqual([0.25, 0.25, 0.25, 0.25]);
});

test("add 1d broadcasts", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.add(tensor([4]));
    expect(await y.toArrayAsync()).toEqual([5, 6, 7]);
});

test("add 2d to 1d broadcasts", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.add(tensor([4]));
    expect(await y.toArrayAsync()).toEqual([[ 5,  6,  7], [ 8,  9, 10]]);
});

test("add 2d to 2d broadcasts", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.add(tensor([[4]]));
    expect(await y.toArrayAsync()).toEqual([[ 5,  6,  7], [ 8,  9, 10]]);
});

test("add 2d to 2d inplace broadcasts", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    x.add_(tensor([[4]]));
    expect(await x.toArrayAsync()).toEqual([[ 5,  6,  7], [ 8,  9, 10]]);
});

// test("gather 1d", async () => {
//     const x = tensor([1, 2, 3]);
//     const y = x.gather(1, tensor([2, 1, 0]));
//     expect(await y.toArrayAsync()).toEqual([3, 2, 1]);
// });

test("numel", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    expect(x.numel()).toEqual(6);
});

test("view [] to []", async () => {
    const a = ones([]);
    expect(() => a.view([])).toThrow("shape '[]' is invalid for input of size 0");
});
test("view [] to [1]", async () => {
    const a = ones([]);
    expect(() => a.view([1])).toThrow("shape '[1]' is invalid for input of size 0");
});
// test("view [1] to []", async () => {
//     const a = tensor([34.0]);
//     expect(a.shape).toEqual([1]);
//     const c = a.view([]);
//     expect(c.shape).toEqual([]);
//     expect(await c.toArrayAsync()).toEqual(34.0);
// });
// test("view [2] to []", async () => {
//     const a = ones([2]);
//     expect(() => a.view([])).toThrow("shape '[]' is invalid for input of size 2");
// });
test("view [1] to [1]", async () => {
    const a = tensor([-116.0]);
    expect(a.shape).toEqual([1]);
    const c = a.view([1]);
    expect(c.shape).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-116.0]);
});
test("view [2] to [-1]", async () => {
    const a = tensor([-50.0, 82.0]);
    expect(a.shape).toEqual([2]);
    const c = a.view([-1]);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([-50.0, 82.0]);
});
// test("view [1, 1] to []", async () => {
//     const a = tensor([[120.0]]);
//     expect(a.shape).toEqual([1, 1]);
//     const c = a.view([]);
//     expect(c.shape).toEqual([]);
//     expect(await c.toArrayAsync()).toEqual(120.0);
// });
test("view [2, 3] to [2, -1]", async () => {
    const a = tensor([[112.0, -103.0, -218.0], [-80.0, -32.0, 237.0]]);
    expect(a.shape).toEqual([2, 3]);
    const c = a.view([2, -1]);
    expect(c.shape).toEqual([2, 3]);
    expect(await c.toArrayAsync()).toEqual([[112.0, -103.0, -218.0], [-80.0, -32.0, 237.0]]);
});
test("view [2, 3] to [-1, 3]", async () => {
    const a = tensor([[-138.0, 103.0, 58.0], [-182.0, 64.0, 8.0]]);
    expect(a.shape).toEqual([2, 3]);
    const c = a.view([-1, 3]);
    expect(c.shape).toEqual([2, 3]);
    expect(await c.toArrayAsync()).toEqual([[-138.0, 103.0, 58.0], [-182.0, 64.0, 8.0]]);
});
// test("view [2, 3] to [3, 2]", async () => {
//     const a = tensor([[-122.0, -74.0, -13.0], [59.0, 70.0, -52.0]]);
//     expect(a.shape).toEqual([2, 3]);
//     const c = a.view([3, 2]);
//     expect(c.shape).toEqual([3, 2]);
//     expect(await c.toArrayAsync()).toEqual([[-122.0, -74.0], [-13.0, 59.0], [70.0, -52.0]]);
// });
