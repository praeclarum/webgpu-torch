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

test("view of same shape", async () => {
    const x = tensor([1, 2, 3]);
    const y = x.view([3]);
    expect(await y.toArrayAsync()).toEqual([1, 2, 3]);
});

test("view of same shape implicit 0/2", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.view([-1, 3]);
    expect(y.shape).toEqual([2, 3]);
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("view of same shape implicit 1/2", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const y = x.view([2, -1]);
    expect(y.shape).toEqual([2, 3]);
    expect(await y.toArrayAsync()).toEqual([[1, 2, 3], [4, 5, 6]]);
});

test("two implicits fails", async () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    expect(() => x.view([-1, -1])).toThrow();
});

// test("view of different shape", async () => {
//     const x = tensor([1, 2, 3]);
//     const y = x.view([1, 3]);
//     expect(await y.toArrayAsync()).toEqual([[1, 2, 3]]);
// });
