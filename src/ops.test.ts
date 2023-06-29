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

test("squeeze [] dim ", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.squeeze();
    expect(c.shape).toEqual([0]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([]);
});
test("squeeze [] dim 0", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([0]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([]);
});
test("squeeze [] dim 1", async () => {
    const a = ones([]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [1] dim ", async () => {
    const a = tensor([-220.0]);
    expect(a.shape).toEqual([1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([]);
    expect(c.strides).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(-220.0);
});
test("squeeze [1] dim 0", async () => {
    const a = tensor([-62.0]);
    expect(a.shape).toEqual([1]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([]);
    expect(c.strides).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(-62.0);
});
test("squeeze [1] dim 1", async () => {
    const a = ones([1]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [2] dim ", async () => {
    const a = tensor([117.0, 131.0]);
    expect(a.shape).toEqual([2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([117.0, 131.0]);
});
test("squeeze [2] dim 0", async () => {
    const a = tensor([-164.0, -24.0]);
    expect(a.shape).toEqual([2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-164.0, -24.0]);
});
test("squeeze [2] dim 1", async () => {
    const a = ones([2]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [1, 1] dim ", async () => {
    const a = tensor([[59.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([]);
    expect(c.strides).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(59.0);
});
test("squeeze [1, 1] dim 1", async () => {
    const a = tensor([[-84.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([1]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-84.0]);
});
test("squeeze [2, 1] dim ", async () => {
    const a = tensor([[-170.0], [62.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-170.0, 62.0]);
});
test("squeeze [2, 1] dim 1", async () => {
    const a = tensor([[52.0], [-157.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([52.0, -157.0]);
});
test("squeeze [1, 2] dim ", async () => {
    const a = tensor([[97.0, -42.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(c.strides).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([97.0, -42.0]);
});
test("squeeze [2, 3, 5] dim ", async () => {
    const a = tensor([[[-71.0, -72.0, -6.0, 49.0, -8.0], [-13.0, 37.0, 132.0, 51.0, 24.0], [-125.0, 56.0, 1.0, -6.0, 31.0]], [[-92.0, 28.0, -110.0, 95.0, 156.0], [-171.0, 18.0, -148.0, -15.0, -166.0], [-90.0, -117.0, 134.0, 139.0, -161.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2, 3, 5]);
    expect(c.strides).toEqual([15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-71.0, -72.0, -6.0, 49.0, -8.0], [-13.0, 37.0, 132.0, 51.0, 24.0], [-125.0, 56.0, 1.0, -6.0, 31.0]], [[-92.0, 28.0, -110.0, 95.0, 156.0], [-171.0, 18.0, -148.0, -15.0, -166.0], [-90.0, -117.0, 134.0, 139.0, -161.0]]]);
});
test("squeeze [2, 3, 5] dim 1", async () => {
    const a = tensor([[[193.0, 28.0, -84.0, -94.0, 23.0], [141.0, -6.0, -29.0, 30.0, 64.0], [-17.0, 82.0, -154.0, -73.0, 84.0]], [[-50.0, -86.0, -155.0, -77.0, -133.0], [51.0, 171.0, 4.0, 92.0, -24.0], [51.0, -80.0, 27.0, -39.0, -76.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(c.strides).toEqual([15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[193.0, 28.0, -84.0, -94.0, 23.0], [141.0, -6.0, -29.0, 30.0, 64.0], [-17.0, 82.0, -154.0, -73.0, 84.0]], [[-50.0, -86.0, -155.0, -77.0, -133.0], [51.0, 171.0, 4.0, 92.0, -24.0], [51.0, -80.0, 27.0, -39.0, -76.0]]]);
});
test("squeeze [2, 3, 5] dim 3", async () => {
    const a = ones([2, 3, 5]);
    expect(() => a.squeeze(3)).toThrow('Dimension out of range (expected to be in range of [-3, 2], but got 3)');
});
test("squeeze [2, 3, 5] dim -2", async () => {
    const a = tensor([[[40.0, -19.0, -112.0, 167.0, 16.0], [151.0, 17.0, 78.0, 14.0, -90.0], [56.0, 154.0, 171.0, 40.0, -65.0]], [[-168.0, 64.0, -74.0, -144.0, 69.0], [78.0, -153.0, -13.0, -263.0, 108.0], [80.0, -124.0, -9.0, -189.0, -125.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(-2);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(c.strides).toEqual([15, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[40.0, -19.0, -112.0, 167.0, 16.0], [151.0, 17.0, 78.0, 14.0, -90.0], [56.0, 154.0, 171.0, 40.0, -65.0]], [[-168.0, 64.0, -74.0, -144.0, 69.0], [78.0, -153.0, -13.0, -263.0, 108.0], [80.0, -124.0, -9.0, -189.0, -125.0]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim ", async () => {
    const a = tensor([[[[[-47.0, 76.0]], [[74.0, -63.0]]]], [[[[-69.0, -139.0]], [[56.0, 39.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2, 2, 2]);
    expect(c.strides).toEqual([4, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-47.0, 76.0], [74.0, -63.0]], [[-69.0, -139.0], [56.0, 39.0]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim 0", async () => {
    const a = tensor([[[[[-15.0, -52.0]], [[23.0, 104.0]]]], [[[[-56.0, -194.0]], [[34.0, 91.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2, 1, 2, 1, 2]);
    expect(c.strides).toEqual([4, 4, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[-15.0, -52.0]], [[23.0, 104.0]]]], [[[[-56.0, -194.0]], [[34.0, 91.0]]]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim 1", async () => {
    const a = tensor([[[[[0.0, 71.0]], [[86.0, 18.0]]]], [[[[-145.0, -26.0]], [[161.0, -40.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2, 2, 1, 2]);
    expect(c.strides).toEqual([4, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[0.0, 71.0]], [[86.0, 18.0]]], [[[-145.0, -26.0]], [[161.0, -40.0]]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim (1, 2, 3)", async () => {
    const a = tensor([[[[[57.0, 33.0]], [[64.0, 69.0]]]], [[[[113.0, 122.0]], [[83.0, -97.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze([1, 2, 3]);
    expect(c.shape).toEqual([2, 2, 2]);
    expect(c.strides).toEqual([4, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[57.0, 33.0], [64.0, 69.0]], [[113.0, 122.0], [83.0, -97.0]]]);
});

test("view [] to []", async () => {
    const a = ones([]);
    expect(() => a.view([])).toThrow("shape '[]' is invalid for input of size 0");
});
test("view [] to [1]", async () => {
    const a = ones([]);
    expect(() => a.view([1])).toThrow("shape '[1]' is invalid for input of size 0");
});
test("view [1] to []", async () => {
    const a = tensor([34.0]);
    expect(a.shape).toEqual([1]);
    const c = a.view([]);
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(34.0);
});
test("view [2] to []", async () => {
    const a = ones([2]);
    expect(() => a.view([])).toThrow("shape '[]' is invalid for input of size 2");
});
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
test("view [1, 1] to []", async () => {
    const a = tensor([[120.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.view([]);
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(120.0);
});
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
