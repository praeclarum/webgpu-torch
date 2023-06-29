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
    expect(await c.toArrayAsync()).toEqual([]);
});
test("squeeze [] dim 0", async () => {
    const a = tensor([]);
    expect(a.shape).toEqual([0]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([0]);
    expect(await c.toArrayAsync()).toEqual([]);
});
test("squeeze [] dim 1", async () => {
    const a = ones([]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [1] dim ", async () => {
    const a = tensor([125.0]);
    expect(a.shape).toEqual([1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(125.0);
});
test("squeeze [1] dim 0", async () => {
    const a = tensor([66.0]);
    expect(a.shape).toEqual([1]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(66.0);
});
test("squeeze [1] dim 1", async () => {
    const a = ones([1]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [2] dim ", async () => {
    const a = tensor([-148.0, -219.0]);
    expect(a.shape).toEqual([2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([-148.0, -219.0]);
});
test("squeeze [2] dim 0", async () => {
    const a = tensor([101.0, 38.0]);
    expect(a.shape).toEqual([2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([101.0, 38.0]);
});
test("squeeze [2] dim 1", async () => {
    const a = ones([2]);
    expect(() => a.squeeze(1)).toThrow('Dimension out of range (expected to be in range of [-1, 0], but got 1)');
});
test("squeeze [1, 1] dim ", async () => {
    const a = tensor([[47.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(47.0);
});
test("squeeze [1, 1] dim 0", async () => {
    const a = tensor([[112.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([112.0]);
});
test("squeeze [1, 1] dim 1", async () => {
    const a = tensor([[0.0]]);
    expect(a.shape).toEqual([1, 1]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([0.0]);
});
test("squeeze [2, 1] dim ", async () => {
    const a = tensor([[-132.0], [110.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([-132.0, 110.0]);
});
test("squeeze [2, 1] dim 0", async () => {
    const a = tensor([[29.0], [-98.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[29.0], [-98.0]]);
});
test("squeeze [2, 1] dim 1", async () => {
    const a = tensor([[-104.0], [66.0]]);
    expect(a.shape).toEqual([2, 1]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([-104.0, 66.0]);
});
test("squeeze [1, 2] dim ", async () => {
    const a = tensor([[178.0, -37.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([178.0, -37.0]);
});
test("squeeze [1, 2] dim 0", async () => {
    const a = tensor([[15.0, 1.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([15.0, 1.0]);
});
test("squeeze [1, 2] dim 1", async () => {
    const a = tensor([[-114.0, -72.0]]);
    expect(a.shape).toEqual([1, 2]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([1, 2]);
    expect(await c.toArrayAsync()).toEqual([[-114.0, -72.0]]);
});
test("squeeze [2, 3, 5] dim ", async () => {
    const a = tensor([[[-141.0, -66.0, 54.0, 178.0, -109.0], [-127.0, -103.0, 20.0, -56.0, -105.0], [43.0, 197.0, 2.0, -116.0, -126.0]], [[-5.0, -46.0, 5.0, 122.0, 39.0], [33.0, 3.0, 45.0, 318.0, -69.0], [-13.0, -33.0, -10.0, 116.0, 167.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-141.0, -66.0, 54.0, 178.0, -109.0], [-127.0, -103.0, 20.0, -56.0, -105.0], [43.0, 197.0, 2.0, -116.0, -126.0]], [[-5.0, -46.0, 5.0, 122.0, 39.0], [33.0, 3.0, 45.0, 318.0, -69.0], [-13.0, -33.0, -10.0, 116.0, 167.0]]]);
});
test("squeeze [2, 3, 5] dim 0", async () => {
    const a = tensor([[[-39.0, 78.0, 2.0, -149.0, 123.0], [17.0, -121.0, 118.0, 69.0, -53.0], [64.0, 66.0, 101.0, 57.0, -63.0]], [[56.0, -24.0, -35.0, 115.0, -26.0], [149.0, 107.0, -182.0, -298.0, -141.0], [138.0, 69.0, 112.0, -142.0, 55.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-39.0, 78.0, 2.0, -149.0, 123.0], [17.0, -121.0, 118.0, 69.0, -53.0], [64.0, 66.0, 101.0, 57.0, -63.0]], [[56.0, -24.0, -35.0, 115.0, -26.0], [149.0, 107.0, -182.0, -298.0, -141.0], [138.0, 69.0, 112.0, -142.0, 55.0]]]);
});
test("squeeze [2, 3, 5] dim 1", async () => {
    const a = tensor([[[62.0, 43.0, -72.0, 146.0, -160.0], [3.0, -143.0, -79.0, 125.0, 40.0], [-77.0, 131.0, -75.0, 121.0, 62.0]], [[31.0, -45.0, 128.0, -144.0, 7.0], [-45.0, -169.0, -50.0, -29.0, -20.0], [77.0, 112.0, 30.0, -82.0, -2.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[62.0, 43.0, -72.0, 146.0, -160.0], [3.0, -143.0, -79.0, 125.0, 40.0], [-77.0, 131.0, -75.0, 121.0, 62.0]], [[31.0, -45.0, 128.0, -144.0, 7.0], [-45.0, -169.0, -50.0, -29.0, -20.0], [77.0, 112.0, 30.0, -82.0, -2.0]]]);
});
test("squeeze [2, 3, 5] dim 2", async () => {
    const a = tensor([[[-45.0, 137.0, -147.0, -40.0, -40.0], [-176.0, 51.0, -127.0, -51.0, 118.0], [-4.0, 38.0, 10.0, 13.0, -67.0]], [[-47.0, -84.0, -120.0, 65.0, 23.0], [160.0, -8.0, 24.0, -83.0, 234.0], [36.0, 25.0, -97.0, -40.0, 145.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(2);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-45.0, 137.0, -147.0, -40.0, -40.0], [-176.0, 51.0, -127.0, -51.0, 118.0], [-4.0, 38.0, 10.0, 13.0, -67.0]], [[-47.0, -84.0, -120.0, 65.0, 23.0], [160.0, -8.0, 24.0, -83.0, 234.0], [36.0, 25.0, -97.0, -40.0, 145.0]]]);
});
test("squeeze [2, 3, 5] dim 3", async () => {
    const a = ones([2, 3, 5]);
    expect(() => a.squeeze(3)).toThrow('Dimension out of range (expected to be in range of [-3, 2], but got 3)');
});
test("squeeze [2, 3, 5] dim -2", async () => {
    const a = tensor([[[-55.0, -134.0, 111.0, -23.0, 42.0], [216.0, 76.0, 88.0, 112.0, -29.0], [-52.0, -191.0, 66.0, 45.0, -122.0]], [[68.0, 30.0, -105.0, -145.0, 10.0], [90.0, 48.0, 65.0, 62.0, -210.0], [149.0, 100.0, -184.0, -49.0, 16.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const c = a.squeeze(-2);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-55.0, -134.0, 111.0, -23.0, 42.0], [216.0, 76.0, 88.0, 112.0, -29.0], [-52.0, -191.0, 66.0, 45.0, -122.0]], [[68.0, 30.0, -105.0, -145.0, 10.0], [90.0, 48.0, 65.0, 62.0, -210.0], [149.0, 100.0, -184.0, -49.0, 16.0]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim ", async () => {
    const a = tensor([[[[[-82.0, 58.0]], [[-79.0, -98.0]]]], [[[[-89.0, -26.0]], [[34.0, 291.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze();
    expect(c.shape).toEqual([2, 2, 2]);
    expect(await c.toArrayAsync()).toEqual([[[-82.0, 58.0], [-79.0, -98.0]], [[-89.0, -26.0], [34.0, 291.0]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim 0", async () => {
    const a = tensor([[[[[-186.0, 67.0]], [[-27.0, 54.0]]]], [[[[-130.0, -155.0]], [[-73.0, 105.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze(0);
    expect(c.shape).toEqual([2, 1, 2, 1, 2]);
    expect(await c.toArrayAsync()).toEqual([[[[[-186.0, 67.0]], [[-27.0, 54.0]]]], [[[[-130.0, -155.0]], [[-73.0, 105.0]]]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim 1", async () => {
    const a = tensor([[[[[-62.0, -59.0]], [[6.0, -150.0]]]], [[[[108.0, -37.0]], [[167.0, 160.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze(1);
    expect(c.shape).toEqual([2, 2, 1, 2]);
    expect(await c.toArrayAsync()).toEqual([[[[-62.0, -59.0]], [[6.0, -150.0]]], [[[108.0, -37.0]], [[167.0, 160.0]]]]);
});
test("squeeze [2, 1, 2, 1, 2] dim (1, 2, 3)", async () => {
    const a = tensor([[[[[-96.0, -104.0]], [[61.0, 36.0]]]], [[[[138.0, -206.0]], [[-93.0, 153.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 1, 2]);
    const c = a.squeeze([1, 2, 3]);
    expect(c.shape).toEqual([2, 2, 2]);
    expect(await c.toArrayAsync()).toEqual([[[-96.0, -104.0], [61.0, 36.0]], [[138.0, -206.0], [-93.0, 153.0]]]);
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
