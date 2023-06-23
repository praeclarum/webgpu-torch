import { ones } from "./factories";
import { tensor, matmul } from "./ops_artisanal";

test("matmul (2,) x (1,)", async () => {
    const a = ones([2]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('inconsistent tensor size, expected tensor [2] and src [1] to have the same number of elements, but got 2 and 1 elements respectively');
});
test("matmul (2,) x (1, 1)", async () => {
    const a = ones([2]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 1x1)');
});
test("matmul (2,) x (1, 3)", async () => {
    const a = ones([2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 1x3)');
});
test("matmul (2,) x (1, 1, 1)", async () => {
    const a = ones([2]);
    const b = ones([1, 1, 1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 1, 1x1,2');
});
test("matmul (2,) x (1, 1, 3)", async () => {
    const a = ones([2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 3, 3x1,2');
});
test("matmul (2,) x (2, 1)", async () => {
    const a = tensor([39.0, -34.0]);
    expect(a.shape).toEqual([2]);
    const b = tensor([[-19.0], [-200.0]]);
    expect(b.shape).toEqual([2, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([6059.0]);
});
test("matmul (2,) x (2, 1, 1)", async () => {
    const a = ones([2]);
    const b = ones([2, 1, 1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x1,2');
});
test("matmul (2,) x (2, 1, 3)", async () => {
    const a = ones([2]);
    const b = ones([2, 1, 3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x1,2');
});
test("matmul (2,) x (2,)", async () => {
    const a = tensor([-12.0, -334.0]);
    expect(a.shape).toEqual([2]);
    const b = tensor([63.0, 22.0]);
    expect(b.shape).toEqual([2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([]);
    expect(await c.toArrayAsync()).toEqual(-8104.0);
});
test("matmul (2,) x (2, 3)", async () => {
    const a = tensor([22.0, 145.0]);
    expect(a.shape).toEqual([2]);
    const b = tensor([[55.0, -2.0, -25.0], [-127.0, 152.0, 2.0]]);
    expect(b.shape).toEqual([2, 3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([3]);
    expect(await c.toArrayAsync()).toEqual([-17205.0, 21996.0, -260.0]);
});
test("matmul (1, 2) x (1,)", async () => {
    const a = ones([1, 2]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 1, 1x2,1');
});
test("matmul (1, 2) x (1, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 1x1)');
});
test("matmul (1, 2) x (1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 1x3)');
});
test("matmul (1, 2) x (1, 1, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x1 and 2x1)');
});
test("matmul (1, 2) x (1, 1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x1 and 2x1)');
});
test("matmul (1, 2) x (2, 1)", async () => {
    const a = tensor([[-148.0, 100.0]]);
    expect(a.shape).toEqual([1, 2]);
    const b = tensor([[9.0], [-68.0]]);
    expect(b.shape).toEqual([2, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 1]);
    expect(await c.toArrayAsync()).toEqual([[-8132.0]]);
});
test("matmul (1, 2) x (2, 1, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([2, 1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x1 and 2x1)');
});
test("matmul (1, 2) x (2, 1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([2, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x1 and 2x1)');
});
test("matmul (1, 2) x (2,)", async () => {
    const a = tensor([[39.0, 122.0]]);
    expect(a.shape).toEqual([1, 2]);
    const b = tensor([-140.0, -82.0]);
    expect(b.shape).toEqual([2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1]);
    expect(await c.toArrayAsync()).toEqual([-15464.0]);
});
test("matmul (1, 2) x (2, 3)", async () => {
    const a = tensor([[-14.0, -116.0]]);
    expect(a.shape).toEqual([1, 2]);
    const b = tensor([[176.0, 54.0, -60.0], [52.0, -31.0, -63.0]]);
    expect(b.shape).toEqual([2, 3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 3]);
    expect(await c.toArrayAsync()).toEqual([[-8496.0, 2840.0, 8148.0]]);
});
test("matmul (2, 2) x (1,)", async () => {
    const a = ones([2, 2]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x2,1');
});
test("matmul (2, 2) x (1, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x1)');
});
test("matmul (2, 2) x (1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x3)');
});
test("matmul (2, 2) x (1, 1, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x1 and 2x2)');
});
test("matmul (2, 2) x (1, 1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x1 and 2x2)');
});
test("matmul (2, 2) x (2, 1)", async () => {
    const a = tensor([[-152.0, -72.0], [-72.0, 134.0]]);
    expect(a.shape).toEqual([2, 2]);
    const b = tensor([[83.0], [-9.0]]);
    expect(b.shape).toEqual([2, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[-11968.0], [-7182.0]]);
});
test("matmul (2, 2) x (2, 1, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([2, 1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x1 and 2x2)');
});
test("matmul (2, 2) x (2, 1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([2, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x1 and 2x2)');
});
test("matmul (2, 2) x (2,)", async () => {
    const a = tensor([[-66.0, 47.0], [-18.0, -50.0]]);
    expect(a.shape).toEqual([2, 2]);
    const b = tensor([-128.0, -89.0]);
    expect(b.shape).toEqual([2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([4265.0, 6754.0]);
});
test("matmul (2, 2) x (2, 3)", async () => {
    const a = tensor([[-82.0, -78.0], [83.0, -17.0]]);
    expect(a.shape).toEqual([2, 2]);
    const b = tensor([[53.0, 115.0, 153.0], [54.0, 101.0, -132.0]]);
    expect(b.shape).toEqual([2, 3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3]);
    expect(await c.toArrayAsync()).toEqual([[-8558.0, -17308.0, -2250.0], [3481.0, 7828.0, 14943.0]]);
});
test("matmul (2, 1, 2) x (1,)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x2,1');
});
test("matmul (2, 1, 2) x (1, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x1)');
});
test("matmul (2, 1, 2) x (1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x3)');
});
test("matmul (2, 1, 2) x (1, 1, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 1, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 1].');
});
test("matmul (2, 1, 2) x (1, 1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 1].');
});
test("matmul (2, 1, 2) x (2, 1)", async () => {
    const a = tensor([[[33.0, 16.0]], [[16.0, 61.0]]]);
    expect(a.shape).toEqual([2, 1, 2]);
    const b = tensor([[29.0], [-97.0]]);
    expect(b.shape).toEqual([2, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-595.0]], [[-5453.0]]]);
});
test("matmul (2, 1, 2) x (2, 1, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([2, 1, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 1].');
});
test("matmul (2, 1, 2) x (2, 1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([2, 1, 3]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 1].');
});
test("matmul (2, 1, 2) x (2,)", async () => {
    const a = tensor([[[-157.0, 106.0]], [[-191.0, 20.0]]]);
    expect(a.shape).toEqual([2, 1, 2]);
    const b = tensor([72.0, 16.0]);
    expect(b.shape).toEqual([2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[-9608.0], [-13432.0]]);
});
test("matmul (2, 1, 2) x (2, 3)", async () => {
    const a = tensor([[[14.0, -41.0]], [[229.0, 50.0]]]);
    expect(a.shape).toEqual([2, 1, 2]);
    const b = tensor([[-37.0, -162.0, 20.0], [160.0, 13.0, 116.0]]);
    expect(b.shape).toEqual([2, 3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 3]);
    expect(await c.toArrayAsync()).toEqual([[[-7078.0, -2801.0, -4476.0]], [[-473.0, -36448.0, 10380.0]]]);
});
test("matmul (2, 3) x (3,)", async () => {
    const a = tensor([[36.0, -107.0, 151.0], [-18.0, -22.0, 133.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([-78.0, 22.0, -282.0]);
    expect(b.shape).toEqual([3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2]);
    expect(await c.toArrayAsync()).toEqual([-47744.0, -36586.0]);
});
test("matmul (2, 3) x (3, 1)", async () => {
    const a = tensor([[19.0, -108.0, -5.0], [-159.0, -120.0, -20.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[-9.0], [109.0], [-100.0]]);
    expect(b.shape).toEqual([3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1]);
    expect(await c.toArrayAsync()).toEqual([[-11443.0], [-9649.0]]);
});
test("matmul (2, 3) x (3, 5)", async () => {
    const a = tensor([[-118.0, -25.0, 8.0], [-70.0, -166.0, -58.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[112.0, 26.0, -134.0, 72.0, -198.0], [3.0, -107.0, -112.0, 80.0, -62.0], [-8.0, -46.0, -63.0, 80.0, 98.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 5]);
    expect(await c.toArrayAsync()).toEqual([[-13355.0, -761.0, 18108.0, -9856.0, 25698.0], [-7874.0, 18610.0, 31626.0, -22960.0, 18468.0]]);
});
test("matmul (2, 3) x (1, 3)", async () => {
    const a = ones([2, 3]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 1x3)');
});
test("matmul (2, 3) x (1, 3, 1)", async () => {
    const a = tensor([[-6.0, 53.0, 109.0], [-38.0, -62.0, 196.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[[-64.0], [-56.0], [301.0]]]);
    expect(b.shape).toEqual([1, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[30225.0], [64900.0]]]);
});
test("matmul (2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[40.0, 102.0, -114.0], [114.0, 284.0, 9.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[[99.0, -185.0, 34.0, -66.0, 9.0], [29.0, 68.0, -32.0, -16.0, -89.0], [79.0, 54.0, -40.0, -111.0, -97.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-2088.0, -6620.0, 2656.0, 8382.0, 2340.0], [20233.0, -1292.0, -5572.0, -13067.0, -25123.0]]]);
});
test("matmul (2, 3) x (2, 3)", async () => {
    const a = ones([2, 3]);
    const b = ones([2, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)');
});
test("matmul (2, 3) x (2, 3, 1)", async () => {
    const a = tensor([[85.0, -69.0, -71.0], [-33.0, -75.0, -82.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[[-30.0], [-213.0], [-69.0]], [[238.0], [-27.0], [-54.0]]]);
    expect(b.shape).toEqual([2, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[17046.0], [22623.0]], [[25927.0], [-1401.0]]]);
});
test("matmul (2, 3) x (2, 3, 5)", async () => {
    const a = tensor([[-90.0, 5.0, 12.0], [-18.0, -71.0, -147.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[[-21.0, -44.0, 74.0, -64.0, -10.0], [268.0, -248.0, 262.0, -7.0, -6.0], [-36.0, 46.0, 145.0, 47.0, 87.0]], [[30.0, -16.0, -56.0, 79.0, 7.0], [-49.0, -111.0, -161.0, 3.0, 7.0], [-107.0, 119.0, -116.0, -93.0, -7.0]]]);
    expect(b.shape).toEqual([2, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[2798.0, 3272.0, -3610.0, 6289.0, 1914.0], [-13358.0, 11638.0, -41249.0, -5260.0, -12183.0]], [[-4229.0, 2313.0, 2843.0, -8211.0, -679.0], [18668.0, -9324.0, 29491.0, 12036.0, 406.0]]]);
});
test("matmul (2, 3) x (1,)", async () => {
    const a = ones([2, 3]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x3,1');
});
test("matmul (2, 3) x (1, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 1x1)');
});
test("matmul (2, 3) x (1, 5)", async () => {
    const a = ones([2, 3]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 1x5)');
});
test("matmul (2, 3) x (2,)", async () => {
    const a = ones([2, 3]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x3,2');
});
test("matmul (2, 3) x (2, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 2x1)');
});
test("matmul (2, 3) x (2, 5)", async () => {
    const a = ones([2, 3]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 2x5)');
});
test("matmul (1, 2, 3) x (3,)", async () => {
    const a = tensor([[[-61.0, -62.0, 177.0], [-13.0, 203.0, 17.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([44.0, -4.0, -32.0]);
    expect(b.shape).toEqual([3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2]);
    expect(await c.toArrayAsync()).toEqual([[-8100.0, -1928.0]]);
});
test("matmul (1, 2, 3) x (3, 1)", async () => {
    const a = tensor([[[19.0, 47.0, -91.0], [-15.0, -20.0, -205.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[46.0], [0.0], [-43.0]]);
    expect(b.shape).toEqual([3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[4787.0], [8125.0]]]);
});
test("matmul (1, 2, 3) x (3, 5)", async () => {
    const a = tensor([[[111.0, -144.0, -68.0], [151.0, 61.0, -113.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[-38.0, 10.0, 28.0, -231.0, -176.0], [44.0, 38.0, -83.0, 114.0, -151.0], [47.0, 132.0, -47.0, 195.0, -29.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-13750.0, -13338.0, 18256.0, -55317.0, 4180.0], [-8365.0, -11088.0, 4476.0, -49962.0, -32510.0]]]);
});
test("matmul (1, 2, 3) x (1, 3)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 1x3)');
});
test("matmul (1, 2, 3) x (1, 3, 1)", async () => {
    const a = tensor([[[-99.0, -83.0, 51.0], [36.0, -97.0, 28.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[[-20.0], [-196.0], [-145.0]]]);
    expect(b.shape).toEqual([1, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[10853.0], [14232.0]]]);
});
test("matmul (1, 2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[[-14.0, 37.0, -18.0], [-35.0, 135.0, 82.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[[-108.0, -66.0, -65.0, 150.0, -97.0], [-44.0, -25.0, 23.0, -209.0, 87.0], [-145.0, 109.0, -102.0, 103.0, 80.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[2494.0, -1963.0, 3597.0, -11687.0, 3137.0], [-14050.0, 7873.0, -2984.0, -25019.0, 21700.0]]]);
});
test("matmul (1, 2, 3) x (2, 3)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([2, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)');
});
test("matmul (1, 2, 3) x (2, 3, 1)", async () => {
    const a = tensor([[[42.0, -48.0, 31.0], [-143.0, 6.0, 78.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[[-51.0], [-17.0], [-289.0]], [[48.0], [-65.0], [-108.0]]]);
    expect(b.shape).toEqual([2, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-10285.0], [-15351.0]], [[1788.0], [-15678.0]]]);
});
test("matmul (1, 2, 3) x (2, 3, 5)", async () => {
    const a = tensor([[[-81.0, 39.0, -15.0], [-46.0, 147.0, -127.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[[103.0, -45.0, -70.0, 100.0, -13.0], [63.0, 64.0, -146.0, 97.0, 45.0], [-66.0, -20.0, 54.0, -155.0, 204.0]], [[95.0, 45.0, -85.0, 14.0, -280.0], [-15.0, 55.0, 26.0, -14.0, -248.0], [22.0, -1.0, -53.0, -123.0, -17.0]]]);
    expect(b.shape).toEqual([2, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-4896.0, 6441.0, -834.0, -1992.0, -252.0], [12905.0, 14018.0, -25100.0, 29344.0, -18695.0]], [[-8610.0, -1485.0, 8694.0, 165.0, 13263.0], [-9369.0, 6142.0, 14463.0, 12919.0, -21417.0]]]);
});
test("matmul (1, 2, 3) x (1,)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x3,1');
});
test("matmul (1, 2, 3) x (1, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 1x1)');
});
test("matmul (1, 2, 3) x (1, 5)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 1x5)');
});
test("matmul (1, 2, 3) x (2,)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x3,2');
});
test("matmul (1, 2, 3) x (2, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 2x1)');
});
test("matmul (1, 2, 3) x (2, 5)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 2x5)');
});
test("matmul (2, 2, 3) x (3,)", async () => {
    const a = tensor([[[-36.0, 38.0, 26.0], [75.0, -91.0, -0.0]], [[183.0, -87.0, 74.0], [-119.0, 48.0, -2.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([132.0, 57.0, 77.0]);
    expect(b.shape).toEqual([3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2]);
    expect(await c.toArrayAsync()).toEqual([[-584.0, 4713.0], [24895.0, -13126.0]]);
});
test("matmul (2, 2, 3) x (3, 1)", async () => {
    const a = tensor([[[-98.0, -46.0, 19.0], [0.0, 212.0, 167.0]], [[43.0, -124.0, 10.0], [-180.0, -105.0, 95.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[124.0], [22.0], [37.0]]);
    expect(b.shape).toEqual([3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-12461.0], [10843.0]], [[2974.0], [-21115.0]]]);
});
test("matmul (2, 2, 3) x (3, 5)", async () => {
    const a = tensor([[[-52.0, 273.0, -165.0], [-99.0, 195.0, -21.0]], [[-75.0, -36.0, -151.0], [-45.0, 46.0, -152.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[65.0, -66.0, -15.0, 162.0, -10.0], [93.0, -16.0, -16.0, 47.0, -138.0], [52.0, 123.0, -31.0, 15.0, -93.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[13429.0, -21231.0, 1527.0, 1932.0, -21809.0], [10608.0, 831.0, -984.0, -7188.0, -23967.0]], [[-16075.0, -13047.0, 6382.0, -16107.0, 19761.0], [-6551.0, -16462.0, 4651.0, -7408.0, 8238.0]]]);
});
test("matmul (2, 2, 3) x (1, 3)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 1x3)');
});
test("matmul (2, 2, 3) x (1, 3, 1)", async () => {
    const a = tensor([[[121.0, 41.0, -236.0], [138.0, -8.0, 22.0]], [[28.0, 28.0, 3.0], [28.0, -137.0, -33.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[[30.0], [31.0], [-25.0]]]);
    expect(b.shape).toEqual([1, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[10801.0], [3342.0]], [[1633.0], [-2582.0]]]);
});
test("matmul (2, 2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[[-34.0, 158.0, -4.0], [59.0, 44.0, 126.0]], [[-85.0, -1.0, 68.0], [-81.0, -28.0, -212.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[[-65.0, -90.0, 119.0, -53.0, -93.0], [-19.0, -15.0, 12.0, 100.0, -14.0], [70.0, 19.0, -86.0, 56.0, -64.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-1072.0, 614.0, -1806.0, 17378.0, 1206.0], [4149.0, -3576.0, -3287.0, 8329.0, -14167.0]], [[10304.0, 8957.0, -15975.0, 8213.0, 3567.0], [-9043.0, 3682.0, 8257.0, -10379.0, 21493.0]]]);
});
test("matmul (2, 2, 3) x (2, 3)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([2, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 2x3)');
});
test("matmul (2, 2, 3) x (2, 3, 1)", async () => {
    const a = tensor([[[-131.0, -0.0, 115.0], [184.0, -84.0, 31.0]], [[-3.0, 128.0, -12.0], [-127.0, 110.0, -35.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[[197.0], [-40.0], [-115.0]], [[69.0], [156.0], [184.0]]]);
    expect(b.shape).toEqual([2, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-39032.0], [36043.0]], [[17553.0], [1957.0]]]);
});
test("matmul (2, 2, 3) x (2, 3, 5)", async () => {
    const a = tensor([[[-27.0, 87.0, 62.0], [-54.0, 63.0, 29.0]], [[90.0, -196.0, -118.0], [68.0, -152.0, -28.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[[-37.0, 179.0, 130.0, -181.0, -33.0], [120.0, -81.0, 31.0, -166.0, -165.0], [5.0, 24.0, -90.0, -9.0, -21.0]], [[90.0, 100.0, -121.0, 94.0, -74.0], [83.0, -81.0, 61.0, 204.0, 40.0], [121.0, 173.0, -35.0, -122.0, 24.0]]]);
    expect(b.shape).toEqual([2, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[11749.0, -10392.0, -6393.0, -10113.0, -14766.0], [9703.0, -14073.0, -7677.0, -945.0, -9222.0]], [[-22446.0, 4462.0, -18716.0, -17128.0, -17332.0], [-9884.0, 14268.0, -16520.0, -21200.0, -11784.0]]]);
});
test("matmul (2, 2, 3) x (1,)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 4, 4x3,1');
});
test("matmul (2, 2, 3) x (1, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 1x1)');
});
test("matmul (2, 2, 3) x (1, 5)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 1x5)');
});
test("matmul (2, 2, 3) x (2,)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 4, 4x3,2');
});
test("matmul (2, 2, 3) x (2, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 2x1)');
});
test("matmul (2, 2, 3) x (2, 5)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 2x5)');
});
test("matmul (2, 1, 2, 3) x (3,)", async () => {
    const a = tensor([[[[79.0, 68.0, 113.0], [-86.0, -114.0, 131.0]]], [[[-20.0, -147.0, -235.0], [-108.0, 144.0, 27.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([-95.0, -109.0, -59.0]);
    expect(b.shape).toEqual([3]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2]);
    expect(await c.toArrayAsync()).toEqual([[[-21584.0, 12867.0]], [[31788.0, -7029.0]]]);
});
test("matmul (2, 1, 2, 3) x (3, 1)", async () => {
    const a = tensor([[[[107.0, 12.0, -40.0], [129.0, 80.0, 23.0]]], [[[-168.0, -30.0, -54.0], [-199.0, 11.0, -50.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[74.0], [-78.0], [-17.0]]);
    expect(b.shape).toEqual([3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[7662.0], [2915.0]]], [[[-9174.0], [-14734.0]]]]);
});
test("matmul (2, 1, 2, 3) x (3, 5)", async () => {
    const a = tensor([[[[-157.0, 55.0, 90.0], [-78.0, 46.0, -35.0]]], [[[33.0, -75.0, 103.0], [131.0, -100.0, -126.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[-33.0, 113.0, -88.0, 70.0, -44.0], [134.0, 30.0, -102.0, -102.0, 111.0], [4.0, 136.0, 112.0, 66.0, 34.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[12911.0, -3851.0, 18286.0, -10660.0, 16073.0], [8598.0, -12194.0, -1748.0, -12462.0, 7348.0]]], [[[-10727.0, 15487.0, 16282.0, 16758.0, -6275.0], [-18227.0, -5333.0, -15440.0, 11054.0, -21148.0]]]]);
});
test("matmul (2, 1, 2, 3) x (1, 3)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 1x3)');
});
test("matmul (2, 1, 2, 3) x (1, 3, 1)", async () => {
    const a = tensor([[[[-1.0, -62.0, 125.0], [56.0, -52.0, 12.0]]], [[[186.0, 23.0, -22.0], [28.0, -0.0, -153.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[[22.0], [-63.0], [146.0]]]);
    expect(b.shape).toEqual([1, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[22134.0], [6260.0]]], [[[-569.0], [-21722.0]]]]);
});
test("matmul (2, 1, 2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[[[69.0, 29.0, 183.0], [-137.0, 6.0, -7.0]]], [[[-258.0, -16.0, -34.0], [-7.0, 118.0, -52.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[[88.0, -53.0, 63.0, 214.0, -5.0], [-76.0, 3.0, 29.0, -9.0, 5.0], [-61.0, 215.0, 58.0, -127.0, -2.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[-7295.0, 35775.0, 15802.0, -8736.0, -566.0], [-12085.0, 5774.0, -8863.0, -28483.0, 729.0]]], [[[-19414.0, 6316.0, -18690.0, -50750.0, 1278.0], [-6412.0, -10455.0, -35.0, 4044.0, 729.0]]]]);
});
test("matmul (2, 1, 2, 3) x (2, 3)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([2, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 2x3)');
});
test("matmul (2, 1, 2, 3) x (2, 3, 1)", async () => {
    const a = tensor([[[[-202.0, -106.0, 118.0], [59.0, 38.0, 143.0]]], [[[14.0, 2.0, 31.0], [-13.0, -19.0, -96.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[[-82.0], [22.0], [-113.0]], [[-45.0], [-132.0], [98.0]]]);
    expect(b.shape).toEqual([2, 3, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 2, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[898.0], [-20161.0]], [[34646.0], [6343.0]]], [[[-4607.0], [11496.0]], [[2144.0], [-6315.0]]]]);
});
test("matmul (2, 1, 2, 3) x (2, 3, 5)", async () => {
    const a = tensor([[[[-66.0, -84.0, 14.0], [-24.0, -80.0, -97.0]]], [[[100.0, -39.0, -27.0], [93.0, 41.0, -67.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[[-32.0, 23.0, 4.0, 183.0, -53.0], [-60.0, 79.0, 149.0, 19.0, 69.0], [-89.0, -129.0, -52.0, 64.0, -67.0]], [[47.0, 122.0, -107.0, -11.0, 95.0], [83.0, -65.0, -74.0, -78.0, 17.0], [82.0, -228.0, -87.0, -208.0, -41.0]]]);
    expect(b.shape).toEqual([2, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[5906.0, -9960.0, -13508.0, -12778.0, -3236.0], [14201.0, 5641.0, -6972.0, -12120.0, 2251.0]], [[-8926.0, -5784.0, 12060.0, 4366.0, -8272.0], [-15722.0, 24388.0, 16927.0, 26680.0, 337.0]]], [[[1543.0, 2702.0, -4007.0, 15831.0, -6182.0], [527.0, 14021.0, 9965.0, 13510.0, 2389.0]], [[-751.0, 20891.0, -5465.0, 7558.0, 9944.0], [2280.0, 23957.0, -7156.0, 9715.0, 12279.0]]]]);
});
test("matmul (2, 1, 2, 3) x (1,)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 4, 4x3,1');
});
test("matmul (2, 1, 2, 3) x (1, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 1x1)');
});
test("matmul (2, 1, 2, 3) x (1, 5)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 1x5)');
});
test("matmul (2, 1, 2, 3) x (2,)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 4, 4x3,2');
});
test("matmul (2, 1, 2, 3) x (2, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 2x1)');
});
test("matmul (2, 1, 2, 3) x (2, 5)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 2x5)');
});
test("matmul (2, 3, 5) x (5,)", async () => {
    const a = tensor([[[-58.0, -207.0, 193.0, 112.0, 100.0], [-152.0, -45.0, 13.0, -218.0, -102.0], [142.0, -157.0, -223.0, -19.0, 55.0]], [[131.0, -37.0, -96.0, 67.0, -176.0], [-273.0, 41.0, 20.0, 180.0, 71.0], [63.0, -137.0, -112.0, 28.0, -89.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([40.0, 48.0, 67.0, -105.0, 42.0]);
    expect(b.shape).toEqual([5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3]);
    expect(await c.toArrayAsync()).toEqual([[-6885.0, 11237.0, -12492.0], [-17395.0, -23530.0, -18238.0]]);
});
test("matmul (2, 3, 5) x (5, 1)", async () => {
    const a = tensor([[[8.0, -22.0, 109.0, -136.0, 59.0], [-62.0, 93.0, 98.0, -46.0, 1.0], [15.0, -132.0, 4.0, 100.0, -91.0]], [[55.0, -50.0, 22.0, -160.0, 36.0], [-10.0, 28.0, 46.0, -71.0, -7.0], [31.0, -82.0, -118.0, 131.0, 31.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[-142.0], [345.0], [-69.0], [-29.0], [82.0]]);
    expect(b.shape).toEqual([5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[-7465.0], [35543.0], [-58308.0]], [[-18986.0], [9391.0], [-25807.0]]]);
});
test("matmul (2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[-168.0, 204.0, 29.0, 21.0, -33.0], [-21.0, 41.0, 138.0, -112.0, 221.0], [21.0, 12.0, -228.0, 1.0, -64.0]], [[-210.0, 37.0, -80.0, -75.0, 127.0], [-83.0, 39.0, -45.0, -202.0, -145.0], [-52.0, 187.0, 120.0, 88.0, 36.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[103.0, 29.0, 103.0, -131.0, 201.0, 4.0, -67.0], [71.0, 35.0, 125.0, 22.0, 132.0, -54.0, 144.0], [-46.0, -4.0, -2.0, -154.0, 45.0, 79.0, -86.0], [-120.0, 113.0, -1.0, -59.0, -40.0, 115.0, 144.0], [160.0, -77.0, -77.0, 22.0, -199.0, -80.0, 204.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[-11954.0, 7066.0, 10658.0, 20065.0, 192.0, -4342.0, 34430.0], [43200.0, -29399.0, -14219.0, -6129.0, -32098.0, -21956.0, 24399.0], [3143.0, 6982.0, 9046.0, 31158.0, 8241.0, -13341.0, 7017.0]], [[13997.0, -22729.0, -26549.0, 47863.0, -63199.0, -27943.0, 41386.0], [-2670.0, -12523.0, 7783.0, 27389.0, 23375.0, -17623.0, -43621.0], [-2399.0, 11729.0, 14919.0, -11954.0, 8948.0, 6414.0, 40108.0]]]);
});
test("matmul (2, 3, 5) x (1, 5)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 1x5)');
});
test("matmul (2, 3, 5) x (1, 5, 1)", async () => {
    const a = tensor([[[-54.0, 28.0, -34.0, 17.0, 149.0], [-97.0, 12.0, 16.0, -231.0, -93.0], [228.0, 86.0, 36.0, 41.0, 53.0]], [[5.0, -25.0, -97.0, -21.0, 137.0], [-41.0, -70.0, 251.0, 73.0, 9.0], [234.0, 113.0, -47.0, 100.0, 234.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[[-121.0], [-96.0], [113.0], [-73.0], [103.0]]]);
    expect(b.shape).toEqual([1, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[14110.0], [19677.0], [-29310.0]], [[6478.0], [35642.0], [-27671.0]]]);
});
test("matmul (2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[43.0, 50.0, -28.0, -85.0, -37.0], [-83.0, -75.0, -54.0, -40.0, 263.0], [-95.0, -69.0, -161.0, 93.0, -19.0]], [[55.0, 82.0, 77.0, 156.0, -20.0], [-74.0, 131.0, -95.0, -236.0, 136.0], [-112.0, 152.0, -36.0, -20.0, 4.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[[-109.0, 50.0, -222.0, 148.0, -127.0, -19.0, 73.0], [-22.0, 132.0, -42.0, -130.0, -31.0, 135.0, -181.0], [-90.0, 16.0, 70.0, 64.0, 68.0, 41.0, 1.0], [26.0, 58.0, 52.0, 22.0, -29.0, -7.0, 7.0], [84.0, 50.0, 8.0, -4.0, 137.0, -237.0, -154.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[-8585.0, 1522.0, -18322.0, -3650.0, -11519.0, 14149.0, -836.0], [36609.0, -4084.0, 17820.0, -7922.0, 46385.0, -72813.0, -33320.0], [27185.0, -11990.0, 17402.0, -13272.0, -2044.0, -10259.0, 8970.0]], [[-12353.0, 22854.0, -2312.0, 5920.0, -11555.0, 16830.0, -6578.0], [19022.0, 5184.0, -6908.0, -39798.0, 24353.0, -15384.0, -51804.0], [11920.0, 12928.0, 14952.0, -39096.0, 8192.0, 20364.0, -36480.0]]]);
});
test("matmul (2, 3, 5) x (2, 5)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 2x5)');
});
test("matmul (2, 3, 5) x (2, 5, 1)", async () => {
    const a = tensor([[[12.0, -118.0, -8.0, -35.0, 93.0], [90.0, -10.0, -78.0, 3.0, -100.0], [-8.0, 130.0, -12.0, 96.0, -49.0]], [[-141.0, -18.0, 124.0, 10.0, -48.0], [-51.0, 21.0, 70.0, 22.0, 10.0], [27.0, 31.0, 106.0, 68.0, -19.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[[-61.0], [-58.0], [206.0], [56.0], [25.0]], [[-111.0], [186.0], [40.0], [-97.0], [-208.0]]]);
    expect(b.shape).toEqual([2, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[4829.0], [-23310.0], [-5373.0]], [[26277.0], [8153.0], [4365.0]]]);
});
test("matmul (2, 3, 5) x (2, 5, 7)", async () => {
    const a = tensor([[[-143.0, -99.0, 55.0, -128.0, 14.0], [-18.0, -147.0, -36.0, 7.0, 34.0], [-122.0, 106.0, -95.0, -84.0, -88.0]], [[23.0, -297.0, 58.0, 113.0, -95.0], [9.0, 10.0, 175.0, 85.0, -12.0], [12.0, 127.0, 121.0, -7.0, 1.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[[-68.0, -56.0, 81.0, -169.0, 49.0, 172.0, 103.0], [41.0, 12.0, 201.0, 77.0, 49.0, 27.0, 45.0], [94.0, -44.0, 67.0, -120.0, -205.0, 213.0, 173.0], [-17.0, 91.0, -74.0, 76.0, 80.0, 170.0, 19.0], [-4.0, 48.0, 134.0, 112.0, 64.0, 43.0, -58.0]], [[144.0, 68.0, 117.0, -88.0, 105.0, -65.0, -20.0], [86.0, 41.0, -42.0, 6.0, -14.0, -44.0, -69.0], [172.0, 135.0, 61.0, -51.0, 237.0, -91.0, 39.0], [-170.0, -114.0, 111.0, -115.0, -27.0, -47.0, 51.0], [-83.0, -74.0, -113.0, 69.0, -94.0, 31.0, 33.0]]]);
    expect(b.shape).toEqual([2, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[12955.0, -6576.0, -16449.0, 1784.0, -32477.0, -36712.0, -12913.0], [-8442.0, 3097.0, -29379.0, 383.0, 2031.0, -12081.0, -16536.0], [5492.0, 416.0, -517.0, 23940.0, 6339.0, -56421.0, -20723.0]], [[-23579.0, -8635.0, 41981.0, -26314.0, 26198.0, -1961.0, 24923.0], [18802.0, 15845.0, 22099.0, -20260.0, 41113.0, -21317.0, 9894.0], [34569.0, 23082.0, 2561.0, -5591.0, 28254.0, -17019.0, -4608.0]]]);
});
test("matmul (2, 3, 5) x (1,)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x5,1');
});
test("matmul (2, 3, 5) x (1, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 1x1)');
});
test("matmul (2, 3, 5) x (1, 7)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 1x7)');
});
test("matmul (2, 3, 5) x (2,)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x5,2');
});
test("matmul (2, 3, 5) x (2, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 2x1)');
});
test("matmul (2, 3, 5) x (2, 7)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 2x7)');
});
test("matmul (1, 2, 3, 5) x (5,)", async () => {
    const a = tensor([[[[-170.0, 219.0, -102.0, 11.0, 33.0], [110.0, 143.0, 29.0, -55.0, 38.0], [-55.0, 62.0, 158.0, -124.0, 27.0]], [[71.0, -80.0, -256.0, 52.0, -61.0], [-193.0, 55.0, -23.0, -147.0, -214.0], [-114.0, -39.0, 27.0, 81.0, 50.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([165.0, -114.0, 26.0, 18.0, -4.0]);
    expect(b.shape).toEqual([5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3]);
    expect(await c.toArrayAsync()).toEqual([[[-55602.0, 1460.0, -14375.0], [15359.0, -40503.0, -12404.0]]]);
});
test("matmul (1, 2, 3, 5) x (5, 1)", async () => {
    const a = tensor([[[[91.0, -88.0, 45.0, 128.0, -42.0], [130.0, -118.0, 53.0, 58.0, 103.0], [40.0, 77.0, -148.0, 48.0, -54.0]], [[36.0, -95.0, 6.0, -112.0, 132.0], [-32.0, 14.0, -255.0, 26.0, -13.0], [71.0, 82.0, 71.0, -119.0, 167.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[-111.0], [-32.0], [68.0], [-145.0], [-133.0]]);
    expect(b.shape).toEqual([5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-17199.0], [-29159.0], [-16746.0]], [[-1864.0], [-16277.0], [-10633.0]]]]);
});
test("matmul (1, 2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[[-73.0, -57.0, 67.0, -65.0, -101.0], [74.0, 54.0, 3.0, 162.0, -120.0], [115.0, 153.0, -8.0, 70.0, -57.0]], [[-45.0, 188.0, 137.0, -75.0, 204.0], [-24.0, -50.0, 39.0, 68.0, 59.0], [-126.0, -114.0, -30.0, -170.0, 154.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[-15.0, -27.0, 65.0, 54.0, -24.0, 91.0, -76.0], [69.0, 21.0, -71.0, -176.0, 129.0, 101.0, -54.0], [-113.0, 61.0, 27.0, 73.0, 15.0, -109.0, -102.0], [-200.0, -27.0, 15.0, 108.0, 111.0, -164.0, 158.0], [172.0, -20.0, -125.0, -31.0, -170.0, 88.0, -156.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-14781.0, 8636.0, 12761.0, 7092.0, 5359.0, -17931.0, 7278.0], [-50763.0, -2655.0, 18487.0, 15927.0, 43617.0, -25267.0, 35470.0], [-14068.0, -1130.0, 4571.0, -11975.0, 34317.0, 10294.0, 3766.0]], [[48254.0, 11465.0, -39199.0, -39941.0, -15618.0, 30212.0, -64380.0], [-10949.0, -1039.0, -3312.0, 15866.0, -7771.0, -17445.0, 2086.0], [57902.0, 688.0, -22706.0, -12064.0, -57182.0, 21722.0, -32092.0]]]]);
});
test("matmul (1, 2, 3, 5) x (1, 5)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 1x5)');
});
test("matmul (1, 2, 3, 5) x (1, 5, 1)", async () => {
    const a = tensor([[[[196.0, 66.0, -89.0, 80.0, -182.0], [-164.0, -114.0, 79.0, 150.0, -128.0], [3.0, -113.0, 64.0, 79.0, 26.0]], [[-93.0, 122.0, -3.0, -19.0, -34.0], [-44.0, -31.0, 2.0, -90.0, -123.0], [-65.0, -170.0, 67.0, 26.0, -3.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[[9.0], [-31.0], [100.0], [-126.0], [-63.0]]]);
    expect(b.shape).toEqual([1, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-7796.0], [-878.0], [-1662.0]], [[-383.0], [19854.0], [8298.0]]]]);
});
test("matmul (1, 2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[[-236.0, 54.0, 38.0, 69.0, -193.0], [-18.0, -250.0, 84.0, 4.0, 24.0], [-125.0, -141.0, 33.0, -66.0, -10.0]], [[81.0, -115.0, -141.0, -42.0, 228.0], [-37.0, 210.0, 171.0, 89.0, 101.0], [31.0, 187.0, 152.0, -170.0, 141.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[[151.0, -38.0, -7.0, 191.0, 99.0, 54.0, 32.0], [94.0, 120.0, -68.0, -301.0, -122.0, -31.0, 36.0], [-55.0, -85.0, -29.0, -19.0, 126.0, 100.0, -56.0], [15.0, -193.0, -131.0, -9.0, -100.0, -34.0, 51.0], [-234.0, -14.0, -46.0, -160.0, 12.0, 18.0, -33.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[13547.0, 1603.0, -3283.0, -31793.0, -34380.0, -16438.0, 2152.0], [-36394.0, -37564.0, 13062.0, 66340.0, 39190.0, 15474.0, -14868.0], [-32594.0, -2097.0, 18612.0, 20133.0, 15465.0, 2985.0, -13960.0]], [[-44806.0, 21.0, 6356.0, 16663.0, 11219.0, -629.0, -3318.0], [-17551.0, -6520.0, -35285.0, -90487.0, -15425.0, 7384.0, -1994.0], [-21645.0, 39178.0, -1557.0, -74284.0, 18099.0, 19395.0, -14111.0]]]]);
});
test("matmul (1, 2, 3, 5) x (2, 5)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 2x5)');
});
test("matmul (1, 2, 3, 5) x (2, 5, 1)", async () => {
    const a = tensor([[[[59.0, -37.0, -270.0, -174.0, -133.0], [-29.0, -51.0, -101.0, 7.0, -38.0], [158.0, 178.0, -132.0, 27.0, -87.0]], [[50.0, -38.0, -16.0, -65.0, -177.0], [-29.0, -200.0, -107.0, 147.0, -86.0], [161.0, -16.0, 86.0, 181.0, -47.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[[-57.0], [56.0], [-50.0], [-48.0], [150.0]], [[92.0], [53.0], [124.0], [-184.0], [-134.0]]]);
    expect(b.shape).toEqual([2, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-3533.0], [-2189.0], [-6784.0]], [[36280.0], [-42060.0], [-2378.0]]]]);
});
test("matmul (1, 2, 3, 5) x (2, 5, 7)", async () => {
    const a = tensor([[[[49.0, -67.0, -55.0, -60.0, -37.0], [-42.0, -98.0, -64.0, -6.0, 114.0], [-4.0, -6.0, -183.0, -23.0, -10.0]], [[97.0, -72.0, 51.0, 42.0, -130.0], [8.0, 132.0, 36.0, -33.0, -20.0], [74.0, 93.0, -68.0, 82.0, -94.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[[-45.0, 82.0, 267.0, 72.0, -27.0, 21.0, 50.0], [97.0, -61.0, -39.0, 25.0, 145.0, 29.0, 2.0], [-57.0, 110.0, -210.0, 110.0, 62.0, -100.0, -36.0], [-54.0, -174.0, -92.0, 26.0, 104.0, 143.0, -4.0], [-38.0, -71.0, -261.0, -15.0, 228.0, -9.0, -141.0]], [[-184.0, 105.0, -45.0, 76.0, -39.0, 116.0, 10.0], [47.0, -121.0, 205.0, -82.0, -200.0, 67.0, 151.0], [53.0, -9.0, -20.0, 24.0, -110.0, 59.0, 81.0], [-130.0, -32.0, 48.0, -71.0, 206.0, 46.0, -157.0], [-95.0, -171.0, -3.0, 143.0, 125.0, -31.0, 139.0]]]);
    expect(b.shape).toEqual([2, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-923.0, 15122.0, 42423.0, -5202.0, -29124.0, -3661.0, 9753.0], [-7976.0, -11556.0, -23154.0, -14380.0, 8324.0, 792.0, -16042.0], [11651.0, -15380.0, 42322.0, -21016.0, -16780.0, 14843.0, 7878.0]], [[-11639.0, 39324.0, -17739.0, -7072.0, -2591.0, 15399.0, -30435.0], [12830.0, -10980.0, 24456.0, -9869.0, -39970.0, 10998.0, 25329.0], [-14579.0, 10579.0, 21313.0, -22898.0, -8864.0, 17489.0, -16665.0]]]]);
});
test("matmul (1, 2, 3, 5) x (1,)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x5,1');
});
test("matmul (1, 2, 3, 5) x (1, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 1x1)');
});
test("matmul (1, 2, 3, 5) x (1, 7)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 1x7)');
});
test("matmul (1, 2, 3, 5) x (2,)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x5,2');
});
test("matmul (1, 2, 3, 5) x (2, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 2x1)');
});
test("matmul (1, 2, 3, 5) x (2, 7)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 2x7)');
});
test("matmul (2, 2, 3, 5) x (5,)", async () => {
    const a = tensor([[[[-24.0, 121.0, -56.0, 103.0, 23.0], [-103.0, -78.0, 125.0, -5.0, 68.0], [-22.0, 17.0, 119.0, -82.0, -169.0]], [[197.0, 137.0, -100.0, 23.0, 30.0], [-152.0, -139.0, 102.0, 47.0, -14.0], [-178.0, 49.0, 100.0, 16.0, 150.0]]], [[[63.0, 17.0, -68.0, -144.0, -47.0], [-27.0, -5.0, 221.0, 14.0, 110.0], [-97.0, 29.0, -164.0, 72.0, -70.0]], [[118.0, -79.0, 107.0, 105.0, -17.0], [-197.0, -60.0, -60.0, 69.0, -14.0], [-104.0, -20.0, 106.0, 13.0, 95.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([64.0, 26.0, 115.0, -229.0, -153.0]);
    expect(b.shape).toEqual([5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3]);
    expect(await c.toArrayAsync()).toEqual([[[-31936.0, -3504.0, 57354.0], [-5187.0, -10233.0, -25232.0]], [[36821.0, 3521.0, -30092.0], [-3641.0, -34727.0, -12498.0]]]);
});
test("matmul (2, 2, 3, 5) x (5, 1)", async () => {
    const a = tensor([[[[23.0, -122.0, -108.0, 83.0, -35.0], [182.0, 189.0, 93.0, 10.0, 101.0], [34.0, 358.0, 19.0, 19.0, -107.0]], [[-77.0, 149.0, 4.0, 81.0, -72.0], [55.0, -130.0, 57.0, 55.0, 72.0], [226.0, -49.0, 172.0, 168.0, -74.0]]], [[[12.0, 60.0, -163.0, 103.0, -148.0], [30.0, 14.0, 13.0, -11.0, -65.0], [-6.0, 3.0, 73.0, 178.0, 121.0]], [[19.0, 43.0, -60.0, -88.0, 58.0], [58.0, -71.0, 321.0, 225.0, -45.0], [-52.0, 59.0, -37.0, 7.0, -108.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[140.0], [121.0], [39.0], [-16.0], [120.0]]);
    expect(b.shape).toEqual([5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-21282.0], [63936.0], [35675.0]], [[-2531.0], [1953.0], [20851.0]]], [[[-16825.0], [-1223.0], [14042.0]], [[13891.0], [3048.0], [-14656.0]]]]);
});
test("matmul (2, 2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[[158.0, -49.0, -40.0, -55.0, 107.0], [-169.0, -12.0, -28.0, 40.0, 28.0], [-39.0, 125.0, -92.0, 7.0, -111.0]], [[-112.0, -194.0, 13.0, 171.0, 84.0], [135.0, -62.0, 7.0, -91.0, -63.0], [-226.0, 53.0, 109.0, -62.0, 54.0]]], [[[39.0, 134.0, -75.0, -54.0, 9.0], [-82.0, 9.0, -31.0, 86.0, -159.0], [-91.0, 75.0, 73.0, 64.0, -14.0]], [[25.0, 93.0, 140.0, -134.0, 103.0], [-92.0, -31.0, -41.0, 100.0, 121.0], [-22.0, 9.0, -40.0, 168.0, -89.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[-34.0, 70.0, 4.0, 63.0, 70.0, -147.0, -32.0], [35.0, -88.0, -41.0, -85.0, -179.0, -40.0, 152.0], [-25.0, -129.0, 199.0, -86.0, 96.0, -65.0, -7.0], [42.0, -67.0, 45.0, 33.0, -128.0, -32.0, 139.0], [-70.0, -149.0, 121.0, -190.0, -101.0, 18.0, -33.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-15887.0, 8274.0, 5153.0, -4586.0, 12224.0, -14980.0, -23400.0], [5746.0, -14014.0, -568.0, -11219.0, -20318.0, 26367.0, 8416.0], [16065.0, 14208.0, -36705.0, 16151.0, -23622.0, 4491.0, 25528.0]], [[-2005.0, -16418.0, 27952.0, -2001.0, -2238.0, 19419.0, -4998.0], [-6347.0, 29487.0, -7243.0, 22140.0, 39231.0, -16042.0, -24363.0], [430.0, -38437.0, 22358.0, -40423.0, -12361.0, 26973.0, 4125.0]]], [[[2341.0, 2890.0, -21604.0, -5975.0, -22453.0, -4328.0, 11842.0], [18620.0, 15396.0, -22235.0, 29783.0, -5276.0, 8095.0, 21410.0], [7562.0, -24589.0, 12274.0, -13614.0, -19565.0, 3332.0, 23159.0]], [[-13933.0, -30863.0, 30580.0, -42362.0, 5292.0, -10353.0, -9669.0], [-1202.0, -23152.0, 11885.0, -19325.0, -29848.0, 16407.0, 8426.0], [15349.0, 4833.0, -11626.0, 23743.0, -19506.0, -1504.0, 28641.0]]]]);
});
test("matmul (2, 2, 3, 5) x (1, 5)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 1x5)');
});
test("matmul (2, 2, 3, 5) x (1, 5, 1)", async () => {
    const a = tensor([[[[-146.0, -80.0, -114.0, 11.0, -75.0], [-151.0, 1.0, -2.0, -77.0, -21.0], [116.0, 135.0, 34.0, 4.0, -109.0]], [[-67.0, 39.0, -104.0, 24.0, 11.0], [113.0, -224.0, 78.0, -264.0, -72.0], [-21.0, -93.0, 156.0, 7.0, 19.0]]], [[[48.0, -30.0, -82.0, 61.0, -68.0], [-52.0, 42.0, 149.0, 37.0, -29.0], [-92.0, 46.0, 31.0, 23.0, 117.0]], [[97.0, -34.0, -194.0, 1.0, -115.0], [-52.0, -3.0, 28.0, 126.0, 45.0], [41.0, 125.0, -84.0, -61.0, -18.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[[16.0], [64.0], [-258.0], [-192.0], [65.0]]]);
    expect(b.shape).toEqual([1, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[14969.0], [11583.0], [-6129.0]], [[24363.0], [13356.0], [-46645.0]]], [[[3872.0], [-45575.0], [-3337.0]], [[41761.0], [-29515.0], [40870.0]]]]);
});
test("matmul (2, 2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[[124.0, -19.0, 97.0, -173.0, -115.0], [55.0, 98.0, -56.0, 13.0, -125.0], [-41.0, 14.0, 87.0, 205.0, -138.0]], [[-44.0, -115.0, 140.0, -120.0, -43.0], [38.0, -59.0, 29.0, -34.0, -26.0], [50.0, 17.0, -105.0, 9.0, -148.0]]], [[[-63.0, 91.0, -65.0, 36.0, 110.0], [-235.0, 121.0, -30.0, -52.0, 103.0], [41.0, -68.0, 144.0, -25.0, -33.0]], [[111.0, -184.0, -7.0, 103.0, 24.0], [-270.0, 16.0, -157.0, 145.0, 262.0], [67.0, 16.0, -125.0, -6.0, 157.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[[4.0, -49.0, 135.0, 11.0, -75.0, -80.0, -160.0], [-103.0, -68.0, -40.0, 133.0, -91.0, -129.0, -92.0], [-167.0, -179.0, 201.0, -92.0, 138.0, 57.0, 71.0], [-200.0, 0.0, 53.0, -47.0, -259.0, -84.0, 42.0], [-71.0, 54.0, 22.0, 169.0, 89.0, 139.0, 14.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[29019.0, -28357.0, 25298.0, -21391.0, 40387.0, -3393.0, -20081.0], [5753.0, -6085.0, -9812.0, -2945.0, -35263.0, -38701.0, -22996.0], [-47337.0, -21968.0, 19221.0, -39550.0, -51570.0, -29969.0, 18127.0]], [[15342.0, -17406.0, 19494.0, -30286.0, 60338.0, 30438.0, 21918.0], [10032.0, -4445.0, 10945.0, -12893.0, 13013.0, 5466.0, -385.0], [24692.0, 7197.0, -17814.0, -12964.0, -35290.0, -33506.0, -18713.0]]], [[[-13780.0, 14474.0, -20882.0, 34288.0, -12060.0, 1862.0, 145.0], [-5306.0, 14219.0, -43085.0, 36119.0, 25109.0, 20166.0, 23596.0], [-9537.0, -24943.0, 35148.0, -26243.0, 26523.0, 11213.0, 8408.0]], [[-1739.0, 9622.0, 26925.0, -23392.0, -17088.0, 9141.0, 3333.0], [-24111.0, 54393.0, -55198.0, 51065.0, -17109.0, 34825.0, 40339.0], [9548.0, 26482.0, -13584.0, 41180.0, -8204.0, 7778.0, -19121.0]]]]);
});
test("matmul (2, 2, 3, 5) x (2, 5)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 2x5)');
});
test("matmul (2, 2, 3, 5) x (2, 5, 1)", async () => {
    const a = tensor([[[[-41.0, -131.0, 83.0, 206.0, -41.0], [6.0, -82.0, 90.0, -13.0, 14.0], [-34.0, -6.0, 28.0, -118.0, -102.0]], [[-63.0, 29.0, 13.0, 118.0, 16.0], [5.0, 36.0, -82.0, -27.0, -39.0], [53.0, -9.0, 58.0, -89.0, -82.0]]], [[[-76.0, -34.0, -293.0, -53.0, -88.0], [2.0, 143.0, 96.0, 46.0, -101.0], [1.0, 155.0, -41.0, -49.0, -76.0]], [[-166.0, -106.0, -58.0, 51.0, -261.0], [-41.0, 5.0, 49.0, 96.0, -95.0], [64.0, 69.0, 124.0, -59.0, -19.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[[-24.0], [184.0], [157.0], [174.0], [52.0]], [[-56.0], [53.0], [-63.0], [104.0], [-46.0]]]);
    expect(b.shape).toEqual([2, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[23623.0], [-2636.0], [-21728.0]], [[15782.0], [5780.0], [-12583.0]]], [[[-64231.0], [44088.0], [9581.0]], [[24642.0], [13828.0], [-13001.0]]]]);
});
test("matmul (2, 2, 3, 5) x (2, 5, 7)", async () => {
    const a = tensor([[[[156.0, -121.0, 22.0, -5.0, 210.0], [87.0, 2.0, 30.0, 209.0, 104.0], [-103.0, -45.0, -31.0, -57.0, 16.0]], [[-157.0, 31.0, 101.0, 82.0, -110.0], [13.0, 59.0, 125.0, -242.0, -182.0], [122.0, -140.0, -102.0, -75.0, 97.0]]], [[[-97.0, 92.0, -39.0, 49.0, 134.0], [141.0, -11.0, 124.0, 100.0, -104.0], [89.0, 122.0, -55.0, -5.0, 85.0]], [[29.0, 28.0, 21.0, -100.0, -151.0], [52.0, 17.0, 12.0, 115.0, 23.0], [-41.0, 62.0, -159.0, 27.0, -255.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[[127.0, 40.0, 165.0, -27.0, 11.0, -27.0, -66.0], [-65.0, 115.0, 170.0, 32.0, -123.0, -1.0, -80.0], [27.0, 16.0, 198.0, 154.0, -182.0, -168.0, -116.0], [138.0, -120.0, 24.0, -25.0, -159.0, 58.0, 129.0], [37.0, 16.0, -15.0, 71.0, 46.0, -56.0, -50.0]], [[-68.0, 124.0, -64.0, -18.0, -46.0, 114.0, 8.0], [32.0, 22.0, -6.0, 69.0, -74.0, -202.0, -37.0], [-17.0, -305.0, 79.0, 40.0, -123.0, 158.0, 77.0], [238.0, 19.0, 144.0, 83.0, 23.0, 26.0, -92.0], [26.0, -87.0, -10.0, -27.0, -161.0, -2.0, 7.0]]]);
    expect(b.shape).toEqual([2, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[35351.0, -3363.0, 6256.0, 10339.0, 23050.0, -19837.0, -14313.0], [44419.0, -19226.0, 24091.0, 4494.0, -33196.0, -1093.0, 12379.0], [-18267.0, -2695.0, -32391.0, -872.0, 19843.0, 3832.0, 5841.0]], [[26607.0, -38463.0, 30749.0, 18781.0, 12101.0, -5850.0, -2940.0], [-63449.0, -23979.0, -24339.0, -6335.0, 3397.0, 3386.0, 28536.0], [-26370.0, 33294.0, -26796.0, -24780.0, -48.0, 23928.0, 5881.0]]], [[[-7632.0, 2340.0, -8921.0, 7846.0, -6912.0, 4417.0, 3187.0], [31922.0, -7305.0, 49907.0, 5053.0, -40348.0, -13004.0, -4710.0], [4343.0, 18670.0, 23140.0, -809.0, 688.0, 1665.0, -14149.0]], [[-29159.0, 9044.0, -13255.0, -1973.0, 16022.0, -1330.0, 8956.0], [24772.0, 3346.0, 13848.0, 9641.0, -6184.0, 7334.0, -9708.0], [7271.0, 67473.0, -3871.0, 7782.0, 58531.0, -41108.0, -19134.0]]]]);
});
test("matmul (2, 2, 3, 5) x (1,)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 12, 12x5,1');
});
test("matmul (2, 2, 3, 5) x (1, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 1x1)');
});
test("matmul (2, 2, 3, 5) x (1, 7)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 1x7)');
});
test("matmul (2, 2, 3, 5) x (2,)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 12, 12x5,2');
});
test("matmul (2, 2, 3, 5) x (2, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 2x1)');
});
test("matmul (2, 2, 3, 5) x (2, 7)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 2x7)');
});
test("matmul (2, 1, 2, 3, 5) x (5,)", async () => {
    const a = tensor([[[[[135.0, -101.0, 71.0, 181.0, 57.0], [-26.0, -173.0, -147.0, 74.0, -24.0], [45.0, 105.0, 24.0, 90.0, -125.0]], [[-108.0, -23.0, -76.0, 91.0, -130.0], [10.0, -92.0, -62.0, 48.0, -2.0], [-135.0, 7.0, -20.0, 183.0, 75.0]]]], [[[[-162.0, -52.0, -73.0, -159.0, 111.0], [-84.0, 58.0, 88.0, 20.0, -233.0], [-42.0, -35.0, 16.0, -107.0, 68.0]], [[2.0, -115.0, 94.0, -0.0, 69.0], [-50.0, -98.0, -19.0, -63.0, 1.0], [-57.0, -166.0, 2.0, 12.0, 177.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([234.0, 150.0, 15.0, 128.0, -0.0]);
    expect(b.shape).toEqual([5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3]);
    expect(await c.toArrayAsync()).toEqual([[[[40673.0, -24767.0, 38160.0], [-18214.0, -6246.0, -7416.0]]], [[[-67155.0, -7076.0, -28534.0], [-15372.0, -34749.0, -36672.0]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (5, 1)", async () => {
    const a = tensor([[[[[107.0, -99.0, 47.0, -178.0, -149.0], [-14.0, -0.0, -315.0, 162.0, 90.0], [-94.0, 68.0, 74.0, -23.0, 13.0]], [[-27.0, -4.0, 119.0, -55.0, -38.0], [-47.0, -115.0, -44.0, -129.0, 115.0], [-5.0, 74.0, -57.0, -122.0, -45.0]]]], [[[[-30.0, -98.0, -63.0, -91.0, -119.0], [-112.0, 101.0, 111.0, 35.0, -75.0], [45.0, -117.0, 228.0, -30.0, 65.0]], [[-44.0, 64.0, 73.0, -83.0, -10.0], [-3.0, -64.0, -173.0, -142.0, 59.0], [152.0, -5.0, 154.0, 77.0, -95.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[-87.0], [79.0], [5.0], [99.0], [-147.0]]);
    expect(b.shape).toEqual([5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[-12614.0], [2451.0], [9732.0]], [[2769.0], [-34892.0], [533.0]]]], [[[[3037.0], [32768.0], [-24543.0]], [[2502.0], [-28391.0], [8739.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[[[64.0, -53.0, -70.0, -91.0, -127.0], [41.0, 87.0, 20.0, -57.0, 99.0], [65.0, 25.0, -162.0, -65.0, -91.0]], [[-4.0, -58.0, -78.0, -2.0, 55.0], [-124.0, 75.0, 130.0, -113.0, -27.0], [97.0, -57.0, -8.0, 64.0, 64.0]]]], [[[[91.0, -95.0, -82.0, 147.0, -110.0], [-10.0, 88.0, 143.0, -141.0, -56.0], [-91.0, 25.0, -106.0, 115.0, 168.0]], [[-17.0, -57.0, -84.0, -56.0, -37.0], [237.0, -104.0, 189.0, 24.0, 105.0], [-59.0, 28.0, 40.0, 113.0, -120.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[18.0, 78.0, -60.0, 158.0, -37.0, -2.0, 31.0], [-209.0, 94.0, 17.0, 0.0, 15.0, -87.0, 166.0], [62.0, 19.0, 94.0, -60.0, -63.0, -86.0, 115.0], [45.0, 32.0, -100.0, 116.0, -183.0, 8.0, 23.0], [-112.0, -10.0, 58.0, -10.0, 94.0, 311.0, -100.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[[18018.0, -2962.0, -9587.0, 5026.0, 5962.0, -29722.0, -4257.0], [-29858.0, 8942.0, 12341.0, -2324.0, 18265.0, 20962.0, 6802.0], [-6832.0, 3172.0, -17481.0, 13360.0, 11517.0, -17194.0, -4860.0]], [[964.0, -7860.0, -4688.0, 3266.0, 9728.0, 28851.0, -24268.0], [-11908.0, -3498.0, 30669.0, -40230.0, 15664.0, -26758.0, 23657.0], [8875.0, 3464.0, -10229.0, 22590.0, -9636.0, 25869.0, -12303.0]]]], [[[[35344.0, 2414.0, -35863.0, 37450.0, -36867.0, -17899.0, -7998.0], [-9779.0, 6257.0, 26390.0, -25956.0, 13220.0, -38478.0, 33100.0], [-27076.0, -4762.0, -5835.0, 3642.0, 5167.0, 60291.0, -25016.0]], [[8023.0, -9702.0, -4391.0, -3772.0, 11836.0, 262.0, -17237.0], [27040.0, 12019.0, 5468.0, 27840.0, -16758.0, 25167.0, 1870.0], [14091.0, 3606.0, -10484.0, 2586.0, -31876.0, -42174.0, 22018.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (1, 5)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 1x5)');
});
test("matmul (2, 1, 2, 3, 5) x (1, 5, 1)", async () => {
    const a = tensor([[[[[-137.0, 163.0, 52.0, 56.0, -144.0], [149.0, 173.0, -57.0, -59.0, 46.0], [-163.0, 2.0, -147.0, -40.0, -119.0]], [[-100.0, 58.0, 52.0, -151.0, 0.0], [-21.0, -73.0, -59.0, -6.0, -57.0], [-22.0, -75.0, 184.0, -3.0, 72.0]]]], [[[[-65.0, -53.0, 203.0, 18.0, 92.0], [-57.0, 120.0, -67.0, 12.0, -56.0], [-105.0, 5.0, 149.0, 130.0, -174.0]], [[-4.0, -121.0, -25.0, -220.0, -7.0], [-160.0, -52.0, -27.0, -4.0, -25.0], [38.0, 98.0, -1.0, -155.0, 111.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[[145.0], [-16.0], [88.0], [88.0], [37.0]]]);
    expect(b.shape).toEqual([1, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[-18297.0], [10331.0], [-44526.0]], [[-24140.0], [-9706.0], [16602.0]]]], [[[[14275.0], [-17097.0], [2809.0]], [[-20463.0], [-26021.0], [-5679.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[[[54.0, -18.0, 55.0, 55.0, 38.0], [39.0, -12.0, 3.0, -34.0, 4.0], [-87.0, 9.0, -215.0, -24.0, 144.0]], [[54.0, 61.0, 171.0, -168.0, -45.0], [117.0, 31.0, 95.0, 39.0, 33.0], [-133.0, -211.0, 83.0, 80.0, -38.0]]]], [[[[23.0, 17.0, 35.0, 68.0, 80.0], [78.0, 54.0, 146.0, 37.0, -100.0], [126.0, 239.0, -210.0, -60.0, 47.0]], [[-19.0, 124.0, -46.0, -51.0, 102.0], [96.0, -94.0, 13.0, 8.0, 8.0], [-139.0, 102.0, 146.0, 74.0, 207.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[[20.0, -77.0, 29.0, -58.0, -54.0, -3.0, 62.0], [-73.0, 52.0, -55.0, -182.0, 51.0, 33.0, -48.0], [-84.0, -21.0, 156.0, 100.0, -68.0, 117.0, 56.0], [-146.0, 50.0, -28.0, -136.0, -146.0, 17.0, 26.0], [89.0, -109.0, -107.0, -25.0, 6.0, -18.0, -221.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[[-6874.0, -7641.0, 5530.0, -2786.0, -15376.0, 5930.0, 324.0], [6724.0, -5826.0, 2783.0, 4746.0, 2066.0, -812.0, 1394.0], [31983.0, -5214.0, -51294.0, -18428.0, 24145.0, -27597.0, -50314.0]], [[2786.0, -8072.0, 34406.0, 26839.0, 12825.0, 19812.0, 15573.0], [-10660.0, -11039.0, 11885.0, -9057.0, -16693.0, 11856.0, 4807.0], [-9291.0, 5668.0, 22522.0, 44486.0, -21131.0, 5191.0, 17008.0]]]], [[[[-6529.0, -6942.0, -5272.0, -12176.0, -12203.0, 4303.0, -13342.0], [-28948.0, 6486.0, 31732.0, -2284.0, -17388.0, 21059.0, 33482.0], [15656.0, -987.0, -45600.0, -64821.0, 28707.0, -18927.0, -27367.0]], [[10956.0, -4791.0, -24033.0, -21680.0, 18536.0, -3936.0, -33574.0], [7234.0, -13025.0, 8902.0, 11552.0, -11982.0, -1877.0, 9632.0], [-14871.0, -5922.0, -11086.0, -11141.0, -6782.0, 18397.0, -49161.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (2, 5)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([2, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 2x5)');
});
test("matmul (2, 1, 2, 3, 5) x (2, 5, 1)", async () => {
    const a = tensor([[[[[-137.0, -83.0, -16.0, 0.0, -7.0], [165.0, -26.0, -108.0, -29.0, -123.0], [15.0, -50.0, -2.0, 37.0, 43.0]], [[54.0, 6.0, 18.0, -74.0, -159.0], [143.0, 62.0, -77.0, -118.0, 92.0], [-187.0, 23.0, 23.0, 28.0, 53.0]]]], [[[[60.0, -26.0, 35.0, -80.0, 41.0], [58.0, -82.0, 41.0, -44.0, 90.0], [84.0, 10.0, 13.0, -69.0, 176.0]], [[142.0, 244.0, 104.0, 96.0, 81.0], [78.0, 249.0, -18.0, -65.0, 31.0], [-1.0, 64.0, 139.0, -3.0, 53.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[[21.0], [-190.0], [-62.0], [46.0], [16.0]], [[-104.0], [-92.0], [36.0], [-72.0], [-152.0]]]);
    expect(b.shape).toEqual([2, 5, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[13773.0], [11799.0], [12329.0]], [[23976.0], [-28836.0], [8088.0]]]], [[[[1006.0], [13672.0], [-1300.0]], [[-52696.0], [-31700.0], [-8620.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (2, 5, 7)", async () => {
    const a = tensor([[[[[-46.0, 23.0, 158.0, -84.0, -88.0], [142.0, -5.0, 77.0, -61.0, 124.0], [108.0, -93.0, 51.0, -20.0, -29.0]], [[57.0, -63.0, -120.0, -181.0, 127.0], [-181.0, 7.0, -95.0, -91.0, -24.0], [-48.0, 13.0, -68.0, 13.0, 112.0]]]], [[[[141.0, -186.0, 50.0, 33.0, 10.0], [-32.0, -31.0, 6.0, -105.0, 42.0], [65.0, 49.0, -38.0, 54.0, -44.0]], [[16.0, 148.0, -93.0, -56.0, 20.0], [3.0, -56.0, 96.0, 40.0, -67.0], [18.0, -195.0, -35.0, 34.0, -83.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[[57.0, -41.0, 47.0, -34.0, 67.0, -35.0, 57.0], [69.0, 144.0, -147.0, 89.0, 54.0, 53.0, 64.0], [117.0, 3.0, 164.0, -180.0, -45.0, -182.0, -97.0], [20.0, 151.0, 127.0, -8.0, -92.0, -76.0, 47.0], [45.0, -138.0, 84.0, 101.0, 1.0, -194.0, -51.0]], [[52.0, 32.0, -137.0, 200.0, -44.0, -28.0, -38.0], [-74.0, -143.0, 22.0, 133.0, 269.0, 57.0, -45.0], [103.0, 150.0, -89.0, 97.0, 124.0, 131.0, 89.0], [15.0, -39.0, 66.0, 59.0, 172.0, 150.0, -44.0], [-51.0, -218.0, 163.0, 49.0, -120.0, -25.0, 105.0]]]);
    expect(b.shape).toEqual([2, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[[11811.0, 5132.0, 2309.0, -33045.0, -1310.0, -2471.0, -15936.0], [21118.0, -32634.0, 22706.0, -6121.0, 11515.0, -38669.0, -8886.0], [4001.0, -16685.0, 22135.0, -23898.0, 1730.0, -10845.0, -4204.0]], [[-13926.0, -27794.0, 10240.0, -13075.0, -80707.0, -51232.0, 11288.0], [-19856.0, -12262.0, 23488.0, -51029.0, -14705.0, -20028.0, -408.0], [-15979.0, -38518.0, 32028.0, -8212.0, -14027.0, -7673.0, 6375.0]]]], [[[[2163.0, -28812.0, 47200.0, -29602.0, -5873.0, -28341.0, -7676.0], [-3471.0, -24785.0, -5770.0, 2331.0, 5614.0, -1783.0, -11467.0], [1740.0, 18503.0, -7218.0, 4115.0, 3699.0, 11670.0, 15309.0]], [[-21559.0, -36778.0, 8905.0, 11539.0, 15544.0, -13095.0, -10981.0], [18205.0, 35550.0, -18468.0, 1541.0, 11628.0, 16975.0, 2155.0], [16504.0, 39979.0, -14926.0, -27791.0, -41779.0, -9029.0, -5235.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (1,)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 12, 12x5,1');
});
test("matmul (2, 1, 2, 3, 5) x (1, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 1x1)');
});
test("matmul (2, 1, 2, 3, 5) x (1, 7)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 1x7)');
});
test("matmul (2, 1, 2, 3, 5) x (2,)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 12, 12x5,2');
});
test("matmul (2, 1, 2, 3, 5) x (2, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 2x1)');
});
test("matmul (2, 1, 2, 3, 5) x (2, 7)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 2x7)');
});
test("matmul (2, 3, 5, 7) x (7,)", async () => {
    const a = tensor([[[[75.0, -71.0, -62.0, -53.0, -31.0, 7.0, -92.0], [-3.0, -1.0, 22.0, -84.0, 87.0, -31.0, -33.0], [-26.0, -29.0, -118.0, 86.0, 55.0, -42.0, -6.0], [-90.0, 45.0, 173.0, 90.0, -1.0, -191.0, -164.0], [-146.0, 163.0, 4.0, -64.0, 23.0, -124.0, 138.0]], [[48.0, -135.0, 120.0, 32.0, -127.0, -99.0, -21.0], [-110.0, -14.0, 190.0, -56.0, 9.0, 46.0, 110.0], [89.0, 163.0, 5.0, 61.0, 43.0, 103.0, -70.0], [91.0, 23.0, 5.0, -10.0, -109.0, -28.0, 146.0], [-91.0, 102.0, -136.0, 111.0, -13.0, 152.0, -121.0]], [[-49.0, -129.0, 122.0, 38.0, -23.0, -105.0, -98.0], [7.0, -78.0, -40.0, -180.0, -44.0, 174.0, 9.0], [-86.0, -64.0, 88.0, 77.0, 73.0, -117.0, -59.0], [-70.0, -20.0, 38.0, 200.0, 10.0, 4.0, 67.0], [117.0, 5.0, -12.0, -134.0, -60.0, 193.0, -93.0]]], [[[-116.0, -5.0, 87.0, -27.0, -22.0, -2.0, 175.0], [106.0, -47.0, 87.0, -257.0, 59.0, 151.0, -8.0], [-49.0, 9.0, 124.0, 103.0, -55.0, -22.0, 244.0], [31.0, 102.0, -143.0, -63.0, 102.0, -61.0, -45.0], [-17.0, -87.0, -78.0, 125.0, -104.0, 11.0, -76.0]], [[203.0, -58.0, 162.0, 47.0, 10.0, -69.0, 91.0], [-320.0, 15.0, -92.0, -63.0, -109.0, 31.0, 79.0], [160.0, 71.0, 60.0, -27.0, 66.0, 281.0, 140.0], [138.0, 84.0, 49.0, 74.0, -64.0, 86.0, 80.0], [-69.0, -227.0, 140.0, -46.0, -50.0, 25.0, 11.0]], [[131.0, -99.0, -25.0, -40.0, 129.0, -62.0, 100.0], [-38.0, 37.0, 73.0, 214.0, -110.0, 37.0, -56.0], [27.0, 44.0, -204.0, -41.0, 5.0, 126.0, 18.0], [107.0, -83.0, -87.0, -58.0, 140.0, 112.0, -70.0], [-28.0, -42.0, 241.0, 52.0, -88.0, 63.0, 1.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([117.0, -31.0, -120.0, 44.0, 14.0, -12.0, -147.0]);
    expect(b.shape).toEqual([7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[29090.0, -215.0, 17957.0, -2339.0, -43907.0], [-694.0, -54296.0, 17100.0, -13758.0, 23176.0], [642.0, -3910.0, -4151.0, -13087.0, 19593.0]], [[-51054.0, -7699.0, -52734.0, 23628.0, 25152.0], [-4232.0, -43148.0, -14897.0, -2770.0, -22477.0], [7486.0, 1311.0, 20383.0, 33886.0, -30741.0]]]);
});
test("matmul (2, 3, 5, 7) x (7, 1)", async () => {
    const a = tensor([[[[-12.0, -230.0, 137.0, -27.0, 33.0, -143.0, -40.0], [126.0, -28.0, 90.0, -36.0, 11.0, -12.0, 38.0], [-42.0, 140.0, 4.0, 44.0, -72.0, -20.0, -192.0], [119.0, -86.0, -212.0, 39.0, 197.0, -9.0, -101.0], [-9.0, -44.0, -140.0, 31.0, -21.0, -61.0, 3.0]], [[-16.0, -103.0, 25.0, 56.0, 18.0, -57.0, -103.0], [-0.0, 54.0, -11.0, -70.0, -4.0, -44.0, -79.0], [212.0, -37.0, -65.0, -42.0, 53.0, -94.0, -65.0], [96.0, -86.0, 72.0, -31.0, -23.0, -126.0, 124.0], [26.0, -105.0, 48.0, -149.0, -33.0, 14.0, 208.0]], [[-95.0, -17.0, -78.0, 164.0, 64.0, 76.0, -75.0], [-121.0, 94.0, 53.0, 46.0, -34.0, 41.0, 130.0], [56.0, -97.0, -21.0, 64.0, -72.0, -5.0, -65.0], [14.0, -56.0, -2.0, 68.0, -120.0, 31.0, 72.0], [168.0, 57.0, -44.0, 96.0, -199.0, -74.0, 62.0]]], [[[-3.0, -13.0, 21.0, 78.0, -63.0, -168.0, -142.0], [51.0, 34.0, -81.0, 141.0, -80.0, -20.0, -86.0], [-128.0, 73.0, 255.0, -55.0, 5.0, -29.0, -84.0], [-146.0, 25.0, -173.0, 178.0, -90.0, 91.0, -102.0], [61.0, -68.0, 24.0, -60.0, 134.0, -151.0, 35.0]], [[12.0, 42.0, 71.0, -13.0, -44.0, 37.0, -76.0], [-19.0, -88.0, -162.0, -267.0, 46.0, -120.0, 150.0], [55.0, -182.0, -167.0, 4.0, 45.0, 95.0, 55.0], [47.0, -110.0, -17.0, 79.0, -17.0, -18.0, -136.0], [26.0, 88.0, 4.0, -158.0, 21.0, 144.0, 160.0]], [[-136.0, -22.0, 137.0, -235.0, 120.0, 106.0, -164.0], [54.0, -132.0, 142.0, 35.0, -49.0, 119.0, -18.0], [54.0, -19.0, 158.0, -71.0, 146.0, 43.0, -96.0], [25.0, 119.0, -6.0, -29.0, 127.0, 111.0, 148.0], [-17.0, 198.0, -3.0, 252.0, -50.0, 15.0, 14.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[-140.0], [-3.0], [-94.0], [-111.0], [96.0], [-171.0], [15.0]]);
    expect(b.shape).toEqual([7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[19510.0], [-18342.0], [-6172.0], [18133.0], [19571.0]], [[3913.0], [14597.0], [1390.0], [4689.0], [6260.0]], [[-5498.0], [-1755.0], [-19711.0], [-24893.0], [-35731.0]]], [[[10377.0], [-20829.0], [4015.0], [-8862.0], [35278.0]], [[-18728.0], [74975.0], [-3000.0], [-14015.0], [-6950.0]], [[23247.0], [-49720.0], [-9251.0], [-4643.0], [-33059.0]]]]);
});
test("matmul (2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[78.0, 29.0, -145.0, 150.0, 177.0, -26.0, -27.0], [55.0, 104.0, -34.0, 65.0, 123.0, 195.0, -21.0], [-37.0, 23.0, 127.0, 144.0, -24.0, 75.0, -215.0], [-52.0, 44.0, 246.0, -37.0, 26.0, 87.0, 60.0], [135.0, 228.0, -117.0, 146.0, 35.0, -38.0, -153.0]], [[53.0, -37.0, 46.0, 28.0, -56.0, 99.0, 41.0], [-2.0, 40.0, -39.0, -53.0, 77.0, 49.0, -117.0], [-112.0, -136.0, 43.0, -80.0, -100.0, -186.0, -83.0], [-124.0, 15.0, 72.0, -46.0, 123.0, 41.0, 127.0], [-78.0, -145.0, -76.0, 111.0, -87.0, 66.0, -157.0]], [[52.0, -56.0, -43.0, 42.0, 71.0, 129.0, 136.0], [-5.0, 43.0, -90.0, -8.0, 65.0, -127.0, 231.0], [-160.0, 26.0, 99.0, 81.0, 110.0, -155.0, 114.0], [-233.0, 30.0, -81.0, -93.0, -96.0, -125.0, 91.0], [-15.0, -146.0, 130.0, 0.0, 68.0, 88.0, 172.0]]], [[[-97.0, 33.0, -14.0, 102.0, -4.0, -22.0, -76.0], [-88.0, 131.0, 56.0, -42.0, -28.0, -6.0, -102.0], [24.0, 111.0, 85.0, 13.0, -23.0, -80.0, 125.0], [-95.0, -15.0, 87.0, -15.0, 173.0, -9.0, -2.0], [196.0, 81.0, 72.0, 46.0, 57.0, 63.0, -16.0]], [[185.0, 67.0, 1.0, -178.0, -109.0, -12.0, 59.0], [45.0, -92.0, 33.0, -28.0, -67.0, 109.0, 60.0], [-148.0, -117.0, -28.0, 138.0, -79.0, 98.0, -11.0], [-112.0, 203.0, -2.0, -53.0, -228.0, -111.0, 152.0], [-76.0, -17.0, 16.0, 25.0, -187.0, -74.0, -54.0]], [[-160.0, 29.0, 230.0, 38.0, -37.0, -94.0, -69.0], [-1.0, 194.0, 120.0, 133.0, 25.0, 158.0, -58.0], [-51.0, -64.0, 13.0, 104.0, 41.0, -47.0, 201.0], [104.0, -123.0, 116.0, -69.0, -160.0, 32.0, -25.0], [52.0, -40.0, 99.0, 82.0, -55.0, 82.0, -59.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[-297.0, 111.0, 60.0, 16.0, 169.0, 22.0, 113.0, 122.0, 129.0, -150.0, -114.0], [-129.0, -172.0, 61.0, -116.0, -85.0, 114.0, 31.0, -109.0, 34.0, 35.0, 101.0], [-6.0, 58.0, 75.0, 73.0, -14.0, 146.0, 13.0, 193.0, 21.0, 28.0, -24.0], [55.0, 15.0, 285.0, 69.0, 76.0, 84.0, -113.0, 37.0, -67.0, 100.0, -81.0], [38.0, 41.0, 151.0, 63.0, -140.0, 129.0, -83.0, -126.0, 158.0, 186.0, 58.0], [-17.0, 1.0, 73.0, 3.0, 11.0, -41.0, 51.0, -68.0, 163.0, 31.0, -109.0], [-31.0, -12.0, -16.0, 139.0, -10.0, -185.0, 8.0, -73.0, 16.0, 105.0, -112.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[-9782.0, 5065.0, 63585.0, 4969.0, -649.0, 25346.0, -25355.0, -34643.0, 21249.0, 29536.0, 1491.0], [-23962.0, -7290.0, 58763.0, -3766.0, -8994.0, 25319.0, 1220.0, -36008.0, 56445.0, 27656.0, -11984.0], [19658.0, 3134.0, 55039.0, -15225.0, 7293.0, 66050.0, -13992.0, 36437.0, -5979.0, -403.0, 6342.0], [3906.0, 806.0, 16786.0, 19708.0, -22067.0, 25367.0, 5626.0, 21397.0, 21682.0, 26361.0, -7230.0], [-54056.0, -25594.0, 59802.0, -41931.0, 12381.0, 58522.0, -1763.0, -16218.0, 9816.0, -11679.0, 21928.0]], [[-14786.0, 12646.0, 10468.0, 12898.0, 22105.0, -12852.0, 12301.0, 17744.0, 12614.0, -8199.0, -31782.0], [-1527.0, -5549.0, 1366.0, -22441.0, -16291.0, 23939.0, 1668.0, -18585.0, 22115.0, -1136.0, 21726.0], [48085.0, 8964.0, -61941.0, -6792.0, -1266.0, -8329.0, -9123.0, 37806.0, -60255.0, -27837.0, 28250.0], [31971.0, -9298.0, 5299.0, 23883.0, -44774.0, -3679.0, -14515.0, -32126.0, 17257.0, 54025.0, 6090.0], [48871.0, 11922.0, 6603.0, -9423.0, 23119.0, -4902.0, -17509.0, 13663.0, -29525.0, -15024.0, -7576.0]], [[-9363.0, 14948.0, 26411.0, 30851.0, 7461.0, -29280.0, 609.0, -21943.0, 35508.0, 24721.0, -39129.0], [-6494.0, -13525.0, -9859.0, 23633.0, -16655.0, -38163.0, -9522.0, -39380.0, -7272.0, 31343.0, -538.0], [51308.0, -12288.0, 25967.0, 29551.0, -42725.0, 20157.0, -41263.0, -11892.0, -29165.0, 63407.0, 22436.0], [56358.0, -42269.0, -69807.0, -13312.0, -36706.0, -45438.0, -13622.0, -36817.0, -58594.0, 12256.0, 36934.0], [18265.0, 31799.0, 13884.0, 54642.0, -2217.0, -24650.0, -4311.0, 12066.0, 23671.0, 34216.0, -41068.0]]], [[[32824.0, -14999.0, 23219.0, -10246.0, -10172.0, 22598.0, -23044.0, -6811.0, -23953.0, 16107.0, 17143.0], [8791.0, -29612.0, -8093.0, -31374.0, -25109.0, 33150.0, 793.0, -4379.0, -9942.0, -951.0, 35775.0], [-24631.0, -13826.0, 6978.0, 10296.0, -4491.0, 3872.0, 4618.0, 6928.0, -6890.0, 10332.0, -1232.0], [35592.0, 3964.0, 21133.0, 16130.0, -41437.0, 30698.0, -23208.0, -14759.0, 15902.0, 46350.0, 19681.0], [-64972.0, 15282.0, 48673.0, 3726.0, 21600.0, 35652.0, 18751.0, 20383.0, 45487.0, -9074.0, -21386.0]], [[-79151.0, 1210.0, -53747.0, -15723.0, 26566.0, -27582.0, 52016.0, 19117.0, 19856.0, -57628.0, -11551.0], [-9494.0, 18955.0, -11537.0, 16315.0, 22814.0, -31244.0, 17426.0, 17501.0, 13387.0, -14629.0, -35433.0], [62480.0, 1133.0, 16614.0, 12470.0, 8061.0, -21264.0, -24842.0, -1508.0, -29588.0, 18310.0, -19483.0], [-7315.0, -59542.0, -54555.0, -22712.0, -11004.0, -37047.0, 14079.0, -12958.0, -55722.0, -11340.0, 19463.0], [21870.0, -11302.0, -30047.0, -15860.0, 16183.0, -10273.0, -417.0, 29130.0, -54193.0, -28993.0, 7806.0]], [[46820.0, -9621.0, 8904.0, 1284.0, -25001.0, 48404.0, -20760.0, 39206.0, -39642.0, 18214.0, 28399.0], [-18072.0, -22645.0, 74916.0, -10596.0, -9413.0, 58263.0, -2049.0, -2847.0, 28852.0, 27058.0, -3221.0], [25171.0, 6883.0, 23195.0, 45114.0, -3724.0, -27753.0, -23522.0, -9532.0, -13417.0, 43448.0, -24397.0], [-25361.0, 32165.0, -33652.0, 6180.0, 44165.0, -17921.0, 31956.0, 65739.0, -4171.0, -54950.0, -31442.0], [-8023.0, 18159.0, 30100.0, 6937.0, 26226.0, 18384.0, 4932.0, 38506.0, 5665.0, -12111.0, -24506.0]]]]);
});
test("matmul (2, 3, 5, 7) x (1, 7)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 1x7)');
});
test("matmul (2, 3, 5, 7) x (1, 7, 1)", async () => {
    const a = tensor([[[[-36.0, 127.0, 84.0, 225.0, 166.0, -207.0, -197.0], [64.0, -97.0, -86.0, -41.0, -69.0, 97.0, -118.0], [3.0, -9.0, -274.0, 61.0, 199.0, -30.0, 40.0], [-40.0, 40.0, -134.0, 87.0, 71.0, -70.0, -45.0], [-80.0, -27.0, 69.0, -25.0, 20.0, -2.0, 250.0]], [[139.0, 34.0, -2.0, -19.0, -28.0, -1.0, 172.0], [19.0, 72.0, 77.0, 92.0, -42.0, 76.0, -122.0], [-221.0, 92.0, -50.0, -192.0, 18.0, 15.0, 260.0], [-147.0, -30.0, 89.0, -1.0, -120.0, 38.0, -155.0], [-90.0, 57.0, -58.0, 128.0, 26.0, -139.0, 22.0]], [[21.0, 48.0, -280.0, 58.0, -52.0, 72.0, 54.0], [7.0, 5.0, -87.0, -3.0, -151.0, 99.0, -2.0], [48.0, -125.0, -57.0, -51.0, -19.0, -128.0, -86.0], [16.0, 29.0, 21.0, -84.0, 68.0, 123.0, -113.0], [-25.0, -117.0, -36.0, -282.0, 63.0, -74.0, 9.0]]], [[[-5.0, 129.0, 131.0, -18.0, -119.0, 47.0, 72.0], [80.0, -71.0, 32.0, 31.0, 106.0, -45.0, -157.0], [53.0, 28.0, 202.0, -85.0, 45.0, 173.0, -64.0], [-3.0, -4.0, 67.0, -120.0, -83.0, -156.0, -30.0], [-264.0, -5.0, -12.0, -189.0, -74.0, -54.0, -38.0]], [[-136.0, 2.0, 28.0, 105.0, 70.0, 118.0, -293.0], [31.0, 275.0, 41.0, -162.0, 40.0, 55.0, 52.0], [205.0, -15.0, 19.0, 42.0, -125.0, -8.0, -15.0], [183.0, 10.0, 61.0, 4.0, -158.0, 107.0, 83.0], [-18.0, -61.0, -134.0, 195.0, 121.0, -116.0, -78.0]], [[97.0, -193.0, 91.0, 134.0, -16.0, 65.0, -105.0], [22.0, -55.0, 100.0, -80.0, -18.0, 181.0, -27.0], [34.0, 98.0, -15.0, -275.0, 76.0, 14.0, -133.0], [10.0, -191.0, -133.0, -221.0, -58.0, -6.0, 14.0], [28.0, 82.0, -48.0, -64.0, -108.0, -133.0, 6.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[[-78.0], [1.0], [-102.0], [-25.0], [-167.0], [122.0], [-34.0]]]);
    expect(b.shape).toEqual([1, 7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[-57536.0], [32077.0], [-12073.0], [-4214.0], [-12284.0]], [[-11423.0], [8870.0], [17214.0], [32329.0], [-12255.0]], [[41152.0], [45771.0], [-6299.0], [6231.0], [-7300.0]]], [[[10766.0], [-28204.0], [-6818.0], [-7755.0], [33598.0]], [[17797.0], [-4013.0], [1416.0], [16032.0], [-21571.0]], [[-6219.0], [16035.0], [-611.0], [26598.0], [6000.0]]]]);
});
test("matmul (2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[-47.0, 104.0, -74.0, 35.0, -101.0, -214.0, -31.0], [229.0, 94.0, -75.0, -8.0, 28.0, 10.0, 190.0], [-58.0, 58.0, 98.0, 156.0, 109.0, 123.0, 18.0], [58.0, -79.0, 127.0, -17.0, 154.0, 113.0, 101.0], [-55.0, 66.0, -189.0, 60.0, 151.0, -246.0, 78.0]], [[-157.0, 78.0, 109.0, 44.0, -119.0, -25.0, 131.0], [169.0, 122.0, 175.0, -205.0, -51.0, 14.0, -112.0], [36.0, 19.0, -115.0, 146.0, 28.0, -60.0, 46.0], [64.0, 43.0, 145.0, -212.0, 80.0, 23.0, 111.0], [113.0, -7.0, 83.0, 81.0, 125.0, 115.0, -132.0]], [[68.0, 17.0, -82.0, -10.0, -14.0, 28.0, -70.0], [-173.0, -33.0, 45.0, -58.0, 88.0, -67.0, 128.0], [-39.0, 65.0, 19.0, 92.0, -71.0, 101.0, -0.0], [-14.0, -31.0, -30.0, -55.0, 1.0, 120.0, -20.0], [46.0, 80.0, -76.0, 208.0, 6.0, -57.0, -162.0]]], [[[113.0, -30.0, 206.0, -94.0, -131.0, -191.0, 119.0], [-59.0, 88.0, -17.0, -40.0, -160.0, 53.0, -131.0], [132.0, 2.0, -84.0, -20.0, -136.0, 106.0, -75.0], [-33.0, -26.0, 54.0, 138.0, 1.0, 183.0, -101.0], [136.0, -90.0, 204.0, -33.0, 47.0, -29.0, -183.0]], [[47.0, 14.0, 168.0, 53.0, 151.0, 98.0, -83.0], [-89.0, -55.0, -117.0, 68.0, -113.0, 111.0, -67.0], [136.0, -28.0, -108.0, -82.0, 49.0, -5.0, -80.0], [106.0, -70.0, -67.0, 9.0, 102.0, 16.0, 54.0], [105.0, -165.0, -77.0, -34.0, 133.0, 44.0, -39.0]], [[184.0, -92.0, 111.0, -53.0, 81.0, 30.0, -112.0], [58.0, -63.0, -98.0, -37.0, 38.0, -115.0, 281.0], [5.0, 32.0, 58.0, -46.0, -119.0, -43.0, 3.0], [165.0, -3.0, -23.0, 84.0, 18.0, 44.0, 116.0], [96.0, 98.0, -104.0, -25.0, -134.0, 115.0, -150.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[[177.0, 51.0, 3.0, 37.0, 79.0, -67.0, -105.0, -53.0, -31.0, 109.0, -122.0], [-156.0, -78.0, -408.0, -66.0, -114.0, -80.0, -125.0, -97.0, 7.0, -73.0, 220.0], [161.0, -151.0, 8.0, 22.0, 93.0, 98.0, 226.0, -49.0, 63.0, -51.0, 222.0], [-202.0, -72.0, -198.0, 32.0, 16.0, -84.0, 130.0, 17.0, -138.0, 235.0, 19.0], [-96.0, -149.0, -71.0, 1.0, -34.0, -210.0, 48.0, 44.0, -108.0, 61.0, -65.0], [-57.0, 88.0, 9.0, -12.0, 20.0, -49.0, -14.0, 65.0, 67.0, -21.0, 77.0], [42.0, 79.0, -60.0, 29.0, 67.0, -204.0, -11.0, 149.0, 18.0, -5.0, 122.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[-22935.0, -8087.0, -42990.0, -7543.0, -24814.0, 22657.0, -21750.0, -26349.0, -11295.0, -2228.0, -844.0], [20132.0, 27966.0, -49979.0, 5781.0, 12250.0, -74671.0, -54671.0, 12476.0, -8996.0, 20592.0, -1930.0], [-51767.0, -37507.0, -61654.0, 329.0, 376.0, -36843.0, 44580.0, 10771.0, -16357.0, 25082.0, 49138.0], [29488.0, -13856.0, 20811.0, 11337.0, 28918.0, -42173.0, 34976.0, 27247.0, 753.0, 8133.0, 14428.0], [-59778.0, -21719.0, -58100.0, -3264.0, -33314.0, -60725.0, -27555.0, 9070.0, -49406.0, 26913.0, -38829.0]], [[-12945.0, -7838.0, -39771.0, -3171.0, 1869.0, 10756.0, 30286.0, 8820.0, 19743.0, -25415.0, 83140.0], [79860.0, -12579.0, 3188.0, -7976.0, 6948.0, 46159.0, -21507.0, -50873.0, 39360.0, -50430.0, 31906.0], [-41935.0, 1389.0, -42760.0, 4302.0, -6751.0, -39790.0, -11487.0, 8552.0, -34592.0, 45450.0, -23796.0], [66460.0, -7848.0, 13651.0, -1041.0, 15424.0, -16281.0, -4588.0, 3282.0, 31607.0, -49536.0, 39927.0], [-6005.0, -30989.0, -12099.0, 3978.0, 7946.0, -10638.0, 24140.0, -14693.0, -17672.0, 33500.0, -10735.0]], [[-4990.0, 14264.0, 38.0, -3110.0, -8006.0, 2736.0, -29391.0, -10631.0, -3647.0, 6911.0, -28424.0], [-5765.0, -17764.0, 10258.0, -485.0, -2404.0, -17796.0, 28674.0, 27768.0, 4282.0, -26238.0, 27471.0], [-31509.0, 2915.0, -38751.0, -3654.0, -2818.0, 1508.0, 7402.0, -164.0, 4600.0, 5203.0, 37416.0], [862.0, 19025.0, 25465.0, -2911.0, -216.0, 3088.0, -9997.0, 9148.0, 13489.0, -13017.0, -6082.0], [-62721.0, -26102.0, -65513.0, -2602.0, -21424.0, 179.0, -2098.0, -30517.0, -41741.0, 54303.0, -25475.0]]], [[[105296.0, -4123.0, 33281.0, 13297.0, 38608.0, 35506.0, 21298.0, -15219.0, 25730.0, -22664.0, 31886.0], [-11991.0, 13729.0, -8600.0, -14240.0, -19191.0, 56334.0, -20828.0, -28370.0, 25367.0, -31606.0, 20523.0], [17432.0, 44367.0, 17978.0, -1319.0, 3787.0, 23110.0, -42881.0, -13683.0, 13830.0, 3679.0, -26840.0], [-35736.0, -9769.0, -8747.0, 975.0, 4446.0, 9418.0, 35456.0, 861.0, -4466.0, 24700.0, 14620.0], [67077.0, -38484.0, 52676.0, 9492.0, 25009.0, 49735.0, 43459.0, -36119.0, 2247.0, 7626.0, -19345.0]], [[-1091.0, -48311.0, -19580.0, 2775.0, 9854.0, -11837.0, 44962.0, -10533.0, -9325.0, 15556.0, 23254.0], [-38039.0, 33834.0, 20815.0, -3449.0, -8981.0, 25144.0, -7623.0, 9201.0, 4054.0, 7372.0, -18206.0], [19837.0, 17271.0, 28480.0, -331.0, -4546.0, -4293.0, -42546.0, -10683.0, -6967.0, 6600.0, -61616.0], [8641.0, 10811.0, 16222.0, 8832.0, 10737.0, -42044.0, -12274.0, 18182.0, -18211.0, 27812.0, -41845.0], [21882.0, 13274.0, 67044.0, 10467.0, 13145.0, -20655.0, -6025.0, 16536.0, -16687.0, 26811.0, -76865.0]], [[61307.0, -14662.0, 50709.0, 10099.0, 24841.0, 14730.0, 15076.0, -18342.0, -795.0, 13527.0, -35672.0], [26499.0, 31751.0, 11827.0, 12531.0, 17293.0, -65011.0, -24830.0, 43276.0, -10058.0, 10552.0, -20438.0], [28524.0, 6497.0, 4413.0, -1639.0, 4792.0, 33138.0, -2540.0, -14577.0, 20096.0, -21930.0, 23222.0], [9638.0, 16428.0, -22939.0, 11339.0, 20622.0, -49725.0, -12256.0, 15037.0, -15085.0, 38711.0, -7930.0], [-9981.0, 32992.0, -16029.0, -11868.0, -16854.0, 30741.0, -55476.0, -30694.0, 14085.0, -7100.0, -14450.0]]]]);
});
test("matmul (2, 3, 5, 7) x (2, 7)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 2x7)');
});
test("matmul (2, 3, 5, 7) x (2, 7, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([2, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1');
});
test("matmul (2, 3, 5, 7) x (2, 7, 11)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([2, 7, 11]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1');
});
test("matmul (2, 3, 5, 7) x (1,)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 30, 30x7,1');
});
test("matmul (2, 3, 5, 7) x (1, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 1x1)');
});
test("matmul (2, 3, 5, 7) x (1, 11)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([1, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 1x11)');
});
test("matmul (2, 3, 5, 7) x (2,)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 30, 30x7,2');
});
test("matmul (2, 3, 5, 7) x (2, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 2x1)');
});
test("matmul (2, 3, 5, 7) x (2, 11)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([2, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 2x11)');
});
test("matmul (1, 2, 3, 5, 7) x (7,)", async () => {
    const a = tensor([[[[[23.0, -123.0, 77.0, -42.0, 80.0, 79.0, -120.0], [-105.0, 6.0, -80.0, 25.0, -2.0, -132.0, 148.0], [-129.0, -311.0, -19.0, 131.0, 32.0, -59.0, -28.0], [-35.0, 51.0, -129.0, 94.0, 111.0, -112.0, 38.0], [-90.0, 87.0, 57.0, 18.0, 25.0, -159.0, -52.0]], [[-49.0, 57.0, 14.0, 101.0, 48.0, 48.0, 29.0], [-36.0, 147.0, -111.0, 189.0, -9.0, -6.0, 50.0], [53.0, -84.0, -40.0, -18.0, 23.0, -57.0, -27.0], [82.0, -112.0, 111.0, 35.0, 30.0, -18.0, 107.0], [-74.0, -201.0, -98.0, 75.0, 108.0, 3.0, -119.0]], [[123.0, -87.0, 75.0, -81.0, -96.0, 65.0, 116.0], [205.0, -137.0, -225.0, 2.0, 106.0, 20.0, -31.0], [-96.0, -61.0, 103.0, 145.0, 154.0, -83.0, -114.0], [-236.0, -39.0, 35.0, -39.0, -38.0, 95.0, 20.0], [-236.0, 125.0, 92.0, -75.0, -37.0, 67.0, 17.0]]], [[[168.0, 4.0, -175.0, 12.0, 71.0, -7.0, -15.0], [-72.0, -16.0, -38.0, -115.0, 149.0, -73.0, -12.0], [-95.0, -56.0, 136.0, -28.0, -24.0, -89.0, -125.0], [-149.0, -80.0, 91.0, -22.0, -115.0, -80.0, 40.0], [45.0, 43.0, -35.0, 69.0, -113.0, -138.0, -84.0]], [[-148.0, -144.0, -128.0, -210.0, -131.0, 39.0, 99.0], [-106.0, 12.0, 49.0, 88.0, -62.0, -53.0, -105.0], [16.0, -142.0, 189.0, 10.0, -157.0, -70.0, -73.0], [31.0, 54.0, 174.0, -313.0, 67.0, -2.0, -83.0], [133.0, 132.0, -126.0, 130.0, 75.0, -201.0, 42.0]], [[51.0, 44.0, 79.0, 13.0, 17.0, 80.0, -58.0], [-97.0, -32.0, -2.0, 89.0, -1.0, -16.0, -63.0], [76.0, -91.0, -152.0, 46.0, -55.0, -44.0, -101.0], [-15.0, -14.0, -124.0, -80.0, 187.0, 17.0, 46.0], [33.0, 132.0, 139.0, -140.0, -39.0, 18.0, 110.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([62.0, -109.0, -46.0, 199.0, -20.0, 70.0, 139.0]);
    expect(b.shape).toEqual([7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[-9817.0, 12863.0, 44182.0, 12133.0, -32961.0], [16635.0, 31172.0, 2497.0, 32164.0, 18263.0], [20134.0, 33362.0, 78.0, -9562.0, -39621.0]], [[16423.0, -33615.0, -34739.0, -6822.0, -5632.0], [-10271.0, -9687.0, -2141.0, -87272.0, 15792.0], [-5483.0, 5420.0, 14758.0, -5776.0, -29266.0]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (7, 1)", async () => {
    const a = tensor([[[[[24.0, -346.0, -70.0, -20.0, -22.0, 36.0, 104.0], [-100.0, 1.0, -154.0, 241.0, -88.0, -59.0, -154.0], [-91.0, 40.0, 59.0, 95.0, 8.0, -27.0, 109.0], [-40.0, -6.0, -94.0, 140.0, -112.0, 92.0, -12.0], [-128.0, -25.0, 79.0, -7.0, 44.0, -61.0, -46.0]], [[20.0, -124.0, -73.0, -9.0, 235.0, 36.0, 64.0], [43.0, 9.0, -121.0, 77.0, 43.0, 48.0, 76.0], [-202.0, 81.0, 34.0, -16.0, 132.0, 95.0, 56.0], [17.0, -104.0, -2.0, -37.0, -56.0, 43.0, 92.0], [-96.0, -7.0, 76.0, -16.0, 35.0, -21.0, 84.0]], [[-215.0, -36.0, -216.0, -174.0, 39.0, 28.0, 39.0], [33.0, 119.0, -91.0, -105.0, -41.0, -189.0, 81.0], [213.0, 163.0, -25.0, 129.0, -133.0, -84.0, -74.0], [70.0, -255.0, 214.0, -229.0, -100.0, -37.0, 112.0], [82.0, 6.0, -47.0, 23.0, 107.0, 49.0, -127.0]]], [[[30.0, 130.0, 5.0, 167.0, 194.0, 224.0, -134.0], [59.0, 19.0, -47.0, -4.0, 134.0, -31.0, -75.0], [24.0, -16.0, -25.0, 154.0, -1.0, -168.0, 35.0], [-163.0, 111.0, -122.0, -24.0, -111.0, 126.0, -93.0], [30.0, 90.0, 85.0, 219.0, 142.0, 73.0, 413.0]], [[89.0, -16.0, -48.0, 38.0, -150.0, 177.0, -2.0], [49.0, 177.0, 67.0, -187.0, 102.0, 103.0, 55.0], [36.0, 103.0, 39.0, 28.0, -47.0, 100.0, -68.0], [117.0, 80.0, 154.0, -167.0, 167.0, -96.0, -97.0], [31.0, 9.0, 138.0, -214.0, -169.0, -45.0, -22.0]], [[51.0, 136.0, 139.0, 47.0, 7.0, 3.0, -51.0], [2.0, 26.0, 204.0, 124.0, -67.0, 10.0, -83.0], [-246.0, 120.0, 108.0, -282.0, -19.0, 149.0, -37.0], [165.0, 215.0, 175.0, 123.0, 56.0, -99.0, 37.0], [-191.0, 62.0, -46.0, 68.0, 22.0, -121.0, -216.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[-102.0], [110.0], [-104.0], [-83.0], [82.0], [30.0], [213.0]]);
    expect(b.shape).toEqual([7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[-10140.0], [-35465.0], [22724.0], [-7404.0], [-5349.0]], [[26641.0], [23951.0], [52908.0], [6399.0], [22578.0]], [[67221.0], [36124.0], [-41091.0], [-23893.0], [-21532.0]]], [[[-9055.0], [-4625.0], [-12057.0], [18385.0], [81626.0]], [[-16416.0], [46194.0], [-14060.0], [-15136.0], [-18656.0]], [[-18798.0], [-51725.0], [45497.0], [-12086.0], [-22392.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[[1.0, 12.0, 22.0, -26.0, -81.0, -7.0, -24.0], [131.0, -78.0, 126.0, 25.0, -80.0, 47.0, 19.0], [21.0, 264.0, 66.0, -76.0, 20.0, 41.0, -3.0], [88.0, 14.0, 14.0, 92.0, 14.0, 107.0, -114.0], [-48.0, 103.0, 54.0, -60.0, -16.0, 60.0, -66.0]], [[-32.0, 64.0, -72.0, 8.0, 71.0, -172.0, 216.0], [145.0, 7.0, -20.0, 246.0, 143.0, -21.0, -65.0], [131.0, -11.0, -87.0, 132.0, 107.0, 43.0, 53.0], [-136.0, 20.0, -17.0, -22.0, -42.0, -36.0, -9.0], [168.0, 10.0, -22.0, 122.0, -48.0, -22.0, -2.0]], [[-88.0, 34.0, 176.0, 107.0, -93.0, -1.0, -75.0], [-78.0, 154.0, -73.0, -113.0, 170.0, -21.0, -87.0], [262.0, -58.0, 23.0, -112.0, -106.0, 5.0, 17.0], [-185.0, 160.0, 12.0, 18.0, 56.0, -122.0, -42.0], [-206.0, -72.0, 36.0, -85.0, -86.0, -113.0, 119.0]]], [[[6.0, -26.0, 127.0, 47.0, -20.0, -2.0, 79.0], [17.0, 6.0, 133.0, 15.0, 85.0, 67.0, 86.0], [-30.0, 12.0, -43.0, 136.0, 33.0, -69.0, 96.0], [89.0, -94.0, -8.0, 3.0, 131.0, 173.0, 138.0], [6.0, 108.0, -44.0, 5.0, 44.0, -23.0, 101.0]], [[-61.0, -115.0, -264.0, -16.0, 63.0, -24.0, 41.0], [-62.0, -222.0, 112.0, -88.0, -40.0, -34.0, -201.0], [-73.0, -43.0, -110.0, -115.0, -71.0, 15.0, 238.0], [-117.0, 3.0, -45.0, 74.0, -55.0, -152.0, 222.0], [2.0, -19.0, -124.0, -17.0, 75.0, 77.0, -150.0]], [[105.0, 83.0, 91.0, 56.0, -75.0, 7.0, -157.0], [147.0, 21.0, 69.0, 91.0, -81.0, -113.0, 217.0], [59.0, -14.0, -98.0, 6.0, -62.0, -3.0, -0.0], [-2.0, -87.0, 60.0, -60.0, -124.0, 58.0, -50.0], [-64.0, 45.0, -84.0, -8.0, -134.0, 168.0, 54.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[167.0, 151.0, -131.0, 185.0, 239.0, 206.0, -72.0, 22.0, -35.0, 23.0, 42.0], [-119.0, -85.0, -40.0, -65.0, 47.0, 91.0, -95.0, -30.0, -11.0, -167.0, -24.0], [47.0, 137.0, 50.0, -120.0, -164.0, -39.0, 86.0, 47.0, 88.0, 103.0, -30.0], [74.0, -116.0, 207.0, 73.0, -38.0, 62.0, 107.0, -2.0, 2.0, -94.0, -52.0], [-64.0, -16.0, 10.0, -30.0, -84.0, 88.0, 65.0, 75.0, -57.0, 52.0, -51.0], [-59.0, -138.0, -142.0, -10.0, -59.0, 137.0, -107.0, 32.0, 138.0, -67.0, -127.0], [93.0, -53.0, 45.0, -16.0, -11.0, -169.0, -78.0, 77.0, -85.0, -84.0, -25.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[1214.0, 8695.0, -5789.0, -2249.0, 5664.0, -5203.0, -4746.0, -7399.0, 7408.0, 1002.0, 6066.0], [43045.0, 34560.0, -9185.0, 17636.0, 9767.0, 12712.0, -222.0, 8061.0, 16842.0, 17762.0, -70.0], [-34409.0, -7230.0, -31500.0, -27705.0, 5425.0, 28948.0, -31901.0, -1623.0, 6790.0, -31118.0, -9634.0], [2685.0, -5604.0, -12528.0, 20740.0, 9663.0, 59717.0, 1735.0, -2314.0, 21840.0, -4385.0, -13297.0], [-30829.0, -6171.0, -19202.0, -25499.0, -14677.0, 11625.0, -10417.0, -5850.0, 19981.0, -6411.0, -8142.0]], [[9940.0, -9912.0, 34542.0, -4722.0, 8672.0, -51284.0, -2941.0, 10429.0, -52047.0, -22520.0, 11687.0], [26688.0, -5921.0, 32134.0, 43688.0, 18858.0, 67231.0, 30109.0, 6596.0, -11944.0, -8715.0, -9271.0], [24409.0, -16970.0, 3602.0, 40538.0, 27936.0, 43912.0, -3525.0, 12341.0, -16526.0, -18288.0, -10731.0], [-23544.0, -15896.0, 15899.0, -24262.0, -22189.0, -34004.0, 5900.0, -9342.0, 1191.0, -5167.0, 2401.0], [39044.0, 11262.0, 4300.0, 43668.0, 44946.0, 37040.0, -2494.0, -2340.0, -7812.0, -12394.0, 6424.0]], [[-3516.0, 1123.0, 36954.0, -27799.0, -43668.0, -10910.0, 29603.0, -7680.0, 29946.0, 1899.0, -8611.0], [-60877.0, -16972.0, -22216.0, -27427.0, -7222.0, 20573.0, -7300.0, -4162.0, -10807.0, -6854.0, -2734.0], [51519.0, 60740.0, -55041.0, 44162.0, 68798.0, 29337.0, -32111.0, 2328.0, -1445.0, 21334.0, 21876.0], [-48331.0, -23813.0, 38155.0, -44539.0, -36391.0, -27590.0, 21048.0, -11280.0, -10651.0, -16817.0, 782.0], [-7194.0, 469.0, 34612.0, -42149.0, -42710.0, -98822.0, 12892.0, -1413.0, -9807.0, 12087.0, 12178.0]]], [[[22288.0, 11472.0, 19972.0, -9653.0, -21473.0, -18554.0, 10741.0, 11306.0, 5495.0, 5601.0, -6079.0], [8091.0, 3374.0, 2494.0, -16706.0, -30076.0, 1916.0, 2897.0, 21556.0, 8164.0, 4385.0, -19194.0], [12492.0, -23311.0, 43900.0, 6922.0, -4479.0, -17752.0, 13914.0, 4346.0, -22157.0, -21632.0, -2650.0], [20138.0, -13299.0, -24724.0, 15886.0, -4678.0, 22185.0, -18605.0, 30383.0, 1898.0, 268.0, -26024.0], [-5614.0, -17765.0, 1980.0, -2971.0, 10086.0, -3258.0, -16498.0, 5155.0, -19527.0, -27555.0, -3128.0]], [[-8897.0, -33617.0, 1962.0, 24396.0, 19593.0, -18400.0, -5634.0, -3154.0, -30252.0, -6446.0, 7760.0], [689.0, 51045.0, -231.0, -12148.0, -32699.0, -17007.0, 42486.0, -8829.0, 28965.0, 72538.0, 15323.0], [5039.0, -22646.0, -10152.0, -7733.0, 5403.0, -66206.0, -37208.0, 8225.0, -20995.0, -19707.0, 3012.0], [16599.0, -22581.0, 59299.0, -11420.0, -12108.0, -80668.0, 7560.0, 3178.0, -36461.0, -26107.0, 9075.0], [-27784.0, -16975.0, -26155.0, 14624.0, 11374.0, 44964.0, -2486.0, -8641.0, 8294.0, 3386.0, -4710.0]], [[5865.0, 23326.0, -9742.0, 11890.0, 19558.0, 49998.0, 4995.0, -13505.0, 22118.0, 1482.0, 3637.0], [64059.0, 24698.0, 27191.0, 24281.0, 32430.0, -24138.0, -7008.0, 12683.0, -28544.0, -16442.0, 11925.0], [11502.0, -2617.0, -11021.0, 25913.0, 34672.0, 9207.0, -14413.0, -7646.0, -7403.0, -9986.0, 8985.0], [8263.0, 18903.0, -17404.0, -2355.0, -4583.0, -8905.0, -3217.0, -5788.0, 25509.0, 20169.0, 3532.0], [-16897.0, -47971.0, -22038.0, -3793.0, 1649.0, -4211.0, -38645.0, -7206.0, 20569.0, -39647.0, -16684.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (1, 7)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 1x7)');
});
test("matmul (1, 2, 3, 5, 7) x (1, 7, 1)", async () => {
    const a = tensor([[[[[119.0, 128.0, -39.0, 89.0, -11.0, 56.0, -14.0], [130.0, 117.0, -45.0, 49.0, 120.0, -2.0, -27.0], [190.0, 47.0, -108.0, -15.0, -31.0, -54.0, 34.0], [103.0, 58.0, -3.0, -103.0, -129.0, 99.0, -1.0], [-114.0, 163.0, -19.0, -182.0, -1.0, -221.0, 29.0]], [[-43.0, -2.0, 130.0, 22.0, 101.0, 22.0, 136.0], [75.0, 4.0, 4.0, -15.0, 78.0, 103.0, 5.0], [13.0, -76.0, 49.0, -85.0, -6.0, 156.0, 7.0], [-59.0, 48.0, -21.0, -6.0, 259.0, 23.0, 218.0], [-209.0, -90.0, 61.0, -72.0, 69.0, 20.0, -28.0]], [[-122.0, -160.0, 25.0, -45.0, -192.0, 51.0, -10.0], [101.0, 56.0, 118.0, -88.0, -24.0, 274.0, 17.0], [-123.0, 113.0, 28.0, -37.0, 65.0, -92.0, -67.0], [-109.0, 120.0, -78.0, -37.0, 113.0, 69.0, -104.0], [75.0, -147.0, -26.0, -182.0, -71.0, 55.0, 20.0]]], [[[192.0, 22.0, 3.0, -108.0, 71.0, -88.0, 30.0], [-45.0, 14.0, -62.0, 158.0, 112.0, -177.0, 35.0], [-91.0, 95.0, 245.0, -161.0, 72.0, 105.0, 52.0], [-26.0, -127.0, 55.0, 126.0, -51.0, -75.0, -98.0], [101.0, 125.0, -44.0, -30.0, -90.0, 36.0, -162.0]], [[-85.0, -23.0, 6.0, -195.0, 78.0, -60.0, 133.0], [201.0, -142.0, -86.0, -135.0, 90.0, -50.0, 55.0], [106.0, -39.0, -28.0, -220.0, 29.0, -77.0, -160.0], [-261.0, 40.0, 18.0, -62.0, -90.0, 123.0, 6.0], [97.0, 172.0, 53.0, -56.0, 19.0, -81.0, 8.0]], [[195.0, -115.0, -95.0, 30.0, 57.0, 38.0, 74.0], [-149.0, 131.0, -42.0, 128.0, -121.0, -75.0, -103.0], [-88.0, -48.0, -217.0, 56.0, 107.0, 53.0, -31.0], [-117.0, -40.0, 72.0, -30.0, 8.0, 29.0, -23.0], [-133.0, -101.0, 87.0, 39.0, 34.0, 5.0, 11.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[[129.0], [-23.0], [167.0], [-75.0], [-5.0], [213.0], [-79.0]]]);
    expect(b.shape).toEqual([1, 7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[12308.0], [3996.0], [-7515.0], [40988.0], [-57337.0]], [[7996.0], [32530.0], [50688.0], [-25390.0], [-3177.0]], [[8105.0], [95186.0], [-25643.0], [-4724.0], [32854.0]]], [[[11394.0], [-69357.0], [56963.0], [-8676.0], [25972.0]], [[-18486.0], [9513.0], [22489.0], [-758.0], [3628.0]], [[11648.0], [-46081.0], [-37484.0], [8055.0], [-3204.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[[-211.0, -226.0, 205.0, 120.0, 38.0, 78.0, -147.0], [-175.0, 62.0, -132.0, 62.0, -66.0, -72.0, -45.0], [226.0, -166.0, -116.0, -110.0, 202.0, -152.0, -154.0], [-45.0, 112.0, -53.0, -38.0, 205.0, 21.0, 198.0], [-25.0, -23.0, 17.0, -236.0, 45.0, 60.0, 36.0]], [[-35.0, 48.0, -114.0, -38.0, 132.0, -130.0, -22.0], [146.0, 89.0, -26.0, 44.0, 86.0, 17.0, -80.0], [23.0, -60.0, -200.0, 27.0, -16.0, -35.0, 38.0], [33.0, 67.0, -31.0, 38.0, -23.0, 177.0, 100.0], [-63.0, 45.0, 59.0, 227.0, -144.0, -23.0, -12.0]], [[16.0, 210.0, -54.0, 31.0, 12.0, 174.0, -14.0], [84.0, -55.0, -30.0, -89.0, -108.0, 55.0, 209.0], [-2.0, -170.0, 249.0, 4.0, -8.0, 149.0, -123.0], [-102.0, 63.0, 116.0, -300.0, 7.0, -42.0, 123.0], [69.0, -42.0, -139.0, 105.0, 219.0, 20.0, -68.0]]], [[[-9.0, 0.0, 87.0, -158.0, -75.0, 2.0, -108.0], [-25.0, 103.0, 12.0, -72.0, 122.0, -44.0, -38.0], [-144.0, 62.0, 39.0, -231.0, 136.0, 40.0, -31.0], [-79.0, 98.0, 19.0, 239.0, 13.0, 68.0, -137.0], [-122.0, -49.0, -17.0, -142.0, 3.0, -40.0, -215.0]], [[-2.0, -121.0, -134.0, 74.0, 233.0, 71.0, 31.0], [-116.0, -1.0, -122.0, -31.0, 7.0, 11.0, 74.0], [117.0, -20.0, -3.0, 163.0, 50.0, -81.0, 26.0], [-30.0, -237.0, 105.0, -57.0, -2.0, -0.0, -47.0], [-210.0, 27.0, 151.0, 156.0, 199.0, -1.0, 109.0]], [[146.0, 76.0, 105.0, 98.0, -195.0, -106.0, -41.0], [-9.0, 151.0, -34.0, -57.0, 39.0, -79.0, 51.0], [-108.0, 18.0, -40.0, 101.0, -246.0, -93.0, -118.0], [-36.0, 10.0, -111.0, -57.0, -87.0, -164.0, -152.0], [-116.0, -208.0, -200.0, -75.0, 154.0, -56.0, -70.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[[-47.0, 46.0, 31.0, -86.0, 78.0, 86.0, 92.0, 22.0, 77.0, -63.0, 144.0], [56.0, -44.0, 98.0, 70.0, 36.0, -38.0, -159.0, -7.0, -162.0, 141.0, -137.0], [205.0, 96.0, -101.0, -194.0, 75.0, -27.0, 12.0, -7.0, -29.0, 26.0, -95.0], [-32.0, -3.0, -12.0, 89.0, 166.0, 63.0, 92.0, 28.0, -66.0, 7.0, -15.0], [6.0, -53.0, -116.0, -160.0, -39.0, 14.0, 222.0, -289.0, 121.0, -23.0, 26.0], [-3.0, 15.0, 96.0, 61.0, 137.0, -53.0, -53.0, 118.0, 63.0, 204.0, -51.0], [94.0, 76.0, 17.0, -82.0, -26.0, 262.0, 31.0, 54.0, -72.0, 70.0, 134.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[21622.0, 7542.0, -50253.0, -16032.0, 23727.0, -49649.0, 29767.0, -10851.0, 26596.0, -7655.0, -43385.0], [-21757.0, -24638.0, 13218.0, 60374.0, -17146.0, -18834.0, -34069.0, 6524.0, -33065.0, 449.0, -26158.0], [-52986.0, -17796.0, -36868.0, -47306.0, -40006.0, -7518.0, 83800.0, -80764.0, 80872.0, -87864.0, 60324.0], [18517.0, -7474.0, -3008.0, -29145.0, -20027.0, 44544.0, 24455.0, -48542.0, -5692.0, 30412.0, 14572.0], [14398.0, 3453.0, -762.0, -30254.0, -35150.0, -9721.0, -12225.0, -11097.0, 23517.0, 10847.0, 4410.0]], [[-18707.0, -25170.0, -12577.0, -2142.0, -38246.0, -1176.0, 19796.0, -56048.0, 4709.0, -25353.0, 6898.0], [-15671.0, -10211.0, 5642.0, -3529.0, 21001.0, -8009.0, 18728.0, -23165.0, 11911.0, -1127.0, 1290.0], [-42724.0, -12372.0, 13851.0, 32334.0, -16043.0, 22946.0, 11221.0, 5628.0, 8632.0, -19032.0, 36588.0], [3361.0, 6954.0, 31624.0, 17525.0, 31515.0, 20020.0, -15880.0, 34471.0, -8754.0, 50465.0, 1723.0], [8389.0, 6480.0, 8066.0, 39946.0, 41590.0, 1639.0, -22480.0, 42496.0, -46843.0, 11217.0, -28426.0]], [[-2820.0, -12871.0, 41232.0, 36401.0, 33638.0, -15915.0, -36706.0, 16436.0, -19846.0, 61655.0, -32239.0], [8503.0, 26104.0, 22673.0, -9678.0, -6139.0, 54848.0, -12487.0, 48939.0, -2529.0, 13884.0, 46209.0], [29434.0, 24591.0, -28778.0, -39223.0, 36986.0, -40418.0, 16716.0, 12767.0, 37176.0, 4628.0, -25002.0], [53432.0, 12919.0, -7857.0, -49790.0, -56013.0, 1352.0, -38016.0, -12234.0, -12279.0, 16106.0, -11033.0], [-42588.0, -25112.0, -13838.0, -807.0, 6842.0, 2088.0, 66468.0, -58878.0, 41873.0, -18865.0, 22882.0]]], [[[12706.0, 4209.0, -114.0, -9188.0, -14398.0, -42529.0, -34424.0, 10848.0, 6039.0, -3704.0, -23715.0], [8999.0, -14328.0, -10051.0, -18464.0, -19092.0, -16840.0, 3081.0, -45873.0, 519.0, 1464.0, -17447.0], [23409.0, -13879.0, -12018.0, -28179.0, -43439.0, -38684.0, -16779.0, -46601.0, 14191.0, 20073.0, -32128.0], [-7556.0, -16920.0, 5059.0, 44541.0, 50836.0, -35290.0, -5599.0, 1004.0, -22563.0, 24945.0, -51680.0], [-16023.0, -21761.0, -13006.0, 12432.0, -36134.0, -71285.0, -20580.0, -23395.0, 21732.0, -23938.0, -33802.0]], [[-32421.0, -16782.0, -18959.0, -11207.0, -2444.0, 20327.0, 73179.0, -53472.0, 48884.0, -8606.0, 34500.0], [-11657.0, -11493.0, 10502.0, 24298.0, -24070.0, 10306.0, -11564.0, 712.0, -6974.0, 11041.0, 5025.0], [-9463.0, 3596.0, -13120.0, -11446.0, 21516.0, 32977.0, 45103.0, -15305.0, 653.0, -24982.0, 26343.0], [7057.0, 15833.0, -34644.0, -35279.0, -11159.0, -12342.0, 29038.0, -3292.0, 39943.0, -32440.0, 12679.0], [48788.0, 902.0, -42314.0, -36299.0, 11081.0, 18062.0, 40161.0, -53241.0, -19051.0, 24904.0, -30793.0]], [[11077.0, 18787.0, 11940.0, 9212.0, 32416.0, 5153.0, -27319.0, 46322.0, -37904.0, -15075.0, -5991.0], [8998.0, -9527.0, 7396.0, -2374.0, -20948.0, 8910.0, -16063.0, -20452.0, -24337.0, 7132.0, -6021.0], [-17637.0, -7228.0, 18846.0, 70660.0, 5911.0, -31960.0, -57327.0, 54354.0, -43867.0, -12565.0, -33198.0], [-32997.0, -21982.0, 3523.0, 36637.0, -35358.0, -36420.0, -26812.0, -4098.0, -7326.0, -41702.0, -9420.0], [-50284.0, -29481.0, -27310.0, 5225.0, -55844.0, -14613.0, 48086.0, -56690.0, 55660.0, -47611.0, 29397.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (2, 7)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 2x7)');
});
test("matmul (1, 2, 3, 5, 7) x (2, 7, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([2, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2');
});
test("matmul (1, 2, 3, 5, 7) x (2, 7, 11)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([2, 7, 11]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2');
});
test("matmul (1, 2, 3, 5, 7) x (1,)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 30, 30x7,1');
});
test("matmul (1, 2, 3, 5, 7) x (1, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 1x1)');
});
test("matmul (1, 2, 3, 5, 7) x (1, 11)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([1, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 1x11)');
});
test("matmul (1, 2, 3, 5, 7) x (2,)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 30, 30x7,2');
});
test("matmul (1, 2, 3, 5, 7) x (2, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 2x1)');
});
test("matmul (1, 2, 3, 5, 7) x (2, 11)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([2, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 2x11)');
});
test("matmul (2, 2, 3, 5, 7) x (7,)", async () => {
    const a = tensor([[[[[101.0, -31.0, 84.0, 159.0, -18.0, -38.0, -117.0], [-126.0, 223.0, 154.0, 35.0, 18.0, -88.0, -177.0], [-161.0, -81.0, -48.0, 13.0, -78.0, 210.0, 111.0], [-52.0, 117.0, -165.0, 12.0, -115.0, -5.0, -14.0], [-39.0, -138.0, 101.0, -35.0, -6.0, -3.0, -35.0]], [[-122.0, 81.0, 17.0, -151.0, 9.0, -150.0, -11.0], [24.0, -141.0, 69.0, -187.0, 59.0, -20.0, 121.0], [73.0, 150.0, 144.0, 129.0, -3.0, 122.0, 32.0], [-15.0, -74.0, 92.0, -88.0, -89.0, 23.0, 192.0], [-31.0, -48.0, -116.0, 30.0, 98.0, -55.0, -73.0]], [[-170.0, 31.0, 4.0, -18.0, 133.0, 93.0, -87.0], [-60.0, -107.0, -145.0, -115.0, -190.0, -8.0, -179.0], [64.0, 158.0, -21.0, -50.0, -69.0, -117.0, -27.0], [-133.0, -77.0, -22.0, -93.0, -60.0, 45.0, -22.0], [15.0, -128.0, 39.0, 207.0, -52.0, -113.0, 14.0]]], [[[-20.0, -96.0, -2.0, -71.0, -89.0, -105.0, -34.0], [42.0, -77.0, 96.0, 90.0, 161.0, -10.0, -26.0], [-30.0, 12.0, 126.0, 87.0, 162.0, -12.0, 36.0], [101.0, -118.0, -70.0, 19.0, -1.0, 76.0, -161.0], [-113.0, -164.0, 104.0, 68.0, 92.0, -111.0, -42.0]], [[29.0, 21.0, -2.0, -70.0, -14.0, 75.0, 146.0], [-27.0, 13.0, -138.0, -51.0, 108.0, 78.0, 92.0], [-42.0, 12.0, 141.0, -45.0, 32.0, -7.0, -128.0], [118.0, -7.0, 11.0, 40.0, 53.0, 14.0, 15.0], [-150.0, 14.0, 60.0, -312.0, -66.0, -22.0, 65.0]], [[-38.0, -94.0, -51.0, 111.0, -88.0, -25.0, -281.0], [114.0, 90.0, -83.0, 61.0, 30.0, 33.0, 35.0], [3.0, 120.0, 36.0, 72.0, -144.0, 32.0, 99.0], [-9.0, 99.0, -47.0, -46.0, 30.0, 67.0, -98.0], [-71.0, 71.0, -124.0, -117.0, -126.0, -80.0, 189.0]]]], [[[[96.0, -142.0, 4.0, 119.0, -8.0, 185.0, 70.0], [-28.0, 12.0, 200.0, -134.0, -24.0, 95.0, 54.0], [-42.0, 91.0, 13.0, -60.0, -105.0, 143.0, -135.0], [90.0, -49.0, 20.0, -60.0, 27.0, -67.0, -124.0], [48.0, -207.0, 44.0, -111.0, 71.0, -74.0, 95.0]], [[29.0, 57.0, -126.0, -170.0, 112.0, -139.0, -96.0], [87.0, -62.0, -272.0, 59.0, 136.0, 14.0, 8.0], [29.0, 26.0, 39.0, -86.0, 170.0, -18.0, -114.0], [184.0, -109.0, -73.0, 58.0, 39.0, 28.0, 15.0], [-117.0, 314.0, -180.0, 65.0, -14.0, 173.0, -84.0]], [[57.0, -75.0, 129.0, -26.0, -54.0, 20.0, -175.0], [331.0, -7.0, 201.0, 168.0, 89.0, 98.0, 154.0], [219.0, -137.0, -171.0, -29.0, 120.0, 33.0, -62.0], [143.0, -116.0, -97.0, -44.0, -61.0, 55.0, 42.0], [33.0, -10.0, -76.0, -64.0, -195.0, -52.0, -115.0]]], [[[-58.0, -88.0, -71.0, -231.0, 12.0, -99.0, -23.0], [166.0, -0.0, 51.0, -70.0, 98.0, -92.0, 40.0], [25.0, -22.0, -47.0, 21.0, -82.0, -32.0, 99.0], [-112.0, -126.0, -117.0, 37.0, -53.0, 33.0, 69.0], [-49.0, 140.0, -135.0, 90.0, -68.0, -43.0, 3.0]], [[-94.0, -155.0, -64.0, -168.0, 76.0, 23.0, -73.0], [-259.0, 62.0, 18.0, 15.0, 4.0, -130.0, 102.0], [-15.0, 109.0, 50.0, -19.0, -32.0, -44.0, -35.0], [17.0, 188.0, 43.0, -176.0, -109.0, -255.0, -75.0], [127.0, -49.0, 134.0, 159.0, 59.0, 16.0, -32.0]], [[-113.0, -116.0, -142.0, -165.0, -149.0, 4.0, 228.0], [-132.0, 10.0, 93.0, 28.0, -54.0, 41.0, -1.0], [7.0, 49.0, -122.0, -51.0, -136.0, -11.0, -192.0], [-148.0, -134.0, 29.0, -132.0, -8.0, -111.0, -15.0], [-252.0, -47.0, -51.0, 11.0, 13.0, 62.0, -62.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([-1.0, -244.0, 151.0, 9.0, -45.0, 50.0, -108.0]);
    expect(b.shape).toEqual([7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[33124.0, -16811.0, 14816.0, -46866.0, 52547.0], [-25151.0, 26393.0, -10989.0, 15590.0, -4779.0], [1109.0, 30720.0, -42066.0, 22088.0, 34147.0]], [[24930.0, 29115.0, 5133.0, 39525.0, 51291.0], [-17473.0, -35338.0, 30034.0, 306.0, -2164.0], [49330.0, -37538.0, -25811.0, -19074.0, -55772.0]]], [[[38277.0, 26092.0, 5716.0, 23173.0, 38950.0], [-36115.0, -31784.0, 2504.0, 13936.0, -84742.0], [59818.0, 17503.0, 10073.0, 14077.0, 8950.0]], [[5724.0, -6425.0, -10167.0, 10105.0, -53100.0], [32352.0, -29712.0, -16182.0, -40725.0, 35095.0], [-12229.0, 16575.0, -4538.0, 32465.0, 13329.0]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (7, 1)", async () => {
    const a = tensor([[[[[18.0, -70.0, -128.0, -147.0, 38.0, 71.0, 33.0], [1.0, 38.0, 19.0, 92.0, -8.0, -48.0, -167.0], [-92.0, 19.0, 106.0, -26.0, 64.0, 132.0, 14.0], [61.0, 148.0, 48.0, 43.0, 37.0, -65.0, 145.0], [215.0, -139.0, -240.0, 73.0, 44.0, -151.0, 93.0]], [[-114.0, 89.0, 84.0, -24.0, 110.0, 112.0, 57.0], [-2.0, 98.0, -17.0, -134.0, 20.0, -108.0, -77.0], [-130.0, 67.0, -106.0, -126.0, -48.0, -133.0, -48.0], [150.0, 84.0, -131.0, -161.0, -192.0, 181.0, -6.0], [141.0, 2.0, -61.0, 10.0, -92.0, 45.0, -87.0]], [[84.0, -3.0, -90.0, -128.0, 49.0, 43.0, -25.0], [-37.0, 61.0, -105.0, 112.0, 55.0, 37.0, -106.0], [-104.0, 91.0, 196.0, 26.0, -88.0, -67.0, -46.0], [-211.0, -57.0, -138.0, 164.0, 48.0, 34.0, -171.0], [-73.0, -87.0, -75.0, -108.0, -181.0, 109.0, 17.0]]], [[[-53.0, -36.0, 139.0, 14.0, -81.0, 41.0, -31.0], [-152.0, 199.0, 187.0, 101.0, 77.0, -56.0, -5.0], [45.0, 3.0, 15.0, -39.0, 56.0, -30.0, -115.0], [92.0, 180.0, 143.0, -52.0, 3.0, -2.0, -56.0], [-116.0, 15.0, -93.0, 135.0, -77.0, 141.0, 11.0]], [[-117.0, 35.0, -124.0, 117.0, -21.0, -12.0, -43.0], [33.0, 188.0, 100.0, -45.0, 124.0, 39.0, -21.0], [35.0, -112.0, 40.0, 58.0, 54.0, 38.0, -74.0], [-49.0, -121.0, -110.0, 15.0, 55.0, 68.0, -3.0], [-1.0, -97.0, 103.0, -188.0, -176.0, 136.0, 50.0]], [[51.0, 75.0, 35.0, -44.0, -61.0, 199.0, 65.0], [-95.0, -129.0, -106.0, 49.0, -57.0, -187.0, -68.0], [-191.0, 175.0, 74.0, -9.0, 53.0, 13.0, 109.0], [56.0, -160.0, 260.0, 97.0, -86.0, 87.0, 29.0], [-77.0, -12.0, 211.0, -164.0, 83.0, 90.0, 161.0]]]], [[[[-88.0, 31.0, 132.0, -94.0, -39.0, -110.0, 48.0], [-116.0, 62.0, 24.0, 91.0, 105.0, 94.0, 94.0], [103.0, -165.0, -128.0, 89.0, -56.0, 168.0, 174.0], [210.0, -30.0, -77.0, -63.0, 18.0, 7.0, 250.0], [-15.0, -52.0, 36.0, 158.0, -68.0, 10.0, 35.0]], [[55.0, -26.0, 80.0, 105.0, -48.0, 141.0, -2.0], [97.0, 52.0, -86.0, -78.0, -95.0, -39.0, 102.0], [33.0, 116.0, 68.0, 142.0, -119.0, -5.0, 3.0], [152.0, -72.0, -5.0, 25.0, -32.0, 11.0, -152.0], [78.0, -247.0, 2.0, -10.0, 267.0, 273.0, -66.0]], [[-82.0, 92.0, -181.0, -18.0, -20.0, -8.0, 52.0], [87.0, -123.0, 158.0, -94.0, 164.0, 134.0, 134.0], [-3.0, 72.0, 142.0, -60.0, -168.0, 47.0, 33.0], [-12.0, 5.0, 124.0, 21.0, 142.0, 1.0, 28.0], [-137.0, 213.0, 29.0, 23.0, -201.0, -120.0, 128.0]]], [[[-65.0, 77.0, 17.0, 87.0, 34.0, -53.0, -16.0], [149.0, -46.0, 22.0, 63.0, -190.0, -53.0, -107.0], [-91.0, 99.0, -156.0, 128.0, -3.0, -80.0, -20.0], [84.0, -125.0, 46.0, -20.0, 52.0, -23.0, -33.0], [-32.0, -121.0, 112.0, 66.0, -119.0, -63.0, -34.0]], [[129.0, 2.0, 175.0, -77.0, -199.0, -68.0, -18.0], [-57.0, -48.0, 34.0, -13.0, -196.0, 30.0, 44.0], [23.0, 56.0, -75.0, -53.0, -15.0, 144.0, 91.0], [-113.0, 37.0, -29.0, -47.0, -109.0, -75.0, -89.0], [-83.0, -1.0, 52.0, -45.0, -26.0, 69.0, -68.0]], [[239.0, -74.0, -147.0, 97.0, -80.0, -78.0, 125.0], [68.0, -231.0, -82.0, 62.0, 200.0, 16.0, -172.0], [-6.0, -123.0, -145.0, 137.0, 114.0, 3.0, 40.0], [-13.0, 137.0, 100.0, 73.0, 89.0, -145.0, -8.0], [-15.0, 111.0, 191.0, 30.0, 18.0, -132.0, 51.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[41.0], [-89.0], [51.0], [-38.0], [-169.0], [53.0], [5.0]]);
    expect(b.shape).toEqual([7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[3532.0], [-7895.0], [-2819.0], [-18830.0], [-8802.0]], [[-19768.0], [-14068.0], [-11088.0], [40122.0], [19610.0]], [[-2142.0], [-24421.0], [7736.0], [-24013.0], [41480.0]]], [[[23295.0], [-34250.0], [-7804.0], [-3872.0], [4577.0]], [[-15984.0], [-27563.0], [3757.0], [-3126.0], [58191.0]], [[20054.0], [-300.0], [-27013.0], [45400.0], [6452.0]]]], [[[[4938.0], [-24801.0], [28236.0], [8326.0], [12042.0]], [[20234.0], [12425.0], [8962.0], [16666.0], [-5321.0]], [[-16881.0], [6200.0], [34039.0], [-19216.0], [4280.0]]], [[[-20592.0], [37697.0], [-29195.0], [7503.0], [29263.0]], [[46899.0], [39097.0], [4770.0], [6382.0], [8759.0]], [[15213.0], [-17003.0], [-20807.0], [-33166.0], [-11676.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[[69.0, -93.0, 79.0, 129.0, -190.0, 92.0, 10.0], [115.0, 55.0, 1.0, -104.0, -78.0, 216.0, -61.0], [-32.0, 108.0, 27.0, -237.0, 94.0, 43.0, 20.0], [83.0, -150.0, 70.0, -120.0, 47.0, -67.0, 157.0], [-41.0, -5.0, 90.0, 122.0, -65.0, 158.0, 65.0]], [[45.0, -17.0, -194.0, -34.0, 37.0, 47.0, 32.0], [-4.0, 29.0, -42.0, -85.0, 28.0, -48.0, -174.0], [37.0, -35.0, -3.0, -11.0, -39.0, 158.0, 14.0], [-2.0, -16.0, 19.0, -182.0, -5.0, 122.0, -69.0], [58.0, -112.0, -72.0, -105.0, 74.0, -136.0, 21.0]], [[-48.0, 92.0, -167.0, -101.0, -15.0, 98.0, 27.0], [164.0, -14.0, -74.0, -40.0, 147.0, 18.0, 27.0], [5.0, -204.0, 40.0, 63.0, 27.0, 293.0, -4.0], [55.0, -146.0, -9.0, -111.0, -170.0, 39.0, -105.0], [-198.0, -106.0, -57.0, 90.0, -20.0, -103.0, 63.0]]], [[[-120.0, -98.0, -59.0, 28.0, -180.0, -148.0, -52.0], [69.0, -194.0, -161.0, 4.0, 20.0, 22.0, -43.0], [41.0, -55.0, -48.0, -8.0, -216.0, 152.0, 8.0], [-75.0, -12.0, -111.0, 59.0, -134.0, 161.0, 25.0], [-30.0, 100.0, -221.0, -124.0, 94.0, 22.0, -31.0]], [[-110.0, 99.0, -68.0, -65.0, 107.0, 264.0, 67.0], [67.0, 16.0, 32.0, 16.0, 31.0, 140.0, -98.0], [124.0, 20.0, -138.0, -59.0, 22.0, -28.0, -100.0], [-110.0, -48.0, -122.0, 31.0, 156.0, 56.0, 84.0], [69.0, -85.0, 153.0, 48.0, -1.0, 97.0, -48.0]], [[-74.0, -95.0, 89.0, -132.0, -71.0, -4.0, 63.0], [-97.0, 132.0, 107.0, -56.0, -37.0, -93.0, 109.0], [131.0, -21.0, 7.0, 11.0, 28.0, -89.0, -38.0], [-74.0, 211.0, 60.0, 111.0, -9.0, 20.0, -126.0], [177.0, 93.0, -84.0, 6.0, 66.0, -111.0, 18.0]]]], [[[[13.0, 90.0, -59.0, 58.0, -51.0, -99.0, 106.0], [8.0, -101.0, 137.0, -45.0, -233.0, -5.0, 128.0], [23.0, -81.0, 53.0, 110.0, 244.0, -159.0, -84.0], [16.0, 117.0, -22.0, -98.0, -134.0, -149.0, -126.0], [79.0, -11.0, -86.0, 79.0, 37.0, 38.0, -18.0]], [[-150.0, 9.0, -113.0, -114.0, -108.0, 71.0, 37.0], [39.0, 106.0, 35.0, -15.0, 25.0, -29.0, 56.0], [103.0, -68.0, 164.0, -51.0, -83.0, -198.0, 167.0], [-10.0, 45.0, -126.0, 83.0, 7.0, -1.0, 148.0], [-160.0, 30.0, 22.0, 19.0, 95.0, 112.0, 168.0]], [[147.0, -51.0, -88.0, 43.0, -9.0, -60.0, 42.0], [-15.0, -2.0, 160.0, 81.0, 82.0, 131.0, 51.0], [36.0, -58.0, -27.0, 87.0, 203.0, -68.0, -105.0], [197.0, -14.0, 46.0, 38.0, 15.0, -43.0, -141.0], [-169.0, -2.0, 39.0, -104.0, 173.0, -169.0, 159.0]]], [[[10.0, 17.0, 155.0, 105.0, 22.0, -33.0, 102.0], [51.0, -20.0, -29.0, -54.0, -56.0, 189.0, 50.0], [-74.0, 157.0, -93.0, -12.0, 71.0, 151.0, 54.0], [60.0, -202.0, 118.0, 163.0, 25.0, -99.0, -32.0], [-50.0, 124.0, 27.0, -65.0, -76.0, -2.0, -22.0]], [[112.0, 46.0, -36.0, 168.0, -107.0, -22.0, 7.0], [-74.0, 79.0, 21.0, -56.0, 138.0, -23.0, 13.0], [109.0, -13.0, 62.0, -159.0, -49.0, 145.0, 8.0], [80.0, 123.0, 8.0, -39.0, -63.0, -19.0, -87.0], [92.0, -41.0, -14.0, 141.0, 107.0, -180.0, 49.0]], [[-86.0, 59.0, -62.0, 176.0, -127.0, 127.0, 142.0], [2.0, 80.0, 17.0, -55.0, 187.0, -25.0, -64.0], [-75.0, 28.0, -74.0, -131.0, -12.0, 111.0, -18.0], [-47.0, -177.0, 19.0, 139.0, 69.0, 135.0, 122.0], [-222.0, 42.0, -68.0, -19.0, 17.0, 78.0, -29.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[59.0, 24.0, -152.0, 89.0, 195.0, -146.0, 100.0, -170.0, 156.0, 108.0, 11.0], [5.0, -89.0, -56.0, 44.0, -74.0, 20.0, 129.0, -164.0, -13.0, 167.0, 124.0], [114.0, 13.0, -45.0, -120.0, 30.0, 102.0, -110.0, -275.0, -16.0, 136.0, -42.0], [51.0, 142.0, 79.0, -11.0, -43.0, -48.0, 60.0, 89.0, -325.0, -32.0, 106.0], [218.0, 166.0, -147.0, 9.0, -187.0, 155.0, 51.0, 13.0, -95.0, -104.0, 44.0], [177.0, 184.0, 112.0, -87.0, -34.0, -144.0, 71.0, -233.0, -77.0, -152.0, 49.0], [-9.0, 72.0, -18.0, 33.0, -118.0, -18.0, 0.0, -86.0, -111.0, 111.0, -284.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[-6035.0, 15386.0, 39410.0, -18234.0, 48382.0, -52946.0, -9205.0, -31488.0, -21360.0, 5423.0, -7109.0], [23647.0, 5514.0, 7935.0, -7828.0, 37297.0, -52692.0, 23603.0, -84197.0, 48558.0, -6422.0, 21495.0], [17566.0, -18727.0, -30484.0, -964.0, -24631.0, 28980.0, 1389.0, -51307.0, 55736.0, 11744.0, -12653.0], [2981.0, 5990.0, -34085.0, 5140.0, 9508.0, 11889.0, -28310.0, -16720.0, 36045.0, 19997.0, -79150.0], [27249.0, 40917.0, 38181.0, -28197.0, -11058.0, -24787.0, 578.0, -49351.0, -60627.0, -6968.0, -5497.0]], [[-5183.0, 12337.0, -595.0, 24211.0, -6618.0, -26675.0, 26831.0, 32240.0, 10709.0, -30715.0, -2226.0], [-10040.0, -32005.0, -12201.0, 5581.0, 16397.0, 15344.0, 881.0, 26421.0, 47646.0, -13511.0, 44602.0], [20443.0, 26008.0, 18779.0, -11401.0, 10457.0, -34929.0, 8084.0, -39229.0, -165.0, -20311.0, -2923.0], [13811.0, -7571.0, 1608.0, -14096.0, 14119.0, -6455.0, -6867.0, -41016.0, 57482.0, -20163.0, 3258.0], [-18830.0, -15714.0, -34087.0, 23220.0, 10261.0, 17664.0, -12910.0, 49807.0, 46892.0, -3565.0, -30728.0]], [[-12728.0, -8367.0, 14375.0, 13157.0, -20548.0, -20261.0, 25571.0, 4657.0, 17695.0, -19639.0, 3662.0], [34119.0, 28198.0, -44053.0, 23948.0, 1229.0, -10145.0, 29109.0, -13399.0, 21602.0, -8437.0, -1382.0], [64831.0, 85848.0, 42760.0, -39404.0, 23.0, -41689.0, -4256.0, -40361.0, -42365.0, -77892.0, -3562.0], [-33384.0, -30169.0, 22700.0, -7616.0, 68886.0, -36616.0, -24905.0, 4923.0, 71499.0, -16017.0, -4636.0], [-37278.0, -1015.0, 35977.0, -5576.0, -36538.0, 27252.0, -30137.0, 93050.0, -55010.0, -24989.0, -27207.0]]], [[[-77836.0, -51805.0, 39415.0, 1320.0, 25706.0, 2546.0, -36160.0, 91805.0, 8666.0, -2802.0, -8430.0], [-6408.0, 21669.0, 8235.0, 13728.0, 23395.0, -29862.0, 2406.0, 63549.0, 15741.0, -57167.0, -1941.0], [-23992.0, -3193.0, 47008.0, -7827.0, 45249.0, -67110.0, 1581.0, -24374.0, 18407.0, -10781.0, -9529.0], [-15070.0, 15383.0, 59008.0, -8920.0, -2970.0, -47848.0, 11299.0, 9089.0, -31385.0, -34849.0, 3496.0], [-8123.0, -12681.0, -11687.0, 27523.0, -29216.0, 1750.0, 33126.0, 37201.0, 30673.0, -29189.0, 22226.0]], [[52389.0, 49597.0, 21734.0, -16353.0, -64912.0, -8413.0, 29552.0, -50504.0, -34164.0, -46334.0, 5648.0], [40917.0, 26722.0, 1631.0, -12484.0, 13160.0, -20557.0, 17725.0, -45179.0, 1685.0, -21634.0, 39129.0], [-10585.0, -17676.0, -22989.0, 28459.0, 29735.0, -19706.0, 25754.0, 23749.0, 51633.0, -9280.0, 31382.0], [24107.0, 46696.0, 9175.0, 1701.0, -63879.0, 15772.0, 10020.0, 44637.0, -53115.0, -52892.0, -13000.0], [40919.0, 32252.0, 3054.0, -26519.0, 24824.0, -11731.0, -11179.0, -54079.0, -8225.0, -7439.0, 7222.0]], [[-18180.0, -18894.0, 10990.0, -18206.0, 6925.0, 12755.0, -41270.0, -13472.0, 31227.0, 7456.0, -51536.0], [-21229.0, -36043.0, -8826.0, -3694.0, -25846.0, 36099.0, -16292.0, -27753.0, -1783.0, 57995.0, -32270.0], [-324.0, -7798.0, -31582.0, 16515.0, 29110.0, -1520.0, 5390.0, 4597.0, 25433.0, 17639.0, 7372.0], [11902.0, -10899.0, 11332.0, -11702.0, -17146.0, 13809.0, 20840.0, -22586.0, -38021.0, 15763.0, 70964.0], [-3783.0, -12441.0, -50316.0, 40704.0, 14163.0, -6948.0, 34782.0, 3465.0, 26076.0, 35037.0, 9996.0]]]], [[[[-32146.0, -19279.0, -5278.0, 23211.0, -7994.0, -4457.0, 13250.0, 17705.0, -16346.0, 38672.0, -17270.0], [-39541.0, -25810.0, 26107.0, -17115.0, 43716.0, -24753.0, -42237.0, -39348.0, 23306.0, 43269.0, -69809.0], [38409.0, 29270.0, -44819.0, 4170.0, -22971.0, 57376.0, -6224.0, 52032.0, -33570.0, -17887.0, 26444.0], [-60428.0, -82963.0, -10458.0, 17889.0, 43008.0, 5418.0, -4180.0, 19231.0, 71366.0, 44009.0, 27807.0], [13785.0, 24813.0, -2140.0, 12431.0, 4155.0, -23731.0, 25266.0, 12230.0, -16275.0, -19151.0, 20093.0]], [[-38811.0, -24258.0, 41537.0, -4068.0, -14988.0, -11604.0, -8716.0, 23824.0, 16027.0, -21870.0, -19653.0], [5869.0, -7327.0, -22555.0, 8696.0, -8841.0, 7759.0, 12040.0, -32708.0, 2663.0, 35178.0, -5712.0], [-32811.0, -34772.0, -36238.0, 9046.0, 34777.0, 15419.0, -37863.0, -25304.0, 35497.0, 80969.0, -80375.0], [-10479.0, 17537.0, 7422.0, 20331.0, -31368.0, -15911.0, 23931.0, 23953.0, -44120.0, 2495.0, -22213.0], [33209.0, 44948.0, 18706.0, -19114.0, -74974.0, 20865.0, -613.0, -21388.0, -68174.0, -18142.0, -34994.0]], [[-12381.0, 3519.0, -18284.0, 27451.0, 26717.0, -27033.0, 15662.0, 21652.0, 11841.0, 8733.0, -11717.0], [62080.0, 54788.0, 3291.0, -30490.0, -27266.0, 7510.0, -1015.0, -67756.0, -54737.0, -5565.0, -3004.0], [36356.0, 31655.0, -29703.0, 7213.0, -16498.0, 29801.0, 9833.0, 46073.0, -23867.0, -34685.0, 38980.0], [15663.0, -3606.0, -32711.0, 10202.0, 54492.0, -15119.0, 12826.0, -18122.0, 35365.0, 13303.0, 41124.0], [-4469.0, -9069.0, -31392.0, 2842.0, -72532.0, 81893.0, -30864.0, 37029.0, -14233.0, 15391.0, -60594.0]]], [[[21737.0, 20576.0, -9918.0, -11682.0, -14201.0, 15976.0, -8778.0, -38565.0, -46137.0, 35689.0, -22779.0], [17644.0, 24039.0, 18907.0, -7564.0, 11023.0, -45008.0, 13033.0, -51286.0, 11447.0, -17402.0, -13828.0], [26924.0, 24796.0, 11196.0, 898.0, -53105.0, -6677.0, 36705.0, -27565.0, -32563.0, -18379.0, 16475.0], [12510.0, 27728.0, -4428.0, -11719.0, 25646.0, 10119.0, -29012.0, 31129.0, -34077.0, -7526.0, -6729.0], [-19291.0, -35683.0, 5650.0, -2755.0, 1555.0, 4558.0, 108.0, -23676.0, 21097.0, 26826.0, 9608.0]], [[-15981.0, 676.0, 8431.0, 15646.0, 30063.0, -40711.0, 24155.0, 1401.0, -26068.0, 24755.0, 18482.0], [21463.0, 3126.0, -21641.0, -1342.0, -43796.0, 41682.0, 2526.0, -5100.0, -7489.0, 436.0, 3417.0], [20236.0, 1123.0, -7892.0, -9354.0, 34203.0, -30837.0, 659.0, -82709.0, 60458.0, 7065.0, -17194.0], [-12056.0, -34679.0, -13790.0, 10216.0, 31108.0, -11995.0, 16085.0, -28353.0, 40533.0, 31300.0, 32667.0], [1843.0, 13867.0, -36690.0, 24753.0, -5180.0, 19175.0, 6588.0, 46600.0, -32460.0, 18344.0, -6566.0]], [[-9356.0, 29381.0, 56799.0, -7060.0, -27889.0, -41565.0, 18931.0, -5796.0, -83867.0, -3833.0, -12063.0], [36568.0, 7173.0, -39031.0, 4009.0, -29222.0, 39419.0, 13112.0, -9270.0, 8139.0, -5104.0, 28577.0], [-2209.0, -6720.0, 17333.0, -5481.0, -12690.0, -7270.0, 3661.0, -7622.0, 26286.0, -26918.0, 1892.0], [43436.0, 79688.0, 29963.0, -22878.0, -33363.0, -12353.0, -8179.0, 3114.0, -81002.0, -50653.0, -33526.0], [-3836.0, 2438.0, 39710.0, -17131.0, -50030.0, 19153.0, -4037.0, 32402.0, -32317.0, -42445.0, 16414.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (1, 7)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 1x7)');
});
test("matmul (2, 2, 3, 5, 7) x (1, 7, 1)", async () => {
    const a = tensor([[[[[190.0, 4.0, 167.0, 132.0, -78.0, -209.0, -5.0], [-78.0, -77.0, 58.0, -24.0, 66.0, -76.0, -134.0], [73.0, -73.0, 200.0, 109.0, -14.0, 55.0, -168.0], [15.0, -47.0, 6.0, -174.0, 26.0, 5.0, -79.0], [-131.0, 156.0, -38.0, -84.0, -27.0, 31.0, -63.0]], [[-13.0, 60.0, 43.0, 129.0, 50.0, 200.0, -103.0], [39.0, 29.0, -47.0, -13.0, -77.0, 31.0, -6.0], [3.0, -266.0, 71.0, -135.0, 14.0, -2.0, -6.0], [43.0, 41.0, -143.0, -42.0, -124.0, -73.0, 100.0], [-205.0, -65.0, -108.0, -60.0, 136.0, 71.0, -138.0]], [[196.0, 65.0, 87.0, -128.0, 192.0, 46.0, -85.0], [41.0, 119.0, -104.0, 69.0, 86.0, -15.0, 38.0], [-127.0, 52.0, -125.0, 61.0, -1.0, 38.0, -35.0], [163.0, 155.0, 59.0, -153.0, 43.0, -115.0, -121.0], [103.0, 33.0, 22.0, 78.0, 67.0, -27.0, -211.0]]], [[[130.0, 29.0, -29.0, -119.0, -86.0, -133.0, -174.0], [-167.0, -64.0, 24.0, 16.0, 53.0, -163.0, 32.0], [-42.0, -59.0, -47.0, 65.0, 254.0, 47.0, -60.0], [-30.0, -12.0, 85.0, -293.0, -110.0, 217.0, -97.0], [-35.0, -147.0, 104.0, 152.0, 93.0, 195.0, -95.0]], [[86.0, 135.0, 77.0, 57.0, 129.0, 54.0, -23.0], [43.0, 34.0, 24.0, -39.0, -66.0, 294.0, -81.0], [-2.0, 124.0, 50.0, 42.0, 155.0, -116.0, 95.0], [14.0, 40.0, 107.0, -56.0, 152.0, -142.0, -46.0], [-52.0, -76.0, -102.0, 3.0, -68.0, 132.0, 22.0]], [[-86.0, -3.0, -221.0, 292.0, -117.0, -63.0, 46.0], [140.0, 11.0, -57.0, 16.0, 27.0, -122.0, 116.0], [47.0, -120.0, 69.0, 1.0, -14.0, 254.0, -20.0], [28.0, -318.0, 284.0, -218.0, -57.0, -111.0, -41.0], [175.0, -1.0, 126.0, -156.0, -26.0, 5.0, -32.0]]]], [[[[31.0, -124.0, 56.0, -33.0, -92.0, 12.0, 145.0], [216.0, -71.0, -1.0, -92.0, -46.0, -88.0, -150.0], [26.0, 74.0, 68.0, -130.0, 37.0, -34.0, -147.0], [62.0, 40.0, 77.0, 45.0, 85.0, 7.0, 22.0], [-64.0, 67.0, 105.0, 7.0, 144.0, -61.0, 144.0]], [[-52.0, -7.0, 33.0, 33.0, 60.0, 111.0, 31.0], [-56.0, 78.0, 43.0, 38.0, 0.0, -181.0, 5.0], [-113.0, -16.0, 62.0, 58.0, -13.0, 56.0, -131.0], [43.0, 138.0, -240.0, 45.0, 152.0, -112.0, -37.0], [1.0, -29.0, -16.0, -107.0, 24.0, 21.0, 20.0]], [[-39.0, 64.0, -2.0, 34.0, -75.0, 13.0, -113.0], [-72.0, 78.0, -6.0, -253.0, 122.0, -32.0, -58.0], [-7.0, 26.0, 58.0, 198.0, 97.0, -60.0, 26.0], [-19.0, -23.0, -69.0, -45.0, 49.0, 102.0, -94.0], [50.0, 58.0, 85.0, 54.0, 151.0, 14.0, 17.0]]], [[[-18.0, 37.0, 39.0, -15.0, -109.0, -74.0, 52.0], [-62.0, 32.0, -86.0, 153.0, 176.0, -21.0, -26.0], [-59.0, -97.0, -163.0, -73.0, 114.0, 12.0, 37.0], [-24.0, -61.0, 86.0, 49.0, -9.0, 227.0, 14.0], [111.0, 82.0, 100.0, 24.0, -126.0, 213.0, -53.0]], [[-199.0, -25.0, -33.0, -28.0, -111.0, -126.0, -54.0], [-53.0, -209.0, 172.0, -91.0, 78.0, 29.0, 25.0], [59.0, 33.0, -73.0, 63.0, -174.0, 23.0, 83.0], [-30.0, -135.0, 9.0, -56.0, -9.0, 117.0, 128.0], [-61.0, 98.0, 2.0, -172.0, 368.0, -49.0, -87.0]], [[34.0, 153.0, -38.0, 101.0, -153.0, -27.0, -84.0], [-123.0, 67.0, 42.0, 223.0, 40.0, -140.0, -35.0], [206.0, -5.0, -73.0, -44.0, 13.0, 228.0, -18.0], [-129.0, 72.0, -27.0, -90.0, -59.0, 91.0, -43.0], [241.0, 275.0, 133.0, -66.0, 161.0, -31.0, -48.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[[-216.0], [-158.0], [-27.0], [-163.0], [-28.0], [-72.0], [-36.0]]]);
    expect(b.shape).toEqual([1, 7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[-50285.0], [39808.0], [-24921.0], [34142.0], [19158.0]], [[-40952.0], [-9478.0], [61436.0], [69.0], [63294.0]], [[-39719.0], [-38793.0], [11200.0], [-24920.0], [-33106.0]]], [[[5766.0], [52028.0], [732.0], [44788.0], [-10022.0]], [[-57948.0], [-25355.0], [-26764.0], [4519.0], [17113.0]], [[-16423.0], [-29195.0], [-10394.0], [83126.0], [-14096.0]]]], [[[[13255.0], [-7391.0], [8750.0], [-32802.0], [-5562.0]], [[-4720.0], [5269.0], [16856.0], [-26807.0], [19335.0]], [[-1944.0], [45605.0], [-35768.0], [11604.0], [-36909.0]]], [[[5942.0], [-16761.0], [38982.0], [-12083.0], [-53444.0]], [[66513.0], [49487.0], [-26028.0], [23915.0], [22030.0]], [[-37703.0], [-11281.0], [-50695.0], [28535.0], [-88887.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[[-77.0, -33.0, -139.0, 64.0, -24.0, -36.0, -131.0], [-190.0, -49.0, 269.0, -15.0, -203.0, -99.0, -175.0], [86.0, -57.0, 1.0, -10.0, 7.0, -106.0, -124.0], [-38.0, -8.0, -133.0, -166.0, -9.0, -61.0, 135.0], [45.0, 16.0, -55.0, 41.0, -7.0, -228.0, 48.0]], [[9.0, -7.0, 38.0, 151.0, -115.0, 133.0, -7.0], [209.0, 28.0, -264.0, -29.0, -14.0, 253.0, 29.0], [-151.0, 10.0, 99.0, -78.0, 19.0, -156.0, -90.0], [-53.0, 32.0, 25.0, -180.0, -35.0, 27.0, -9.0], [-21.0, -153.0, 29.0, 7.0, -148.0, -70.0, 260.0]], [[-29.0, -111.0, -82.0, -133.0, -255.0, 54.0, -37.0], [-42.0, -52.0, 179.0, 80.0, -6.0, -70.0, -109.0], [45.0, -145.0, -55.0, 60.0, 125.0, -62.0, 215.0], [-56.0, -9.0, 7.0, -104.0, -128.0, 236.0, -23.0], [-106.0, -230.0, -55.0, 85.0, 66.0, 150.0, -75.0]]], [[[13.0, 13.0, -25.0, -112.0, -27.0, -146.0, 23.0], [178.0, -137.0, -79.0, 26.0, -80.0, -17.0, -2.0], [91.0, -158.0, -2.0, -117.0, -183.0, 0.0, 61.0], [-167.0, -25.0, 9.0, 57.0, 58.0, -28.0, 22.0], [-92.0, 21.0, 27.0, -95.0, 181.0, 52.0, 24.0]], [[-189.0, 34.0, 35.0, 49.0, -63.0, -211.0, -136.0], [108.0, 42.0, -99.0, -19.0, 54.0, -101.0, -103.0], [-112.0, 120.0, -5.0, -83.0, 40.0, -364.0, 190.0], [-142.0, 67.0, -44.0, -91.0, -19.0, -9.0, -12.0], [84.0, 8.0, -59.0, -58.0, 202.0, 124.0, -8.0]], [[95.0, -24.0, 110.0, -32.0, 82.0, -19.0, 45.0], [-180.0, 232.0, 23.0, 124.0, -20.0, 102.0, 100.0], [26.0, 10.0, 44.0, 166.0, -82.0, -98.0, 0.0], [70.0, -28.0, 195.0, -52.0, 92.0, 132.0, -42.0], [248.0, 164.0, -127.0, 24.0, -7.0, 117.0, 66.0]]]], [[[[200.0, -55.0, -49.0, -11.0, 89.0, -37.0, 43.0], [41.0, -24.0, 44.0, -218.0, -13.0, -177.0, -87.0], [-116.0, 7.0, -7.0, -11.0, -1.0, -132.0, 58.0], [-202.0, -126.0, -103.0, -72.0, 53.0, -56.0, -68.0], [-39.0, -66.0, -125.0, 84.0, -108.0, 19.0, 96.0]], [[-55.0, -99.0, 37.0, 69.0, 7.0, -5.0, 57.0], [-29.0, -49.0, -50.0, -19.0, -102.0, 191.0, -79.0], [-26.0, 125.0, -72.0, -220.0, 12.0, 115.0, -78.0], [74.0, -36.0, 52.0, -89.0, 35.0, -228.0, -68.0], [185.0, -36.0, 21.0, -149.0, -263.0, 147.0, -51.0]], [[124.0, 4.0, -133.0, 33.0, 34.0, 73.0, 14.0], [-84.0, -85.0, -54.0, 68.0, -12.0, -21.0, 60.0], [103.0, 32.0, -18.0, 166.0, 106.0, -47.0, 95.0], [87.0, -21.0, -69.0, 61.0, -84.0, -58.0, 73.0], [-142.0, -46.0, -31.0, -149.0, 129.0, -0.0, -98.0]]], [[[-97.0, -10.0, 20.0, -25.0, 18.0, -8.0, 160.0], [-164.0, 22.0, -2.0, -147.0, -126.0, 119.0, 20.0], [-153.0, -59.0, 24.0, 125.0, -17.0, -37.0, 37.0], [181.0, -73.0, 62.0, 93.0, -90.0, 63.0, -2.0], [13.0, 122.0, -135.0, -65.0, -123.0, 29.0, -44.0]], [[-3.0, 6.0, 2.0, -95.0, -42.0, -132.0, 122.0], [-63.0, 2.0, -83.0, 112.0, -215.0, 68.0, -73.0], [253.0, -25.0, 220.0, 56.0, 42.0, -45.0, -105.0], [-58.0, -109.0, 102.0, 70.0, 168.0, -86.0, -15.0], [-162.0, -74.0, 109.0, 30.0, 2.0, 76.0, 43.0]], [[-4.0, -93.0, 33.0, 37.0, -157.0, 10.0, 104.0], [193.0, 81.0, 127.0, 225.0, -126.0, 111.0, -47.0], [10.0, -13.0, 101.0, 24.0, -47.0, 34.0, -57.0], [-18.0, -107.0, 21.0, 125.0, -26.0, -4.0, 52.0], [-162.0, -43.0, -1.0, -180.0, 95.0, 107.0, 42.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[[50.0, 49.0, 117.0, -107.0, -111.0, 226.0, 133.0, 126.0, 19.0, 131.0, 65.0], [21.0, -77.0, 13.0, 20.0, -34.0, -129.0, 15.0, 29.0, -23.0, -173.0, 30.0], [66.0, 164.0, -165.0, -79.0, -87.0, -57.0, 93.0, -16.0, -48.0, 95.0, 92.0], [26.0, -114.0, -43.0, 174.0, -105.0, 4.0, 126.0, -8.0, 7.0, -93.0, -32.0], [56.0, -232.0, -102.0, -74.0, 90.0, 22.0, -53.0, -41.0, 214.0, -56.0, -15.0], [7.0, -23.0, -28.0, -14.0, -121.0, -39.0, -13.0, -55.0, 21.0, -9.0, -2.0], [33.0, 74.0, -27.0, 124.0, -119.0, -34.0, -6.0, -105.0, -109.0, 54.0, -99.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[-17972.0, -34622.0, 17738.0, 15732.0, 32827.0, 364.0, -13073.0, 7772.0, 14803.0, -28941.0, -7430.0], [-11001.0, 76712.0, -38404.0, -9803.0, 15462.0, -46667.0, 10218.0, 2598.0, -41946.0, 13346.0, 31976.0], [-1533.0, 1545.0, 15188.0, -26571.0, 21567.0, 35196.0, 11167.0, 27810.0, 15615.0, 16018.0, 16675.0], [-11638.0, 9347.0, 23514.0, 3789.0, 23997.0, -3048.0, -37999.0, -12015.0, -13238.0, 7552.0, -22742.0], [-382.0, -2301.0, 18587.0, 16646.0, 16187.0, 18511.0, 9323.0, 14473.0, -8104.0, -875.0, -7158.0]], [[997.0, 13101.0, -3606.0, 27949.0, -45532.0, -6104.0, 28060.0, -2750.0, -21489.0, -3178.0, 1191.0], [-5196.0, -32330.0, 63185.0, -4903.0, -33462.0, 47393.0, -2710.0, 15216.0, 14952.0, 225.0, -12102.0], [-5832.0, 9479.0, -25658.0, -15418.0, 47294.0, -31809.0, -18993.0, -2445.0, 2203.0, -9372.0, 11026.0], [-7076.0, 26392.0, 887.0, -25888.0, 16174.0, -19768.0, -25366.0, -3815.0, -10145.0, 7867.0, 6937.0], [-2365.0, 69896.0, 504.0, 42286.0, -31515.0, 4000.0, 5685.0, -24985.0, -59705.0, 48780.0, -26891.0]], [[-27774.0, 64020.0, 39910.0, -2255.0, 3011.0, 5449.0, -16871.0, 6873.0, -44396.0, 31779.0, -1123.0], [6279.0, 17118.0, -33050.0, -8859.0, 3358.0, -6363.0, 22243.0, 5237.0, 1493.0, 8139.0, 20639.0], [10796.0, -14154.0, -6944.0, 25348.0, -8413.0, 30108.0, -854.0, -22425.0, 9263.0, 25343.0, -31441.0], [-11506.0, 33519.0, 3717.0, -9521.0, -20506.0, -23548.0, -16182.0, -11914.0, -21850.0, 8360.0, 3787.0], [-9279.0, -30506.0, -18879.0, 9593.0, 12161.0, 7341.0, -16951.0, -22907.0, 31960.0, 3678.0, -15435.0]]], [[[-5414.0, 19628.0, 16852.0, -11750.0, 24549.0, 6556.0, -11322.0, 10033.0, -10987.0, 11563.0, 939.0], [-3180.0, 22154.0, 39652.0, -5111.0, -15862.0, 61479.0, 22021.0, 23936.0, -6752.0, 41621.0, 792.0], [-10177.0, 76605.0, 30973.0, -11991.0, -15999.0, 34494.0, 4138.0, 8950.0, -41171.0, 63488.0, 1441.0], [-3021.0, -22464.0, -29526.0, 25404.0, 18609.0, -33182.0, -17409.0, -25515.0, 6795.0, -23806.0, -15593.0], [6445.0, -32279.0, -31427.0, -19545.0, 24266.0, -24282.0, -31793.0, -23456.0, 33018.0, -13593.0, -5021.0]], [[-14645.0, -2320.0, -13547.0, 17416.0, 47678.0, -37432.0, -8300.0, 4688.0, -8799.0, -33790.0, 5218.0], [-1828.0, -29839.0, 30435.0, -21555.0, 26530.0, 33186.0, 2462.0, 30718.0, 26367.0, -8433.0, 9369.0], [394.0, 7066.0, -6168.0, 26033.0, 42536.0, -32223.0, -22547.0, -11458.0, -25023.0, -16892.0, -20166.0], [-12486.0, -5232.0, -2056.0, 4220.0, 27674.0, -38250.0, -32243.0, -11983.0, -5711.0, -25413.0, -6865.0], [10882.0, -49872.0, -1699.0, -31935.0, 5755.0, 20963.0, -13773.0, -2038.0, 50542.0, -3451.0, -358.0]], [[16618.0, 12934.0, -15018.0, -25125.0, -11615.0, 19183.0, 14104.0, 2728.0, 9097.0, 28032.0, 10952.0], [3508.0, -27354.0, -30687.0, 56111.0, -28971.0, -79241.0, -3563.0, -32602.0, -22030.0, -67461.0, -16396.0], [3452.0, 10074.0, -118.0, 30266.0, -20006.0, 4760.0, 34236.0, 10286.0, -20292.0, -4108.0, 2152.0], [19120.0, 16006.0, -34059.0, -46367.0, -21017.0, 6413.0, 14133.0, -1318.0, 19288.0, 28767.0, 25828.0], [10691.0, -20223.0, 46727.0, -1983.0, -47216.0, 35266.0, 25111.0, 24766.0, 969.0, -7278.0, 1925.0]]]], [[[[11469.0, -9362.0, 22040.0, -21279.0, -7542.0, 56983.0, 15338.0, 18348.0, 20922.0, 29754.0, 1676.0], [-6056.0, 36574.0, 15230.0, -53623.0, 45927.0, 18557.0, -14771.0, 24913.0, 677.0, 31600.0, 22131.0], [-5467.0, 1443.0, -9621.0, 20305.0, 23382.0, -23610.0, -15939.0, -13002.0, -11414.0, -11673.0, -13085.0], [-21084.0, -24920.0, -7183.0, 3133.0, 62865.0, -18153.0, -49080.0, -18835.0, 21078.0, -13889.0, -18033.0], [-12149.0, 4818.0, 19484.0, 46974.0, -14815.0, 780.0, -2317.0, -12197.0, -25812.0, -2317.0, -26625.0]], [[1645.0, 5839.0, -18907.0, 19608.0, -6541.0, -3081.0, 2687.0, -16942.0, -4881.0, 9751.0, -11087.0], [-13255.0, 9743.0, 12226.0, -2155.0, -11660.0, -4466.0, -8239.0, -2151.0, -6363.0, 1422.0, 1622.0], [-10244.0, -8828.0, 17585.0, -39480.0, 24447.0, -20346.0, -37662.0, 4634.0, 12032.0, -17330.0, 9788.0], [2182.0, 17164.0, 8087.0, -36062.0, 36661.0, 30022.0, 4441.0, 26405.0, 9229.0, 25559.0, 18025.0], [-9376.0, 86128.0, 48206.0, -37020.0, -40881.0, 34876.0, 19578.0, 31175.0, -45344.0, 56966.0, 26345.0]], [[1241.0, -28337.0, 29196.0, 1259.0, -13233.0, 32646.0, 5506.0, 10725.0, 16162.0, -1957.0, -7154.0], [-6620.0, -6472.0, -4755.0, 32008.0, 4093.0, -6154.0, -8352.0, -17382.0, -6122.0, -3652.0, -20872.0], [17692.0, -35774.0, -3762.0, 24519.0, -24463.0, 21775.0, 27844.0, 1130.0, 14589.0, -9574.0, -10214.0], [-1760.0, 13834.0, 26889.0, 22416.0, -18574.0, 24480.0, 17293.0, 9938.0, -21276.0, 11970.0, -9126.0], [-9996.0, -28694.0, -16202.0, -30901.0, 58940.0, -18817.0, -47482.0, -12537.0, 37093.0, -12248.0, -927.0]]], [[[1842.0, 9995.0, -19636.0, 22869.0, -4460.0, -26604.0, -16151.0, -29730.0, -16504.0, 952.0, -20059.0], [-17255.0, 34675.0, -3271.0, 2706.0, 4946.0, -48469.0, -35179.0, -22297.0, -31200.0, -4744.0, -5808.0], [-4045.0, -5735.0, -26232.0, 41409.0, 2320.0, -28024.0, -2092.0, -23526.0, -10275.0, -15898.0, -16841.0], [9362.0, 33339.0, 13469.0, -4013.0, -48253.0, 42792.0, 44425.0, 19388.0, -14926.0, 37946.0, 13725.0], [-15525.0, 1126.0, 41099.0, 3644.0, 3636.0, -7706.0, -10780.0, 15924.0, -17451.0, -21932.0, 308.0]], [[-1612.0, 32357.0, 8168.0, 3837.0, 7604.0, -1870.0, -8883.0, -3304.0, -26014.0, 17722.0, -7975.0], [-19647.0, 13293.0, 23531.0, 38732.0, -16505.0, -14217.0, 8993.0, 5292.0, -33100.0, -19414.0, -4939.0], [26673.0, 27539.0, -9621.0, -50705.0, -30533.0, 54336.0, 59779.0, 38963.0, 14702.0, 45543.0, 43998.0], [11674.0, -23809.0, -42366.0, -4940.0, 21231.0, 2979.0, 1261.0, -13244.0, 32780.0, 4995.0, -759.0], [383.0, 13186.0, -42684.0, 16583.0, -6268.0, -37541.0, -10091.0, -33319.0, -9061.0, 671.0, -8121.0]], [[-4303.0, 52049.0, 4213.0, 26773.0, -30866.0, 1980.0, 13371.0, -9058.0, -43986.0, 29577.0, -9159.0], [17753.0, 21599.0, 4017.0, 12028.0, -78029.0, 21327.0, 72562.0, 26831.0, -22227.0, 5929.0, 25780.0], [3242.0, 21223.0, -11315.0, -9199.0, -13536.0, -2146.0, 15943.0, 5117.0, -7322.0, 10170.0, 15064.0], [1721.0, 6523.0, -10977.0, 28305.0, -17360.0, 6854.0, 14822.0, -10881.0, -9330.0, 10823.0, -11198.0], [-6294.0, -5664.0, -25428.0, -18087.0, 29036.0, -35239.0, -51642.0, -34393.0, 14698.0, -1153.0, -11949.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (2, 7)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 2x7)');
});
test("matmul (2, 2, 3, 5, 7) x (2, 7, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([2, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2');
});
test("matmul (2, 2, 3, 5, 7) x (2, 7, 11)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([2, 7, 11]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2');
});
test("matmul (2, 2, 3, 5, 7) x (1,)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 60, 60x7,1');
});
test("matmul (2, 2, 3, 5, 7) x (1, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 1x1)');
});
test("matmul (2, 2, 3, 5, 7) x (1, 11)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([1, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 1x11)');
});
test("matmul (2, 2, 3, 5, 7) x (2,)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 60, 60x7,2');
});
test("matmul (2, 2, 3, 5, 7) x (2, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 2x1)');
});
test("matmul (2, 2, 3, 5, 7) x (2, 11)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([2, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 2x11)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (7,)", async () => {
    const a = tensor([[[[[[2.0, -0.0, -180.0, 42.0, -152.0, -2.0, 99.0], [-105.0, -43.0, 81.0, 111.0, -42.0, -4.0, 28.0], [11.0, -56.0, 32.0, 144.0, -53.0, 104.0, 88.0], [121.0, 144.0, -33.0, 125.0, 125.0, 89.0, 73.0], [-35.0, 155.0, 62.0, 86.0, 15.0, -108.0, 15.0]], [[-173.0, -119.0, 120.0, 35.0, 47.0, 174.0, 133.0], [51.0, 188.0, -91.0, 69.0, -47.0, 99.0, 17.0], [-73.0, 167.0, -132.0, 6.0, -190.0, -16.0, 65.0], [-105.0, 19.0, -50.0, 4.0, 129.0, 26.0, -142.0], [79.0, 2.0, -230.0, -7.0, 107.0, 247.0, -45.0]], [[-204.0, 137.0, 27.0, -56.0, -152.0, -42.0, -86.0], [124.0, -69.0, -51.0, -90.0, -109.0, 1.0, -3.0], [-59.0, -88.0, -65.0, 214.0, 72.0, 117.0, 50.0], [-13.0, -174.0, 61.0, 3.0, -76.0, 174.0, 43.0], [138.0, 88.0, 13.0, 47.0, -65.0, -149.0, -5.0]]], [[[-172.0, -80.0, -48.0, 139.0, 56.0, 141.0, -76.0], [18.0, -21.0, 3.0, 90.0, -45.0, 13.0, 147.0], [32.0, 66.0, 91.0, 25.0, -70.0, 76.0, 93.0], [164.0, -45.0, 38.0, -121.0, -79.0, -34.0, 90.0], [59.0, 59.0, -123.0, 133.0, 77.0, 132.0, -165.0]], [[-2.0, -104.0, 37.0, -33.0, -10.0, -69.0, 34.0], [79.0, 91.0, -51.0, -115.0, -111.0, -9.0, 4.0], [2.0, 67.0, 122.0, 130.0, 32.0, -116.0, 34.0], [168.0, -70.0, -127.0, 142.0, 9.0, 44.0, -1.0], [-59.0, 40.0, -67.0, -108.0, 1.0, 19.0, 3.0]], [[21.0, -6.0, -104.0, 19.0, -21.0, 162.0, 258.0], [11.0, 123.0, -62.0, -27.0, -296.0, -114.0, 47.0], [-45.0, -7.0, -253.0, -138.0, 31.0, 80.0, -34.0], [-156.0, -81.0, -7.0, 59.0, -87.0, -10.0, -39.0], [-122.0, -9.0, 28.0, -20.0, -81.0, 29.0, 86.0]]]]], [[[[[48.0, 38.0, 18.0, 57.0, 55.0, -94.0, 140.0], [21.0, 107.0, 19.0, -117.0, 117.0, 2.0, 125.0], [-30.0, 79.0, 147.0, 189.0, -256.0, 49.0, 39.0], [37.0, 15.0, 54.0, -28.0, -46.0, 37.0, -88.0], [-5.0, 45.0, 19.0, 29.0, -170.0, 210.0, -158.0]], [[18.0, 64.0, 185.0, -59.0, -51.0, 58.0, -249.0], [-224.0, 95.0, 163.0, -80.0, -20.0, -131.0, -90.0], [64.0, 10.0, -81.0, -20.0, -0.0, 91.0, -39.0], [-13.0, 89.0, -42.0, -107.0, -241.0, -60.0, -18.0], [57.0, -235.0, 102.0, -143.0, -208.0, 15.0, 18.0]], [[-59.0, 140.0, -227.0, -67.0, -31.0, 154.0, 226.0], [-59.0, 88.0, -182.0, -22.0, 63.0, -132.0, -32.0], [163.0, 74.0, -108.0, 116.0, 37.0, -148.0, -19.0], [78.0, -65.0, 142.0, -79.0, 74.0, -59.0, -91.0], [-93.0, -10.0, 74.0, 12.0, 116.0, 41.0, 50.0]]], [[[-51.0, -151.0, -15.0, -45.0, -47.0, -52.0, -123.0], [10.0, 133.0, -167.0, -122.0, 114.0, -14.0, 21.0], [-22.0, -67.0, -42.0, 75.0, -148.0, 70.0, 126.0], [38.0, -160.0, -60.0, -117.0, 115.0, 50.0, 7.0], [19.0, 100.0, 162.0, 82.0, 107.0, -8.0, -24.0]], [[15.0, 133.0, -28.0, -38.0, -69.0, 65.0, -34.0], [-40.0, 2.0, -159.0, -10.0, 9.0, -81.0, -78.0], [-235.0, 195.0, -175.0, -21.0, -59.0, 46.0, -135.0], [48.0, -4.0, -111.0, 3.0, 146.0, 108.0, 66.0], [-47.0, 124.0, 21.0, -79.0, -24.0, -22.0, -66.0]], [[71.0, 37.0, -126.0, 104.0, -37.0, -116.0, -165.0], [67.0, 18.0, 169.0, -82.0, 54.0, 43.0, 61.0], [78.0, -3.0, -8.0, -11.0, 85.0, 60.0, 1.0], [123.0, 64.0, -43.0, -173.0, 20.0, -53.0, -109.0], [161.0, 141.0, 38.0, -88.0, 149.0, 28.0, -78.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([-63.0, -7.0, 89.0, 12.0, 39.0, -4.0, 211.0]);
    expect(b.shape).toEqual([7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[[-673.0, 19743.0, 20360.0, 9854.0, 11852.0], [52032.0, -10442.0, -1877.0, -22955.0, -31855.0], [-10282.0, -17836.0, 14006.0, 12915.0, -10583.0]], [[-5624.0, 29570.0, 22510.0, 7958.0, -45821.0], [10811.0, -14982.0, 20709.0, -19729.0, -3226.0], [42662.0, -8567.0, -27574.0, -1102.0, 24872.0]]]], [[[[31057.0, 29145.0, 14737.0, -18476.0, -38769.0], [-40585.0, 7748.0, -20144.0, -17783.0, 1042.0], [27591.0, -17128.0, -20981.0, -8848.0, 27569.0]], [[-25183.0, -8955.0, 19551.0, -2256.0, 12646.0], [-14949.0, -27548.0, -33357.0, 6349.0, -11760.0], [-50492.0, 24515.0, -2451.0, -36107.0, -19563.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (7, 1)", async () => {
    const a = tensor([[[[[[-49.0, -110.0, 53.0, 15.0, -73.0, 50.0, -126.0], [-138.0, -119.0, -14.0, -14.0, 64.0, 51.0, -45.0], [-85.0, 24.0, -109.0, 56.0, 77.0, -67.0, 38.0], [-42.0, -2.0, 113.0, 130.0, 172.0, -90.0, -55.0], [-163.0, -107.0, -9.0, 100.0, 58.0, 61.0, -171.0]], [[79.0, 59.0, 25.0, -22.0, -98.0, -74.0, -106.0], [-23.0, 18.0, -110.0, -64.0, -46.0, 216.0, -31.0], [-169.0, 8.0, -66.0, -47.0, 89.0, -18.0, 8.0], [189.0, -23.0, 4.0, 3.0, 29.0, -83.0, 0.0], [-24.0, 28.0, -38.0, 72.0, -178.0, 56.0, 44.0]], [[126.0, -20.0, -180.0, -103.0, 103.0, -83.0, 1.0], [-21.0, 3.0, -24.0, 59.0, 21.0, 86.0, -117.0], [87.0, -332.0, 85.0, -29.0, -151.0, 69.0, -65.0], [-36.0, 25.0, 59.0, 44.0, 40.0, -75.0, 28.0], [39.0, -193.0, -52.0, -75.0, -8.0, -254.0, 39.0]]], [[[-26.0, -69.0, 12.0, 87.0, -18.0, -33.0, -40.0], [124.0, 31.0, -219.0, -66.0, -8.0, 20.0, -151.0], [21.0, 4.0, 38.0, -60.0, 58.0, -101.0, 104.0], [213.0, 248.0, -102.0, -240.0, 82.0, -85.0, 37.0], [-19.0, -67.0, 125.0, -128.0, 146.0, 21.0, -14.0]], [[-192.0, -25.0, -247.0, 64.0, -92.0, 93.0, -15.0], [-82.0, 58.0, 113.0, -126.0, 52.0, 209.0, -246.0], [47.0, 34.0, -189.0, 48.0, -33.0, 40.0, -4.0], [-33.0, -33.0, -75.0, 113.0, 112.0, 17.0, 10.0], [-160.0, -9.0, 25.0, 10.0, -46.0, 33.0, -8.0]], [[-12.0, 135.0, -98.0, 12.0, 123.0, 41.0, -22.0], [28.0, -52.0, -14.0, -30.0, -161.0, -196.0, -213.0], [-130.0, 284.0, 54.0, 28.0, -58.0, -52.0, -49.0], [170.0, 183.0, -63.0, -219.0, 144.0, 70.0, 139.0], [27.0, -43.0, -124.0, -125.0, 38.0, 23.0, 154.0]]]]], [[[[[-157.0, -192.0, -85.0, 7.0, 10.0, 162.0, 21.0], [-3.0, 116.0, -85.0, 122.0, 27.0, -18.0, -66.0], [258.0, 97.0, 60.0, 61.0, -199.0, -29.0, -130.0], [79.0, -39.0, -160.0, -175.0, -49.0, -224.0, 52.0], [32.0, -137.0, -249.0, -35.0, 33.0, -1.0, -102.0]], [[-32.0, 44.0, 63.0, 73.0, -72.0, -52.0, -190.0], [-47.0, -10.0, 70.0, -48.0, -87.0, 138.0, 33.0], [-101.0, 136.0, 76.0, 6.0, -97.0, 37.0, -25.0], [103.0, 59.0, 116.0, -22.0, -27.0, 143.0, 29.0], [-63.0, -84.0, 93.0, 96.0, -58.0, -57.0, 49.0]], [[-63.0, 87.0, -137.0, 88.0, -2.0, 185.0, 1.0], [175.0, 102.0, 82.0, -60.0, -164.0, 75.0, -279.0], [141.0, 120.0, -22.0, 12.0, -30.0, 2.0, 13.0], [-36.0, -2.0, 70.0, -29.0, 65.0, -28.0, -20.0], [-52.0, -32.0, -32.0, -34.0, 14.0, 120.0, 42.0]]], [[[-35.0, -92.0, -135.0, -135.0, 236.0, -122.0, -206.0], [-40.0, -21.0, 22.0, -93.0, -83.0, 51.0, 113.0], [-99.0, -99.0, -33.0, -261.0, 92.0, 24.0, 50.0], [251.0, 58.0, -7.0, -105.0, 157.0, -102.0, 24.0], [50.0, -60.0, -180.0, 117.0, 46.0, 123.0, -196.0]], [[-165.0, 7.0, 93.0, 21.0, 163.0, 68.0, -175.0], [-113.0, -33.0, 2.0, 24.0, 182.0, -80.0, 103.0], [-36.0, 58.0, 205.0, -51.0, -99.0, -66.0, -38.0], [-38.0, -80.0, -11.0, -211.0, -89.0, 18.0, -37.0], [75.0, 42.0, -21.0, 239.0, 149.0, 63.0, 305.0]], [[-161.0, 60.0, 106.0, -23.0, 64.0, 45.0, 152.0], [-28.0, 142.0, -141.0, -66.0, -221.0, -23.0, 69.0], [109.0, -99.0, 15.0, 65.0, -65.0, -59.0, -185.0], [-27.0, -29.0, -12.0, -10.0, 62.0, 41.0, 139.0], [145.0, -91.0, 148.0, -14.0, -53.0, 194.0, -34.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[31.0], [-46.0], [55.0], [100.0], [209.0], [141.0], [-71.0]]);
    expect(b.shape).toEqual([7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[[8695.0], [22788.0], [-186.0], [45168.0], [42238.0]], [[-24480.0], [9052.0], [1558.0], [1795.0], [-29352.0]], [[-5621.0], [28613.0], [2529.0], [1176.0], [-40528.0]]], [[[6153.0], [-4358.0], [-12946.0], [-31889.0], [31037.0]], [[-17037.0], [46208.0], [-6675.0], [32765.0], [-6564.0]], [[22278.0], [-46672.0], [-27299.0], [1584.0], [-16254.0]]]]], [[[[[23431.0], [9887.0], [-23514.0], [-67574.0], [4097.0]], [[-1141.0], [-3015.0], [-17888.0], [17120.0], [-7012.0]], [[20906.0], [-4649.0], [-8070.0], [10983.0], [11564.0]]], [[[28970.0], [-26543.0], [-7368.0], [10955.0], [46983.0]], [[57858.0], [19970.0], [-24908.0], [-32639.0], [41507.0]], [[4708.0], [-76086.0], [6489.0], [7707.0], [34112.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[[[73.0, 31.0, -131.0, 112.0, 11.0, 14.0, -28.0], [-33.0, -19.0, 38.0, 138.0, -85.0, 23.0, 131.0], [-44.0, -15.0, 72.0, -84.0, 109.0, -229.0, -32.0], [-109.0, -44.0, -37.0, -92.0, -78.0, -222.0, -56.0], [-50.0, -83.0, 10.0, 114.0, 126.0, -154.0, 78.0]], [[-12.0, -130.0, -155.0, -58.0, -112.0, -88.0, 45.0], [6.0, 33.0, 83.0, 47.0, 48.0, -161.0, 43.0], [-110.0, -160.0, 266.0, 37.0, -36.0, -184.0, -43.0], [-126.0, 7.0, 93.0, -0.0, 30.0, -209.0, 135.0], [-85.0, 38.0, 103.0, -132.0, 9.0, -166.0, -50.0]], [[167.0, 19.0, 93.0, -101.0, 27.0, -28.0, 35.0], [-73.0, 112.0, -16.0, -64.0, 19.0, 247.0, 72.0], [71.0, 50.0, -133.0, -45.0, -23.0, 47.0, 147.0], [9.0, -12.0, 10.0, -181.0, 10.0, -63.0, -13.0], [-41.0, 7.0, 159.0, 23.0, -94.0, -10.0, 96.0]]], [[[-22.0, 58.0, -43.0, 49.0, 72.0, 8.0, 210.0], [6.0, 85.0, 114.0, 14.0, 52.0, -64.0, -46.0], [43.0, 43.0, -37.0, -125.0, -18.0, 71.0, 76.0], [128.0, -22.0, -34.0, 73.0, -231.0, -52.0, -44.0], [-61.0, 49.0, -140.0, 64.0, -68.0, 30.0, -70.0]], [[134.0, -113.0, -65.0, 146.0, 34.0, 210.0, 183.0], [-41.0, 97.0, -208.0, -186.0, -30.0, 196.0, 11.0], [21.0, 121.0, -47.0, -86.0, -180.0, 16.0, -156.0], [126.0, -42.0, -73.0, 122.0, 168.0, 39.0, -47.0], [84.0, -21.0, 75.0, 16.0, -107.0, -80.0, -168.0]], [[-7.0, -56.0, 52.0, -45.0, 187.0, 34.0, -2.0], [-219.0, 15.0, -101.0, -301.0, 327.0, -203.0, -33.0], [-2.0, -34.0, -60.0, -55.0, 25.0, -99.0, 146.0], [35.0, -87.0, -170.0, 6.0, 27.0, -42.0, -21.0], [16.0, 130.0, 32.0, 42.0, -41.0, 121.0, -116.0]]]]], [[[[[27.0, 52.0, -75.0, -23.0, 93.0, -156.0, 199.0], [-112.0, 87.0, 18.0, -123.0, 164.0, 38.0, 12.0], [-62.0, -136.0, -178.0, -102.0, -17.0, -53.0, 96.0], [88.0, -141.0, -4.0, 60.0, 28.0, -121.0, 43.0], [-85.0, 64.0, 18.0, 102.0, -6.0, 76.0, 25.0]], [[210.0, 59.0, 40.0, -29.0, 29.0, 155.0, -179.0], [97.0, -139.0, -143.0, 26.0, 39.0, 2.0, 85.0], [-130.0, 34.0, 78.0, 76.0, -171.0, -80.0, -119.0], [66.0, 156.0, -53.0, 158.0, 99.0, 50.0, -36.0], [-63.0, -169.0, -64.0, -35.0, 17.0, 53.0, -110.0]], [[153.0, -64.0, 28.0, -122.0, -121.0, -81.0, 37.0], [5.0, 20.0, -29.0, -22.0, 30.0, -53.0, -95.0], [36.0, -20.0, 22.0, 133.0, -54.0, -195.0, -44.0], [-155.0, 140.0, -19.0, 5.0, -122.0, -46.0, -18.0], [-42.0, -46.0, -180.0, -24.0, -16.0, -51.0, 21.0]]], [[[-24.0, -135.0, -91.0, 46.0, -117.0, 139.0, -47.0], [68.0, 66.0, 80.0, -29.0, -29.0, 113.0, -13.0], [41.0, -18.0, -105.0, -23.0, 16.0, 5.0, -70.0], [-4.0, 54.0, -30.0, -24.0, 109.0, -50.0, 90.0], [27.0, -98.0, 34.0, -68.0, 22.0, -68.0, 219.0]], [[-9.0, -135.0, 58.0, -13.0, 175.0, 84.0, -18.0], [194.0, 56.0, 0.0, -22.0, -66.0, 81.0, 113.0], [-7.0, 255.0, -34.0, -22.0, -126.0, 110.0, 70.0], [29.0, -79.0, -122.0, -45.0, 41.0, 115.0, -39.0], [150.0, 185.0, -126.0, -63.0, -164.0, 29.0, 126.0]], [[187.0, 14.0, -47.0, 172.0, -147.0, 36.0, -19.0], [90.0, 23.0, 4.0, -75.0, 141.0, 167.0, -27.0], [54.0, -24.0, -36.0, -106.0, 165.0, 103.0, 24.0], [30.0, -3.0, 173.0, -149.0, 94.0, 42.0, 98.0], [-3.0, 35.0, -59.0, -57.0, 134.0, 68.0, 30.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[72.0, 105.0, 17.0, -136.0, -120.0, 83.0, -110.0, -49.0, 66.0, 69.0, 24.0], [58.0, 48.0, -28.0, -52.0, 132.0, -58.0, 118.0, -5.0, -56.0, 91.0, 31.0], [-141.0, 12.0, 208.0, -45.0, -21.0, -38.0, -32.0, 175.0, 92.0, 73.0, -53.0], [-33.0, -4.0, -28.0, -30.0, -12.0, 55.0, -5.0, -37.0, -7.0, 39.0, 27.0], [12.0, -126.0, 27.0, -16.0, 5.0, -145.0, -12.0, -30.0, 86.0, 133.0, -75.0], [-129.0, 10.0, -6.0, -62.0, 188.0, 41.0, 99.0, 64.0, 26.0, -163.0, 150.0], [9.0, -61.0, 49.0, 139.0, -67.0, 134.0, -34.0, 27.0, -74.0, -116.0, 59.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[[19903.0, 7595.0, -31170.0, -13941.0, 1302.0, 10626.0, 1466.0, -30991.0, -6372.0, 5092.0, 12303.0], [-16198.0, -1524.0, 7997.0, 17769.0, -5880.0, 35331.0, -1675.0, 10815.0, -14990.0, -26100.0, 17885.0], [19143.0, -18212.0, 19749.0, 14050.0, -37567.0, -39620.0, -21705.0, -851.0, 10936.0, 53115.0, -52018.0], [25051.0, -2609.0, -9259.0, 28765.0, -29221.0, -15445.0, -10696.0, -10890.0, -15826.0, 14494.0, -35257.0], [8494.0, -31744.0, 8510.0, 25620.0, -40082.0, -7578.0, -24594.0, -11133.0, 2530.0, 26985.0, -29173.0]], [[25778.0, 1359.0, -27471.0, 30610.0, -31888.0, 27906.0, -17668.0, -24798.0, -22616.0, -32007.0, 186.0], [10824.0, -7259.0, 19495.0, 7514.0, -31580.0, -9784.0, -17634.0, 1744.0, 2615.0, 38948.0, -27176.0], [-33010.0, -10867.0, 54927.0, 16207.0, -45841.0, -16009.0, -31799.0, 39514.0, 21215.0, 28903.0, -48136.0], [6757.0, -25883.0, 25685.0, 43830.0, -34096.0, -9227.0, -13931.0, 11783.0, -12996.0, 21129.0, -33371.0], [6989.0, -5081.0, 21400.0, 12107.0, -13176.0, -35244.0, -3644.0, 14640.0, 2820.0, 34019.0, -38410.0]], [[7597.0, 14150.0, 27091.0, -18686.0, -25747.0, 3297.0, -22885.0, 10077.0, 18225.0, 20197.0, -7219.0], [-25379.0, -6541.0, -3354.0, 1134.0, 66355.0, 1553.0, 43855.0, 19767.0, -9386.0, -44595.0, 40713.0], [23234.0, 2840.0, -20297.0, 12966.0, 285.0, 30532.0, 2502.0, -17672.0, -21669.0, -29787.0, 26536.0], [12645.0, 116.0, 7648.0, 6319.0, -11625.0, -14667.0, -7736.0, 3383.0, 3637.0, 6307.0, -16540.0], [-24698.0, 3735.0, 33761.0, 12835.0, -6553.0, 17498.0, -2993.0, 33720.0, -4079.0, -11696.0, 2641.0]]], [[[7948.0, -22040.0, -128.0, 27983.0, -1595.0, 17167.0, 3183.0, -4528.0, -18139.0, -13556.0, 13062.0], [-2708.0, 1636.0, 20576.0, -14044.0, -752.0, -24322.0, 256.0, 11815.0, 12238.0, 39701.0, -19099.0], [6241.0, 4977.0, -1857.0, 3781.0, 10959.0, 11311.0, 6814.0, 2964.0, -7425.0, -23479.0, 17435.0], [13865.0, 42954.0, -14405.0, -16120.0, -26409.0, 42674.0, -16833.0, -12399.0, -11921.0, -9948.0, 13092.0], [10762.0, 7149.0, -38767.0, -374.0, 25950.0, 2645.0, 22818.0, -22054.0, -19986.0, -13288.0, 14673.0]], [[-17594.0, -6065.0, -3541.0, -1930.0, -3994.0, 56378.0, -12564.0, -5417.0, 3012.0, -51024.0, 46847.0], [12595.0, 3668.0, -42916.0, 5329.0, 60285.0, 2505.0, 42932.0, -14253.0, -24270.0, -53654.0, 40324.0], [12367.0, 40149.0, -22999.0, -24249.0, 28031.0, -2367.0, 22950.0, -4465.0, -12632.0, -2777.0, 11120.0], [9465.0, -8061.0, -13283.0, -26966.0, -9274.0, -6681.0, -13647.0, -27066.0, 22038.0, 25740.0, -638.0], [1251.0, 31578.0, 6527.0, -30867.0, -18938.0, -4057.0, -15122.0, 2076.0, 14658.0, 28281.0, -16065.0]], [[-11759.0, -25719.0, 18272.0, -2504.0, 357.0, -27773.0, -6087.0, 7900.0, 24887.0, 16023.0, -14918.0], [39090.0, -63502.0, -8293.0, 45346.0, -325.0, -91924.0, 7698.0, -19575.0, 2807.0, 47550.0, -64487.0], [22544.0, -15388.0, -1599.0, 32422.0, -30597.0, 12941.0, -16662.0, -11341.0, -14591.0, -7231.0, -7518.0], [26799.0, -5106.0, -32545.0, 6487.0, -18540.0, 6290.0, -12474.0, -35317.0, -5716.0, -4805.0, -2249.0], [-14351.0, 21588.0, -5405.0, -34606.0, 44379.0, -9756.0, 28761.0, 8454.0, 4630.0, 5188.0, 18233.0]]]]], [[[[[39325.0, -20894.0, -2755.0, 33534.0, -36721.0, 7595.0, -17645.0, -21258.0, -18653.0, 14936.0, -13020.0], [-4323.0, -27892.0, 7636.0, 10276.0, 33182.0, -42405.0, 24011.0, 10590.0, 4457.0, 10932.0, -10158.0], [23609.0, -19010.0, -26851.0, 43476.0, -22031.0, 17052.0, -11329.0, -23948.0, -22082.0, -38384.0, -35.0], [13074.0, -5177.0, 6521.0, 6775.0, -55297.0, 15675.0, -40267.0, -13950.0, 8996.0, 13748.0, -18140.0], [-17963.0, -6054.0, -1742.0, 3221.0, 29629.0, 1495.0, 22562.0, 8940.0, -8642.0, -10835.0, 15069.0]], [[-7399.0, 34293.0, 2132.0, -67513.0, 23374.0, -10943.0, 3810.0, 1705.0, 34209.0, 21004.0, 14480.0], [19202.0, -8386.0, -19725.0, 10758.0, -32421.0, 28794.0, -25786.0, -28792.0, -2036.0, -20380.0, 8690.0], [-13697.0, 16619.0, 966.0, 1277.0, 9616.0, -5977.0, 13614.0, 13835.0, -11820.0, 6883.0, -10344.0], [10473.0, 3372.0, -18085.0, -29131.0, 24196.0, -9995.0, 17040.0, -19877.0, 2116.0, 30236.0, 11446.0], [-11782.0, -10257.0, -13920.0, 2438.0, 4435.0, -9952.0, -2006.0, -6061.0, 10643.0, -19381.0, -4119.0]], [[16712.0, 25996.0, 12665.0, -2979.0, -44244.0, 27819.0, -32493.0, 1682.0, 1862.0, -5163.0, -3982.0], [12677.0, 2710.0, -9418.0, -10154.0, -536.0, -20106.0, 471.0, -11463.0, 4928.0, 22839.0, -14122.0], [18052.0, 10090.0, -420.0, -1998.0, -43000.0, 4566.0, -24850.0, -14783.0, -1869.0, 37164.0, -25127.0], [3782.0, 6207.0, -14547.0, 16807.0, 29367.0, -6596.0, 31675.0, 3615.0, -30209.0, -5787.0, 2950.0], [27056.0, -8457.0, -35291.0, 23261.0, -8039.0, 7745.0, -499.0, -30541.0, -20844.0, -17411.0, 1247.0]]], [[[-18003.0, 8723.0, -23140.0, -280.0, 15115.0, 28192.0, 6155.0, -4639.0, -5688.0, -51556.0, 28156.0], [-16641.0, 16961.0, 14662.0, -23759.0, 21190.0, 4277.0, 9870.0, 19162.0, 9761.0, -5361.0, 17013.0], [16389.0, 4577.0, -23023.0, -9521.0, 895.0, -4323.0, -476.0, -21493.0, 901.0, 2062.0, 790.0], [16434.0, -17816.0, 505.0, 13672.0, -6359.0, -9439.0, -1426.0, -8476.0, -4466.0, 13719.0, -7845.0], [4717.0, -18000.0, 23912.0, 36239.0, -43421.0, 26261.0, -29724.0, 8534.0, -5208.0, -18619.0, -4957.0]], [[-25125.0, -26789.0, 19394.0, -4486.0, 71.0, -20179.0, -9903.0, 11387.0, 30959.0, 2492.0, -9413.0], [7718.0, 25379.0, 5615.0, -16895.0, -8297.0, 39677.0, -9653.0, 1243.0, -2110.0, -17465.0, 29565.0], [4734.0, 23891.0, -14347.0, -5192.0, 50838.0, 16871.0, 42080.0, 6642.0, -30872.0, -23426.0, 39025.0], [1499.0, -3668.0, -22905.0, -6203.0, 13632.0, 2694.0, 3836.0, -15634.0, 4831.0, -24617.0, 15372.0], [36800.0, 36638.0, -25502.0, -4120.0, 6012.0, 44896.0, 10232.0, -17816.0, -34285.0, -25625.0, 38396.0]], [[8648.0, 39096.0, -16921.0, -31726.0, -14363.0, 46200.0, -12300.0, -17621.0, -4270.0, -5761.0, 27361.0], [-10369.0, -3547.0, 5300.0, -27729.0, 26962.0, -15357.0, 8820.0, 4679.0, 24011.0, 334.0, 13518.0], [-21.0, -16714.0, 2083.0, -6986.0, 10961.0, -15074.0, 311.0, -2614.0, 17430.0, -2848.0, 4089.0], [-20898.0, -11724.0, 47838.0, 2275.0, -4041.0, -10881.0, -8747.0, 36847.0, 21031.0, 2903.0, -7533.0], [5120.0, -17149.0, -7027.0, 763.0, 18347.0, -15794.0, 10737.0, -7102.0, 3885.0, -294.0, 4521.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 7)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([1, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 1x7)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 7, 1)", async () => {
    const a = tensor([[[[[[-76.0, 94.0, -106.0, -45.0, 31.0, 4.0, -39.0], [-91.0, -130.0, 98.0, 18.0, -124.0, 172.0, 41.0], [56.0, -98.0, 182.0, -6.0, -18.0, 53.0, -46.0], [-101.0, -56.0, 1.0, 19.0, 100.0, -185.0, -41.0], [-201.0, 62.0, -138.0, 99.0, -54.0, 82.0, 51.0]], [[106.0, -151.0, -26.0, -56.0, -40.0, -120.0, 5.0], [62.0, 18.0, 21.0, 10.0, 10.0, 36.0, -25.0], [229.0, -128.0, 8.0, -7.0, 136.0, -79.0, 184.0], [-1.0, -41.0, 140.0, 82.0, 15.0, 114.0, 46.0], [29.0, 38.0, 40.0, -42.0, -16.0, -143.0, -32.0]], [[10.0, 197.0, -64.0, 161.0, -164.0, 69.0, 59.0], [-5.0, 94.0, 78.0, 76.0, 4.0, 53.0, -177.0], [73.0, -209.0, 117.0, 256.0, 61.0, -84.0, 0.0], [18.0, -151.0, 140.0, 132.0, 25.0, -54.0, -268.0], [44.0, 36.0, -147.0, 30.0, 5.0, -7.0, 246.0]]], [[[78.0, -23.0, 28.0, -125.0, 158.0, 67.0, 12.0], [-81.0, -102.0, 116.0, 78.0, -145.0, 143.0, -291.0], [-6.0, -159.0, 114.0, 88.0, 162.0, -2.0, 79.0], [121.0, -73.0, -5.0, 49.0, -106.0, -198.0, 76.0], [189.0, 185.0, 69.0, 96.0, 92.0, -86.0, 26.0]], [[66.0, -155.0, -36.0, 65.0, -77.0, 46.0, -97.0], [-16.0, -98.0, -116.0, -200.0, 14.0, -47.0, 113.0], [-61.0, 133.0, 60.0, -88.0, 72.0, 143.0, 14.0], [-31.0, -132.0, -69.0, 34.0, -65.0, -114.0, 52.0], [166.0, -53.0, -15.0, -16.0, 28.0, -114.0, 40.0]], [[-93.0, 81.0, -61.0, 16.0, 8.0, 14.0, -11.0], [96.0, -15.0, -18.0, -67.0, -142.0, -106.0, -24.0], [-75.0, -210.0, 48.0, -15.0, 15.0, 94.0, -30.0], [-50.0, 24.0, 58.0, -53.0, -34.0, 39.0, 51.0], [-21.0, 80.0, -71.0, -17.0, -249.0, -114.0, 108.0]]]]], [[[[[190.0, 30.0, -84.0, 31.0, -111.0, 20.0, 72.0], [-55.0, 10.0, -26.0, -102.0, 41.0, -152.0, 35.0], [-115.0, -43.0, -22.0, 31.0, -77.0, 90.0, 50.0], [-89.0, -79.0, -78.0, 222.0, -16.0, -21.0, -16.0], [46.0, 58.0, 17.0, -11.0, 34.0, 38.0, 193.0]], [[-123.0, 13.0, 126.0, 32.0, 54.0, -86.0, 31.0], [-31.0, -367.0, 48.0, 52.0, 200.0, 119.0, -51.0], [103.0, 94.0, 36.0, -75.0, -49.0, -151.0, -27.0], [-5.0, -71.0, 82.0, 16.0, 33.0, 48.0, -214.0], [-76.0, -125.0, 23.0, 16.0, -42.0, -25.0, -32.0]], [[-53.0, 20.0, -65.0, -123.0, -52.0, 2.0, -48.0], [-25.0, -46.0, -7.0, -127.0, 15.0, -72.0, -44.0], [2.0, 26.0, -32.0, -167.0, -18.0, 108.0, -29.0], [-42.0, -40.0, 90.0, 111.0, 24.0, -35.0, 68.0], [54.0, -109.0, -34.0, -58.0, -9.0, 122.0, -168.0]]], [[[-90.0, -163.0, 166.0, -154.0, -94.0, 17.0, -90.0], [16.0, 115.0, 74.0, -33.0, -19.0, -28.0, 16.0], [-29.0, 4.0, -80.0, 55.0, -88.0, 260.0, -46.0], [-159.0, 16.0, -148.0, 67.0, -106.0, -19.0, -162.0], [-38.0, 205.0, -46.0, -55.0, -60.0, -36.0, 72.0]], [[-53.0, 93.0, 232.0, -43.0, 61.0, 41.0, -153.0], [67.0, 79.0, 77.0, 41.0, -139.0, -94.0, -68.0], [-8.0, 44.0, -14.0, 36.0, -42.0, 9.0, 58.0], [57.0, 49.0, 75.0, 8.0, 22.0, -76.0, -22.0], [111.0, 91.0, 104.0, -74.0, -175.0, 199.0, -72.0]], [[25.0, 57.0, -141.0, 91.0, 20.0, 126.0, -86.0], [-9.0, 117.0, -135.0, -147.0, 70.0, -109.0, -51.0], [-31.0, -121.0, 78.0, -79.0, 30.0, 15.0, 17.0], [26.0, -196.0, 9.0, 129.0, 290.0, -119.0, 6.0], [57.0, -37.0, -64.0, -101.0, -30.0, -19.0, -84.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[[-29.0], [35.0], [74.0], [136.0], [65.0], [93.0], [-59.0]]]);
    expect(b.shape).toEqual([1, 7, 1]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 1]);
    expect(await c.toArrayAsync()).toEqual([[[[[[-3782.0], [13306.0], [14071.0], [-4659.0], [12358.0]], [[-31954.0], [7219.0], [-20844.0], [28969.0], [-14714.0]], [[16041.0], [35175.0], [30195.0], [34920.0], [-21654.0]]], [[[-2202.0], [39014.0], [20696.0], [-29558.0], [15604.0]], [[3833.0], [-48878.0], [16049.0], [-22098.0], [-21097.0]], [[5665.0], [-31425.0], [7824.0], [-2218.0], [-37316.0]]]]], [[[[[-16063.0], [-27387.0], [4833.0], [22187.0], [-5185.0]], [[11381.0], [25754.0], [-22868.0], [25139.0], [-1460.0]], [[-19663.0], [-21800.0], [-13643.0], [15867.0], [4888.0]]], [[[-10974.0], [-234.0], [23715.0], [4232.0], [-14103.0]], [[32917.0], [-1669.0], [317.0], [2360.0], [8978.0]], [[21304.0], [-28204.0], [-5966.0], [18025.0], [-20181.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[[[5.0, 77.0, 119.0, -31.0, -3.0, -192.0, -76.0], [-74.0, 4.0, 7.0, -52.0, 59.0, -49.0, -59.0], [49.0, -186.0, 245.0, -59.0, 140.0, 261.0, -104.0], [17.0, -61.0, -136.0, 48.0, 38.0, -37.0, -66.0], [21.0, 18.0, 8.0, -22.0, 95.0, -18.0, 38.0]], [[119.0, 42.0, -33.0, -231.0, -66.0, 36.0, -196.0], [-93.0, -50.0, 124.0, -28.0, -13.0, -27.0, 106.0], [11.0, 67.0, 18.0, 9.0, 40.0, -18.0, -109.0], [74.0, 2.0, -0.0, 106.0, 89.0, 93.0, 1.0], [18.0, -100.0, 149.0, -145.0, -88.0, 113.0, -44.0]], [[91.0, -157.0, -120.0, 60.0, 90.0, 4.0, 76.0], [62.0, -7.0, 139.0, -210.0, 6.0, 110.0, -108.0], [-16.0, -36.0, -23.0, 151.0, 289.0, 77.0, -66.0], [-95.0, 18.0, 50.0, 94.0, -112.0, 38.0, -82.0], [82.0, 109.0, 94.0, -17.0, -6.0, 73.0, -41.0]]], [[[56.0, 189.0, 79.0, -4.0, -66.0, 104.0, -28.0], [-41.0, 118.0, 4.0, -25.0, 98.0, 116.0, -217.0], [180.0, 27.0, 34.0, 59.0, -56.0, -12.0, -153.0], [-45.0, 30.0, 77.0, 31.0, 77.0, 116.0, -162.0], [-56.0, 37.0, -210.0, 36.0, -89.0, 54.0, -192.0]], [[-26.0, -40.0, 112.0, -20.0, -52.0, 168.0, 26.0], [0.0, 170.0, -70.0, 70.0, 43.0, -158.0, 3.0], [-47.0, 9.0, -58.0, 85.0, -67.0, 122.0, -29.0], [-67.0, 107.0, 197.0, -116.0, -79.0, 122.0, -80.0], [19.0, 44.0, -61.0, -42.0, -74.0, -22.0, -25.0]], [[-75.0, -108.0, -2.0, 214.0, -35.0, 48.0, 33.0], [23.0, 146.0, -18.0, 123.0, -32.0, -51.0, 178.0], [53.0, -48.0, 46.0, -13.0, -56.0, -164.0, -68.0], [-32.0, -6.0, -35.0, 402.0, -39.0, 127.0, 34.0], [58.0, 121.0, 274.0, 113.0, -161.0, 31.0, -155.0]]]]], [[[[[-22.0, 18.0, 68.0, 52.0, -150.0, -25.0, 60.0], [17.0, 67.0, -21.0, 56.0, 126.0, -28.0, 156.0], [-174.0, -10.0, -64.0, -124.0, 122.0, -46.0, 29.0], [27.0, 37.0, 59.0, -79.0, -103.0, -88.0, -50.0], [85.0, -55.0, -35.0, -15.0, -83.0, -26.0, -170.0]], [[173.0, -30.0, 80.0, 92.0, 70.0, -5.0, -108.0], [-85.0, 70.0, 125.0, 173.0, -35.0, -38.0, -39.0], [240.0, 145.0, -102.0, -92.0, -30.0, 5.0, -49.0], [7.0, -4.0, -23.0, -289.0, -113.0, 36.0, -166.0], [119.0, -91.0, -10.0, 40.0, -6.0, -28.0, -152.0]], [[-143.0, 120.0, -17.0, 127.0, 1.0, 7.0, 39.0], [27.0, 55.0, 120.0, -84.0, 9.0, 146.0, -13.0], [-57.0, 19.0, 10.0, 57.0, 58.0, 77.0, -246.0], [-203.0, -70.0, -31.0, 82.0, 34.0, 25.0, -73.0], [-80.0, -65.0, 183.0, -96.0, 108.0, 169.0, -77.0]]], [[[-127.0, -2.0, 7.0, -27.0, -20.0, -92.0, 137.0], [85.0, 75.0, -26.0, 64.0, 85.0, 9.0, 62.0], [-142.0, -1.0, -2.0, -133.0, -62.0, -4.0, 209.0], [43.0, -7.0, -170.0, -53.0, -29.0, 61.0, -10.0], [156.0, 57.0, -149.0, 175.0, -115.0, 57.0, -51.0]], [[15.0, 125.0, 197.0, -59.0, -9.0, -185.0, 41.0], [-67.0, 14.0, -181.0, 23.0, 44.0, -14.0, -191.0], [73.0, -205.0, -191.0, 106.0, 90.0, -4.0, -69.0], [27.0, -28.0, -6.0, -205.0, 6.0, 57.0, -12.0], [18.0, 218.0, -36.0, -85.0, 61.0, -118.0, 14.0]], [[-215.0, 96.0, -17.0, 9.0, -194.0, -21.0, 40.0], [7.0, -25.0, 67.0, 68.0, 14.0, 55.0, 36.0], [116.0, -163.0, -40.0, 237.0, -36.0, 27.0, 52.0], [65.0, 208.0, -123.0, -14.0, 30.0, 50.0, 5.0], [43.0, -25.0, -68.0, 80.0, 45.0, -25.0, 69.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[[-63.0, 41.0, 61.0, -56.0, -5.0, -78.0, 74.0, -39.0, -130.0, -50.0, -239.0], [19.0, 12.0, -117.0, 4.0, 14.0, -28.0, 178.0, -50.0, 128.0, -13.0, -331.0], [147.0, -37.0, -9.0, 35.0, -67.0, -49.0, -38.0, 46.0, -11.0, 82.0, 1.0], [2.0, 254.0, 92.0, -191.0, -46.0, 122.0, -52.0, 71.0, -156.0, 44.0, 1.0], [-91.0, -60.0, -137.0, -39.0, -77.0, -105.0, -34.0, -57.0, 159.0, -101.0, 221.0], [99.0, -48.0, -50.0, -4.0, 129.0, -17.0, -49.0, 104.0, -18.0, 149.0, -41.0], [47.0, 162.0, -21.0, -53.0, 12.0, -15.0, -200.0, 15.0, -17.0, -76.0, -117.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[[-3728.0, -14064.0, -1020.0, 15027.0, -30943.0, -7440.0, 35876.0, -21709.0, 17004.0, -15386.0, -10493.0], [-7330.0, -27199.0, -14223.0, 15359.0, -9223.0, -5504.0, 9869.0, -10028.0, 29433.0, -6842.0, 38268.0], [37487.0, -62050.0, -12928.0, 15364.0, 5091.0, -35394.0, -32473.0, 32074.0, -4339.0, 50115.0, 82448.0], [-32349.0, 5993.0, 11844.0, -12960.0, -2526.0, 10531.0, 6793.0, -7465.0, -8180.0, -13432.0, 33677.0], [-8490.0, -3487.0, -15834.0, -2269.0, -8558.0, -15457.0, -4350.0, -9630.0, 17701.0, -16761.0, 6296.0]], [[-11654.0, -81590.0, -7252.0, 49288.0, 20204.0, -27765.0, 69228.0, -20094.0, 18495.0, 7560.0, -35737.0], [26573.0, 3135.0, -2610.0, 9693.0, -8465.0, -604.0, -38473.0, 9366.0, 5311.0, 3470.0, 24705.0], [-7301.0, -16319.0, -8793.0, 2852.0, -7447.0, -4777.0, 32910.0, -8099.0, 14081.0, 2013.0, -2448.0], [-3257.0, 20340.0, -2832.0, -28278.0, -62.0, -3837.0, -7463.0, 9154.0, -13440.0, 5730.0, -2503.0], [35706.0, -50077.0, 5447.0, 36814.0, 16022.0, -15616.0, -8335.0, 16965.0, -9437.0, 35307.0, 9869.0]], [[-30458.0, 28247.0, 16394.0, -28938.0, -2875.0, -160.0, -38228.0, -533.0, -27020.0, -23979.0, 40992.0], [21242.0, -79161.0, -20024.0, 46525.0, 12371.0, -37951.0, 24986.0, -1106.0, 23085.0, 23141.0, -3120.0], [-24533.0, 6389.0, -24722.0, -36975.0, -18941.0, -8859.0, -14969.0, 2632.0, 19856.0, -6674.0, 84302.0], [23965.0, 9959.0, 15463.0, -2250.0, 5595.0, 28268.0, 7732.0, 20885.0, -17658.0, 35958.0, 175.0], [16535.0, -12912.0, -12128.0, 4496.0, 4987.0, -16124.0, 27609.0, 1788.0, 3339.0, 16042.0, -55122.0]]], [[[26654.0, -4943.0, -15346.0, 4791.0, 15419.0, -8437.0, 37740.0, 5874.0, 4777.0, 25335.0, -91442.0], [-2270.0, -53365.0, -33312.0, 14898.0, 7553.0, -12359.0, 53502.0, -2669.0, 41473.0, 23622.0, 13011.0], [-8994.0, 582.0, 24428.0, -9710.0, -4586.0, -885.0, 46858.0, -2968.0, -35609.0, 11529.0, -45847.0], [11649.0, -32892.0, -17043.0, 4533.0, 1151.0, -4948.0, 21570.0, 11243.0, 16916.0, 31357.0, 32148.0], [-22146.0, -13294.0, 10982.0, 2489.0, 24727.0, 29321.0, 47330.0, 1039.0, -3149.0, 18310.0, 1544.0]], [[39888.0, -11502.0, -1576.0, 9014.0, 18974.0, -2566.0, -23924.0, 27572.0, -11586.0, 38432.0, -1876.0], [-26334.0, 27900.0, -10874.0, -16344.0, -19807.0, 5336.0, 34960.0, -25588.0, 21240.0, -32983.0, -40640.0], [11588.0, 15383.0, 8110.0, -11935.0, 20886.0, 22022.0, -1992.0, 20822.0, -17716.0, 28366.0, -8135.0], [50488.0, -52292.0, -22648.0, 40064.0, 14831.0, -14154.0, 25342.0, 14080.0, 24938.0, 45246.0, -32424.0], [-6031.0, -5658.0, 4459.0, 9298.0, 9100.0, 3670.0, 22334.0, -7174.0, -560.0, -2276.0, -31735.0]], [[12295.0, 55201.0, 29469.0, -37752.0, -1564.0, 37444.0, -43588.0, 30909.0, -44426.0, 22585.0, 40321.0], [5154.0, 67807.0, -1005.0, -32809.0, -4502.0, 11563.0, -10035.0, -1102.0, -10488.0, -17007.0, -79525.0], [-11851.0, -3191.0, 24539.0, 7377.0, -21081.0, 3058.0, 17846.0, -13358.0, -16308.0, -12438.0, 5558.0], [15281.0, 103771.0, 34328.0, -77028.0, 3723.0, 54849.0, -34707.0, 44421.0, -68000.0, 36774.0, -7803.0], [49584.0, 5456.0, 21073.0, -387.0, -7616.0, 11151.0, 44497.0, 22391.0, -36216.0, 55627.0, -72243.0]]]]], [[[[[25823.0, 29926.0, 21264.0, -3478.0, 2459.0, 19499.0, -9387.0, 13628.0, -28116.0, 15595.0, -39725.0], [-9679.0, 35558.0, -20599.0, -25185.0, -11758.0, -10435.0, -23042.0, -8757.0, 15747.0, -29733.0, -15463.0], [-13177.0, -36796.0, -35299.0, 25037.0, -4258.0, -10603.0, -13470.0, -15765.0, 61121.0, -23254.0, 70163.0], [5828.0, -18394.0, 9080.0, 22809.0, -3957.0, -2610.0, 28264.0, -9829.0, -1042.0, 622.0, -32025.0], [-14586.0, -21002.0, 26796.0, 9011.0, 2837.0, 6502.0, 36706.0, -3763.0, -25204.0, 10364.0, 453.0]], [[-11466.0, 5685.0, 14735.0, -21566.0, -18208.0, -10995.0, 19103.0, -1165.0, -28506.0, 2741.0, -2934.0], [22996.0, 34278.0, 8930.0, -20044.0, -17603.0, 24557.0, 3276.0, 15306.0, -12571.0, 22039.0, -4171.0], [-26621.0, -14392.0, -4982.0, 4889.0, 14263.0, -25206.0, 62805.0, -26339.0, -1193.0, -18798.0, -106651.0], [1569.0, -94156.0, -8319.0, 67047.0, 26097.0, -20822.0, 50986.0, -13955.0, 28122.0, 14493.0, -7688.0], [-19986.0, -8603.0, 27090.0, -6616.0, -8013.0, 2022.0, 22884.0, -2561.0, -31114.0, 4159.0, 19316.0]], [[11479.0, 34386.0, -12232.0, -18498.0, -1014.0, 23312.0, -3357.0, 9068.0, 13695.0, 7762.0, -10062.0], [29840.0, -33663.0, -21856.0, 18706.0, 14444.0, -23006.0, 6736.0, 10229.0, 14338.0, 25912.0, -27098.0], [-3681.0, -35029.0, -7176.0, 3199.0, -226.0, 6669.0, 39275.0, 6792.0, 12858.0, 30242.0, 45844.0], [3016.0, -2254.0, -745.0, -3216.0, -1929.0, 26417.0, -18349.0, 15380.0, 11176.0, 17965.0, 86768.0], [33798.0, -62281.0, -29383.0, 28154.0, 4206.0, -25677.0, -16005.0, 18237.0, 30482.0, 35752.0, 66670.0]]], [[[8089.0, 15462.0, -5597.0, 6393.0, -7304.0, 7934.0, -30828.0, -2915.0, 16536.0, -16338.0, 14318.0], [-11554.0, 26115.0, -10865.0, -24231.0, -5217.0, -9656.0, 1569.0, -6696.0, 1151.0, -16497.0, -33940.0], [23436.0, -1772.0, -16458.0, 24638.0, 13714.0, -1581.0, -43190.0, 2306.0, 25763.0, -9121.0, -3857.0], [-19730.0, -8301.0, 1229.0, 3154.0, 23497.0, 864.0, 11149.0, -5063.0, -1887.0, -5553.0, -15923.0], [-16587.0, 52945.0, 34264.0, -40188.0, 17547.0, 26758.0, 29569.0, 8355.0, -57089.0, 10925.0, -77910.0]], [[14702.0, -4098.0, -11289.0, 16742.0, -31490.0, -18046.0, 20113.0, -20074.0, 22289.0, -18589.0, -44023.0], [-36441.0, -22950.0, -3297.0, 1543.0, 4114.0, 14992.0, 40606.0, -11609.0, 19400.0, -2676.0, 43866.0], [-48188.0, 18138.0, 29228.0, -31676.0, -3588.0, 13990.0, -18406.0, -438.0, -34610.0, -16425.0, 78450.0], [1008.0, -56117.0, -17303.0, 37495.0, 16052.0, -27457.0, 7305.0, -9078.0, 25084.0, -1699.0, 2997.0], [-19029.0, -12632.0, -34655.0, 12190.0, -10467.0, -20723.0, 46832.0, -34832.0, 50805.0, -35233.0, -59900.0]], [[30343.0, 14380.0, 3422.0, 15640.0, 15853.0, 36140.0, 981.0, 12916.0, 7873.0, 21929.0, -27092.0], [14932.0, 17132.0, 3581.0, -13809.0, -1553.0, 2222.0, -20385.0, 14349.0, -14831.0, 12506.0, 3364.0], [-7418.0, 73766.0, 50801.0, -55275.0, -4205.0, 28931.0, -41733.0, 24253.0, -79570.0, 7174.0, 11279.0], [-15797.0, 2766.0, -27267.0, -6074.0, 15672.0, -10650.0, 42766.0, -16022.0, 25496.0, -12616.0, -80525.0], [-16347.0, 33977.0, 7156.0, -25480.0, -5551.0, 5103.0, -16949.0, -2005.0, -14090.0, -17395.0, 907.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (2, 7)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([2, 7]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 2x7)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (2, 7, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([2, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 3');
});
test("matmul (2, 1, 2, 3, 5, 7) x (2, 7, 11)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([2, 7, 11]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 3');
});
test("matmul (2, 1, 2, 3, 5, 7) x (1,)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 60, 60x7,1');
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([1, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 1x1)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 11)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([1, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 1x11)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (2,)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([2]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 60, 60x7,2');
});
test("matmul (2, 1, 2, 3, 5, 7) x (2, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([2, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 2x1)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (2, 11)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([2, 11]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 2x11)');
});
