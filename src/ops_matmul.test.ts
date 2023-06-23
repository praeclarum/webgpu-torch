import { ones } from "./factories";
import { tensor, matmul } from "./ops_artisanal";

test("matmul (2,) x (3,)", async () => {
    const a = ones([2]);
    const b = ones([3]);
    expect(() => matmul(a, b)).toThrow('inconsistent tensor size, expected tensor [2] and src [3] to have the same number of elements, but got 2 and 3 elements respectively');
});
test("matmul (2,) x (3, 1)", async () => {
    const a = ones([2]);
    const b = ones([3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 3x1)');
});
test("matmul (2,) x (1, 3)", async () => {
    const a = ones([2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 1x3)');
});
test("matmul (2,) x (1, 3, 1)", async () => {
    const a = ones([2]);
    const b = ones([1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 1, 1x3,2');
});
test("matmul (2,) x (1, 1, 3)", async () => {
    const a = ones([2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 3, 3x1,2');
});
test("matmul (2,) x (1, 1, 3, 1)", async () => {
    const a = ones([2]);
    const b = ones([1, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 1, 1x3,2');
});
test("matmul (2,) x (3, 1, 3)", async () => {
    const a = ones([2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 9, 9x1,2');
});
test("matmul (2,) x (3, 1, 3, 1)", async () => {
    const a = ones([2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 3, 3x3,2');
});
test("matmul (2,) x (3, 1, 3)", async () => {
    const a = ones([2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 9, 9x1,2');
});
test("matmul (2,) x (3, 1, 3, 1)", async () => {
    const a = ones([2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 3, 3x3,2');
});
test("matmul (1, 2) x (3,)", async () => {
    const a = ones([1, 2]);
    const b = ones([3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 1, 1x2,3');
});
test("matmul (1, 2) x (3, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 3x1)');
});
test("matmul (1, 2) x (1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x2 and 1x3)');
});
test("matmul (1, 2) x (1, 3, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x3 and 2x1)');
});
test("matmul (1, 2) x (1, 1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x1 and 2x1)');
});
test("matmul (1, 2) x (1, 1, 3, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([1, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x3 and 2x1)');
});
test("matmul (1, 2) x (3, 1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (9x1 and 2x1)');
});
test("matmul (1, 2) x (3, 1, 3, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x3 and 2x1)');
});
test("matmul (1, 2) x (3, 1, 3)", async () => {
    const a = ones([1, 2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (9x1 and 2x1)');
});
test("matmul (1, 2) x (3, 1, 3, 1)", async () => {
    const a = ones([1, 2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x3 and 2x1)');
});
test("matmul (2, 2) x (3,)", async () => {
    const a = ones([2, 2]);
    const b = ones([3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x2,3');
});
test("matmul (2, 2) x (3, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 3x1)');
});
test("matmul (2, 2) x (1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x3)');
});
test("matmul (2, 2) x (1, 3, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x3 and 2x2)');
});
test("matmul (2, 2) x (1, 1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x1 and 2x2)');
});
test("matmul (2, 2) x (1, 1, 3, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([1, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (1x3 and 2x2)');
});
test("matmul (2, 2) x (3, 1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (9x1 and 2x2)');
});
test("matmul (2, 2) x (3, 1, 3, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x3 and 2x2)');
});
test("matmul (2, 2) x (3, 1, 3)", async () => {
    const a = ones([2, 2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (9x1 and 2x2)');
});
test("matmul (2, 2) x (3, 1, 3, 1)", async () => {
    const a = ones([2, 2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x3 and 2x2)');
});
test("matmul (2, 1, 2) x (3,)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([3]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x2,3');
});
test("matmul (2, 1, 2) x (3, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([3, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 3x1)');
});
test("matmul (2, 1, 2) x (1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 3]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x2 and 1x3)');
});
test("matmul (2, 1, 2) x (1, 3, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 3].');
});
test("matmul (2, 1, 2) x (1, 1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 1, 3]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 1].');
});
test("matmul (2, 1, 2) x (1, 1, 3, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([1, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 3].');
});
test("matmul (2, 1, 2) x (3, 1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 1, 2) x (3, 1, 3, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 2] but got: [6, 3].');
});
test("matmul (2, 1, 2) x (3, 1, 3)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([3, 1, 3]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 1, 2) x (3, 1, 3, 1)", async () => {
    const a = ones([2, 1, 2]);
    const b = ones([3, 1, 3, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 2] but got: [6, 3].');
});
test("matmul (2, 3) x (5,)", async () => {
    const a = ones([2, 3]);
    const b = ones([5]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x3,5');
});
test("matmul (2, 3) x (5, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 5x1)');
});
test("matmul (2, 3) x (3, 5)", async () => {
    const a = tensor([[-92.0, 110.0, -10.0], [26.0, -56.0, -4.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[29.0, -76.0, -162.0, -71.0, 83.0], [44.0, 133.0, -137.0, -97.0, 202.0], [-139.0, -124.0, -112.0, -94.0, -72.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 5]);
    expect(await c.toArrayAsync()).toEqual([[3562.0, 22862.0, 954.0, -3198.0, 15304.0], [-1154.0, -8928.0, 3908.0, 3962.0, -8866.0]]);
});
test("matmul (2, 3) x (3, 5, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x5 and 3x2)');
});
test("matmul (2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[-16.0, -210.0, 129.0], [-59.0, 82.0, -1.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[[40.0, -21.0, -69.0, -2.0, -132.0], [-127.0, 197.0, 93.0, 135.0, 3.0], [220.0, -60.0, -61.0, -37.0, 89.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[54410.0, -48774.0, -26295.0, -33091.0, 12963.0], [-12994.0, 17453.0, 11758.0, 11225.0, 7945.0]]]);
});
test("matmul (2, 3) x (1, 3, 5, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([1, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x5 and 3x2)');
});
test("matmul (2, 3) x (3, 3, 5)", async () => {
    const a = tensor([[-176.0, 30.0, 121.0], [-123.0, 141.0, 215.0]]);
    expect(a.shape).toEqual([2, 3]);
    const b = tensor([[[0.0, -151.0, -111.0, -38.0, -74.0], [-73.0, 73.0, 176.0, -61.0, -89.0], [-4.0, 68.0, 6.0, -155.0, -34.0]], [[26.0, -103.0, -128.0, -21.0, 20.0], [-166.0, -135.0, -41.0, -148.0, 1.0], [-32.0, 74.0, 92.0, 51.0, 39.0]], [[14.0, 113.0, -70.0, -208.0, 30.0], [-96.0, 88.0, 120.0, -179.0, 36.0], [49.0, 120.0, -94.0, -3.0, 19.0]]]);
    expect(b.shape).toEqual([3, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([3, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-2674.0, 36994.0, 25542.0, -13897.0, 6240.0], [-11153.0, 43486.0, 39759.0, -37252.0, -10757.0]], [[-13428.0, 23032.0, 32430.0, 5427.0, 1229.0], [-33484.0, 9544.0, 29743.0, -7320.0, 6066.0]], [[585.0, -2728.0, 4546.0, 30875.0, -1901.0], [-4723.0, 24309.0, 5320.0, -300.0, 5471.0]]]);
});
test("matmul (2, 3) x (3, 3, 5, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([3, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (9x5 and 3x2)');
});
test("matmul (2, 3) x (3, 1, 5)", async () => {
    const a = ones([2, 3]);
    const b = ones([3, 1, 5]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (15x1 and 3x2)');
});
test("matmul (2, 3) x (3, 1, 5, 1)", async () => {
    const a = ones([2, 3]);
    const b = ones([3, 1, 5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (3x5 and 3x2)');
});
test("matmul (1, 2, 3) x (5,)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([5]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 2, 2x3,5');
});
test("matmul (1, 2, 3) x (5, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (2x3 and 5x1)');
});
test("matmul (1, 2, 3) x (3, 5)", async () => {
    const a = tensor([[[-123.0, 124.0, -21.0], [-7.0, 40.0, 111.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[-23.0, -141.0, 33.0, -207.0, -104.0], [-67.0, 49.0, 78.0, 52.0, -47.0], [-88.0, -70.0, 49.0, 174.0, 17.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-3631.0, 24889.0, 4584.0, 28255.0, 6607.0], [-12287.0, -4823.0, 8328.0, 22843.0, 735.0]]]);
});
test("matmul (1, 2, 3) x (3, 5, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [3, 3] but got: [3, 5].');
});
test("matmul (1, 2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[[-134.0, 102.0, 37.0], [-72.0, -29.0, -180.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[[96.0, -151.0, -43.0, 75.0, -32.0], [69.0, 96.0, 124.0, 1.0, -150.0], [-76.0, 15.0, -35.0, -104.0, 27.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-8638.0, 30581.0, 17115.0, -13796.0, -10013.0], [4767.0, 5388.0, 5800.0, 13291.0, 1794.0]]]);
});
test("matmul (1, 2, 3) x (1, 3, 5, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([1, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [3, 3] but got: [3, 5].');
});
test("matmul (1, 2, 3) x (3, 3, 5)", async () => {
    const a = tensor([[[186.0, 27.0, 57.0], [47.0, 41.0, -18.0]]]);
    expect(a.shape).toEqual([1, 2, 3]);
    const b = tensor([[[-74.0, -41.0, 67.0, 192.0, 36.0], [24.0, 38.0, -127.0, 5.0, 15.0], [-108.0, 64.0, -55.0, -13.0, 37.0]], [[-68.0, 29.0, 81.0, -41.0, -65.0], [83.0, -95.0, 17.0, 51.0, -9.0], [-21.0, 174.0, -82.0, -52.0, 95.0]], [[16.0, 26.0, 68.0, 65.0, 96.0], [-43.0, -65.0, -7.0, -118.0, -39.0], [56.0, 183.0, -26.0, -20.0, -15.0]]]);
    expect(b.shape).toEqual([3, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([3, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-19272.0, -2952.0, 5898.0, 35106.0, 9210.0], [-550.0, -1521.0, -1068.0, 9463.0, 1641.0]], [[-11604.0, 12747.0, 10851.0, -9213.0, -6918.0], [585.0, -5664.0, 5980.0, 1100.0, -5134.0]], [[5007.0, 13512.0, 10977.0, 7764.0, 15948.0], [-2019.0, -4737.0, 3377.0, -1423.0, 3183.0]]]);
});
test("matmul (1, 2, 3) x (3, 3, 5, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([3, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [9, 3] but got: [9, 5].');
});
test("matmul (1, 2, 3) x (3, 1, 5)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([3, 1, 5]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [3, 3] but got: [3, 1].');
});
test("matmul (1, 2, 3) x (3, 1, 5, 1)", async () => {
    const a = ones([1, 2, 3]);
    const b = ones([3, 1, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [3, 3] but got: [3, 5].');
});
test("matmul (2, 2, 3) x (5,)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([5]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 4, 4x3,5');
});
test("matmul (2, 2, 3) x (5, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 5x1)');
});
test("matmul (2, 2, 3) x (3, 5)", async () => {
    const a = tensor([[[-2.0, -13.0, -132.0], [-126.0, 287.0, 160.0]], [[5.0, -73.0, 185.0], [61.0, 45.0, 0.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[-3.0, -92.0, 204.0, -177.0, -96.0], [191.0, 70.0, -161.0, 64.0, -58.0], [-77.0, -1.0, 139.0, 85.0, -38.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[7687.0, -594.0, -16663.0, -11698.0, 5962.0], [42875.0, 31522.0, -49671.0, 54270.0, -10630.0]], [[-28203.0, -5755.0, 38488.0, 10168.0, -3276.0], [8412.0, -2462.0, 5199.0, -7917.0, -8466.0]]]);
});
test("matmul (2, 2, 3) x (3, 5, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[[-38.0, -64.0, -110.0], [62.0, -0.0, 41.0]], [[-39.0, -3.0, -5.0], [-277.0, -3.0, -123.0]]]);
    expect(a.shape).toEqual([2, 2, 3]);
    const b = tensor([[[61.0, -24.0, -97.0, -11.0, 194.0], [-84.0, 59.0, -137.0, -116.0, -15.0], [35.0, -14.0, 175.0, -84.0, -32.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[-792.0, -1324.0, -6796.0, 17082.0, -2892.0], [5217.0, -2062.0, 1161.0, -4126.0, 10716.0]], [[-2302.0, 829.0, 3319.0, 1197.0, -7361.0], [-20950.0, 8193.0, 5755.0, 13727.0, -49757.0]]]);
});
test("matmul (2, 2, 3) x (1, 3, 5, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([1, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (2, 2, 3) x (3, 3, 5)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([3, 3, 5]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 2, 3) x (3, 3, 5, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([3, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (2, 2, 3) x (3, 1, 5)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([3, 1, 5]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 2, 3) x (3, 1, 5, 1)", async () => {
    const a = ones([2, 2, 3]);
    const b = ones([3, 1, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 3] but got: [6, 5].');
});
test("matmul (2, 1, 2, 3) x (5,)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([5]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 4, 4x3,5');
});
test("matmul (2, 1, 2, 3) x (5, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([5, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (4x3 and 5x1)');
});
test("matmul (2, 1, 2, 3) x (3, 5)", async () => {
    const a = tensor([[[[6.0, 35.0, 189.0], [2.0, 100.0, 116.0]]], [[[115.0, -8.0, 52.0], [76.0, 6.0, -22.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[-94.0, -25.0, 159.0, 162.0, 50.0], [-128.0, 106.0, -18.0, -37.0, -75.0], [234.0, 23.0, -73.0, -6.0, -135.0]]);
    expect(b.shape).toEqual([3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[39182.0, 7907.0, -13473.0, -1457.0, -27840.0], [14156.0, 13218.0, -9950.0, -4072.0, -23060.0]]], [[[2382.0, -2527.0, 14633.0, 18614.0, -670.0], [-13060.0, -1770.0, 13582.0, 12222.0, 6320.0]]]]);
});
test("matmul (2, 1, 2, 3) x (3, 5, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 3] but got: [6, 5].');
});
test("matmul (2, 1, 2, 3) x (1, 3, 5)", async () => {
    const a = tensor([[[[-73.0, -163.0, 50.0], [-17.0, 151.0, 114.0]]], [[[105.0, 94.0, 98.0], [55.0, -69.0, -32.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[[7.0, -110.0, 39.0, -133.0, -122.0], [-102.0, -2.0, -7.0, 148.0, 42.0], [-4.0, -111.0, -116.0, 61.0, -75.0]]]);
    expect(b.shape).toEqual([1, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[15915.0, 2806.0, -7506.0, -11365.0, -1690.0], [-15977.0, -11086.0, -14944.0, 31563.0, -134.0]]], [[[-9245.0, -22616.0, -7931.0, 5925.0, -16212.0], [7551.0, -2360.0, 6340.0, -19479.0, -7208.0]]]]);
});
test("matmul (2, 1, 2, 3) x (1, 3, 5, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([1, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 3] but got: [6, 5].');
});
test("matmul (2, 1, 2, 3) x (3, 3, 5)", async () => {
    const a = tensor([[[[-67.0, 108.0, -125.0], [148.0, -54.0, -10.0]]], [[[-190.0, 27.0, 35.0], [35.0, -95.0, -122.0]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3]);
    const b = tensor([[[-297.0, 11.0, -64.0, 37.0, -23.0], [72.0, 53.0, -64.0, -54.0, 59.0], [-27.0, -5.0, -203.0, 14.0, -113.0]], [[32.0, -136.0, -20.0, -87.0, -13.0], [-26.0, 9.0, -37.0, 11.0, 51.0], [-30.0, 17.0, -22.0, -24.0, 31.0]], [[-152.0, -62.0, 58.0, 149.0, -6.0], [63.0, -50.0, 90.0, 17.0, 187.0], [-29.0, 106.0, 19.0, 103.0, -64.0]]]);
    expect(b.shape).toEqual([3, 3, 5]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 2, 5]);
    expect(await c.toArrayAsync()).toEqual([[[[31050.0, 5612.0, 22751.0, -10061.0, 22038.0], [-47574.0, -1184.0, -3986.0, 8252.0, -5460.0]], [[-1202.0, 7959.0, 94.0, 10017.0, 2504.0], [6440.0, -20784.0, -742.0, -13230.0, -4988.0]], [[20613.0, -14496.0, 3459.0, -21022.0, 28598.0], [-25608.0, -7536.0, 3534.0, 20104.0, -10346.0]]], [[[57429.0, -834.0, 3327.0, -7998.0, 2008.0], [-13941.0, -4040.0, 28606.0, 4717.0, 7376.0]], [[-7832.0, 26678.0, 2031.0, 15987.0, 4932.0], [7250.0, -7689.0, 5499.0, -1162.0, -9082.0]], [[29566.0, 14140.0, -7925.0, -24246.0, 3949.0], [-7767.0, -10352.0, -8838.0, -8966.0, -10167.0]]]]);
});
test("matmul (2, 1, 2, 3) x (3, 3, 5, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([3, 3, 5, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 1, 2, 3) x (3, 1, 5)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([3, 1, 5]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 3] but got: [6, 1].');
});
test("matmul (2, 1, 2, 3) x (3, 1, 5, 1)", async () => {
    const a = ones([2, 1, 2, 3]);
    const b = ones([3, 1, 5, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 3, 5) x (7,)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([7]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x5,7');
});
test("matmul (2, 3, 5) x (7, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([7, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 7x1)');
});
test("matmul (2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[-12.0, 70.0, -32.0, 29.0, 47.0], [-66.0, -138.0, 65.0, 37.0, 14.0], [-189.0, -41.0, -39.0, 54.0, -140.0]], [[96.0, -45.0, 56.0, -163.0, 210.0], [-107.0, 276.0, 37.0, -100.0, 79.0], [-4.0, -87.0, 136.0, 133.0, -101.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[-44.0, -43.0, 81.0, 7.0, -126.0, -51.0, -1.0], [-162.0, 201.0, -111.0, -33.0, -188.0, -92.0, 281.0], [-147.0, 23.0, 6.0, -141.0, -19.0, 203.0, 181.0], [-29.0, -97.0, -131.0, 96.0, -3.0, -133.0, -128.0], [66.0, -64.0, 131.0, -75.0, 106.0, -95.0, -31.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[-3847.0, 8029.0, -6576.0, 1377.0, -6145.0, -20646.0, 8721.0], [15556.0, -27890.0, 7349.0, -2571.0, 34398.0, 23006.0, -32117.0], [9885.0, 2711.0, -36406.0, 21213.0, 17261.0, 11612.0, -20963.0]], [[13421.0, -9514.0, 61970.0, -37137.0, 18049.0, 12341.0, 11749.0], [-37329.0, 65572.0, -15632.0, -30599.0, -30435.0, -6629.0, 94711.0], [-16245.0, -20624.0, -20505.0, 4010.0, 3171.0, 27722.0, -13720.0]]]);
});
test("matmul (2, 3, 5) x (5, 7, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 0');
});
test("matmul (2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[46.0, 207.0, -108.0, 75.0, 40.0], [148.0, -128.0, -39.0, -10.0, -62.0], [-96.0, -47.0, -12.0, 56.0, 82.0]], [[-41.0, 154.0, -173.0, -69.0, 123.0], [-115.0, 8.0, -29.0, 48.0, -24.0], [-76.0, 10.0, 110.0, 26.0, -160.0]]]);
    expect(a.shape).toEqual([2, 3, 5]);
    const b = tensor([[[22.0, -81.0, -45.0, -113.0, -71.0, -41.0, -19.0], [82.0, -50.0, -88.0, -31.0, -30.0, -27.0, 44.0], [-27.0, 125.0, -46.0, 58.0, 80.0, -97.0, -33.0], [72.0, 40.0, 11.0, -11.0, -22.0, -42.0, 8.0], [12.0, -13.0, 114.0, -76.0, -201.0, -182.0, 76.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[26782.0, -25096.0, -9933.0, -21744.0, -27806.0, -7429.0, 15438.0], [-7651.0, -10057.0, -780.0, -10196.0, 2894.0, 12875.0, -11949.0], [-626.0, 9800.0, 18972.0, 4761.0, -10448.0, -10907.0, 6832.0]], [[12905.0, -30363.0, 9514.0, -18764.0, -38754.0, -5184.0, 22060.0], [2077.0, 7522.0, 3597.0, 12361.0, 9373.0, 9664.0, 2054.0], [-3870.0, 22526.0, -20474.0, 26532.0, 45484.0, 20204.0, -13698.0]]]);
});
test("matmul (2, 3, 5) x (1, 5, 7, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([1, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (2, 3, 5) x (3, 5, 7)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([3, 5, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 3, 5) x (3, 5, 7, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([3, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (2, 3, 5) x (3, 1, 7)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([3, 1, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 3, 5) x (3, 1, 7, 1)", async () => {
    const a = ones([2, 3, 5]);
    const b = ones([3, 1, 7, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 5] but got: [6, 7].');
});
test("matmul (1, 2, 3, 5) x (7,)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([7]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 6, 6x5,7');
});
test("matmul (1, 2, 3, 5) x (7, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([7, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (6x5 and 7x1)');
});
test("matmul (1, 2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[[10.0, 38.0, -31.0, 43.0, -36.0], [-63.0, 133.0, -89.0, 9.0, 83.0], [14.0, 64.0, 52.0, -111.0, 72.0]], [[-112.0, -62.0, -144.0, 46.0, 76.0], [70.0, -180.0, -4.0, -44.0, -42.0], [-209.0, -2.0, 177.0, 51.0, -97.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[35.0, -67.0, 192.0, -64.0, 174.0, -82.0, -130.0], [92.0, 78.0, -39.0, -65.0, -220.0, 82.0, 37.0], [91.0, 20.0, 157.0, 123.0, 258.0, 38.0, 62.0], [-143.0, 71.0, 77.0, 19.0, -273.0, 41.0, 125.0], [-121.0, 25.0, -106.0, 24.0, 14.0, 4.0, 72.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-768.0, 3827.0, 2698.0, -6970.0, -26861.0, 2737.0, 967.0], [-9398.0, 15529.0, -39361.0, -13397.0, -64479.0, 13391.0, 14694.0], [18271.0, -987.0, -7823.0, 959.0, 33083.0, 1813.0, -4919.0]], [[-38502.0, 4954.0, -46208.0, -3816.0, -54494.0, 818.0, 14560.0], [-3100.0, -22984.0, 20896.0, 4884.0, 62172.0, -22624.0, -24532.0], [13052.0, 18583.0, 1948.0, 33918.0, -5541.0, 25403.0, 37461.0]]]]);
});
test("matmul (1, 2, 3, 5) x (5, 7, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (1, 2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[[50.0, -84.0, 244.0, -10.0, 212.0], [13.0, -77.0, 76.0, 27.0, -14.0], [-57.0, -65.0, -18.0, -117.0, -125.0]], [[9.0, 78.0, 152.0, 39.0, -190.0], [110.0, 107.0, -91.0, -0.0, 51.0], [-115.0, 60.0, -51.0, 12.0, -58.0]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5]);
    const b = tensor([[[239.0, -109.0, -57.0, -163.0, 72.0, -160.0, -10.0], [6.0, -37.0, 167.0, 200.0, -184.0, -22.0, 15.0], [-127.0, 97.0, -147.0, -12.0, 15.0, 58.0, -39.0], [37.0, 161.0, 54.0, 151.0, 7.0, -82.0, -25.0], [9.0, -35.0, -14.0, -356.0, 151.0, -66.0, -24.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-18004.0, 12296.0, -56254.0, -104860.0, 54658.0, -5172.0, -16114.0], [-6134.0, 13641.0, -23118.0, -9370.0, 14319.0, 2732.0, -4588.0], [-17181.0, -7590.0, -9528.0, 23340.0, -12108.0, 27350.0, 6222.0]], [[-16952.0, 23806.0, -5065.0, 85838.0, -39841.0, 15002.0, -1263.0], [38948.0, -26561.0, 24262.0, -13594.0, -5432.0, -28598.0, 2830.0], [-20726.0, 9330.0, 25532.0, 53817.0, -28759.0, 16966.0, 5131.0]]]]);
});
test("matmul (1, 2, 3, 5) x (1, 5, 7, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([1, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (1, 2, 3, 5) x (3, 5, 7)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([3, 5, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (1, 2, 3, 5) x (3, 5, 7, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([3, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (1, 2, 3, 5) x (3, 1, 7)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([3, 1, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (1, 2, 3, 5) x (3, 1, 7, 1)", async () => {
    const a = ones([1, 2, 3, 5]);
    const b = ones([3, 1, 7, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 5] but got: [6, 7].');
});
test("matmul (2, 2, 3, 5) x (7,)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([7]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 12, 12x5,7');
});
test("matmul (2, 2, 3, 5) x (7, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([7, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 7x1)');
});
test("matmul (2, 2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[[127.0, 110.0, -65.0, -21.0, 81.0], [123.0, -188.0, -202.0, -57.0, 3.0], [29.0, -136.0, 244.0, -36.0, 5.0]], [[55.0, 65.0, -43.0, -19.0, 169.0], [-22.0, -47.0, -60.0, 114.0, 78.0], [-62.0, -80.0, -216.0, 60.0, -29.0]]], [[[72.0, -135.0, -136.0, -102.0, 104.0], [-162.0, 66.0, 65.0, -19.0, -60.0], [-69.0, 54.0, -44.0, 46.0, -62.0]], [[50.0, 21.0, -42.0, -7.0, -75.0], [-82.0, -33.0, -172.0, 34.0, -23.0], [241.0, 32.0, 112.0, 30.0, -8.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[-51.0, -26.0, -95.0, 86.0, 53.0, -76.0, 42.0], [-152.0, 141.0, -220.0, -56.0, 68.0, -86.0, 43.0], [30.0, -37.0, 85.0, -38.0, 81.0, -46.0, 13.0], [25.0, -61.0, 11.0, 109.0, -37.0, -82.0, 1.0], [-94.0, -56.0, 104.0, 6.0, -40.0, 63.0, 67.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-33286.0, 11358.0, -33597.0, 5429.0, 6483.0, -9297.0, 14625.0], [14536.0, -18923.0, 12190.0, 22587.0, -20638.0, 20975.0, -5400.0], [25143.0, -27042.0, 48029.0, -3056.0, 13185.0, 1535.0, -1159.0]], [[-30336.0, 1021.0, -5813.0, 1667.0, -2205.0, 4413.0, 15850.0], [1984.0, -15157.0, 16696.0, 15914.0, -16560.0, 4040.0, 1615.0], [13068.0, -3712.0, 2774.0, 13722.0, -27282.0, 14781.0, -10735.0]]], [[[442.0, -15477.0, 20994.0, 8426.0, -16766.0, 27310.0, 2317.0], [5345.0, 15632.0, -54.0, -22529.0, 4270.0, 1424.0, -7160.0], [969.0, 11702.0, -15007.0, -2644.0, -2771.0, -5054.0, -5256.0]], [[-127.0, 7842.0, -20817.0, 3507.0, 3935.0, -7825.0, -2575.0], [7050.0, 3057.0, -1588.0, 4900.0, -20860.0, 12745.0, -8606.0], [-12293.0, -7280.0, -20917.0, 17900.0, 23231.0, -29184.0, 12448.0]]]]);
});
test("matmul (2, 2, 3, 5) x (5, 7, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (2, 2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[[-22.0, -74.0, -97.0, -32.0, -1.0], [93.0, 123.0, -11.0, -27.0, -138.0], [-104.0, 50.0, -44.0, 101.0, 2.0]], [[-29.0, 73.0, -144.0, -63.0, 136.0], [60.0, 37.0, -49.0, 99.0, -40.0], [160.0, 36.0, 52.0, 71.0, -81.0]]], [[[-23.0, -37.0, 64.0, 19.0, 38.0], [63.0, -36.0, -153.0, 60.0, -45.0], [89.0, 72.0, 98.0, 115.0, 112.0]], [[-214.0, 89.0, 137.0, 35.0, -244.0], [107.0, 39.0, -18.0, -123.0, 8.0], [119.0, 111.0, 100.0, -31.0, 25.0]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5]);
    const b = tensor([[[-154.0, 1.0, 8.0, -29.0, -77.0, 71.0, -197.0], [152.0, 156.0, -2.0, -76.0, -74.0, 129.0, -15.0], [-78.0, 46.0, 62.0, 135.0, -126.0, 85.0, -101.0], [40.0, 125.0, 93.0, -38.0, -71.0, -9.0, -47.0], [-186.0, 68.0, 49.0, 74.0, 106.0, 44.0, 14.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[-1388.0, -20096.0, -9067.0, -5691.0, 21558.0, -19109.0, 16731.0], [29820.0, 6016.0, -9457.0, -22716.0, -27588.0, 15706.0, -19718.0], [30716.0, 18433.0, 5831.0, -10414.0, 2893.0, -5495.0, 19463.0]], [[-1022.0, 6108.0, -8501.0, -11689.0, 33864.0, 1669.0, 24027.0], [11606.0, 13233.0, 4615.0, -17889.0, -12453.0, 2217.0, -12639.0], [-5318.0, 11535.0, 7066.0, -9048.0, -35163.0, 16221.0, -41783.0]]], [[[-13382.0, 2108.0, 7487.0, 14209.0, -876.0, 535.0, -1739.0], [7530.0, -8151.0, -5535.0, -25356.0, 8061.0, -15696.0, 132.0], [-26638.0, 37820.0, 22827.0, 9095.0, -20822.0, 27830.0, -32348.0]], [[82582.0, 7755.0, -2097.0, -1449.0, -35719.0, -3119.0, 21925.0], [-15554.0, -9468.0, -11385.0, -3231.0, 724.0, 12557.0, -13953.0], [-15144.0, 19860.0, 5272.0, 4641.0, -25126.0, 32647.0, -33401.0]]]]);
});
test("matmul (2, 2, 3, 5) x (1, 5, 7, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([1, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (2, 2, 3, 5) x (3, 5, 7)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([3, 5, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (2, 2, 3, 5) x (3, 5, 7, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([3, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 1');
});
test("matmul (2, 2, 3, 5) x (3, 1, 7)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([3, 1, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (2, 2, 3, 5) x (3, 1, 7, 1)", async () => {
    const a = ones([2, 2, 3, 5]);
    const b = ones([3, 1, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (2, 1, 2, 3, 5) x (7,)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([7]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 12, 12x5,7');
});
test("matmul (2, 1, 2, 3, 5) x (7, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([7, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (12x5 and 7x1)');
});
test("matmul (2, 1, 2, 3, 5) x (5, 7)", async () => {
    const a = tensor([[[[[-175.0, 19.0, 63.0, -31.0, 7.0], [-35.0, -10.0, 91.0, -24.0, 75.0], [-43.0, -70.0, -27.0, -145.0, -97.0]], [[-58.0, -90.0, 57.0, 96.0, 271.0], [-17.0, 12.0, 36.0, -47.0, 242.0], [145.0, -71.0, -53.0, -90.0, 27.0]]]], [[[[-20.0, 104.0, 35.0, -95.0, -30.0], [-72.0, 50.0, 21.0, -34.0, -225.0], [-190.0, 121.0, -6.0, 129.0, -4.0]], [[-203.0, -170.0, 21.0, 52.0, -132.0], [-114.0, 19.0, -163.0, 196.0, 71.0], [8.0, -32.0, 23.0, -46.0, -104.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[-101.0, -65.0, -85.0, -104.0, -13.0, 224.0, -39.0], [-21.0, -3.0, -4.0, 32.0, -68.0, 137.0, 10.0], [-145.0, 151.0, -116.0, 19.0, -48.0, 68.0, -8.0], [-14.0, 111.0, 187.0, -34.0, 241.0, -1.0, -98.0], [36.0, -52.0, 137.0, 90.0, 17.0, -34.0, 137.0]]);
    expect(b.shape).toEqual([5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[[8827.0, 17026.0, 2653.0, 21689.0, -9393.0, -32520.0, 10508.0], [-6414.0, 9482.0, -1754.0, 12615.0, -7742.0, -5548.0, 13164.0], [8266.0, -12123.0, -33337.0, -2081.0, -29979.0, -17615.0, 2114.0]], [[7895.0, 9211.0, 53757.0, 25361.0, 31881.0, -30756.0, 28625.0], [5615.0, -11296.0, 21586.0, 26214.0, -9536.0, -7897.0, 38255.0], [-3237.0, -28609.0, -19024.0, -12869.0, -15744.0, 18321.0, 6578.0]]]], [[[[-4989.0, -2712.0, -24651.0, 6603.0, -31897.0, 13263.0, 6740.0], [-4447.0, 15627.0, -33699.0, -9607.0, -15491.0, -166.0, -24353.0], [15569.0, 25608.0, 39937.0, 18772.0, 25551.0, -26384.0, -4522.0]], [[15548.0, 29512.0, 7139.0, 2423.0, 23479.0, -62898.0, -17131.0], [34562.0, 804.0, 74901.0, 9093.0, 56457.0, -36627.0, -3541.0], [-6571.0, 3351.0, -26070.0, -9215.0, -11886.0, 2554.0, -10556.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (5, 7, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 2');
});
test("matmul (2, 1, 2, 3, 5) x (1, 5, 7)", async () => {
    const a = tensor([[[[[27.0, 105.0, -56.0, -5.0, 197.0], [174.0, 25.0, -47.0, 98.0, -1.0], [103.0, 118.0, 13.0, -39.0, -133.0]], [[-140.0, -22.0, -199.0, 107.0, 156.0], [96.0, -155.0, 18.0, 3.0, 186.0], [-143.0, -122.0, 107.0, -39.0, 146.0]]]], [[[[-164.0, -13.0, 259.0, 68.0, 48.0], [108.0, -114.0, 44.0, 51.0, -75.0], [16.0, -32.0, -140.0, -98.0, -110.0]], [[-255.0, -144.0, -14.0, 27.0, 183.0], [-67.0, -4.0, -100.0, -4.0, 223.0], [28.0, -68.0, 92.0, -48.0, 42.0]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5]);
    const b = tensor([[[207.0, -41.0, -78.0, -191.0, -31.0, 35.0, 114.0], [-29.0, -82.0, 65.0, -77.0, -124.0, 18.0, -132.0], [-41.0, 34.0, 61.0, -4.0, 149.0, 11.0, -141.0], [102.0, -12.0, -168.0, 59.0, 36.0, 106.0, 5.0], [74.0, 87.0, 55.0, -113.0, -16.0, -96.0, -170.0]]]);
    expect(b.shape).toEqual([1, 5, 7]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 7]);
    expect(await c.toArrayAsync()).toEqual([[[[[18908.0, 5578.0, 12978.0, -35574.0, -25533.0, -17223.0, -36401.0], [47142.0, -12045.0, -31333.0, -29076.0, -11953.0, 16507.0, 23823.0], [3546.0, -24560.0, -334.0, -16083.0, -15164.0, 14506.0, 16748.0]], [[2275.0, 13066.0, -12045.0, 17915.0, -21227.0, -11119.0, -10982.0], [37699.0, 25532.0, -6739.0, -27314.0, 16058.0, -16770.0, -2739.0], [-23624.0, 32675.0, 24333.0, 17480.0, 31764.0, -24174.0, -40300.0]]]], [[[[-33702.0, 19956.0, 18962.0, 29877.0, 46967.0, -525.0, -61319.0], [23510.0, -721.0, -25843.0, -542.0, 20380.0, 14818.0, 34161.0], [-8156.0, -11186.0, -1454.0, 6616.0, -19156.0, -1384.0, 43998.0]], [[-31739.0, 37384.0, 15205.0, 40763.0, 21719.0, -26377.0, -39063.0], [6441.0, 19124.0, 11803.0, -11930.0, -16039.0, -25349.0, -30940.0], [2208.0, 11786.0, 9382.0, -8058.0, 18872.0, -8352.0, -8184.0]]]]]);
});
test("matmul (2, 1, 2, 3, 5) x (1, 5, 7, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([1, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 2');
});
test("matmul (2, 1, 2, 3, 5) x (3, 5, 7)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([3, 5, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2');
});
test("matmul (2, 1, 2, 3, 5) x (3, 5, 7, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([3, 5, 7, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (5) at non-singleton dimension 2');
});
test("matmul (2, 1, 2, 3, 5) x (3, 1, 7)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([3, 1, 7]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2');
});
test("matmul (2, 1, 2, 3, 5) x (3, 1, 7, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5]);
    const b = ones([3, 1, 7, 1]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [12, 5] but got: [12, 7].');
});
test("matmul (2, 3, 5, 7) x (11,)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([11]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 30, 30x7,11');
});
test("matmul (2, 3, 5, 7) x (11, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([11, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 11x1)');
});
test("matmul (2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[-15.0, 125.0, 98.0, -109.0, -140.0, -189.0, -4.0], [-119.0, 151.0, -63.0, -56.0, 42.0, 168.0, -46.0], [61.0, -73.0, 149.0, -87.0, 27.0, -8.0, -65.0], [-9.0, 138.0, -205.0, -127.0, -15.0, -28.0, -54.0], [-42.0, 242.0, -1.0, -73.0, 63.0, -143.0, 17.0]], [[116.0, -6.0, -10.0, 27.0, -33.0, -60.0, -16.0], [-234.0, 68.0, 5.0, -188.0, 117.0, 16.0, -129.0], [-11.0, 74.0, -44.0, -0.0, -142.0, 132.0, -14.0], [-180.0, -52.0, 2.0, -27.0, 53.0, -116.0, 45.0], [-116.0, -119.0, -67.0, -97.0, 3.0, -70.0, 117.0]], [[106.0, -37.0, -41.0, -116.0, -56.0, -201.0, -177.0], [25.0, 26.0, -43.0, 79.0, 133.0, -16.0, -29.0], [-33.0, -97.0, 41.0, 45.0, -89.0, 7.0, 149.0], [-60.0, 202.0, 140.0, 154.0, 102.0, 172.0, 80.0], [20.0, 9.0, 115.0, 19.0, -24.0, 53.0, -138.0]]], [[[17.0, -76.0, 110.0, -131.0, -125.0, -42.0, 2.0], [19.0, 107.0, -124.0, 76.0, -53.0, -57.0, -32.0], [104.0, -11.0, 15.0, 46.0, -40.0, 78.0, -73.0], [-225.0, -232.0, -9.0, -164.0, -184.0, 62.0, 91.0], [208.0, -125.0, -72.0, -102.0, -71.0, 117.0, 38.0]], [[118.0, -238.0, 21.0, -172.0, -72.0, -65.0, -61.0], [172.0, -31.0, -69.0, -70.0, -189.0, -99.0, 83.0], [16.0, 70.0, -103.0, 22.0, -36.0, -170.0, 60.0], [8.0, 1.0, 57.0, 132.0, 20.0, -146.0, 105.0], [85.0, 169.0, -1.0, 12.0, -45.0, -70.0, -117.0]], [[7.0, -178.0, -115.0, 164.0, -14.0, 139.0, -38.0], [17.0, 24.0, -100.0, -66.0, 141.0, -247.0, -82.0], [-122.0, -72.0, -37.0, -18.0, 197.0, 37.0, 20.0], [-288.0, 25.0, -140.0, -229.0, -74.0, -234.0, -25.0], [-233.0, -150.0, 132.0, -82.0, 217.0, 34.0, 182.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[-95.0, 149.0, 37.0, 4.0, -129.0, -45.0, 43.0, 46.0, -172.0, 81.0, -18.0], [5.0, -28.0, 181.0, 124.0, 54.0, -30.0, 1.0, 99.0, 29.0, -119.0, -21.0], [-235.0, -39.0, 33.0, -57.0, -167.0, 125.0, -43.0, -73.0, -55.0, -85.0, 9.0], [17.0, -24.0, -24.0, 165.0, 123.0, -26.0, -209.0, -130.0, -163.0, -27.0, -70.0], [3.0, 129.0, 83.0, 118.0, -102.0, -22.0, -126.0, 38.0, 30.0, -3.0, -68.0], [-91.0, 361.0, -66.0, 55.0, -72.0, 47.0, 165.0, -126.0, -23.0, -88.0, 231.0], [-118.0, -1.0, 63.0, -159.0, -172.0, 80.0, 2.0, -31.0, -69.0, -100.0, 23.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[-5582.0, -93226.0, 28522.0, -34410.0, 7488.0, 5886.0, 4494.0, 37319.0, 19005.0, -4025.0, -28074.0], [16179.0, 47954.0, 11693.0, 34109.0, 18670.0, -2302.0, 31783.0, 3208.0, 38010.0, -31051.0, 37218.0], [-34175.0, 8070.0, -5277.0, -18575.0, -38393.0, 14162.0, 9474.0, 61.0, -1144.0, 10435.0, 2687.0], [56436.0, -6151.0, 18129.0, 13082.0, 40061.0, -31364.0, 32271.0, 49355.0, 41446.0, 11612.0, -2381.0], [15390.0, -54756.0, 59705.0, 14718.0, 10620.0, -10344.0, -17763.0, 51474.0, 30202.0, -19449.0, -36151.0]], [[-992.0, -8707.0, 2441.0, 95.0, 141.0, -10366.0, -6005.0, 8764.0, -22483.0, 17210.0, -15926.0], [32316.0, -11455.0, 8855.0, 11388.0, 19001.0, 1861.0, 16723.0, 26472.0, 84632.0, -11254.0, 8762.0], [969.0, 27353.0, -9845.0, 4370.0, 20151.0, 983.0, 41137.0, -11562.0, 128.0, -15747.0, 38074.0], [21316.0, -59878.0, -468.0, -19018.0, 11963.0, 7594.0, -27963.0, 5171.0, 34896.0, -2284.0, -23125.0], [17094.0, -34011.0, -13474.0, -49505.0, -7594.0, 8941.0, 6353.0, 5691.0, 29624.0, 7530.0, -2909.0]], [[36417.0, -58395.0, -3877.0, -10487.0, 27535.0, -28144.0, 4065.0, 47971.0, 17014.0, 55162.0, -40074.0], [14480.0, 14188.0, 12584.0, 38235.0, 7651.0, -15332.0, -33017.0, 4562.0, -7699.0, 4362.0, -20320.0], [-24706.0, -13983.0, -16967.0, -40880.0, -19347.0, 22557.0, -17.0, -28847.0, -19839.0, -11079.0, 10946.0], [-48358.0, 51418.0, 37420.0, 51014.0, -22338.0, 22376.0, -24896.0, -33278.0, -23040.0, -68398.0, 21954.0], [-17168.0, 13962.0, -8476.0, 19801.0, 3406.0, 4690.0, 3446.0, -12366.0, -5018.0, -531.0, 9857.0]]], [[[-26861.0, -27774.0, -13830.0, -54619.0, -25350.0, 19607.0, 32128.0, 2738.0, 7253.0, 8479.0, 10294.0], [37966.0, -24535.0, 11501.0, 28651.0, 48397.0, -25614.0, -12419.0, 16799.0, -3804.0, 5669.0, -19324.0], [-11282.0, 37186.0, -11819.0, 16964.0, 163.0, -4965.0, 11966.0, -12465.0, -24487.0, 7772.0, 14333.0], [2610.0, -24187.0, -60309.0, -88986.0, -3520.0, 34466.0, 58352.0, -28966.0, 45974.0, 572.0, 49248.0], [-20543.0, 72788.0, -26078.0, -35379.0, -41822.0, -1857.0, 61560.0, -2909.0, -26258.0, 26714.0, 38102.0]], [[-7362.0, -5137.0, -39420.0, -60989.0, -30221.0, 2576.0, 38106.0, 10038.0, 3227.0, 52775.0, 3581.0], [-2822.0, -29336.0, -3768.0, -51717.0, -8819.0, -7470.0, 32607.0, 21699.0, -24398.0, 26355.0, -6274.0], [31691.0, -62161.0, 21347.0, -4893.0, 27215.0, -18665.0, -22805.0, 30517.0, 47.0, 10195.0, -39667.0], [-10950.0, -54458.0, 17101.0, -3678.0, -3849.0, 4401.0, -56094.0, -4953.0, -29285.0, -5592.0, -41563.0], [13250.0, -23274.0, 26927.0, 32776.0, 29558.0, -20992.0, -4755.0, 29891.0, -3287.0, 4530.0, -21729.0]], [[20051.0, 54987.0, -52420.0, 23606.0, 26818.0, -9813.0, -4585.0, -47093.0, -27768.0, 18706.0, 23284.0], [53459.0, -63551.0, 26096.0, 13945.0, 25191.0, -33540.0, -39836.0, 58060.0, 29599.0, 38316.0, -65621.0], [14483.0, 24463.0, -3166.0, 11824.0, -10383.0, 2498.0, -18642.0, -5495.0, 27544.0, -3530.0, 246.0], [80514.0, -126651.0, 2472.0, -45484.0, 62411.0, -10706.0, 12186.0, 56664.0, 100175.0, 15094.0, -30168.0], [-34948.0, 6388.0, -2214.0, -42048.0, -66059.0, 45001.0, -20075.0, -26224.0, 35002.0, -31872.0, 11556.0]]]]);
});
test("matmul (2, 3, 5, 7) x (7, 11, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1');
});
test("matmul (2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[19.0, -22.0, -2.0, -73.0, 34.0, 107.0, -132.0], [-98.0, -146.0, 141.0, 10.0, -240.0, -120.0, 88.0], [-131.0, 64.0, 36.0, -38.0, 116.0, 37.0, 27.0], [34.0, -155.0, -51.0, -41.0, 41.0, 0.0, 31.0], [31.0, 137.0, 123.0, -89.0, -87.0, -140.0, -66.0]], [[194.0, 47.0, -43.0, -45.0, -3.0, 99.0, 29.0], [38.0, -125.0, 130.0, -110.0, -225.0, -55.0, 36.0], [-5.0, 97.0, 222.0, 5.0, 49.0, 97.0, 281.0], [-163.0, 145.0, 2.0, 44.0, 21.0, -52.0, -37.0], [-55.0, -15.0, -17.0, 32.0, -103.0, -100.0, 110.0]], [[-91.0, 181.0, 139.0, 212.0, -57.0, -47.0, 127.0], [-39.0, 50.0, 74.0, 15.0, -13.0, 24.0, 93.0], [-68.0, -28.0, 151.0, -120.0, -61.0, 159.0, 117.0], [-149.0, -52.0, 34.0, -75.0, 43.0, 73.0, -69.0], [-102.0, -96.0, 31.0, -50.0, -26.0, 107.0, 43.0]]], [[[117.0, -89.0, 159.0, 110.0, 62.0, 32.0, 46.0], [-95.0, 10.0, -156.0, -84.0, -116.0, 103.0, 122.0], [-87.0, -90.0, 45.0, 9.0, -114.0, 44.0, -116.0], [37.0, -50.0, 70.0, -86.0, -46.0, 34.0, 83.0], [333.0, -29.0, -172.0, -80.0, 108.0, 186.0, -121.0]], [[-217.0, 33.0, -164.0, 167.0, 84.0, 14.0, 168.0], [190.0, 57.0, -18.0, 103.0, -19.0, -56.0, -27.0], [-29.0, -125.0, -223.0, 30.0, 13.0, -83.0, -155.0], [-46.0, 97.0, 39.0, -8.0, 24.0, -76.0, 102.0], [-35.0, 8.0, 66.0, 163.0, 156.0, -44.0, -57.0]], [[148.0, 57.0, 74.0, 216.0, 15.0, 54.0, 45.0], [137.0, -80.0, -47.0, -85.0, 72.0, 51.0, -116.0], [-93.0, 27.0, -105.0, -33.0, 35.0, -45.0, 43.0], [91.0, -98.0, -106.0, 85.0, 32.0, 69.0, -29.0], [-56.0, 216.0, 15.0, -173.0, 146.0, 103.0, 183.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[[-133.0, -25.0, -52.0, -15.0, -88.0, -105.0, -14.0, 6.0, 2.0, -214.0, -16.0], [-80.0, -350.0, 52.0, 125.0, -51.0, 127.0, -7.0, 30.0, -69.0, 109.0, 25.0], [-41.0, -32.0, 150.0, 65.0, -28.0, -16.0, 44.0, -81.0, -28.0, 57.0, 28.0], [-5.0, 38.0, 58.0, 96.0, 128.0, -159.0, -19.0, 34.0, 109.0, 100.0, 13.0], [54.0, -101.0, 81.0, -22.0, 78.0, 101.0, 30.0, 165.0, 52.0, -7.0, -24.0], [-102.0, -14.0, -19.0, 52.0, 43.0, -10.0, -41.0, -68.0, 6.0, 29.0, 25.0], [66.0, 216.0, -98.0, -36.0, -135.0, -79.0, 211.0, -6.0, -42.0, -33.0, 27.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[-18110.0, -28929.0, 6991.0, -605.0, 15235.0, 19642.0, -30032.0, -3740.0, 1609.0, -6657.0, -3564.0], [23971.0, 94346.0, -6550.0, -10783.0, -22358.0, -42090.0, 24696.0, -48017.0, -9876.0, 9391.0, 7132.0], [15289.0, -28123.0, 19383.0, 7057.0, 9386.0, 36562.0, 11352.0, 13388.0, -4708.0, 32632.0, 3080.0], [14434.0, 56029.0, -19573.0, -29154.0, 106.0, -14228.0, 6915.0, 4870.0, 8552.0, -32488.0, -6527.0], [-14455.0, -59552.0, 20881.0, 13121.0, -28447.0, 24154.0, -5086.0, -13132.0, -25128.0, 5137.0, 2022.0]], [[-35920.0, -16453.0, -21670.0, 20.0, -23917.0, -10142.0, -2112.0, -2874.0, -7336.0, -41409.0, -388.0], [-3998.0, 65731.0, -16064.0, -17511.0, -39464.0, -29474.0, 11254.0, -51393.0, -20471.0, -26555.0, 3474.0], [-4924.0, 13650.0, 13482.0, 20960.0, -40025.0, -9723.0, 65848.0, -15129.0, -21046.0, 17994.0, 17622.0], [13773.0, -54452.0, 25183.0, 23090.0, 16922.0, 34066.0, -4526.0, 11929.0, -3257.0, 54767.0, 4058.0], [20950.0, 43948.0, -15837.0, -5977.0, -17007.0, -19039.0, 23739.0, -9170.0, -5687.0, 6557.0, 3387.0]], [[962.0, -23620.0, 31120.0, 47615.0, -1591.0, -18710.0, 29109.0, -6138.0, -2035.0, 63171.0, 16251.0], [1066.0, 2742.0, 5975.0, 11271.0, -11807.0, -2024.0, 21416.0, -8553.0, -8403.0, 17232.0, 7564.0], [-6097.0, 31315.0, -1658.0, 1213.0, -25892.0, 3254.0, 26410.0, -39138.0, -22644.0, 9284.0, 11654.0], [13280.0, -2282.0, 14652.0, -3921.0, 21020.0, 29486.0, -10891.0, -5213.0, -265.0, 24749.0, -9.0], [10745.0, 43674.0, -6291.0, -8667.0, 3372.0, -1121.0, 8320.0, -19527.0, -2414.0, 9997.0, 3910.0]]], [[[-12390.0, 30543.0, 19424.0, 6659.0, 3873.0, -41314.0, 14145.0, -3329.0, 15397.0, -15700.0, 2339.0], [9933.0, 37301.0, -46121.0, -12013.0, -19623.0, 4713.0, 14031.0, -17366.0, -16206.0, 3901.0, 4963.0], [-1419.0, 18419.0, 8414.0, 2816.0, 20798.0, -7236.0, -26043.0, -27667.0, 4965.0, 18175.0, 1223.0], [-3835.0, 33165.0, -11518.0, -10719.0, -27005.0, -9224.0, 19285.0, -20272.0, -13484.0, -19409.0, 3195.0], [-55643.0, -35359.0, -32192.0, -15828.0, -492.0, -4569.0, -40424.0, 18238.0, 10577.0, -83596.0, -13118.0]], [[46306.0, 33077.0, -11840.0, 5584.0, 27855.0, -1881.0, 29812.0, 30550.0, 17480.0, 51661.0, 4746.0], [-26703.0, -23339.0, -1471.0, 11471.0, -6184.0, -28026.0, -9779.0, 8645.0, 7988.0, -25773.0, -2453.0], [21788.0, 19120.0, -18882.0, -25827.0, 37381.0, 356.0, -38013.0, 23878.0, 24769.0, -14513.0, -15087.0], [12579.0, -13680.0, 6214.0, 6430.0, -18181.0, 12923.0, 27191.0, 7719.0, -12241.0, 16102.0, 4427.0], [9644.0, -25295.0, 40648.0, 17795.0, 39659.0, -1583.0, -5302.0, 29300.0, 25539.0, 27937.0, -1656.0]], [[-30086.0, -10361.0, 14675.0, 31309.0, 7062.0, -46409.0, 4412.0, 2481.0, 17049.0, 335.0, 6142.0], [-18439.0, -10193.0, -7033.0, -18026.0, 5929.0, 5648.0, -26218.0, 8447.0, 6767.0, -44414.0, -10198.0], [23997.0, 1364.0, -11948.0, -9881.0, 513.0, 20709.0, 9088.0, 16212.0, -2962.0, 10591.0, -2010.0], [-7566.0, 28185.0, -16675.0, -8417.0, 20216.0, -28987.0, -14855.0, 9844.0, 22473.0, -24964.0, -5595.0], [-126.0, -57914.0, -1705.0, 7763.0, -37540.0, 59838.0, 41989.0, 15035.0, -33769.0, 15009.0, 8479.0]]]]);
});
test("matmul (2, 3, 5, 7) x (1, 7, 11, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([1, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1');
});
test("matmul (2, 3, 5, 7) x (3, 7, 11)", async () => {
    const a = tensor([[[[-115.0, 19.0, 182.0, 4.0, 208.0, 7.0, 99.0], [30.0, -134.0, 127.0, -9.0, 32.0, -112.0, -133.0], [-39.0, 57.0, -53.0, -175.0, -81.0, -172.0, -117.0], [121.0, -12.0, -91.0, -126.0, -175.0, -26.0, 81.0], [151.0, -152.0, -50.0, -71.0, -132.0, 6.0, -113.0]], [[-80.0, -66.0, -11.0, -78.0, -59.0, 44.0, -36.0], [30.0, -19.0, -35.0, 244.0, 85.0, 131.0, 122.0], [55.0, -59.0, 138.0, -100.0, -66.0, 62.0, 258.0], [-45.0, -216.0, -27.0, 70.0, -53.0, -2.0, 103.0], [-153.0, -37.0, -50.0, 197.0, -64.0, 58.0, -69.0]], [[126.0, -4.0, -44.0, 111.0, -172.0, -197.0, 23.0], [50.0, 168.0, 71.0, 94.0, 77.0, 82.0, -212.0], [-5.0, -152.0, -83.0, 22.0, -76.0, 127.0, 46.0], [-225.0, 12.0, 54.0, -78.0, 146.0, -20.0, -9.0], [-10.0, -11.0, -103.0, -142.0, -128.0, -53.0, -20.0]]], [[[-115.0, 169.0, 144.0, 22.0, -115.0, 188.0, -213.0], [-21.0, 157.0, -106.0, -14.0, -49.0, -3.0, -23.0], [2.0, -46.0, -67.0, 128.0, -30.0, -87.0, 91.0], [3.0, -37.0, 52.0, -134.0, -54.0, 10.0, 44.0], [-22.0, 169.0, -2.0, 172.0, 57.0, 76.0, 138.0]], [[-119.0, -152.0, -10.0, -28.0, -119.0, -20.0, 18.0], [-144.0, -34.0, -81.0, 104.0, -52.0, -51.0, -50.0], [-115.0, -29.0, -134.0, 309.0, 45.0, -51.0, 50.0], [-240.0, -60.0, 4.0, 46.0, -33.0, 85.0, -36.0], [-48.0, 127.0, -142.0, 2.0, -50.0, -31.0, 29.0]], [[224.0, 88.0, -77.0, 10.0, -109.0, 65.0, -31.0], [33.0, -59.0, 112.0, -63.0, 180.0, 143.0, -196.0], [19.0, -5.0, 79.0, -88.0, -137.0, 39.0, 45.0], [-93.0, -3.0, -22.0, 12.0, 62.0, -29.0, 10.0], [111.0, 2.0, -9.0, 32.0, 80.0, -37.0, -39.0]]]]);
    expect(a.shape).toEqual([2, 3, 5, 7]);
    const b = tensor([[[-37.0, 16.0, 38.0, -65.0, -232.0, -77.0, 58.0, -222.0, 48.0, -177.0, -79.0], [12.0, -43.0, -183.0, 88.0, 9.0, -186.0, -45.0, -179.0, -138.0, -124.0, -28.0], [-49.0, -8.0, 181.0, -83.0, -93.0, 82.0, 18.0, -17.0, -27.0, -37.0, -25.0], [6.0, -80.0, 46.0, 230.0, 120.0, -79.0, 9.0, 24.0, 134.0, 136.0, -46.0], [146.0, -108.0, 8.0, -21.0, 127.0, -52.0, -38.0, -177.0, 56.0, -174.0, 54.0], [41.0, -24.0, 139.0, -78.0, -71.0, -79.0, 12.0, 73.0, -149.0, 145.0, -3.0], [121.0, 98.0, -156.0, 99.0, -119.0, -43.0, -4.0, -158.0, 32.0, -127.0, -58.0]], [[-73.0, 43.0, -47.0, -67.0, 32.0, -93.0, 160.0, 57.0, 180.0, 11.0, -79.0], [-149.0, -121.0, 163.0, -49.0, 56.0, 46.0, 137.0, 36.0, 15.0, 130.0, 131.0], [129.0, -96.0, 8.0, -36.0, 134.0, 115.0, 139.0, 74.0, -4.0, 31.0, 101.0], [-173.0, 9.0, 13.0, 28.0, -72.0, 179.0, -115.0, -62.0, -156.0, 283.0, 187.0], [-39.0, -211.0, 22.0, -103.0, -192.0, -32.0, 60.0, -229.0, -91.0, 37.0, 13.0], [82.0, 115.0, 29.0, 24.0, -85.0, 120.0, -9.0, -135.0, 228.0, -64.0, 15.0], [-14.0, -74.0, 97.0, -11.0, 31.0, 71.0, 28.0, 35.0, -20.0, 57.0, 35.0]], [[-4.0, 4.0, 3.0, 29.0, -65.0, -75.0, 70.0, 103.0, 77.0, -11.0, 86.0], [-25.0, -80.0, -113.0, 37.0, -56.0, 50.0, 123.0, -181.0, -152.0, 39.0, 9.0], [-116.0, -44.0, -83.0, -18.0, -88.0, 22.0, -151.0, -148.0, 204.0, -32.0, 45.0], [8.0, 48.0, 80.0, -86.0, 46.0, -170.0, 59.0, -61.0, -178.0, 118.0, -84.0], [-78.0, -81.0, 57.0, 119.0, -136.0, 12.0, 63.0, 73.0, -65.0, 63.0, -46.0], [101.0, -48.0, -53.0, 23.0, -91.0, 65.0, 76.0, -120.0, 68.0, 28.0, 43.0], [-106.0, 93.0, 258.0, -32.0, 53.0, -76.0, 80.0, 0.0, -31.0, -114.0, -76.0]]]);
    expect(b.shape).toEqual([3, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[38223.0, -17363.0, 12472.0, -152.0, 24543.0, 4303.0, -12429.0, -32816.0, 1253.0, -35941.0, 9288.0], [-25008.0, -7856.0, 53671.0, -31456.0, 6786.0, 46642.0, 7947.0, 22125.0, 29521.0, 466.0, 8399.0], [-29361.0, 12759.0, -35860.0, -24766.0, 9338.0, 24711.0, -5874.0, 15423.0, -14409.0, -17991.0, 13788.0], [-17733.0, 40722.0, -33123.0, -16626.0, -64855.0, 3078.0, 10800.0, -9912.0, -10297.0, -17305.0, -15222.0], [-38086.0, 18070.0, 38644.0, -44254.0, -44013.0, 29403.0, 19599.0, 34488.0, 8158.0, 22504.0, -3749.0]], [[34162.0, 25073.0, -11614.0, 14335.0, 4358.0, -6211.0, -19345.0, 3397.0, 12943.0, -38926.0, -19390.0], [-40367.0, -2753.0, 15888.0, 60.0, -46035.0, 57649.0, -23391.0, -49572.0, -13116.0, 67542.0, 44574.0], [43924.0, -2680.0, 12974.0, -3114.0, 39548.0, 18011.0, 34105.0, 33197.0, 39045.0, -22791.0, -7734.0], [20337.0, 30754.0, -23632.0, 20809.0, -8655.0, 12443.0, -48873.0, -667.0, -19845.0, -5564.0, -11492.0], [-15631.0, 29751.0, -3098.0, 28123.0, -22633.0, 46149.0, -65448.0, -21556.0, -38199.0, 37695.0, 36652.0]], [[-3331.0, 33615.0, 19933.0, -30983.0, 43550.0, -46105.0, -2447.0, 24527.0, -21353.0, -6010.0, -2811.0], [12864.0, -41741.0, -71860.0, 16137.0, -43752.0, 12598.0, 13112.0, -45719.0, -16791.0, 46137.0, 17207.0], [27503.0, 21186.0, 26615.0, -13762.0, 18370.0, -8944.0, 3329.0, 17151.0, 14021.0, -7097.0, -1920.0], [-18742.0, -19683.0, -5693.0, 16857.0, -12900.0, 33059.0, -20072.0, -15523.0, -4820.0, 1675.0, -17152.0], [17878.0, 9608.0, -11245.0, -2442.0, 24969.0, 18613.0, -8570.0, 21883.0, 10502.0, -21047.0, 11463.0]]], [[[-35496.0, -24985.0, 50219.0, -17881.0, 14843.0, -12222.0, -4007.0, 61092.0, -71050.0, 71384.0, 5321.0], [-2289.0, -2009.0, -46580.0, 19745.0, 11190.0, -31397.0, -8399.0, -9887.0, -24721.0, -2721.0, -746.0], [6489.0, 6552.0, -24274.0, 47248.0, 12251.0, -2684.0, 1864.0, -3418.0, 39600.0, 6285.0, -9720.0], [-6057.0, 21847.0, 4227.0, -33877.0, -34749.0, 21627.0, 3565.0, 5193.0, -17216.0, -10833.0, 165.0], [32108.0, -15819.0, -34721.0, 62565.0, 12872.0, -58394.0, -9175.0, -47550.0, -4992.0, -10020.0, -16010.0]], [[37638.0, 35460.0, -21079.0, 26576.0, 13462.0, 599.0, -44490.0, 19322.0, -13383.0, -31400.0, -17974.0], [-14317.0, 15441.0, -5543.0, 21824.0, -12085.0, 13123.0, -54978.0, -4831.0, -48226.0, 19407.0, 14998.0], [-64664.0, -4851.0, 7984.0, 16193.0, -48263.0, 45252.0, -71975.0, -38343.0, -85526.0, 86037.0, 51105.0], [27779.0, 16372.0, 377.0, 25999.0, -15821.0, 36954.0, -55107.0, -23574.0, -28189.0, -6011.0, 19692.0], [-35081.0, 1058.0, 22661.0, 6248.0, -462.0, -5727.0, -12158.0, 7854.0, -9577.0, 13933.0, 6361.0]], [[24269.0, 550.0, -19737.0, -206.0, -4986.0, -10521.0, 34314.0, 2173.0, -1150.0, 3099.0, 25916.0], [9026.0, -42772.0, -55457.0, 33157.0, -59476.0, 34100.0, -19048.0, -2675.0, 49671.0, 24006.0, 25404.0], [36.0, 6186.0, -11241.0, -10334.0, 5513.0, 12494.0, -18473.0, -18143.0, 44165.0, -25985.0, 17095.0], [-5730.0, -1288.0, 10497.0, 2947.0, 3438.0, 2400.0, -347.0, 1494.0, -19641.0, 4980.0, -14882.0], [-5037.0, -6115.0, -127.0, 10620.0, -14643.0, -12344.0, 10371.0, 20731.0, -5796.0, 11371.0, 4164.0]]]]);
});
test("matmul (2, 3, 5, 7) x (3, 7, 11, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([3, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1');
});
test("matmul (2, 3, 5, 7) x (3, 1, 11)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([3, 1, 11]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 7] but got: [6, 1].');
});
test("matmul (2, 3, 5, 7) x (3, 1, 11, 1)", async () => {
    const a = ones([2, 3, 5, 7]);
    const b = ones([3, 1, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0');
});
test("matmul (1, 2, 3, 5, 7) x (11,)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([11]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 30, 30x7,11');
});
test("matmul (1, 2, 3, 5, 7) x (11, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([11, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (30x7 and 11x1)');
});
test("matmul (1, 2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[[76.0, 64.0, -62.0, -8.0, 44.0, 49.0, -28.0], [59.0, 34.0, 95.0, -25.0, -25.0, -42.0, 49.0], [-194.0, 61.0, -144.0, 57.0, 66.0, 180.0, 78.0], [-153.0, 10.0, -19.0, -114.0, -52.0, -17.0, -86.0], [91.0, 170.0, -35.0, 52.0, 46.0, -82.0, -4.0]], [[-248.0, 41.0, -28.0, -16.0, 51.0, 112.0, -35.0], [7.0, -58.0, -15.0, 132.0, 85.0, 89.0, -53.0], [-63.0, -10.0, 1.0, -15.0, 147.0, 13.0, -146.0], [-27.0, -157.0, -94.0, -98.0, 36.0, -76.0, 65.0], [8.0, -75.0, 113.0, 87.0, -131.0, -45.0, 63.0]], [[-69.0, -123.0, -177.0, -79.0, 17.0, 49.0, -22.0], [146.0, 13.0, -1.0, 25.0, -100.0, 105.0, -243.0], [46.0, 119.0, -3.0, -4.0, 48.0, -73.0, 166.0], [-89.0, 32.0, 7.0, 107.0, 66.0, -228.0, 119.0], [-55.0, -306.0, 89.0, 52.0, 17.0, -79.0, -31.0]]], [[[-17.0, 159.0, 160.0, 45.0, -8.0, -44.0, -100.0], [36.0, 21.0, 89.0, 104.0, 134.0, -177.0, -46.0], [177.0, 48.0, -139.0, -64.0, 9.0, 105.0, -2.0], [3.0, 73.0, 44.0, -50.0, 0.0, -142.0, -161.0], [-14.0, 3.0, -81.0, 97.0, -99.0, 144.0, 15.0]], [[88.0, -12.0, -166.0, 97.0, -34.0, 97.0, 70.0], [-69.0, -93.0, -152.0, 88.0, -93.0, 10.0, 62.0], [-106.0, 142.0, -53.0, -260.0, 100.0, 133.0, 129.0], [-53.0, -0.0, 151.0, 92.0, -40.0, 109.0, -138.0], [-98.0, -87.0, -24.0, 126.0, -29.0, -24.0, 126.0]], [[-42.0, -71.0, -20.0, 47.0, -116.0, -24.0, -59.0], [-45.0, -173.0, -17.0, -91.0, -2.0, -16.0, 45.0], [171.0, 32.0, 164.0, -27.0, 85.0, -149.0, -32.0], [-66.0, 147.0, -58.0, 77.0, 9.0, 11.0, 60.0], [47.0, 14.0, -14.0, 112.0, -27.0, -93.0, 29.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[63.0, -47.0, 194.0, 127.0, 75.0, 45.0, 8.0, -57.0, 116.0, 20.0, -25.0], [-94.0, -49.0, 0.0, -38.0, -7.0, 20.0, -168.0, -90.0, 28.0, -17.0, -13.0], [-141.0, 86.0, -19.0, -153.0, 8.0, 19.0, 186.0, -107.0, 62.0, 71.0, 48.0], [47.0, -166.0, -169.0, -56.0, -151.0, -102.0, 94.0, -74.0, -17.0, -197.0, -84.0], [-121.0, -49.0, 75.0, -94.0, 20.0, 214.0, 22.0, 93.0, 129.0, -49.0, -2.0], [42.0, 72.0, -40.0, -15.0, -90.0, -166.0, -13.0, 93.0, -34.0, 149.0, -37.0], [-183.0, 46.0, 40.0, 111.0, -45.0, 82.0, 123.0, -158.0, 76.0, 58.0, 18.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[8996.0, -10628.0, 17494.0, 9175.0, 3694.0, 3324.0, -25541.0, 10207.0, 8782.0, 1127.0, -7441.0], [-21755.0, 8336.0, 15631.0, 1485.0, 9797.0, 13330.0, 16103.0, -28711.0, 16038.0, 10081.0, 7229.0], [-9673.0, -2403.0, -43663.0, -8362.0, -43126.0, -25420.0, -24520.0, 27312.0, -22371.0, 1740.0, -13031.0], [8058.0, 21359.0, -16715.0, -14923.0, 9877.0, -10776.0, -28655.0, 25461.0, -29374.0, 12906.0, 11544.0], [-11146.0, -32591.0, 16101.0, 4002.0, 5983.0, 24654.0, -27868.0, -23306.0, 20680.0, -28503.0, -7663.0]], [[-11344.0, 13850.0, -46931.0, -38233.0, -24180.0, -19788.0, -20223.0, 35315.0, -28973.0, 7666.0, 791.0], [17364.0, -20884.0, -19970.0, -17212.0, -23046.0, -15524.0, 13612.0, 21214.0, -75.0, -19921.0, -15646.0], [5602.0, -6956.0, -5041.0, -37153.0, 5958.0, 15842.0, -14941.0, 43442.0, 154.0, -11798.0, -390.0], [2262.0, 12900.0, 21450.0, 27378.0, 17755.0, 29505.0, 9239.0, 18989.0, 478.0, 5443.0, 10346.0], [-1858.0, 4652.0, -20803.0, 1687.0, -12513.0, -23265.0, 47312.0, -38557.0, -6226.0, -4313.0, 1952.0]], [[32486.0, 8845.0, 1763.0, 22641.0, 3119.0, -7170.0, -23205.0, 49402.0, -22224.0, 8899.0, -779.0], [70271.0, -10453.0, 2698.0, -2347.0, 6561.0, -54495.0, -32306.0, 27624.0, -18125.0, 4154.0, -14026.0], [-47305.0, -7559.0, 22817.0, 17012.0, 3257.0, 40803.0, 1865.0, -41268.0, 29840.0, -4129.0, 3088.0], [-43912.0, -28721.0, -16652.0, -9157.0, -6515.0, 47584.0, 24325.0, -40342.0, 14497.0, -53210.0, 3603.0], [15492.0, 8654.0, -17954.0, -15740.0, -278.0, 2002.0, 69998.0, 16436.0, -7791.0, -14225.0, 7588.0]]], [[[-19042.0, -8078.0, -16783.0, -44889.0, 397.0, -1743.0, -4762.0, -22827.0, 4499.0, -12512.0, 2102.0], [-22597.0, -33757.0, 3007.0, -30714.0, 8241.0, 47409.0, 22681.0, -17892.0, 28322.0, -49413.0, -184.0], [26917.0, -4974.0, 44190.0, 42863.0, 12311.0, -2856.0, -39931.0, 16118.0, 11785.0, 20551.0, -10284.0], [8272.0, -9264.0, 7436.0, -22066.0, 27641.0, 17901.0, -26713.0, 4483.0, -1438.0, -18703.0, 7644.0], [30098.0, -6648.0, -30155.0, 13880.0, -31981.0, -55863.0, -8769.0, 3832.0, -24738.0, 1986.0, -16585.0]], [[30015.0, -22056.0, 203.0, 41109.0, -21851.0, -26966.0, -12437.0, 1447.0, -4433.0, -8752.0, -20421.0], [30290.0, -11751.0, -30265.0, 28573.0, -24578.0, -33307.0, 522.0, 4540.0, -29153.0, -18284.0, -10822.0], [-54894.0, 47236.0, 31723.0, 6735.0, 14117.0, 33483.0, -42664.0, 19460.0, 10996.0, 65322.0, 17301.0], [14366.0, 3665.0, -41579.0, -48179.0, -21059.0, -46870.0, 17039.0, 8277.0, -17704.0, 1734.0, -5592.0], [-9247.0, -8622.0, -36025.0, 4548.0, -30049.0, -11348.0, 36384.0, -18177.0, -10783.0, -21854.0, -4941.0]], [[32882.0, -2827.0, -25811.0, 2507.0, -7415.0, -34162.0, 2793.0, 3748.0, -27531.0, -11626.0, -2877.0], [2882.0, 25252.0, 9262.0, 13979.0, 10816.0, 9392.0, 22687.0, 17904.0, -5865.0, 19085.0, 11608.0], [-27315.0, -7384.0, 45676.0, -12386.0, 34540.0, 54505.0, 23829.0, -29073.0, 44958.0, -8383.0, 10216.0], [-17786.0, -18760.0, -22080.0, -3757.0, -21580.0, -3966.0, -21339.0, -16580.0, -3098.0, -18428.0, -8858.0], [2937.0, -26730.0, -6689.0, 8459.0, -7072.0, 2743.0, 10130.0, -26471.0, 4955.0, -33208.0, -7420.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (7, 11, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2');
});
test("matmul (1, 2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[[229.0, 24.0, 58.0, -100.0, -5.0, -185.0, 91.0], [-172.0, -55.0, -82.0, 168.0, 1.0, -144.0, 35.0], [-67.0, -75.0, 33.0, -77.0, 112.0, 11.0, 25.0], [152.0, 37.0, 0.0, -106.0, 70.0, 165.0, -6.0], [3.0, 55.0, -48.0, -18.0, 89.0, -1.0, 131.0]], [[-33.0, -77.0, 60.0, 32.0, 48.0, -118.0, -180.0], [-42.0, -45.0, 93.0, -30.0, 15.0, -8.0, 186.0], [-34.0, 4.0, -89.0, 102.0, -33.0, -113.0, -12.0], [-34.0, -69.0, -5.0, -38.0, 41.0, -1.0, -92.0], [87.0, 87.0, 103.0, -23.0, -92.0, -19.0, 152.0]], [[-18.0, 169.0, -74.0, -229.0, -25.0, -83.0, -76.0], [-41.0, 70.0, 42.0, 47.0, 183.0, -66.0, -93.0], [-46.0, 30.0, 151.0, -8.0, -67.0, 3.0, 108.0], [63.0, -82.0, -352.0, -161.0, -69.0, -111.0, -63.0], [37.0, 25.0, 137.0, 4.0, -44.0, -1.0, 40.0]]], [[[66.0, 148.0, -44.0, 77.0, 125.0, -38.0, 99.0], [50.0, 4.0, -134.0, -55.0, 9.0, 204.0, 41.0], [135.0, -29.0, -66.0, 15.0, -59.0, -26.0, -9.0], [-92.0, 67.0, -99.0, 71.0, -15.0, 109.0, -72.0], [-46.0, 5.0, -28.0, 145.0, -101.0, 6.0, -85.0]], [[-69.0, -97.0, 117.0, 61.0, 232.0, 103.0, -134.0], [-91.0, -111.0, 154.0, 106.0, -68.0, 58.0, 33.0], [22.0, -50.0, 114.0, 93.0, 63.0, -177.0, 104.0], [-90.0, 168.0, 117.0, -143.0, -16.0, 183.0, 107.0], [4.0, -21.0, 13.0, -98.0, 37.0, 398.0, -27.0]], [[-20.0, -24.0, 159.0, 144.0, -129.0, -266.0, 150.0], [-1.0, 43.0, -59.0, 171.0, -114.0, 22.0, 213.0], [3.0, 100.0, -81.0, -135.0, 53.0, 38.0, 34.0], [-23.0, 129.0, 100.0, 32.0, 75.0, 28.0, -49.0], [81.0, 116.0, 14.0, 95.0, -20.0, 12.0, -58.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[[-127.0, -92.0, 230.0, 31.0, 157.0, 39.0, 117.0, -9.0, -30.0, 73.0, -257.0], [-20.0, -46.0, -15.0, 46.0, 115.0, 1.0, 137.0, -37.0, 7.0, 5.0, -78.0], [70.0, 93.0, -153.0, -50.0, 48.0, 137.0, -182.0, 20.0, 73.0, -63.0, 62.0], [-89.0, 37.0, 8.0, -30.0, 72.0, 11.0, 136.0, -179.0, -56.0, -197.0, -86.0], [-153.0, 33.0, -11.0, -57.0, 142.0, -1.0, 43.0, 69.0, 223.0, 51.0, -68.0], [57.0, 27.0, -45.0, 184.0, 98.0, 3.0, -35.0, 51.0, 204.0, -151.0, -8.0], [-71.0, 61.0, -118.0, -37.0, -18.0, -279.0, 234.0, 240.0, -30.0, -10.0, -83.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[-32844.0, -20087.0, 40278.0, -28819.0, 13819.0, -10138.0, 33479.0, 28171.0, -38453.0, 59653.0, -54262.0], [-8594.0, 15224.0, -22506.0, -36650.0, -39769.0, -26347.0, 23386.0, -27004.0, -40822.0, -19316.0, 27141.0], [888.0, 15352.0, -24627.0, -10152.0, -6572.0, -6068.0, -24311.0, 32110.0, 34676.0, 11625.0, 21958.0], [-11489.0, -13209.0, 26070.0, 36186.0, 46705.0, 6898.0, 4268.0, 28042.0, 51085.0, 10878.0, -38416.0], [-26214.0, 2965.0, -9327.0, -4541.0, 13378.0, -43243.0, 48690.0, 37730.0, 13512.0, 10444.0, -23406.0]], [[5793.0, 760.0, 10663.0, -26313.0, -10360.0, 57026.0, -56904.0, -47288.0, -4929.0, 9188.0, 28075.0], [-543.0, 25098.0, -45207.0, -16331.0, -11467.0, -41205.0, 12364.0, 54540.0, 5547.0, -3127.0, 6256.0], [-11610.0, -6431.0, 13417.0, -17947.0, -17350.0, -9351.0, 26368.0, -30800.0, -41212.0, -1449.0, -1720.0], [8932.0, 145.0, 4126.0, -1955.0, -8869.0, 23126.0, -37419.0, -9741.0, 13999.0, 8136.0, 21934.0], [-1331.0, 2445.0, -13307.0, -1637.0, 9290.0, -25035.0, 32501.0, 31338.0, -22146.0, 1485.0, -26989.0]], [[18597.0, -29175.0, 15793.0, 6751.0, -13747.0, 7790.0, -12583.0, 9222.0, -11082.0, 61324.0, 15222.0], [-22594.0, 4781.0, -4599.0, -20695.0, 28205.0, 30308.0, -8042.0, -22853.0, 32289.0, 5681.0, -558.0], [19278.0, 21057.0, -46339.0, -6981.0, -8264.0, -11221.0, -7556.0, 25206.0, -4508.0, -16095.0, 15100.0], [-7969.0, -49834.0, 81476.0, 6451.0, -47569.0, -30307.0, 24481.0, -1296.0, -55285.0, 71954.0, -6964.0], [7870.0, 9296.0, -16985.0, -3829.0, 8482.0, 9162.0, -9133.0, 7279.0, -2374.0, -9086.0, -3629.0]]], [[[-49595.0, -4985.0, 8961.0, -9036.0, 43058.0, -30319.0, 76349.0, 9714.0, 8685.0, 4284.0, -54269.0], [-3575.0, -10975.0, 17385.0, 45590.0, 18450.0, -27845.0, 26147.0, 27432.0, 34219.0, -7808.0, -22387.0], [-14336.0, -19867.0, 44584.0, 4613.0, 5008.0, -1149.0, 22141.0, -11704.0, -28102.0, 11920.0, -32848.0], [10715.0, -3142.0, -2694.0, 26625.0, 3469.0, 4127.0, 4781.0, -29096.0, 13077.0, -30635.0, 12298.0], [12707.0, -1593.0, 5660.0, 5860.0, -9775.0, 19804.0, -4324.0, -53349.0, -27498.0, -35341.0, 11101.0]], [[-6647.0, 26211.0, -23203.0, -3595.0, 33470.0, 51375.0, -59345.0, -15268.0, 83284.0, -27291.0, 21829.0], [26490.0, 33057.0, -47735.0, -5480.0, -16594.0, 9639.0, -36698.0, -4782.0, 2937.0, -50338.0, 33898.0], [-29203.0, 17963.0, -15888.0, -50115.0, -400.0, -12161.0, 20864.0, 7565.0, -23075.0, 4753.0, -14184.0], [34269.0, 17082.0, -62950.0, 34003.0, 14246.0, -18174.0, -10311.0, 56440.0, 50979.0, -14449.0, 20321.0], [28486.0, 8501.0, -16669.0, 73570.0, 36525.0, 9528.0, -36760.0, 34914.0, 96423.0, -39267.0, 6385.0]], [[-4741.0, 20770.0, -31726.0, -61135.0, -34986.0, -19956.0, 23881.0, -7995.0, -83556.0, -7878.0, 2936.0], [-16509.0, 8779.0, -15350.0, 2432.0, -3598.0, -65445.0, 83938.0, 11005.0, -40876.0, -41094.0, -31564.0], [-4393.0, -12555.0, 4198.0, 15506.0, 9001.0, -21790.0, 19338.0, 32573.0, 20808.0, 29042.0, -8713.0], [-1907.0, 6908.0, -18572.0, 1951.0, 32604.0, 26964.0, -8087.0, -13451.0, 31008.0, -13551.0, -1960.0], [-12220.0, -11845.0, 22032.0, 9791.0, 32949.0, 22476.0, 20889.0, -36434.0, -6188.0, -15356.0, -31089.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (1, 7, 11, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([1, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2');
});
test("matmul (1, 2, 3, 5, 7) x (3, 7, 11)", async () => {
    const a = tensor([[[[[48.0, -127.0, -39.0, -43.0, -197.0, 113.0, -69.0], [203.0, -11.0, -9.0, 71.0, 136.0, -17.0, 14.0], [43.0, 63.0, 5.0, -72.0, -138.0, 111.0, 17.0], [92.0, 115.0, -93.0, 118.0, 143.0, -37.0, 257.0], [-193.0, 28.0, 53.0, 11.0, -13.0, -119.0, -62.0]], [[-82.0, -148.0, -207.0, -5.0, 7.0, 229.0, 54.0], [-35.0, -130.0, -40.0, 9.0, -22.0, -23.0, -63.0], [-268.0, 85.0, 30.0, -98.0, 197.0, 106.0, -104.0], [15.0, -10.0, -67.0, -89.0, 23.0, 22.0, 29.0], [186.0, -124.0, -37.0, -12.0, 0.0, -2.0, -100.0]], [[101.0, 1.0, 55.0, 133.0, 113.0, -140.0, -22.0], [-95.0, -79.0, -105.0, 126.0, -233.0, -132.0, 158.0], [43.0, 2.0, -86.0, -56.0, -4.0, 45.0, 167.0], [-97.0, -56.0, -225.0, 33.0, 29.0, 25.0, 138.0], [-96.0, -54.0, -84.0, 133.0, 19.0, -32.0, -206.0]]], [[[-42.0, -98.0, -17.0, 69.0, -38.0, 54.0, 24.0], [15.0, -127.0, 5.0, -20.0, -61.0, 23.0, -9.0], [173.0, -22.0, -8.0, 67.0, -24.0, -15.0, 201.0], [41.0, -54.0, -94.0, -45.0, 80.0, 132.0, 151.0], [22.0, 87.0, 180.0, 61.0, 40.0, 57.0, 102.0]], [[73.0, -62.0, 0.0, 72.0, 23.0, 33.0, 6.0], [9.0, 18.0, 89.0, 84.0, -88.0, -47.0, -58.0], [29.0, -27.0, -31.0, -20.0, 49.0, 16.0, -81.0], [-6.0, 44.0, -2.0, 7.0, 134.0, -60.0, -60.0], [-96.0, 64.0, 58.0, -153.0, -3.0, 75.0, 15.0]], [[60.0, -31.0, 17.0, 100.0, -1.0, -113.0, 53.0], [107.0, 73.0, 56.0, -94.0, 30.0, -184.0, 35.0], [-43.0, -11.0, 20.0, -86.0, 2.0, 115.0, 53.0], [-55.0, -102.0, -50.0, 115.0, -14.0, 138.0, 120.0], [-53.0, 44.0, 150.0, -34.0, -75.0, -60.0, -172.0]]]]]);
    expect(a.shape).toEqual([1, 2, 3, 5, 7]);
    const b = tensor([[[-116.0, -189.0, -4.0, -1.0, 17.0, 42.0, 177.0, 74.0, -73.0, -64.0, -62.0], [-13.0, -193.0, -176.0, -5.0, -29.0, -57.0, -51.0, 55.0, -73.0, 11.0, 163.0], [-64.0, -82.0, 63.0, 25.0, -82.0, -52.0, -49.0, 26.0, 190.0, 128.0, -171.0], [-51.0, -68.0, -161.0, -12.0, 228.0, 43.0, -36.0, 49.0, -68.0, -62.0, -50.0], [-4.0, -81.0, 59.0, 103.0, -9.0, 59.0, 61.0, -58.0, -123.0, 38.0, 79.0], [41.0, 112.0, 48.0, -16.0, 44.0, -4.0, 183.0, 168.0, 68.0, -79.0, 20.0], [-23.0, -32.0, 46.0, 120.0, 15.0, 53.0, -9.0, 50.0, -35.0, 173.0, -72.0]], [[-112.0, -9.0, 16.0, 204.0, 32.0, 32.0, -49.0, -51.0, -80.0, -59.0, -12.0], [-98.0, 78.0, 4.0, -14.0, -81.0, 86.0, 61.0, -133.0, 157.0, 66.0, -218.0], [-251.0, 108.0, -5.0, 29.0, 35.0, 88.0, 57.0, 144.0, 48.0, -148.0, -72.0], [212.0, -7.0, -3.0, -36.0, -19.0, -37.0, -127.0, -33.0, 171.0, -85.0, -54.0], [-17.0, -217.0, -87.0, 67.0, 67.0, -16.0, -83.0, 142.0, 37.0, -206.0, -102.0], [219.0, -73.0, -32.0, -38.0, 109.0, -239.0, -77.0, 73.0, 19.0, 10.0, 107.0], [107.0, 28.0, 110.0, -143.0, -11.0, -44.0, 80.0, 64.0, 97.0, -94.0, 167.0]], [[-24.0, 123.0, -101.0, 138.0, -5.0, -79.0, -234.0, -35.0, -67.0, -167.0, -156.0], [-135.0, 125.0, -62.0, 45.0, 68.0, 34.0, -50.0, 14.0, -26.0, -94.0, -19.0], [166.0, -108.0, 46.0, -17.0, 48.0, -81.0, -76.0, -64.0, 78.0, 185.0, 23.0], [24.0, 101.0, 6.0, -130.0, 141.0, 170.0, -14.0, -12.0, -83.0, 18.0, -167.0], [63.0, 33.0, -17.0, 31.0, -173.0, -120.0, 75.0, 34.0, 209.0, 51.0, -61.0], [59.0, 31.0, -56.0, 5.0, -88.0, 52.0, 137.0, 105.0, -94.0, -82.0, -135.0], [-141.0, -143.0, -60.0, 28.0, -58.0, -86.0, 65.0, 56.0, -167.0, -144.0, -53.0]]]);
    expect(b.shape).toEqual([3, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[7780.0, 52382.0, 17253.0, -30251.0, 3603.0, -6298.0, 27715.0, 20406.0, 35611.0, -35145.0, -23193.0], [-28013.0, -53702.0, -3022.0, 14735.0, 18934.0, 21508.0, 39436.0, 7618.0, -38928.0, -9734.0, -6994.0], [2257.0, 7266.0, -1385.0, -13319.0, -11541.0, -12826.0, 18487.0, 30751.0, 22035.0, -8027.0, 442.0], [-20233.0, -63932.0, -26982.0, 41753.0, 33699.0, 29425.0, 10367.0, 14837.0, -69905.0, 28975.0, 15097.0], [14670.0, 15688.0, -11919.0, -5629.0, -11980.0, -15562.0, -60594.0, -33163.0, 17044.0, 16943.0, 7974.0]], [[130395.0, -49851.0, -2851.0, -36434.0, 27050.0, -90602.0, -30068.0, 15390.0, -17619.0, 21903.0, 81229.0], [17204.0, -9519.0, -5187.0, 1605.0, 4551.0, -7532.0, -11081.0, 4183.0, -25353.0, 8864.0, 20416.0], [2117.0, -40431.0, -35775.0, -27421.0, 13348.0, -18910.0, -360.0, 38973.0, 18682.0, -4434.0, -38302.0], [4779.0, -13313.0, 1287.0, 1019.0, 4256.0, -9885.0, 4856.0, 582.0, -17123.0, 8692.0, 16481.0], [-13075.0, -17912.0, -8235.0, 53415.0, 15811.0, -2646.0, -25109.0, -4472.0, -47914.0, -3282.0, 11198.0]], [[11724.0, 22576.0, 304.0, -2055.0, 15003.0, -8738.0, -41861.0, -20727.0, 26909.0, 16019.0, -23548.0], [-46206.0, -31869.0, 12292.0, -34719.0, 50590.0, 42252.0, 7107.0, -5507.0, -72904.0, -17677.0, 16523.0], [-38066.0, -13447.0, -21231.0, 19543.0, -25057.0, -17425.0, 13878.0, 18640.0, -37948.0, -52229.0, -14054.0], [-42826.0, -9300.0, -7056.0, -11483.0, -24691.0, 15546.0, 56706.0, 27954.0, -31669.0, -40011.0, -6948.0], [27197.0, 33040.0, 23807.0, -36879.0, 23006.0, 48934.0, 13337.0, -7866.0, 31626.0, 41219.0, 5938.0]]], [[[5529.0, 31912.0, 6690.0, -2619.0, 22332.0, 6487.0, 3261.0, 6917.0, 9804.0, -6402.0, -17563.0], [2005.0, 30431.0, 22918.0, -6746.0, 394.0, 2581.0, 10176.0, 227.0, 19868.0, -6169.0, -25197.0], [-27829.0, -38519.0, -1001.0, 20821.0, 22082.0, 21114.0, 23705.0, 23589.0, -22202.0, 18554.0, -32962.0], [5876.0, 16913.0, 28665.0, 22667.0, 7064.0, 19948.0, 43914.0, 20501.0, -20000.0, 6275.0, 5068.0], [-18483.0, -39977.0, -4093.0, 18759.0, 677.0, -3234.0, 394.0, 26438.0, 17481.0, 33470.0, -24057.0]], [[20642.0, -13229.0, -1693.0, 12597.0, 11062.0, -14179.0, -20473.0, 8206.0, -1202.0, -19491.0, 10939.0], [-22306.0, 31250.0, 2299.0, 5325.0, -10032.0, 21753.0, 1345.0, -12448.0, 10967.0, 3455.0, -20715.0], [-3057.0, -19644.0, -13114.0, 20373.0, 8328.0, -4426.0, -14074.0, 1250.0, -17207.0, 475.0, -7963.0], [-23492.0, -23157.0, -16269.0, 17688.0, -861.0, 17993.0, -9327.0, 4743.0, 6487.0, -19605.0, -39862.0], [-24433.0, 8787.0, -1600.0, -18486.0, 4490.0, -5340.0, 27019.0, 15794.0, -2882.0, 14267.0, 2122.0]], [[-6236.0, 654.0, 409.0, -5516.0, 19551.0, -485.0, -27293.0, -13753.0, -8626.0, -578.0, -12573.0], [-19284.0, -2975.0, -5627.0, 30309.0, 2835.0, -42665.0, -52311.0, -21519.0, 20824.0, -4485.0, 20062.0], [3211.0, -21458.0, -4225.0, 6532.0, -25239.0, -12035.0, 29646.0, 16214.0, -7378.0, -6593.0, 3283.0], [-110.0, -15844.0, -4421.0, -22664.0, -9528.0, 23013.0, 45816.0, 23051.0, -43046.0, -17717.0, -33973.0], [35403.0, -392.0, 24276.0, -10905.0, 33894.0, 8425.0, -25747.0, -25203.0, 35618.0, 57716.0, 38351.0]]]]]);
});
test("matmul (1, 2, 3, 5, 7) x (3, 7, 11, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([3, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2');
});
test("matmul (1, 2, 3, 5, 7) x (3, 1, 11)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([3, 1, 11]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [6, 7] but got: [6, 1].');
});
test("matmul (1, 2, 3, 5, 7) x (3, 1, 11, 1)", async () => {
    const a = ones([1, 2, 3, 5, 7]);
    const b = ones([3, 1, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (2, 2, 3, 5, 7) x (11,)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([11]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 60, 60x7,11');
});
test("matmul (2, 2, 3, 5, 7) x (11, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([11, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 11x1)');
});
test("matmul (2, 2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[[150.0, -36.0, 97.0, 54.0, 135.0, -57.0, 114.0], [197.0, -58.0, 127.0, 37.0, 102.0, 34.0, -185.0], [97.0, -61.0, -38.0, -141.0, -76.0, -32.0, -57.0], [147.0, 19.0, 18.0, 160.0, 50.0, 128.0, 6.0], [-74.0, 119.0, 98.0, 174.0, 71.0, -59.0, -110.0]], [[-108.0, 54.0, 60.0, 19.0, 16.0, -47.0, -46.0], [104.0, -105.0, 184.0, -184.0, 32.0, -74.0, -158.0], [119.0, 62.0, -73.0, -43.0, 2.0, -58.0, 0.0], [-23.0, 17.0, 109.0, 205.0, -0.0, 125.0, -4.0], [62.0, -135.0, -59.0, 44.0, 11.0, -38.0, 106.0]], [[-36.0, 52.0, 188.0, -8.0, -45.0, 64.0, 48.0], [135.0, 100.0, 43.0, -100.0, 87.0, -99.0, -90.0], [124.0, -155.0, -13.0, -43.0, -85.0, 119.0, 85.0], [11.0, -127.0, -83.0, -123.0, -132.0, 8.0, -58.0], [-39.0, 12.0, -92.0, -68.0, -67.0, 42.0, 15.0]]], [[[-17.0, -197.0, 41.0, 146.0, -106.0, 11.0, -100.0], [-48.0, -7.0, -34.0, -44.0, -62.0, 15.0, 185.0], [31.0, -10.0, 45.0, -44.0, -57.0, 90.0, 35.0], [-70.0, 88.0, 74.0, -84.0, 115.0, 43.0, -126.0], [177.0, 57.0, -137.0, -156.0, 109.0, 118.0, 21.0]], [[71.0, 100.0, -104.0, 104.0, 11.0, 74.0, -114.0], [-70.0, 48.0, -76.0, -105.0, -160.0, -3.0, 86.0], [-92.0, 155.0, 16.0, -244.0, -82.0, -117.0, 138.0], [-9.0, 72.0, -32.0, -58.0, 6.0, 121.0, 103.0], [-107.0, -0.0, 37.0, -165.0, 10.0, 126.0, -16.0]], [[-92.0, 106.0, 83.0, 76.0, 85.0, -8.0, -55.0], [-114.0, -168.0, -70.0, 165.0, 231.0, 19.0, -65.0], [61.0, 116.0, 13.0, -16.0, -28.0, -41.0, -175.0], [89.0, 182.0, -96.0, 38.0, 58.0, -223.0, 155.0], [-9.0, -109.0, 14.0, -16.0, 32.0, -41.0, -50.0]]]], [[[[71.0, 48.0, 21.0, -54.0, -144.0, 108.0, 93.0], [-26.0, 37.0, -37.0, -91.0, -150.0, -28.0, -95.0], [299.0, -98.0, -209.0, 37.0, 88.0, -271.0, -71.0], [-57.0, 85.0, 13.0, -35.0, 38.0, 16.0, 66.0], [-129.0, -145.0, -60.0, -20.0, 91.0, -197.0, -70.0]], [[-79.0, 29.0, 42.0, -5.0, -83.0, 38.0, -70.0], [206.0, -162.0, 55.0, 25.0, 96.0, 130.0, 53.0], [43.0, -17.0, 119.0, 35.0, -53.0, -101.0, -30.0], [30.0, 101.0, 50.0, 38.0, -48.0, -5.0, 7.0], [-24.0, -18.0, -51.0, 132.0, 161.0, -167.0, 80.0]], [[203.0, 24.0, -86.0, 190.0, -70.0, 260.0, 28.0], [83.0, 159.0, 35.0, 76.0, -13.0, 30.0, -104.0], [-163.0, 46.0, -53.0, -9.0, -20.0, -70.0, -84.0], [240.0, 147.0, -180.0, -27.0, 12.0, 11.0, 77.0], [-66.0, 27.0, 12.0, -88.0, 28.0, 55.0, -43.0]]], [[[-38.0, 157.0, 12.0, -98.0, -135.0, 6.0, 1.0], [174.0, -121.0, -14.0, 148.0, 104.0, 119.0, -62.0], [241.0, 74.0, 117.0, -95.0, 165.0, 125.0, -82.0], [-19.0, 86.0, -77.0, 11.0, -321.0, 13.0, 101.0], [3.0, 85.0, 71.0, -51.0, -56.0, 150.0, -62.0]], [[-84.0, -125.0, 39.0, 72.0, 34.0, 119.0, 22.0], [35.0, -116.0, -59.0, 177.0, 112.0, -86.0, 166.0], [193.0, -69.0, -79.0, 99.0, -90.0, -57.0, -59.0], [40.0, -29.0, -185.0, 199.0, 113.0, 179.0, 30.0], [-103.0, -178.0, -61.0, 88.0, 95.0, 107.0, -10.0]], [[69.0, -169.0, -49.0, 59.0, 22.0, 2.0, -81.0], [-62.0, 26.0, 189.0, -64.0, -73.0, 96.0, 67.0], [-220.0, 1.0, -14.0, 57.0, -53.0, 89.0, 8.0], [123.0, -66.0, -168.0, 188.0, 154.0, 39.0, 155.0], [25.0, -69.0, -3.0, -37.0, -43.0, -121.0, 11.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[-8.0, 89.0, 85.0, -108.0, 15.0, 119.0, 197.0, -98.0, 30.0, -31.0, 87.0], [-23.0, -29.0, 66.0, 51.0, 194.0, -22.0, 102.0, -21.0, 54.0, 465.0, -94.0], [-3.0, -92.0, 1.0, 62.0, -128.0, -96.0, 62.0, 65.0, 117.0, 185.0, 63.0], [82.0, 9.0, -62.0, 27.0, 217.0, 213.0, -27.0, 145.0, -86.0, -50.0, -29.0], [36.0, 74.0, 36.0, 50.0, 227.0, -47.0, -41.0, 177.0, -60.0, 105.0, 97.0], [-110.0, -155.0, 164.0, -62.0, -149.0, -23.0, 84.0, 164.0, -8.0, 39.0, -41.0], [-7.0, 31.0, 122.0, -63.0, 62.0, 157.0, 141.0, -77.0, -13.0, -2.0, -165.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[14097.0, 28315.0, 16543.0, -7462.0, 40774.0, 33696.0, 36185.0, 5960.0, 135.0, 5579.0, 17601.0], [3638.0, 4407.0, -2572.0, -714.0, -9906.0, -14213.0, 12357.0, 33407.0, 10468.0, 974.0, 68544.0], [-9638.0, 10198.0, -2015.0, -17975.0, -52130.0, -18141.0, 6729.0, -45451.0, 12853.0, -40466.0, 19213.0], [-869.0, -3638.0, 27371.0, -15285.0, 30957.0, 45075.0, 37241.0, 38945.0, -10320.0, 9838.0, 6109.0], [21645.0, -6498.0, -29666.0, 38973.0, 65278.0, -3020.0, -24439.0, 47714.0, -1650.0, 72433.0, 10960.0]], [[7068.0, -9484.0, -19478.0, 25263.0, 13082.0, -22646.0, -23651.0, 14771.0, 5076.0, 38547.0, -174.0], [-3659.0, 2657.0, -16758.0, 5995.0, -73796.0, -66778.0, -3652.0, -17013.0, 35528.0, -8019.0, 68054.0], [767.0, 24260.0, 7360.0, -11681.0, 22922.0, 11886.0, 21448.0, -33102.0, 2419.0, 11734.0, 3745.0], [2554.0, -30222.0, 6578.0, 8146.0, 14613.0, 26587.0, 8362.0, 59515.0, -5597.0, 23416.0, -7142.0], [10228.0, 25247.0, 669.0, -19823.0, 6571.0, 42383.0, 4901.0, -13143.0, -17851.0, -78351.0, -1774.0]], [[-11124.0, -33842.0, 15788.0, 8738.0, -33027.0, -17001.0, 24073.0, 12331.0, 25976.0, 58151.0, -10853.0], [2943.0, 23252.0, 234.0, 6644.0, 23141.0, -27505.0, 17588.0, -20942.0, 19823.0, 60724.0, 35302.0], [-17659.0, -5760.0, 29789.0, -40247.0, -67633.0, 24858.0, 34439.0, -18051.0, 570.0, -80628.0, -1363.0], [-12230.0, -1615.0, -10420.0, -19574.0, -75292.0, -17214.0, -14706.0, -39227.0, 2949.0, -82033.0, 7671.0], [-12401.0, -6970.0, 7907.0, -9615.0, -21774.0, -6019.0, -1937.0, -18396.0, -1949.0, -12258.0, -19041.0]]], [[[12190.0, -10907.0, -37670.0, -1409.0, -43940.0, 18502.0, -33673.0, 20380.0, -11335.0, -101294.0, 21155.0], [-8138.0, -2515.0, 20950.0, -14154.0, -12113.0, 19948.0, 18797.0, -26498.0, -817.0, -12152.0, -41538.0], [-15958.0, -18570.0, 21726.0, -12891.0, -40962.0, -3679.0, 23897.0, -4307.0, 11684.0, 2369.0, -7246.0], [-8282.0, -18407.0, 960.0, 25390.0, 208.0, -61438.0, -16827.0, 34751.0, 12928.0, 74984.0, 22918.0], [-24311.0, 15727.0, 54180.0, -32104.0, 5860.0, -4807.0, 44805.0, -13040.0, -1982.0, 19478.0, 8204.0]], [[-974.0, -267.0, 4707.0, -3064.0, 40748.0, 18268.0, 4622.0, 22123.0, -13352.0, 24128.0, 4052.0], [-14958.0, -10284.0, 7892.0, -10771.0, -35336.0, -3364.0, 7663.0, -49747.0, 9136.0, -1409.0, -41932.0], [-13933.0, -6.0, 12250.0, 6705.0, -18931.0, -39655.0, 18258.0, -72907.0, 32528.0, 76638.0, -40417.0], [-20059.0, -15585.0, 40177.0, -12597.0, -4938.0, 1169.0, 29594.0, 1855.0, 2195.0, 35882.0, -29259.0], [-26173.0, -33698.0, 20244.0, 3091.0, -59642.0, -57310.0, -6412.0, 12632.0, 13909.0, 24408.0, -3749.0]], [[8606.0, -12389.0, -10415.0, 30751.0, 42129.0, -17506.0, -16130.0, 41173.0, 1818.0, 72420.0, 2705.0], [25197.0, 14785.0, -27576.0, 18326.0, 56039.0, 10496.0, -65429.0, 83083.0, -48039.0, -70660.0, 29032.0], [220.0, -417.0, -15236.0, 11869.0, 7186.0, -25165.0, -1884.0, -8094.0, 15274.0, 51065.0, 23526.0], [24039.0, 55479.0, 1551.0, 1705.0, 113180.0, 50635.0, 29864.0, -51515.0, -5713.0, 59294.0, -27321.0], [7237.0, 8101.0, -18625.0, 3141.0, -16272.0, -11836.0, -23397.0, 4551.0, -4084.0, -45155.0, 23844.0]]]], [[[[-23878.0, -22004.0, 36446.0, -25131.0, -47043.0, 12760.0, 49732.0, -29368.0, 18390.0, 15610.0, -29187.0], [-9649.0, -10507.0, -15745.0, 165.0, -43991.0, -26960.0, -10782.0, -37656.0, 15174.0, -936.0, -3159.0], [36998.0, 95330.0, -33494.0, -23574.0, 76207.0, 56632.0, -1433.0, -58865.0, -26146.0, -96541.0, 52347.0], [-5262.0, -6671.0, 14992.0, 7102.0, 16710.0, -9148.0, 8284.0, 3839.0, 4145.0, 49929.0, -18975.0], [28343.0, 33163.0, -56927.0, 23451.0, 18945.0, -21397.0, -73532.0, -1924.0, -19974.0, -71514.0, 27661.0]], [[-7249.0, -25983.0, -9745.0, 10384.0, -30863.0, -23099.0, -13141.0, 6069.0, 10126.0, 16861.0, -4867.0], [-7252.0, 6794.0, 36565.0, -33024.0, -24245.0, 28942.0, 41250.0, 24645.0, -5772.0, -57747.0, 31127.0], [11972.0, 4490.0, -21650.0, 8314.0, -9132.0, 1626.0, 2629.0, -14682.0, 15663.0, 1583.0, 15771.0], [-824.0, -7077.0, 5216.0, 3506.0, 12173.0, 8112.0, 20821.0, -6156.0, 11765.0, 48136.0, -10442.0], [35189.0, 44545.0, -23295.0, 15440.0, 97710.0, 39386.0, -22639.0, 13504.0, -28375.0, -13429.0, 1827.0]], [[-17654.0, -17619.0, 50509.0, -42286.0, 7045.0, 74061.0, 60635.0, 29656.0, -17260.0, -17809.0, -17593.0], [-1234.0, -8596.0, 4636.0, 7409.0, 30234.0, 2800.0, 21076.0, 12449.0, 10527.0, 74050.0, 6945.0], [7235.0, -4280.0, -32762.0, 25053.0, 11992.0, -27876.0, -47366.0, 1706.0, -4981.0, 12426.0, -6793.0], [-8292.0, 34984.0, 43226.0, -35245.0, 55158.0, 48127.0, 63132.0, -44223.0, -5409.0, 30500.0, -15487.0], [-12086.0, -16339.0, 6422.0, 7572.0, -20889.0, -37676.0, -9719.0, 11208.0, 6889.0, 26392.0, 2584.0]]], [[[-16906.0, -20810.0, 9466.0, 3024.0, -24391.0, -23638.0, 18098.0, -35991.0, 25209.0, 67360.0, -27972.0], [4657.0, 8944.0, 13310.0, -20107.0, 15077.0, 38877.0, 14062.0, 48737.0, -22066.0, -55964.0, 36777.0], [-19007.0, -2023.0, 47812.0, -11899.0, -3874.0, -27920.0, 57017.0, 24677.0, 23251.0, 75698.0, 48547.0], [-14386.0, -19640.0, 6200.0, -21258.0, -39900.0, 36227.0, 28452.0, -65816.0, 11962.0, -7616.0, -63242.0], [-24456.0, -38505.0, 24118.0, -1158.0, -42526.0, -29744.0, 21194.0, 14603.0, 20339.0, 55211.0, -3129.0]], [[-2686.0, -22038.0, 3609.0, -5.0, -23527.0, 3465.0, -17120.0, 47672.0, -14177.0, -43739.0, -400.0], [29409.0, 40264.0, -5534.0, -8101.0, 72512.0, 72858.0, -1784.0, 13774.0, -35529.0, -66716.0, -7901.0], [11841.0, 27683.0, -14152.0, -23837.0, 5509.0, 49434.0, 13995.0, -28980.0, -9070.0, -69188.0, 18771.0], [1388.0, 4759.0, 26047.0, -19234.0, 62677.0, 60827.0, 2712.0, 60566.0, -47727.0, -40114.0, -12548.0], [4037.0, -7466.0, -6272.0, -614.0, -4171.0, 7763.0, -40922.0, 57760.0, -33833.0, -81094.0, 7854.0]], [[9459.0, 14888.0, -17758.0, -11437.0, -13002.0, 15403.0, -20431.0, 12616.0, -18146.0, -90189.0, 32508.0], [-19574.0, -42441.0, 21893.0, 4189.0, -60687.0, -27984.0, 24388.0, 6199.0, 29902.0, 48122.0, -16147.0], [-5301.0, -35277.0, -8518.0, 15810.0, -13741.0, -11017.0, -34868.0, 33493.0, -10722.0, -265.0, -31879.0], [16623.0, 40165.0, 25125.0, -26473.0, 90098.0, 88461.0, 20824.0, 27391.0, -47265.0, -57602.0, -11367.0], [10047.0, 20083.0, -20188.0, -2745.0, -11706.0, 3431.0, -8150.0, -34863.0, 3260.0, -40821.0, 8520.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (7, 11, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2');
});
test("matmul (2, 2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[[-47.0, -147.0, 3.0, 31.0, 21.0, -97.0, 66.0], [196.0, -93.0, 107.0, -45.0, -96.0, -12.0, 117.0], [-72.0, -124.0, -35.0, -64.0, 126.0, -76.0, -6.0], [105.0, 63.0, -101.0, -25.0, -37.0, -164.0, 84.0], [105.0, 78.0, -69.0, 130.0, -66.0, 42.0, -144.0]], [[83.0, 2.0, 17.0, -124.0, 23.0, -112.0, -58.0], [-113.0, -108.0, 169.0, -171.0, -50.0, -37.0, 72.0], [-80.0, 207.0, -127.0, -57.0, -51.0, 94.0, -36.0], [-24.0, -30.0, -24.0, -51.0, -128.0, -166.0, -33.0], [155.0, -83.0, -20.0, -47.0, 8.0, 40.0, -3.0]], [[20.0, -17.0, 38.0, 57.0, 108.0, 5.0, 324.0], [101.0, -171.0, 15.0, -52.0, -23.0, -36.0, -149.0], [-93.0, -33.0, 110.0, 77.0, 81.0, 53.0, 42.0], [-80.0, -28.0, 97.0, -106.0, -0.0, -2.0, 85.0], [70.0, 55.0, -164.0, 16.0, 80.0, -132.0, -172.0]]], [[[-123.0, 43.0, 51.0, 119.0, 64.0, -59.0, 150.0], [-150.0, -103.0, -92.0, -19.0, 94.0, 35.0, -15.0], [181.0, -5.0, 66.0, 112.0, -102.0, 97.0, -10.0], [49.0, -201.0, -103.0, -152.0, 18.0, 35.0, 116.0], [142.0, -54.0, 71.0, -73.0, -85.0, 126.0, -32.0]], [[69.0, 39.0, -141.0, -136.0, -47.0, 19.0, -102.0], [127.0, -174.0, 95.0, -38.0, 34.0, -15.0, 81.0], [3.0, -189.0, -174.0, 29.0, -68.0, -85.0, 127.0], [75.0, 124.0, 120.0, 48.0, 142.0, 13.0, -97.0], [69.0, -121.0, -133.0, 33.0, 148.0, -44.0, 5.0]], [[-111.0, -44.0, 96.0, 69.0, 149.0, -28.0, 109.0], [-75.0, 173.0, 103.0, 74.0, 57.0, 133.0, 45.0], [-139.0, 52.0, 39.0, -36.0, 19.0, 133.0, 32.0], [117.0, -73.0, 72.0, -117.0, 34.0, 87.0, 62.0], [-60.0, 284.0, 15.0, -150.0, -108.0, 43.0, 42.0]]]], [[[[-29.0, 127.0, -115.0, -112.0, -128.0, -26.0, 45.0], [55.0, -43.0, 61.0, 107.0, -0.0, -14.0, -97.0], [8.0, 105.0, -54.0, -17.0, -54.0, -77.0, -51.0], [38.0, 141.0, -23.0, 6.0, 4.0, -75.0, -280.0], [-39.0, 108.0, -123.0, -175.0, -85.0, 86.0, 60.0]], [[106.0, -60.0, 69.0, -123.0, -217.0, 55.0, -58.0], [93.0, 127.0, -52.0, -44.0, 98.0, -113.0, -54.0], [144.0, 107.0, 8.0, -123.0, -14.0, -78.0, 35.0], [-204.0, -39.0, -46.0, 87.0, 138.0, 4.0, 15.0], [-177.0, -122.0, 32.0, 160.0, -51.0, -179.0, -6.0]], [[-38.0, 10.0, -51.0, 111.0, -0.0, 80.0, -101.0], [-62.0, -199.0, 79.0, 27.0, -10.0, -174.0, 89.0], [-0.0, -51.0, 137.0, 55.0, -133.0, -193.0, -55.0], [-30.0, 20.0, -131.0, 150.0, 38.0, 49.0, -164.0], [-184.0, -85.0, -107.0, 73.0, 157.0, -76.0, -187.0]]], [[[126.0, -15.0, 167.0, 171.0, 31.0, -55.0, 45.0], [-205.0, 71.0, -31.0, 36.0, -50.0, -39.0, 232.0], [-196.0, 103.0, -64.0, -102.0, -4.0, 85.0, 89.0], [207.0, 75.0, -124.0, 70.0, -139.0, 33.0, 221.0], [-51.0, -33.0, -16.0, -51.0, 100.0, 275.0, -55.0]], [[117.0, -63.0, -10.0, -26.0, -158.0, 108.0, 12.0], [-55.0, 138.0, -25.0, -56.0, 30.0, 73.0, 91.0], [-135.0, 11.0, -53.0, -46.0, -32.0, -46.0, -9.0], [-6.0, 57.0, -33.0, 212.0, -143.0, -19.0, 267.0], [-144.0, -68.0, -103.0, 28.0, -275.0, -29.0, -36.0]], [[35.0, -57.0, -13.0, -21.0, 28.0, 46.0, -10.0], [-25.0, -91.0, -195.0, 144.0, 8.0, 43.0, 30.0], [-67.0, -2.0, -13.0, 103.0, -63.0, -231.0, -262.0], [161.0, -151.0, -35.0, 96.0, -130.0, 28.0, 161.0], [98.0, 96.0, -189.0, -0.0, -42.0, 23.0, -25.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[[99.0, -60.0, -5.0, 114.0, -119.0, 39.0, 111.0, -164.0, 59.0, -50.0, 120.0], [79.0, -41.0, 81.0, 134.0, 4.0, -61.0, 88.0, -62.0, -24.0, 55.0, 64.0], [138.0, 56.0, -41.0, -5.0, 107.0, 86.0, -108.0, 18.0, 171.0, 109.0, 84.0], [-118.0, 157.0, 239.0, 66.0, 80.0, 3.0, 0.0, 34.0, -126.0, 156.0, -133.0], [-167.0, 31.0, 103.0, -15.0, 54.0, 1.0, 46.0, -0.0, 45.0, 122.0, -30.0], [-67.0, 48.0, -54.0, -146.0, -99.0, 109.0, 12.0, -165.0, 85.0, 36.0, 167.0], [-94.0, -129.0, 57.0, -118.0, -51.0, 108.0, -146.0, -99.0, 61.0, -226.0, 55.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[-22722.0, 1363.0, 6777.0, -16966.0, 15177.0, 4061.0, -28311.0, 27401.0, -5912.0, -16418.0, -32118.0], [37971.0, -27665.0, -26226.0, -4237.0, -25810.0, 33616.0, -19626.0, -35585.0, 39560.0, -48858.0, 39852.0], [-29588.0, -1572.0, -6805.0, -18959.0, 13841.0, -7252.0, -9364.0, 29824.0, -349.0, -3027.0, -27806.0], [13655.0, -38319.0, 12577.0, 33854.0, -15096.0, -17350.0, 12173.0, -5050.0, -19919.0, -46096.0, -10185.0], [13439.0, 25594.0, 22418.0, 43197.0, -9544.0, -17247.0, 44463.0, -11552.0, -32040.0, 37803.0, -4420.0]], [[34468.0, -20759.0, -25475.0, 24312.0, -2682.0, -14244.0, 15735.0, 6576.0, 11357.0, -9649.0, 5424.0], [27842.0, -18789.0, -55029.0, -41829.0, 14709.0, 19895.0, -53555.0, 21433.0, 45367.0, -32249.0, 15748.0], [3236.0, -12173.0, -3630.0, 6780.0, -18025.0, -20533.0, 27090.0, -15884.0, -20724.0, -2052.0, 15809.0], [33560.0, -14360.0, -19616.0, 20048.0, 7293.0, -23109.0, -5774.0, 34287.0, -20257.0, -25156.0, -25730.0], [7840.0, -11841.0, -19418.0, -2060.0, -28052.0, 13291.0, 13347.0, -28535.0, 17216.0, -18733.0, 24134.0]], [[-49672.0, -27634.0, 39910.0, -37008.0, -5009.0, 40901.0, -45656.0, -32505.0, 25953.0, -48769.0, 12338.0], [24955.0, 10407.0, -36317.0, 8276.0, -5337.0, -4535.0, 14807.0, 13231.0, 5996.0, 8640.0, -4165.0], [-26746.0, 24819.0, 19560.0, -24401.0, 25850.0, 18471.0, -26877.0, 8993.0, 15125.0, 29135.0, -5542.0], [7906.0, -16323.0, -26226.0, -30091.0, 7170.0, 15574.0, -34254.0, 4913.0, 30910.0, -22785.0, 15195.0], [-1593.0, 5205.0, 20217.0, 55594.0, 1782.0, -47565.0, 57530.0, 21510.0, -45362.0, 28025.0, -37888.0]]], [[[-36619.0, 6958.0, 48776.0, -10707.0, 31433.0, 7156.0, -35041.0, 17355.0, -7547.0, 4422.0, -27074.0], [-50074.0, 11617.0, -1425.0, -36446.0, 8450.0, -5247.0, -8844.0, 24394.0, -13426.0, 4961.0, -27593.0], [24891.0, 13409.0, 6438.0, 15574.0, -20138.0, 22767.0, 10455.0, -39393.0, 11018.0, 8649.0, 30757.0], [-23561.0, -37057.0, -42055.0, -49933.0, -38225.0, 21219.0, -16813.0, -19855.0, 20115.0, -71204.0, 16265.0], [36965.0, -6250.0, -42825.0, -9566.0, -30789.0, 24912.0, 5616.0, -38766.0, 35946.0, -12321.0, 51089.0]], [[22666.0, -22374.0, -35590.0, 14788.0, -33239.0, -21214.0, 39277.0, -13933.0, -10562.0, -19888.0, 15993.0], [4134.0, -11247.0, -18777.0, -19699.0, -9494.0, 30770.0, -21917.0, -15166.0, 37898.0, -26191.0, 18068.0], [-36955.0, -20193.0, 3566.0, -23756.0, -19145.0, 1152.0, -20197.0, 10532.0, -31233.0, -65045.0, -35379.0], [12650.0, 22211.0, 24616.0, 35152.0, 19579.0, -3092.0, 27127.0, -8738.0, 17499.0, 63352.0, 13208.0], [-47214.0, 385.0, 21099.0, -1891.0, -8193.0, -5375.0, 16925.0, 1679.0, -16701.0, -4112.0, -26538.0]], [[-42612.0, 13887.0, 32618.0, -25485.0, 34084.0, 15687.0, -35957.0, 18835.0, 13203.0, 16894.0, -20400.0], [-10936.0, 17139.0, 29105.0, -6582.0, 14174.0, 15016.0, -6577.0, -20456.0, 16327.0, 37608.0, 23858.0], [-15115.0, 5585.0, -8697.0, -34928.0, 4269.0, 12625.0, -17267.0, -6063.0, 15868.0, 8319.0, 18113.0], [12223.0, -21132.0, -35075.0, -25054.0, -25810.0, 31070.0, -7657.0, -37837.0, 48416.0, -27001.0, 47896.0], [47473.0, -37456.0, -24213.0, 11627.0, -14350.0, -9709.0, 6128.0, -23851.0, 12466.0, -24265.0, 44917.0]]]], [[[[23396.0, -38512.0, -20836.0, 7301.0, -23939.0, -17206.0, 7607.0, -9161.0, -15537.0, -48294.0, 11857.0], [7896.0, 30519.0, 14541.0, 20755.0, 14703.0, -1667.0, 9727.0, 10295.0, -5881.0, 39644.0, -12932.0], [22612.0, -9269.0, 2305.0, 32200.0, -362.0, -24743.0, 19998.0, 8382.0, -21226.0, -997.0, -8639.0], [41696.0, 24237.0, 2110.0, 67667.0, 15982.0, -47490.0, 59274.0, 24911.0, -29106.0, 65352.0, -17191.0], [11140.0, -42698.0, -37818.0, -19270.0, -38252.0, -3443.0, 6821.0, -28594.0, 3269.0, -53651.0, 35387.0]], [[67796.0, -15952.0, -66243.0, -2350.0, -29516.0, 12873.0, -1820.0, -19937.0, 26363.0, -31653.0, 43540.0], [13537.0, -16027.0, 14556.0, 46376.0, -410.0, -26775.0, 38151.0, -1567.0, -9398.0, 9895.0, -4009.0], [42601.0, -40583.0, -17013.0, 30064.0, -20511.0, -5328.0, 17846.0, -24883.0, 17669.0, -32057.0, 30478.0], [-64615.0, 27457.0, 35393.0, -26934.0, 32449.0, -7078.0, -16902.0, 35859.0, -22463.0, 30203.0, -45058.0], [-20551.0, 33135.0, 32002.0, 1481.0, 52072.0, -16439.0, -37457.0, 72737.0, -40079.0, 19278.0, -76333.0]], [[-18974.0, 33310.0, 19543.0, 4827.0, 5216.0, -8333.0, 17876.0, 5267.0, -24550.0, 39913.0, -15162.0], [-9181.0, 399.0, 844.0, -17295.0, 29342.0, 7232.0, -48468.0, 44745.0, 1414.0, -22620.0, -40994.0], [48699.0, 12106.0, -3015.0, 32774.0, 33585.0, -12052.0, -19688.0, 44788.0, -8024.0, 9964.0, -30337.0], [-31381.0, 41880.0, 34911.0, 21443.0, 7198.0, -25539.0, 38858.0, 14573.0, -47680.0, 55185.0, -35251.0], [-51860.0, 45336.0, 25485.0, 3794.0, 41486.0, -39297.0, 17264.0, 67055.0, -47113.0, 62930.0, -73904.0]]], [[[8435.0, 21770.0, 40905.0, 25060.0, 21319.0, 19600.0, -11174.0, -6294.0, 14270.0, 29386.0, -2195.0], [-34057.0, -20045.0, 26831.0, -32257.0, 13571.0, 5871.0, -49799.0, 13351.0, -15049.0, -43544.0, -19701.0], [-21456.0, -19586.0, -12360.0, -37806.0, -4442.0, -864.0, -17938.0, -1698.0, 346.0, -24965.0, 10472.0], [1274.0, -42683.0, 23352.0, 10077.0, -54045.0, 20370.0, 4705.0, -65774.0, -9580.0, -74537.0, 31750.0], [-33801.0, 18905.0, -21636.0, -48682.0, -18875.0, 22630.0, 9093.0, -31542.0, 25993.0, 25565.0, 37107.0]], [[26316.0, -10341.0, -32914.0, -11584.0, -37161.0, 20378.0, 799.0, -35354.0, 12783.0, -32561.0, 36062.0], [-9840.0, -19855.0, 3429.0, -13195.0, -10306.0, 4934.0, -2291.0, -22944.0, 9330.0, -15399.0, 23876.0], [-5110.0, -4580.0, -8580.0, -8429.0, 10043.0, -16650.0, -9003.0, 27421.0, -17395.0, -9124.0, -21047.0], [-25605.0, -10329.0, 58184.0, -5476.0, -5087.0, 20709.0, -37874.0, -19234.0, -25840.0, -45562.0, -12238.0], [14106.0, 4783.0, -22684.0, -10558.0, -2060.0, -17566.0, -18586.0, 35279.0, -45041.0, -29857.0, -32581.0]], [[-7172.0, 578.0, -9448.0, -10925.0, -9996.0, 7623.0, 3573.0, -9754.0, 8416.0, -2246.0, 8545.0], [-60603.0, 15361.0, 35377.0, -14503.0, -12089.0, -3827.0, 6781.0, 1063.0, -44935.0, -6802.0, -35765.0], [29887.0, 40302.0, 16374.0, 64544.0, 47643.0, -56838.0, 26373.0, 78433.0, -57558.0, 61101.0, -74056.0], [-7448.0, -13812.0, 5618.0, -16505.0, -33831.0, 33078.0, -20787.0, -34967.0, 1393.0, -56432.0, 11379.0], [-973.0, -17373.0, 8042.0, 25203.0, -34771.0, -18523.0, 41732.0, -26746.0, -30301.0, -18867.0, 5754.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (1, 7, 11, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([1, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2');
});
test("matmul (2, 2, 3, 5, 7) x (3, 7, 11)", async () => {
    const a = tensor([[[[[144.0, 31.0, -5.0, 39.0, -18.0, 59.0, 86.0], [-78.0, -27.0, -195.0, 151.0, -114.0, -13.0, -162.0], [2.0, -180.0, -39.0, 282.0, 46.0, -122.0, 38.0], [-38.0, -76.0, -78.0, 103.0, 59.0, 253.0, -79.0], [-315.0, -174.0, -5.0, 54.0, 4.0, -99.0, -38.0]], [[34.0, 21.0, 77.0, -123.0, -92.0, -86.0, 128.0], [-169.0, -133.0, -170.0, -67.0, 107.0, 182.0, -91.0], [145.0, -59.0, 105.0, 113.0, -108.0, -167.0, -32.0], [-89.0, -98.0, 86.0, -63.0, -137.0, 196.0, -145.0], [-83.0, -91.0, 53.0, -75.0, 32.0, 51.0, 96.0]], [[-177.0, 3.0, 75.0, 162.0, -161.0, -99.0, 115.0], [-74.0, 26.0, -71.0, 244.0, -8.0, 68.0, -35.0], [-5.0, -44.0, 122.0, -173.0, -182.0, 62.0, -69.0], [-56.0, 79.0, 27.0, -59.0, -85.0, 93.0, -138.0], [127.0, -51.0, -265.0, 23.0, 227.0, 44.0, 124.0]]], [[[-132.0, 158.0, 26.0, -125.0, -114.0, -13.0, -112.0], [105.0, -23.0, 91.0, 93.0, 134.0, 49.0, -225.0], [119.0, -141.0, 11.0, -107.0, 78.0, -15.0, 178.0], [39.0, 28.0, -113.0, -17.0, 32.0, -55.0, -155.0], [71.0, -66.0, 72.0, 91.0, 23.0, -0.0, 202.0]], [[-6.0, 218.0, 7.0, 2.0, -119.0, 167.0, -5.0], [-10.0, 53.0, 219.0, 69.0, 129.0, 109.0, 28.0], [259.0, -3.0, 50.0, 15.0, -3.0, 137.0, -73.0], [-80.0, 45.0, -85.0, 80.0, -114.0, -194.0, 49.0], [89.0, 246.0, -46.0, -123.0, 7.0, 33.0, -26.0]], [[-164.0, -38.0, 136.0, -190.0, -114.0, -88.0, 138.0], [-165.0, 168.0, 63.0, -110.0, 134.0, 47.0, -276.0], [84.0, -57.0, 70.0, -38.0, -14.0, -79.0, 144.0], [108.0, -39.0, 20.0, -69.0, -197.0, 302.0, 89.0], [-41.0, 61.0, 45.0, 15.0, 39.0, -117.0, 40.0]]]], [[[[93.0, -168.0, 90.0, -451.0, -130.0, 55.0, -101.0], [-28.0, -145.0, -59.0, -59.0, -60.0, 80.0, -13.0], [133.0, -18.0, -185.0, 121.0, -13.0, -26.0, -25.0], [65.0, -134.0, 61.0, -84.0, -64.0, 11.0, 115.0], [-126.0, -190.0, 203.0, -59.0, 145.0, -42.0, 15.0]], [[0.0, -232.0, 65.0, 78.0, -10.0, -68.0, -84.0], [74.0, -66.0, -15.0, 17.0, -120.0, 108.0, 49.0], [-3.0, 0.0, 55.0, -161.0, -167.0, -235.0, 338.0], [189.0, 36.0, 100.0, -66.0, -29.0, -160.0, 153.0], [-52.0, 18.0, 6.0, -218.0, -178.0, -89.0, -1.0]], [[-38.0, 253.0, 134.0, -99.0, -169.0, 110.0, -42.0], [4.0, 61.0, -18.0, 75.0, 62.0, 6.0, 183.0], [-112.0, 34.0, 103.0, 48.0, 92.0, 179.0, -88.0], [90.0, -15.0, 58.0, 61.0, 157.0, 54.0, -75.0], [-99.0, -74.0, 62.0, 71.0, -161.0, -22.0, 66.0]]], [[[-220.0, 94.0, -38.0, -131.0, -24.0, 0.0, -53.0], [30.0, 79.0, 0.0, -38.0, -5.0, -45.0, 63.0], [-79.0, 81.0, -7.0, 26.0, -255.0, -96.0, -49.0], [-12.0, 175.0, -114.0, -93.0, -27.0, 125.0, -2.0], [55.0, -94.0, 219.0, -39.0, 0.0, 121.0, 1.0]], [[-92.0, -114.0, 165.0, -57.0, -52.0, -33.0, -86.0], [18.0, 79.0, -171.0, 84.0, -50.0, -46.0, -26.0], [118.0, 191.0, 7.0, -130.0, 179.0, 55.0, 66.0], [-11.0, 13.0, -166.0, -178.0, 13.0, 86.0, 48.0], [102.0, 87.0, 180.0, -112.0, -211.0, -58.0, -93.0]], [[66.0, -161.0, 93.0, 79.0, -35.0, 216.0, 70.0], [51.0, -15.0, 118.0, 10.0, 11.0, 4.0, -91.0], [-85.0, -8.0, 124.0, 49.0, 16.0, -142.0, -128.0], [36.0, -44.0, -142.0, -139.0, -120.0, 44.0, -88.0], [9.0, -13.0, 49.0, -52.0, 78.0, -45.0, -174.0]]]]]);
    expect(a.shape).toEqual([2, 2, 3, 5, 7]);
    const b = tensor([[[186.0, -121.0, -168.0, -57.0, 172.0, -167.0, -39.0, 104.0, 78.0, 129.0, 83.0], [-80.0, -146.0, -36.0, -49.0, 72.0, -102.0, 54.0, 60.0, -109.0, 62.0, -56.0], [67.0, -55.0, -130.0, -66.0, -86.0, 17.0, 120.0, 79.0, -142.0, 113.0, 15.0], [-41.0, -55.0, 173.0, -54.0, 13.0, -33.0, 100.0, -183.0, 106.0, -17.0, -98.0], [-65.0, -87.0, 33.0, 39.0, -102.0, 51.0, 37.0, 66.0, 35.0, -159.0, -41.0], [-104.0, 60.0, -71.0, 32.0, 54.0, -4.0, -35.0, 65.0, 1.0, -206.0, -8.0], [69.0, 37.0, -20.0, -139.0, 117.0, -164.0, -44.0, 248.0, -23.0, -225.0, -56.0]], [[-8.0, -14.0, 30.0, 19.0, 26.0, 43.0, -77.0, 100.0, -58.0, 56.0, 70.0], [17.0, -150.0, -37.0, 81.0, -119.0, 93.0, -54.0, 86.0, -7.0, -67.0, -7.0], [-30.0, -138.0, -62.0, -11.0, -7.0, -5.0, 35.0, -12.0, 84.0, 326.0, 94.0], [199.0, 98.0, -43.0, 110.0, -108.0, 2.0, 113.0, -89.0, -87.0, 67.0, -145.0], [-21.0, -64.0, 200.0, 45.0, -37.0, 6.0, -23.0, 42.0, -117.0, 200.0, 6.0], [-212.0, -31.0, -36.0, 152.0, -17.0, 126.0, 17.0, -16.0, 46.0, 31.0, -87.0], [-115.0, -155.0, 18.0, -139.0, 25.0, 9.0, -3.0, -47.0, -143.0, 192.0, -20.0]], [[-243.0, -89.0, 12.0, 31.0, -103.0, 201.0, -88.0, 26.0, -8.0, 103.0, -134.0], [-132.0, 15.0, 34.0, -41.0, 31.0, -11.0, 51.0, -6.0, 64.0, 153.0, -12.0], [60.0, -35.0, 103.0, 15.0, -70.0, -63.0, -3.0, 70.0, 97.0, -68.0, 61.0], [-162.0, -21.0, 39.0, 37.0, 35.0, 18.0, 27.0, -76.0, 8.0, 74.0, -99.0], [-150.0, -30.0, -30.0, 156.0, 56.0, -29.0, 110.0, -0.0, 51.0, -7.0, 196.0], [8.0, 121.0, -54.0, 149.0, -58.0, -163.0, 76.0, 147.0, -94.0, 60.0, 36.0], [-169.0, 39.0, 21.0, -48.0, 71.0, 27.0, 184.0, -142.0, 55.0, -122.0, 94.0]]]);
    expect(b.shape).toEqual([3, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[23338.0, -15532.0, -24414.0, -22271.0, 43021.0, -43840.0, -7157.0, 33279.0, 10148.0, -9372.0, 1769.0], [-34020.0, 18944.0, 65950.0, 28141.0, -4655.0, 28288.0, -3351.0, -101315.0, 40278.0, 20916.0, -8835.0], [12917.0, 2757.0, 69420.0, -11340.0, -12430.0, 4659.0, 18022.0, -60749.0, 55820.0, -10835.0, -21013.0], [-46035.0, 21443.0, 22643.0, 26854.0, -5560.0, 24326.0, -4878.0, -32776.0, 31449.0, -63903.0, -10181.0], [-39805.0, 53130.0, 77097.0, 26165.0, -75776.0, 75318.0, 12974.0, -69072.0, 1745.0, -24598.0, -19012.0]], [[-21258.0, -37592.0, -12242.0, -47034.0, 19196.0, -7452.0, -14686.0, 6725.0, 3554.0, 20868.0, 31676.0], [-39508.0, 40825.0, 26482.0, 25644.0, 10531.0, 3835.0, 7580.0, -14476.0, 11148.0, -50892.0, -30536.0], [58526.0, 20453.0, -21000.0, -16545.0, 3887.0, -21529.0, 8206.0, -2251.0, 522.0, 20953.0, 18569.0], [-38071.0, 23071.0, -38733.0, 26277.0, 13662.0, 9072.0, 14954.0, -14828.0, 64333.0, -23767.0, -3299.0], [-39922.0, -18361.0, 7108.0, -21933.0, 16749.0, -4965.0, 4528.0, -14071.0, 1302.0, 40115.0, 4519.0]], [[24794.0, 7107.0, 24612.0, -43878.0, 23635.0, -13508.0, 15804.0, -42565.0, 17599.0, -29727.0, -12091.0], [-21579.0, 11440.0, -1968.0, 15167.0, 15061.0, -18092.0, 12487.0, -10628.0, -11404.0, 27646.0, -21293.0], [81826.0, 9419.0, 4926.0, -18764.0, -34131.0, -18012.0, -34845.0, 40734.0, -11231.0, -14933.0, -14159.0], [51174.0, 14884.0, -2876.0, 468.0, -15690.0, -31308.0, -20391.0, 37711.0, -13016.0, 23128.0, -12240.0], [-98409.0, 74.0, -33190.0, 38920.0, 23657.0, 32790.0, 38769.0, -27830.0, -15540.0, 10923.0, 22884.0]]], [[[-29291.0, 3343.0, -9116.0, 15522.0, -17367.0, 23101.0, 5465.0, -15464.0, -45887.0, 43835.0, 3886.0], [-5677.0, -36510.0, -7110.0, 22183.0, -27560.0, 26827.0, 28026.0, -44061.0, 17547.0, 40046.0, 8968.0], [47310.0, 10367.0, -34778.0, -17002.0, 20039.0, -26927.0, -26056.0, 72683.0, 10368.0, -39691.0, 15378.0], [-8915.0, -13476.0, 12250.0, 25814.0, -6148.0, 16543.0, -5340.0, -39983.0, 18864.0, 35404.0, 9448.0], [32022.0, -2447.0, -6450.0, -37660.0, 23739.0, -38859.0, 3370.0, 44073.0, 8313.0, -37451.0, -10504.0]], [[-28388.0, -30172.0, -38668.0, 38411.0, -24924.0, 40268.0, -5248.0, 10451.0, 21556.0, -32109.0, -16721.0], [-20895.0, -47245.0, 3574.0, 27765.0, -21478.0, 18302.0, 12172.0, -2853.0, -1481.0, 106461.0, 241.0], [-21224.0, -1346.0, -2710.0, 36614.0, 1078.0, 27225.0, -13719.0, 24820.0, 4986.0, 21641.0, 10199.0], [57762.0, 19655.0, -17169.0, -29569.0, -6739.0, -23357.0, 8972.0, -14217.0, -12368.0, -49251.0, -10291.0], [-23780.0, -41293.0, 1453.0, 17538.0, -14824.0, 30655.0, -35168.0, 42543.0, 4370.0, -37304.0, 15710.0]], [[76882.0, 11410.0, 14408.0, -46036.0, 8062.0, -23158.0, 13120.0, -12608.0, 20600.0, -67332.0, 36998.0], [66439.0, 8213.0, -6423.0, 26027.0, -875.0, -59961.0, -12543.0, 53573.0, 4539.0, 31839.0, 36839.0], [-25400.0, -13506.0, 12508.0, -16282.0, -2627.0, 29588.0, 7417.0, -21747.0, 16798.0, -29851.0, 5408.0], [8207.0, 36475.0, -9190.0, 12688.0, -38377.0, -21475.0, 4242.0, 41442.0, -35512.0, 7332.0, -25327.0], [-11365.0, -11093.0, 12790.0, -15811.0, 15299.0, 7543.0, 9747.0, -22301.0, 23904.0, -9013.0, 13214.0]]]], [[[[51020.0, 44003.0, -105474.0, 32074.0, -5290.0, 27732.0, -49290.0, 59182.0, -37192.0, 51483.0, 73221.0], [-459.0, 40587.0, -13.0, 17808.0, -2030.0, 19162.0, -24166.0, -7460.0, 14024.0, -22281.0, 13241.0], [10646.0, -11299.0, 25204.0, 1113.0, 36060.0, -23972.0, -14730.0, -32754.0, 51526.0, 6127.0, -445.0], [41292.0, 23447.0, -33751.0, -14758.0, 15771.0, -15546.0, -18664.0, 43922.0, -2764.0, -9567.0, 18142.0], [3762.0, 20486.0, -1122.0, 8506.0, -68880.0, 50923.0, 19289.0, 12890.0, -19510.0, -21870.0, 2560.0]], [[33914.0, 49242.0, 136.0, -10037.0, 18155.0, -31129.0, 22943.0, -23058.0, 10352.0, 21724.0, 3960.0], [-23892.0, 9337.0, -22145.0, 2300.0, 11876.0, 10482.0, 3711.0, -8680.0, 5432.0, -6429.0, -9329.0], [-19208.0, -57743.0, -15433.0, -108589.0, 35549.0, -28296.0, -17205.0, -5771.0, -20804.0, 31186.0, 40988.0], [-100.0, -45213.0, 3690.0, -48745.0, 14676.0, -8114.0, -22967.0, 20821.0, -22918.0, 54966.0, 42634.0], [-20119.0, -9858.0, -25638.0, -44975.0, 28082.0, -13319.0, -18808.0, 9673.0, 39235.0, -55319.0, 35103.0]], [[33244.0, 21308.0, 16335.0, -21162.0, -19914.0, -34808.0, -4786.0, 36532.0, 7433.0, 31264.0, -13081.0], [-62433.0, 5617.0, 4852.0, 1910.0, 21481.0, 4782.0, 45786.0, -32326.0, 15389.0, -5881.0, 19779.0], [23636.0, 21332.0, -1981.0, 43702.0, -4418.0, -62732.0, 20109.0, 39255.0, -3527.0, 11046.0, 32335.0], [-36735.0, -12647.0, -278.0, 42670.0, -11325.0, 319.0, 362.0, 20442.0, 3240.0, 18836.0, 11285.0], [38863.0, 8782.0, 12855.0, -28040.0, 2994.0, -11676.0, -569.0, -15792.0, 125.0, -28726.0, -15237.0]]], [[[-47712.0, 22318.0, 16121.0, 23947.0, -33260.0, 38297.0, -2560.0, -10997.0, -35517.0, -8878.0, -7304.0], [10170.0, -13008.0, -12688.0, -13921.0, 15805.0, -22221.0, -2086.0, 27183.0, -11968.0, 5304.0, -1173.0], [469.0, 11300.0, 15145.0, -6614.0, 8277.0, -631.0, 5296.0, -43889.0, -19135.0, 64944.0, 221.0], [-31440.0, -2938.0, -15279.0, 7880.0, 28401.0, -16264.0, -18348.0, 23112.0, -14455.0, -23006.0, -3173.0], [21507.0, 4466.0, -49684.0, -7144.0, -9998.0, 4765.0, 10880.0, 32631.0, -20598.0, 1526.0, 15912.0]], [[483.0, 7713.0, -17081.0, -14469.0, 16510.0, -20741.0, 13467.0, -13525.0, 41817.0, 24522.0, 22412.0], [36837.0, 28384.0, -4205.0, 12234.0, -14826.0, 2814.0, -1699.0, 3028.0, -15817.0, -70821.0, -23325.0], [-46786.0, -67399.0, 36637.0, 10577.0, -11578.0, 31140.0, -37225.0, 43248.0, -24134.0, 37560.0, 21400.0], [-54158.0, -7270.0, 17503.0, -9925.0, 17810.0, 12556.0, -24760.0, 14766.0, -2340.0, -53047.0, 981.0], [397.0, -20577.0, -48289.0, -10699.0, 9603.0, 1942.0, -14762.0, 21927.0, 53657.0, -10795.0, 45331.0]], [[-6856.0, 16713.0, -1166.0, 36329.0, -25052.0, -21703.0, 13281.0, 25000.0, -19418.0, -13648.0, -1564.0], [8808.0, -12499.0, 10189.0, 11016.0, -19705.0, -266.0, -20567.0, 22426.0, 5338.0, 6939.0, -6700.0], [39309.0, -20578.0, 17891.0, -11152.0, 1586.0, -4701.0, -24561.0, 96.0, 19712.0, -7801.0, 191.0], [44282.0, 9517.0, -21735.0, -12293.0, -15517.0, 8096.0, -34787.0, 20788.0, -33086.0, 10562.0, -29405.0], [28239.0, -16190.0, -879.0, 13438.0, -11956.0, -1696.0, -29862.0, 25787.0, 2071.0, 9740.0, 4399.0]]]]]);
});
test("matmul (2, 2, 3, 5, 7) x (3, 7, 11, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([3, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2');
});
test("matmul (2, 2, 3, 5, 7) x (3, 1, 11)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([3, 1, 11]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [12, 7] but got: [12, 1].');
});
test("matmul (2, 2, 3, 5, 7) x (3, 1, 11, 1)", async () => {
    const a = ones([2, 2, 3, 5, 7]);
    const b = ones([3, 1, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1');
});
test("matmul (2, 1, 2, 3, 5, 7) x (11,)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([11]);
    expect(() => matmul(a, b)).toThrow('size mismatch, got 60, 60x7,11');
});
test("matmul (2, 1, 2, 3, 5, 7) x (11, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([11, 1]);
    expect(() => matmul(a, b)).toThrow('mat1 and mat2 shapes cannot be multiplied (60x7 and 11x1)');
});
test("matmul (2, 1, 2, 3, 5, 7) x (7, 11)", async () => {
    const a = tensor([[[[[[-99.0, 251.0, -168.0, 253.0, 36.0, 68.0, 28.0], [-33.0, -19.0, -91.0, 246.0, -32.0, -56.0, 233.0], [145.0, 32.0, 19.0, 82.0, -17.0, -144.0, 5.0], [-31.0, 2.0, -54.0, 27.0, -47.0, 68.0, -12.0], [73.0, 57.0, 2.0, 127.0, 136.0, 127.0, -47.0]], [[-87.0, -63.0, 95.0, 44.0, 12.0, -64.0, 153.0], [-175.0, 93.0, 118.0, -56.0, 36.0, -121.0, 104.0], [88.0, -46.0, 60.0, -184.0, 46.0, 63.0, 285.0], [38.0, -127.0, 159.0, 13.0, -47.0, 27.0, -77.0], [-130.0, -84.0, -27.0, -37.0, -41.0, 66.0, -97.0]], [[-61.0, 91.0, 54.0, -19.0, -111.0, 59.0, -13.0], [-178.0, 80.0, -114.0, -17.0, 141.0, -168.0, -108.0], [217.0, -51.0, -0.0, 75.0, -21.0, 2.0, 88.0], [21.0, 100.0, -131.0, -198.0, -119.0, -206.0, 213.0], [75.0, 62.0, -141.0, -84.0, -117.0, -24.0, -81.0]]], [[[42.0, 112.0, 4.0, -50.0, 161.0, -4.0, 177.0], [-32.0, -117.0, -31.0, -22.0, 54.0, -46.0, 9.0], [-23.0, -36.0, -76.0, -8.0, -49.0, 92.0, -39.0], [-125.0, 31.0, 93.0, -37.0, 111.0, -115.0, -60.0], [278.0, 70.0, 95.0, -2.0, -64.0, 7.0, 196.0]], [[49.0, 27.0, 61.0, 143.0, 117.0, 4.0, -80.0], [-107.0, -75.0, 83.0, 139.0, -55.0, 2.0, 6.0], [-52.0, -229.0, -62.0, 121.0, 4.0, 66.0, 27.0], [-69.0, -60.0, 130.0, 7.0, -76.0, -59.0, -180.0], [83.0, -103.0, -51.0, -133.0, -58.0, 97.0, 74.0]], [[109.0, 141.0, 1.0, 128.0, -131.0, -250.0, 130.0], [-152.0, -37.0, 28.0, 74.0, -20.0, 38.0, -40.0], [38.0, -50.0, -76.0, 24.0, -3.0, -65.0, -11.0], [-3.0, 88.0, -86.0, 61.0, -122.0, -103.0, 34.0], [54.0, 96.0, -100.0, -53.0, -9.0, 41.0, -124.0]]]]], [[[[[-106.0, -85.0, 57.0, 92.0, 32.0, -99.0, 122.0], [-53.0, 8.0, -162.0, -56.0, -47.0, -11.0, -7.0], [69.0, 38.0, 91.0, 51.0, 55.0, -14.0, -35.0], [123.0, 48.0, -214.0, 71.0, 231.0, 82.0, -15.0], [-91.0, -160.0, -99.0, -113.0, -10.0, 97.0, 29.0]], [[181.0, -23.0, 37.0, 31.0, -83.0, -151.0, -191.0], [162.0, 40.0, -38.0, -43.0, 36.0, 126.0, -84.0], [9.0, -99.0, 23.0, 107.0, -111.0, -187.0, 8.0], [-89.0, -139.0, -10.0, -8.0, 160.0, 3.0, -75.0], [127.0, -116.0, -95.0, 49.0, 116.0, -69.0, 61.0]], [[-29.0, -55.0, 90.0, 89.0, -99.0, -74.0, -123.0], [-94.0, 75.0, -90.0, 170.0, -55.0, -20.0, -141.0], [23.0, 19.0, -71.0, 215.0, -181.0, 65.0, -108.0], [-136.0, -116.0, -16.0, 54.0, -83.0, -37.0, -146.0], [-178.0, -54.0, 34.0, -80.0, -78.0, 55.0, -18.0]]], [[[102.0, 61.0, -11.0, 58.0, -277.0, 380.0, -52.0], [104.0, 54.0, 34.0, 7.0, -2.0, -28.0, -35.0], [33.0, -92.0, -86.0, -52.0, 28.0, -54.0, -128.0], [-13.0, 33.0, 46.0, 83.0, -209.0, -217.0, 25.0], [64.0, -18.0, 116.0, -102.0, -100.0, -240.0, 194.0]], [[15.0, 122.0, -26.0, -46.0, -10.0, -121.0, -46.0], [73.0, 33.0, -3.0, 175.0, 47.0, -148.0, 59.0], [140.0, 49.0, -149.0, 44.0, 128.0, 150.0, 76.0], [154.0, 53.0, -99.0, 216.0, 39.0, 29.0, -318.0], [64.0, 2.0, -114.0, -21.0, -62.0, -5.0, -55.0]], [[-29.0, 8.0, 49.0, 98.0, -23.0, -2.0, 179.0], [189.0, -18.0, -95.0, 83.0, 118.0, -8.0, -177.0], [183.0, 44.0, 32.0, 128.0, -12.0, 32.0, -57.0], [-131.0, -20.0, 17.0, 177.0, -131.0, 6.0, -23.0], [-30.0, 72.0, 102.0, -136.0, -12.0, 169.0, 15.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[130.0, -162.0, 37.0, -143.0, 8.0, -49.0, -21.0, 129.0, 227.0, -23.0, 117.0], [150.0, 15.0, -69.0, 154.0, -118.0, -12.0, -37.0, -49.0, 12.0, 15.0, 18.0], [-342.0, 25.0, -81.0, 13.0, -84.0, 64.0, -37.0, -161.0, -149.0, 136.0, 53.0], [14.0, -73.0, -169.0, 31.0, 61.0, -57.0, -98.0, -162.0, 243.0, 61.0, -113.0], [-77.0, -188.0, -14.0, 155.0, 91.0, -30.0, -49.0, 48.0, 63.0, -43.0, -37.0], [-92.0, 190.0, -219.0, -132.0, 28.0, 63.0, 142.0, -17.0, -73.0, -171.0, 7.0], [-112.0, -72.0, 70.0, 15.0, -4.0, -248.0, -87.0, -109.0, 168.0, 148.0, -111.0]]);
    expect(b.shape).toEqual([7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[[73614.0, 1270.0, -63567.0, 55494.0, 4203.0, -27074.0, -20330.0, -41488.0, 69058.0, -10405.0, -48522.0], [8946.0, -36572.0, -5091.0, 14163.0, 19216.0, -78353.0, -46000.0, -54508.0, 106834.0, 48540.0, -61895.0], [32297.0, -53045.0, 19884.0, 3430.0, -4809.0, -20749.0, -33018.0, 1881.0, 60675.0, 30826.0, 8348.0], [13823.0, 24351.0, -16548.0, -11565.0, 3374.0, 5170.0, 12932.0, -1881.0, -2347.0, -16337.0, -5957.0], [2242.0, -18246.0, -55864.0, 5913.0, 17557.0, 4205.0, -703.0, -4780.0, 39219.0, -27326.0, -3604.0]], [[-64806.0, -13120.0, 10555.0, 17941.0, 130.0, -33745.0, -26656.0, -45572.0, 7164.0, 49732.0, -29125.0], [-53228.0, -463.0, 20289.0, 62257.0, -26230.0, -16292.0, -26638.0, -44609.0, -41226.0, 52587.0, -19942.0], [-59814.0, -17212.0, 38175.0, -21503.0, -5322.0, -57523.0, -2437.0, 3826.0, 11951.0, 23651.0, 544.0], [-58547.0, 14475.0, -15552.0, -34526.0, -486.0, 31304.0, 9580.0, -10902.0, -31298.0, 5646.0, 19593.0], [-12835.0, 49058.0, -11244.0, -12366.0, 7388.0, 37203.0, 30283.0, 5170.0, -59183.0, -28078.0, -1226.0]], [[-8439.0, 46998.0, -21976.0, -2338.0, -25318.0, 16707.0, 12726.0, -22858.0, -38902.0, 1713.0, 5473.0], [44305.0, -22225.0, 27259.0, 78176.0, 6234.0, 13405.0, -14707.0, 15622.0, -23588.0, -4566.0, -17912.0], [13187.0, -43402.0, 4889.0, -38759.0, 10122.0, -35364.0, -16363.0, 7708.0, 80187.0, 12404.0, 7019.0], [64019.0, -22827.0, 99640.0, 16498.0, -30155.0, -61559.0, -21842.0, 25549.0, 20697.0, 42990.0, -994.0], [86385.0, 14655.0, 25338.0, -21796.0, -10991.0, 13431.0, 18952.0, 46567.0, -861.0, -27948.0, 25062.0]]], [[[-11661.0, -45146.0, 12964.0, 37882.0, -2435.0, -49274.0, -24130.0, -4111.0, 38303.0, 18165.0, -12840.0], [-12350.0, -15280.0, 23066.0, 50.0, 18402.0, -4508.0, -1657.0, 12553.0, -1123.0, 299.0, -8326.0], [17167.0, 31370.0, -13051.0, -23815.0, 8233.0, 14089.0, 24269.0, 12664.0, -12628.0, -30232.0, 323.0], [-35171.0, -12657.0, 11387.0, 54196.0, -7606.0, 18119.0, -14886.0, -12800.0, -45543.0, 19743.0, -3209.0], [-3546.0, -42215.0, 11182.0, -35705.0, -20550.0, -54515.0, -24669.0, -7094.0, 77690.0, 38017.0, 19708.0]], [[-8857.0, -31923.0, -37272.0, 18784.0, 11884.0, 9610.0, -16504.0, -13721.0, 30746.0, -1258.0, -2128.0], [-48221.0, 18425.0, -28246.0, 440.0, 4528.0, 3820.0, -9214.0, -49337.0, -6382.0, 24014.0, -23794.0], [-27616.0, 4450.0, -14170.0, -32572.0, 41299.0, -8227.0, 6828.0, -8980.0, 24059.0, -10752.0, -29848.0], [-30892.0, 29055.0, -8741.0, -4158.0, -11813.0, 55225.0, 9179.0, -11050.0, -64773.0, 5511.0, 19325.0], [-1826.0, 17449.0, 21535.0, -53201.0, 6131.0, -9015.0, 27167.0, 33012.0, -5418.0, -21644.0, 14794.0]], [[55297.0, -57094.0, 38275.0, 24753.0, -27483.0, -58325.0, -60478.0, -29953.0, 89227.0, 75175.0, -10453.0], [-31326.0, 33227.0, -28687.0, 9980.0, 4716.0, 18380.0, 6129.0, -31537.0, -31892.0, -295.0, -19882.0], [31211.0, -21552.0, 20463.0, -5428.0, 12003.0, -8771.0, -6614.0, 17860.0, 27890.0, -880.0, -2317.0], [58138.0, -3879.0, 17119.0, 9950.0, -13585.0, -21151.0, -17595.0, -8546.0, 33557.0, 21305.0, -10199.0], [65687.0, 12471.0, -5102.0, -4548.0, -4904.0, 26428.0, 21259.0, 39335.0, -8961.0, -41611.0, 23119.0]]]]], [[[[[-51756.0, -23004.0, 11551.0, 25519.0, 9658.0, -32835.0, -31994.0, -43669.0, 18520.0, 48136.0, -36726.0], [54345.0, 15994.0, 22650.0, -969.0, 4267.0, -2222.0, 13649.0, 26619.0, -4739.0, -21243.0, -5876.0], [-14765.0, -22536.0, -16213.0, 8597.0, -3712.0, 5228.0, -12858.0, -9181.0, 13560.0, 9319.0, 9569.0], [73721.0, -56507.0, -15668.0, 13978.0, 41004.0, -22390.0, -1769.0, 47796.0, 83683.0, -53057.0, -10418.0], [-14956.0, 36338.0, 15716.0, -30336.0, 21265.0, 5703.0, 34309.0, 25056.0, -38124.0, -32529.0, -8175.0]], [[49535.0, -30339.0, 20909.0, -23781.0, -8072.0, 32353.0, -8115.0, 32899.0, 16537.0, 3537.0, 42436.0], [34498.0, -235.0, -20399.0, -31145.0, 4285.0, 19291.0, 24174.0, 40764.0, 11425.0, -46443.0, 31393.0], [4807.0, -25417.0, 30285.0, -5318.0, 980.0, -14315.0, -29674.0, -18046.0, 31431.0, 45897.0, -9691.0], [-33308.0, -11443.0, 313.0, 14222.0, 30986.0, 19834.0, 7277.0, 14040.0, -25064.0, -20379.0, -10115.0], [22870.0, -67576.0, 29874.0, -7738.0, 34053.0, -36659.0, -20451.0, 29516.0, 76092.0, 1247.0, -9347.0]], [[-13347.0, 13034.0, -10627.0, -7816.0, -6462.0, 31580.0, -4364.0, -20041.0, -20525.0, 16218.0, 7128.0], [54057.0, 18385.0, -34813.0, 21092.0, 3327.0, 23614.0, -2009.0, -15782.0, 8589.0, -13666.0, -16082.0], [53185.0, 33243.0, -50305.0, -32876.0, 2802.0, 18155.0, 7866.0, -19384.0, 33981.0, -16101.0, -5885.0], [-2705.0, 35036.0, -5813.0, -7121.0, 9233.0, 40321.0, 13963.0, -5473.0, -43814.0, -9206.0, -5932.0], [-41026.0, 61126.0, -4307.0, -4520.0, -8274.0, 26375.0, 25516.0, -15547.0, -77513.0, -5687.0, -5687.0]]], [[[19177.0, 107902.0, -92328.0, -97412.0, -16279.0, 35406.0, 62381.0, -11544.0, -14308.0, -60154.0, 24576.0], [16740.0, -18123.0, -105.0, -3036.0, -8795.0, 3009.0, -6959.0, 8357.0, 16929.0, 3163.0, 17914.0], [36322.0, -11388.0, 25797.0, -12069.0, 16720.0, 24449.0, 13085.0, 47249.0, -9233.0, -27921.0, 16317.0], [21947.0, -6046.0, 31688.0, 6736.0, -27994.0, -15147.0, -33532.0, -33214.0, 17634.0, 61907.0, -4429.0], [-27428.0, -41060.0, 78992.0, 5512.0, -29926.0, -49914.0, -41032.0, -14880.0, 16054.0, 81864.0, 5324.0]], [[45552.0, -15690.0, 25436.0, 28611.0, -19012.0, 2844.0, -12049.0, 14186.0, -1960.0, 9456.0, 12400.0], [21305.0, -65385.0, 6976.0, 27735.0, 7514.0, -39506.0, -48245.0, -21726.0, 83616.0, 41102.0, -20123.0], [44956.0, -29918.0, -22890.0, -11867.0, 26082.0, -32730.0, 4864.0, 27830.0, 75143.0, -39971.0, -7729.0], [94797.0, -21322.0, -55601.0, -11004.0, 22103.0, 52691.0, 7173.0, 34257.0, 49749.0, -56735.0, 23375.0], [58708.0, 3011.0, 13126.0, -20752.0, 3009.0, 5926.0, 11971.0, 33018.0, 13654.0, -22846.0, 12219.0]], [[-36049.0, -10055.0, -8866.0, 8438.0, -2179.0, -44953.0, -25834.0, -48479.0, 38795.0, 41252.0, -30758.0], [66996.0, -50282.0, -10387.0, -11770.0, 27901.0, 19996.0, 559.0, 52205.0, 55293.0, -42376.0, 22600.0], [25602.0, -25090.0, -31319.0, -21948.0, 1424.0, 1769.0, -9108.0, 656.0, 55737.0, -4781.0, 16430.0], [-11255.0, 35850.0, -35847.0, -81.0, -980.0, 7670.0, -5212.0, -51213.0, -2054.0, 17025.0, -27345.0], [-46192.0, 51704.0, -27149.0, -11455.0, -22020.0, 22173.0, 30801.0, -6872.0, -64765.0, -18817.0, 18522.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (7, 11, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 3');
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 7, 11)", async () => {
    const a = tensor([[[[[[-77.0, -173.0, 28.0, 38.0, 83.0, 77.0, 66.0], [5.0, -152.0, 29.0, -20.0, 148.0, 12.0, 163.0], [53.0, -34.0, 79.0, 2.0, -208.0, -52.0, 50.0], [48.0, 51.0, 70.0, 1.0, -23.0, -56.0, -18.0], [-150.0, -42.0, -139.0, 18.0, 15.0, 135.0, -111.0]], [[62.0, 83.0, 21.0, -124.0, -56.0, -24.0, -146.0], [-154.0, 90.0, -45.0, -10.0, 148.0, -10.0, -158.0], [136.0, -84.0, 108.0, 112.0, 4.0, 196.0, 37.0], [-2.0, -159.0, -100.0, -153.0, -2.0, 66.0, -144.0], [-73.0, 16.0, -9.0, -120.0, 17.0, 149.0, -214.0]], [[25.0, 105.0, 132.0, -131.0, -22.0, -131.0, -57.0], [-74.0, 102.0, -43.0, -50.0, 189.0, 136.0, -52.0], [129.0, -84.0, -67.0, -157.0, 14.0, 55.0, -97.0], [-97.0, -58.0, 65.0, -112.0, 45.0, 193.0, -0.0], [144.0, 9.0, 10.0, -17.0, -69.0, 19.0, 60.0]]], [[[-120.0, 93.0, 18.0, -4.0, -27.0, -77.0, -112.0], [99.0, 172.0, -134.0, -257.0, 139.0, 76.0, -51.0], [10.0, -140.0, 60.0, 121.0, 36.0, -76.0, -109.0], [5.0, -17.0, -55.0, -20.0, 76.0, -81.0, -63.0], [-148.0, -106.0, -169.0, 18.0, -10.0, 76.0, 39.0]], [[-131.0, -51.0, 73.0, -92.0, 127.0, 118.0, -103.0], [-34.0, -60.0, 51.0, 130.0, -68.0, 32.0, -85.0], [-98.0, -132.0, 59.0, 30.0, 216.0, -31.0, 43.0], [-95.0, 122.0, -12.0, 60.0, 152.0, 55.0, -58.0], [-195.0, 138.0, -120.0, 22.0, -40.0, -161.0, 86.0]], [[57.0, -21.0, -41.0, 66.0, -48.0, -79.0, -70.0], [6.0, 148.0, 56.0, -198.0, 120.0, 151.0, 96.0], [-10.0, 1.0, 11.0, 9.0, -103.0, 130.0, 59.0], [154.0, 93.0, -84.0, 53.0, 161.0, -71.0, 16.0], [-141.0, 15.0, 61.0, -158.0, -10.0, -16.0, 68.0]]]]], [[[[[81.0, -105.0, 81.0, -32.0, -68.0, -74.0, -126.0], [65.0, 95.0, -6.0, 141.0, 40.0, 80.0, 51.0], [-68.0, 199.0, 90.0, -181.0, 58.0, -56.0, 39.0], [-22.0, -32.0, -5.0, 54.0, -17.0, -38.0, -13.0], [-111.0, -72.0, 168.0, 19.0, -47.0, -126.0, 23.0]], [[-56.0, 23.0, -80.0, 23.0, 38.0, -30.0, -29.0], [-228.0, 74.0, 41.0, 33.0, -47.0, 54.0, -32.0], [-75.0, 147.0, 75.0, -98.0, -109.0, -84.0, 164.0], [17.0, -103.0, -21.0, -86.0, -50.0, -25.0, 286.0], [-3.0, -81.0, 123.0, -69.0, 16.0, -67.0, 113.0]], [[68.0, -75.0, 17.0, 29.0, -130.0, 31.0, 106.0], [41.0, 303.0, 7.0, 76.0, 114.0, 65.0, -48.0], [90.0, 53.0, -115.0, -147.0, 36.0, -17.0, -59.0], [42.0, 16.0, -28.0, -15.0, -168.0, -149.0, 38.0], [-106.0, 121.0, -35.0, -62.0, 74.0, 104.0, 133.0]]], [[[-36.0, -17.0, 69.0, -135.0, -43.0, -59.0, 225.0], [36.0, -95.0, 236.0, 37.0, -48.0, -4.0, -108.0], [-164.0, 40.0, 21.0, 64.0, -240.0, 153.0, -150.0], [-43.0, -140.0, -163.0, -82.0, -54.0, 3.0, 48.0], [143.0, 225.0, 57.0, -153.0, 29.0, 89.0, 35.0]], [[53.0, 62.0, -161.0, -86.0, -47.0, -4.0, -90.0], [-27.0, 181.0, 30.0, -70.0, 13.0, -50.0, 2.0], [-58.0, 84.0, -80.0, 164.0, -27.0, 50.0, 31.0], [229.0, 157.0, 326.0, -205.0, -42.0, -24.0, 57.0], [0.0, -177.0, 171.0, -37.0, 23.0, 129.0, 51.0]], [[-205.0, 250.0, 179.0, 166.0, 158.0, 65.0, 38.0], [76.0, -30.0, -44.0, 78.0, 173.0, 19.0, -63.0], [94.0, -0.0, -139.0, 65.0, 85.0, -39.0, 7.0], [-37.0, 59.0, 84.0, -125.0, -210.0, 37.0, -78.0], [-66.0, -91.0, 139.0, -36.0, 87.0, -200.0, -155.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[[-59.0, 72.0, -260.0, 39.0, 63.0, -55.0, -173.0, -30.0, 52.0, 5.0, 13.0], [-55.0, -180.0, 88.0, 19.0, -106.0, -25.0, -41.0, -17.0, -28.0, 22.0, -6.0], [-102.0, -122.0, 1.0, -104.0, -19.0, 102.0, 95.0, 65.0, -108.0, -212.0, 201.0], [3.0, -27.0, 50.0, 233.0, 216.0, -140.0, 77.0, -3.0, 12.0, 136.0, -68.0], [-79.0, 132.0, -34.0, 197.0, -94.0, -40.0, 142.0, 63.0, -45.0, -75.0, 85.0], [106.0, -95.0, 52.0, -21.0, 32.0, -82.0, 274.0, 13.0, -81.0, 10.0, -28.0], [-151.0, 98.0, -138.0, 11.0, -133.0, -61.0, -68.0, -166.0, -22.0, 122.0, 183.0]]]);
    expect(b.shape).toEqual([1, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[[2955.0, 31263.0, -1202.0, 15112.0, 7047.0, -7564.0, 54396.0, 2231.0, -13152.0, -2362.0, 20058.0], [-29986.0, 59092.0, -42549.0, 20328.0, -23651.0, -7564.0, 19802.0, -13199.0, -10074.0, -3281.0, 50239.0], [-5939.0, -17372.0, -19125.0, -45663.0, 17112.0, 15247.0, -47300.0, -17963.0, 7672.0, 4221.0, 9562.0], [-14175.0, -13771.0, -7518.0, -7759.0, -732.0, 9695.0, -21054.0, 3051.0, -513.0, -14373.0, 10639.0], [55278.0, -8491.0, 57893.0, 10901.0, 19204.0, -12297.0, 62521.0, 17251.0, -564.0, 16925.0, -53679.0]], [[13189.0, -29110.0, 5809.0, -39215.0, -8161.0, 27131.0, -26282.0, 18862.0, 4820.0, -33032.0, -17845.0], [19802.0, -16526.0, 63667.0, 25682.0, -13765.0, 7568.0, 46927.0, 35617.0, -8162.0, -21086.0, -26961.0], [789.0, -5754.0, -32094.0, 15651.0, 40587.0, -28533.0, 50556.0, 690.0, -17766.0, -2658.0, 17987.0], [47502.0, 24161.0, 2150.0, -31712.0, 7032.0, 18757.0, 13176.0, 21358.0, 11224.0, -19874.0, -37138.0], [50750.0, -36681.0, 51081.0, -31701.0, -412.0, 19653.0, 59670.0, 40225.0, -12838.0, -40318.0, -36583.0]], [[-24648.0, -25712.0, -1876.0, -43491.0, -34902.0, 42903.0, -41319.0, 12811.0, -4613.0, -49979.0, 26502.0], [10329.0, -10160.0, 33495.0, 25679.0, -31955.0, -11406.0, 68323.0, 20148.0, -21037.0, -14969.0, -4076.0], [22743.0, 23938.0, -33079.0, -25642.0, -2263.0, 10998.0, -13673.0, 11373.0, 11461.0, -20685.0, -18711.0], [18850.0, -13845.0, 23087.0, -32929.0, -23444.0, 11469.0, 75982.0, 13801.0, -29442.0, -32218.0, 18189.0], [-11657.0, 2954.0, -42434.0, -12546.0, 3370.0, -7203.0, -34312.0, -17832.0, 6198.0, 9171.0, 9567.0]]], [[[11000.0, -34693.0, 51572.0, -10651.0, -3654.0, 20897.0, 1033.0, 19091.0, -920.0, -15323.0, -18863.0], [2372.0, 5585.0, -17324.0, -13590.0, -68812.0, 3886.0, -12668.0, 4378.0, 431.0, -18152.0, -8849.0], [6912.0, 16623.0, 1056.0, 27172.0, 49147.0, 3571.0, 10727.0, 24991.0, 6346.0, -16052.0, -9957.0], [1113.0, 22223.0, -1953.0, 16912.0, -2515.0, 4785.0, -14051.0, 10817.0, 10963.0, -5605.0, -12329.0], [34811.0, 23838.0, 28793.0, 10847.0, 7196.0, -17179.0, 32033.0, -10913.0, 7176.0, 41472.0, -32322.0]], [[20840.0, -11214.0, 41077.0, -13698.0, -18569.0, 20333.0, 81975.0, 36451.0, -27379.0, -50676.0, 8174.0], [22093.0, -21726.0, 25817.0, 7517.0, 50050.0, -4347.0, 28089.0, 15207.0, -1698.0, 428.0, -20902.0], [-19729.0, 44367.0, 533.0, 38200.0, -13838.0, 1787.0, 49535.0, 14996.0, -15567.0, -23086.0, 36434.0], [2879.0, -19801.0, 44120.0, 41992.0, -10543.0, -14501.0, 55511.0, 19735.0, -16359.0, -5013.0, -7693.0], [-10671.0, -6391.0, 44944.0, 9070.0, -32711.0, 1511.0, -37271.0, -23251.0, 12169.0, 42375.0, -12133.0]], [[8160.0, 5413.0, -6225.0, 12899.0, 32146.0, -3364.0, -31515.0, 3353.0, 18871.0, 11761.0, -26540.0], [-22770.0, -16791.0, -7856.0, -27387.0, -78358.0, 6364.0, 34854.0, -4875.0, -31999.0, -31292.0, 47450.0], [12448.0, -22649.0, 5269.0, -21790.0, 6994.0, -9752.0, 20409.0, -13622.0, -8821.0, 15087.0, -135.0], [-28135.0, 32730.0, -40664.0, 62242.0, -6646.0, -28377.0, -32034.0, -5256.0, 13266.0, 16999.0, -443.0], [-10376.0, -9164.0, 20265.0, -49258.0, -54376.0, 33286.0, 6979.0, -3712.0, -15986.0, -25909.0, 33124.0]]]]], [[[[[9192.0, 1420.0, -15967.0, -27944.0, 28564.0, 27386.0, -25841.0, 20386.0, 9846.0, -34441.0, -6626.0], [-10406.0, -12817.0, -5734.0, 44578.0, 16612.0, -37573.0, 19279.0, -9284.0, -6342.0, 26885.0, -26.0], [-33063.0, -30011.0, 15966.0, -37373.0, -78615.0, 33178.0, -11542.0, 1502.0, -19932.0, -39810.0, 41955.0], [3008.0, 3420.0, 5995.0, 8942.0, 15876.0, -1471.0, -3141.0, 1310.0, 5069.0, 6899.0, -7531.0], [-19686.0, -8021.0, 15514.0, -25102.0, -1122.0, 33190.0, -3184.0, 7000.0, -9857.0, -30100.0, 35207.0]], [[8465.0, 5991.0, 18804.0, 19729.0, -153.0, -6166.0, 2064.0, 2838.0, 6718.0, 13626.0, -19747.0], [19568.0, -50099.0, 76305.0, -14806.0, -5457.0, 9656.0, 53144.0, 11201.0, -19515.0, -3555.0, -8774.0], [-36661.0, -28700.0, 4317.0, -48671.0, -57154.0, 23064.0, -43119.0, -30263.0, -9191.0, 974.0, 42981.0], [-35340.0, 48451.0, -56873.0, -25327.0, -40326.0, -1858.0, -40733.0, -50817.0, 2987.0, 28967.0, 51254.0], [-33550.0, 20772.0, -29297.0, -24723.0, -27521.0, 22357.0, -13558.0, -8952.0, -9779.0, -25341.0, 53777.0]], [[-3984.0, 5822.0, -31409.0, -18879.0, 17289.0, -7999.0, -22015.0, -25130.0, 5155.0, 22022.0, 10259.0], [-14438.0, -50325.0, 25939.0, 44901.0, -15504.0, -26718.0, 24263.0, 9841.0, -15535.0, 1967.0, -5960.0], [7327.0, 15524.0, -20167.0, -10974.0, -25596.0, 6128.0, -35521.0, 1206.0, 14907.0, -4064.0, -19528.0], [-8807.0, -332.0, -17570.0, -28190.0, 4212.0, 13154.0, -79003.0, -22136.0, 23373.0, 20204.0, -7312.0], [-11922.0, -10546.0, 19611.0, 1216.0, -53548.0, -11686.0, 35238.0, -17030.0, -20544.0, 12836.0, 22794.0]]], [[[-41216.0, 17674.0, -31473.0, -45115.0, -58708.0, 21176.0, -34487.0, -34567.0, -8704.0, -3457.0, 61855.0], [-1184.0, -26639.0, 694.0, -26884.0, 34594.0, 28123.0, 22368.0, 30616.0, -15652.0, -56526.0, 22226.0], [63354.0, -84213.0, 86197.0, -45051.0, 46259.0, 7406.0, 51697.0, 17182.0, -9441.0, 5542.0, -54637.0], [23953.0, 41495.0, -10035.0, -16664.0, -3696.0, -295.0, -18730.0, -18010.0, 19435.0, 30045.0, -22796.0], [-25227.0, -34224.0, -26161.0, -27496.0, -53505.0, 3151.0, -14206.0, -6777.0, -16140.0, -24242.0, 28748.0]], [[26506.0, -24.0, 1025.0, -10214.0, -2490.0, -1149.0, -35278.0, -924.0, 21795.0, 16570.0, -46549.0], [-18261.0, -29632.0, 16160.0, -13411.0, -39665.0, 13278.0, -17280.0, -270.0, -7131.0, -13264.0, 12224.0], [10206.0, -19240.0, 29832.0, 39838.0, 24401.0, -34941.0, 19376.0, -11577.0, 1723.0, 47129.0, -26512.0], [-63846.0, -43687.0, -63334.0, -76898.0, -57090.0, 45603.0, -47285.0, -154.0, -27576.0, -82529.0, 89034.0], [-3662.0, 7776.0, -18367.0, -27385.0, 2704.0, 12438.0, 55797.0, 8895.0, -26562.0, -39391.0, 45625.0]], [[-30745.0, -67675.0, 76543.0, 46996.0, -24786.0, -13925.0, 92664.0, 17528.0, -48211.0, -17461.0, 39090.0], [-252.0, 28991.0, -14744.0, 58133.0, 18377.0, -23473.0, 23964.0, 16740.0, 2542.0, -815.0, -10336.0], [-3079.0, 37582.0, -27213.0, 50908.0, 12434.0, -29077.0, -23554.0, -8364.0, 19860.0, 32867.0, -21539.0], [22285.0, -59036.0, 28474.0, -81188.0, -5883.0, 36752.0, -12041.0, 6141.0, -5979.0, -27091.0, -8611.0], [-10055.0, 10936.0, 15523.0, -7513.0, 1108.0, 47498.0, -6324.0, 41281.0, -633.0, -64131.0, 14705.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (1, 7, 11, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([1, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 3');
});
test("matmul (2, 1, 2, 3, 5, 7) x (3, 7, 11)", async () => {
    const a = tensor([[[[[[-21.0, 267.0, -30.0, -20.0, 232.0, 44.0, -65.0], [-63.0, 62.0, -107.0, -32.0, -54.0, 53.0, 154.0], [193.0, 81.0, -113.0, -7.0, -137.0, -48.0, -214.0], [111.0, -6.0, 73.0, 33.0, 54.0, -113.0, -116.0], [-33.0, 2.0, 194.0, -213.0, -177.0, 173.0, 63.0]], [[44.0, -20.0, 94.0, -24.0, 126.0, -118.0, -85.0], [44.0, 103.0, 9.0, 203.0, 205.0, -74.0, -93.0], [-124.0, 32.0, -19.0, 5.0, -53.0, 97.0, 153.0], [-174.0, -140.0, 11.0, -17.0, 157.0, 102.0, -27.0], [16.0, -89.0, 28.0, -95.0, 57.0, 51.0, 133.0]], [[-68.0, -163.0, -21.0, 152.0, -144.0, 60.0, -173.0], [-68.0, -15.0, 80.0, -69.0, 24.0, -34.0, 105.0], [19.0, 10.0, 36.0, 50.0, 50.0, 75.0, 155.0], [44.0, -84.0, -85.0, -67.0, -151.0, 138.0, 17.0], [-140.0, 28.0, -141.0, -84.0, 242.0, 124.0, 14.0]]], [[[292.0, 37.0, -113.0, -45.0, -107.0, 1.0, 109.0], [-67.0, 39.0, 6.0, 85.0, 69.0, -164.0, 74.0], [-27.0, -49.0, 17.0, -70.0, -140.0, 175.0, 62.0], [-29.0, 117.0, 114.0, -68.0, -101.0, -101.0, 66.0], [241.0, 66.0, -94.0, -8.0, -25.0, -24.0, 59.0]], [[-82.0, -107.0, 7.0, -14.0, 102.0, -175.0, 58.0], [-40.0, 41.0, 141.0, -15.0, 61.0, 28.0, -13.0], [-39.0, 116.0, -48.0, 76.0, 119.0, 10.0, -74.0], [173.0, -146.0, -92.0, -12.0, 32.0, 10.0, 2.0], [-64.0, 15.0, -7.0, -30.0, 152.0, 25.0, 84.0]], [[-43.0, 124.0, -93.0, 217.0, 31.0, 208.0, 55.0], [2.0, 51.0, -117.0, -25.0, 48.0, -84.0, -14.0], [-74.0, -19.0, -18.0, -57.0, 75.0, 94.0, -29.0], [-118.0, -86.0, 11.0, -48.0, 12.0, -245.0, -36.0], [81.0, -121.0, 24.0, -171.0, 70.0, 35.0, -51.0]]]]], [[[[[-50.0, 48.0, -49.0, 47.0, 160.0, -51.0, 186.0], [-13.0, 36.0, 71.0, -100.0, 126.0, 62.0, 201.0], [216.0, -23.0, -110.0, -20.0, -183.0, -42.0, 21.0], [38.0, 96.0, 17.0, 64.0, 147.0, 21.0, -34.0], [-222.0, -43.0, -11.0, -52.0, -73.0, 144.0, 43.0]], [[130.0, 6.0, 48.0, -124.0, 50.0, 5.0, -19.0], [141.0, -61.0, -131.0, 123.0, 55.0, -45.0, 60.0], [-12.0, -110.0, 157.0, -185.0, 150.0, 9.0, 22.0], [36.0, 76.0, -59.0, 341.0, 42.0, -0.0, -36.0], [198.0, -54.0, 68.0, -125.0, -236.0, -32.0, -44.0]], [[27.0, -27.0, -3.0, -35.0, -32.0, 208.0, 89.0], [71.0, 24.0, -145.0, -22.0, -32.0, 102.0, -30.0], [-158.0, 68.0, -6.0, -41.0, 172.0, -20.0, -1.0], [-76.0, 70.0, -60.0, 36.0, -81.0, 35.0, 2.0], [-176.0, -6.0, -11.0, 40.0, -117.0, 25.0, 44.0]]], [[[-138.0, -89.0, -73.0, 72.0, 17.0, -145.0, 146.0], [84.0, -49.0, -98.0, -53.0, -57.0, 311.0, -10.0], [-158.0, 44.0, 122.0, -189.0, 74.0, 65.0, -262.0], [-35.0, -140.0, -74.0, 47.0, 45.0, 101.0, 59.0], [151.0, -59.0, 10.0, -91.0, 3.0, 269.0, -28.0]], [[29.0, 63.0, 67.0, 67.0, 142.0, 10.0, -113.0], [20.0, 66.0, -1.0, 39.0, 89.0, 80.0, 32.0], [-46.0, 175.0, 27.0, 52.0, -153.0, -46.0, 17.0], [20.0, -113.0, 128.0, 64.0, 7.0, -58.0, -106.0], [-110.0, -118.0, 61.0, 86.0, 47.0, 4.0, 13.0]], [[68.0, 47.0, 106.0, -143.0, -115.0, -60.0, -5.0], [-26.0, 144.0, 13.0, 42.0, -1.0, -220.0, 50.0], [98.0, -49.0, -42.0, 55.0, 192.0, -100.0, 85.0], [-21.0, 82.0, -28.0, 8.0, 24.0, 27.0, 57.0], [15.0, 57.0, 150.0, -26.0, 230.0, 96.0, -49.0]]]]]]);
    expect(a.shape).toEqual([2, 1, 2, 3, 5, 7]);
    const b = tensor([[[85.0, -1.0, 124.0, 47.0, 34.0, -183.0, -89.0, 92.0, 68.0, 41.0, -55.0], [113.0, 3.0, -74.0, -74.0, 68.0, -109.0, 104.0, 18.0, -6.0, 27.0, 178.0], [11.0, -4.0, 95.0, -99.0, 128.0, -106.0, 21.0, -4.0, 25.0, 130.0, -184.0], [232.0, 163.0, 23.0, -11.0, 124.0, -42.0, -22.0, -62.0, -79.0, -65.0, -4.0], [76.0, -119.0, -60.0, 24.0, 143.0, -52.0, 138.0, 31.0, -38.0, -52.0, 10.0], [207.0, -36.0, 35.0, -16.0, 25.0, 37.0, 35.0, 103.0, -30.0, -46.0, -61.0], [-35.0, 152.0, 107.0, 40.0, 83.0, -101.0, -47.0, 5.0, -72.0, 48.0, 31.0]], [[-191.0, 36.0, -49.0, 109.0, 83.0, -67.0, 7.0, 314.0, -20.0, -71.0, -101.0], [59.0, 142.0, -23.0, -2.0, -38.0, 116.0, 145.0, 141.0, 83.0, -9.0, 58.0], [135.0, 54.0, -23.0, -92.0, 24.0, -33.0, 108.0, -193.0, -52.0, -13.0, -29.0], [219.0, 97.0, 28.0, -35.0, 109.0, 146.0, 48.0, 92.0, -158.0, -47.0, 13.0], [-66.0, -4.0, 177.0, 40.0, -34.0, -51.0, 35.0, 97.0, -18.0, -110.0, -31.0], [-163.0, -89.0, 125.0, 96.0, -11.0, -18.0, 76.0, 199.0, 164.0, 11.0, 20.0], [116.0, -23.0, -33.0, 98.0, 63.0, 168.0, -78.0, -193.0, -63.0, 60.0, 37.0]], [[-73.0, 126.0, -27.0, 253.0, 146.0, 149.0, -44.0, -23.0, 55.0, -154.0, -147.0], [-19.0, -21.0, 91.0, 7.0, 100.0, -62.0, -13.0, -180.0, -114.0, 2.0, -132.0], [-52.0, 3.0, 68.0, 119.0, -175.0, -117.0, 127.0, 5.0, 125.0, -74.0, 195.0], [-68.0, 119.0, 87.0, 25.0, -24.0, 135.0, 3.0, -24.0, -102.0, -33.0, -203.0], [26.0, 51.0, -85.0, -19.0, -89.0, -213.0, -19.0, -128.0, 80.0, -92.0, 76.0], [43.0, 11.0, -11.0, -27.0, -151.0, -1.0, 188.0, -149.0, -65.0, 125.0, -5.0], [-6.0, -114.0, 116.0, 96.0, 114.0, 287.0, -1.0, -160.0, 19.0, 106.0, 36.0]]]);
    expect(b.shape).toEqual([3, 7, 11]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 1, 2, 3, 5, 11]);
    expect(await c.toArrayAsync()).toEqual([[[[[[52431.0, -41390.0, -45007.0, -15291.0, 40003.0, -25111.0, 66058.0, 15633.0, -7656.0, -13460.0, 51902.0], [-5473.0, 23387.0, -1728.0, 7412.0, -9205.0, 6672.0, -2323.0, 2287.0, -15429.0, -4977.0, 35318.0], [9833.0, -15136.0, -9316.0, 3261.0, -41815.0, -4914.0, -21500.0, 9839.0, 32420.0, -5075.0, 19547.0], [1989.0, -15032.0, 2295.0, -3465.0, 12071.0, -24056.0, -747.0, -2779.0, 16492.0, 8556.0, -16900.0], [-29707.0, -11045.0, 32707.0, -23058.0, -18323.0, 3445.0, -9427.0, 22077.0, 16421.0, 42036.0, -43043.0]], [[-1092.0, 13445.0, 5827.0, -17590.0, -4289.0, -30456.0, 8480.0, -4209.0, -19901.0, -23296.0, -18053.0], [31089.0, 44292.0, 31056.0, -11361.0, 10066.0, 13594.0, 34764.0, 68386.0, -34840.0, -42653.0, -7368.0], [29537.0, -12401.0, 3612.0, 10179.0, -1045.0, 40038.0, -4457.0, -45664.0, 12557.0, 24605.0, 24240.0], [-7384.0, -36284.0, 52447.0, -5677.0, -18872.0, -21806.0, -5793.0, -37325.0, 9577.0, -3498.0, 5088.0], [-21979.0, -27591.0, 10034.0, 22881.0, 907.0, -7671.0, -18832.0, -31660.0, 4806.0, 6037.0, -4651.0]], [[-1309.0, 25918.0, -9689.0, -32536.0, -42167.0, 3912.0, 17089.0, 64323.0, -21994.0, 9094.0, -20911.0], [4313.0, -27344.0, 10422.0, 1028.0, -8804.0, -2820.0, 6187.0, -8486.0, 21133.0, 11471.0, 47357.0], [-3254.0, -6053.0, 20100.0, 22316.0, -1831.0, 38509.0, 16751.0, -45632.0, 1375.0, 13985.0, 1762.0], [9266.0, -9041.0, -7152.0, -471.0, 9046.0, 49568.0, 16956.0, 11337.0, -12522.0, 34501.0, -9908.0], [34272.0, -16537.0, -30878.0, -60705.0, -29615.0, -65091.0, 6337.0, -52201.0, -8383.0, 29542.0, 24717.0]]], [[[5578.0, 22201.0, 39818.0, 24444.0, -13829.0, -49009.0, -43377.0, 28103.0, 16552.0, 11956.0, 13746.0], [-12796.0, 22956.0, -10631.0, -324.0, 23591.0, -13326.0, 8579.0, -25139.0, -14385.0, 1069.0, 22171.0], [-470.0, 8186.0, 21442.0, -2236.0, -21253.0, 18913.0, -16905.0, 14901.0, 19.0, 6536.0, -20238.0], [-34659.0, 14527.0, 6599.0, -18727.0, 1640.0, -21825.0, -1936.0, -10006.0, 7664.0, 34276.0, 8914.0], [16120.0, 11836.0, 22859.0, 17981.0, 380.0, -46544.0, -23446.0, 21280.0, 11696.0, 5199.0, 18864.0]], [[35749.0, -5293.0, 191.0, -15914.0, -1987.0, -1501.0, -30259.0, -79599.0, -39583.0, -2313.0, -2825.0], [15711.0, 8104.0, 12080.0, -13035.0, -6330.0, -5206.0, 25450.0, -21374.0, 3554.0, -5839.0, 322.0], [6389.0, 20184.0, 27230.0, -4259.0, -9331.0, 10068.0, 25708.0, 48181.0, 5056.0, -18643.0, 6820.0], [-60215.0, -21700.0, 3509.0, 30469.0, 15319.0, -28719.0, -28747.0, 55096.0, -7960.0, -12499.0, -24147.0], [1231.0, -8227.0, 29369.0, 11400.0, -9471.0, 7789.0, 199.0, -15883.0, 3701.0, -5495.0, 6043.0]], [[283.0, 15121.0, 26457.0, -16578.0, -10708.0, 35055.0, 27580.0, -70764.0, -60255.0, 35569.0, -68937.0], [4389.0, -1025.0, -10324.0, -13673.0, 33283.0, -6708.0, -32375.0, -599.0, -8745.0, -7123.0, -21202.0], [16741.0, -7597.0, -16687.0, -29169.0, -32361.0, -39829.0, 17322.0, -12566.0, 999.0, 16347.0, 25633.0], [2933.0, -16720.0, -10569.0, -27416.0, 5222.0, -32660.0, -38689.0, 60130.0, 25786.0, -16775.0, 41428.0], [10397.0, 2239.0, -38694.0, 11056.0, -17699.0, -35904.0, 5845.0, 18126.0, 41047.0, -16320.0, 46767.0]]]]], [[[[[6632.0, 19119.0, -4809.0, 10528.0, 38163.0, -21855.0, 18932.0, -5817.0, -26568.0, -7225.0, 30599.0], [-4081.0, -3137.0, 16286.0, 868.0, 34945.0, -29430.0, 18703.0, 16665.0, -12545.0, 16413.0, -1832.0], [-13426.0, 23376.0, 29333.0, 20084.0, -36256.0, -18680.0, -51197.0, 11244.0, 20358.0, 7691.0, 5729.0], [45822.0, -12803.0, -11028.0, -5873.0, 36656.0, -25341.0, 28170.0, 7738.0, -6391.0, -8042.0, 10749.0], [-13159.0, 1700.0, -12566.0, -7927.0, -21598.0, 53444.0, 9144.0, -5146.0, -15647.0, -9077.0, -1393.0]], [[-51471.0, -4112.0, -982.0, 14700.0, -4754.0, -33534.0, 4624.0, 30506.0, 16111.0, -10665.0, -17939.0], [-10613.0, 3676.0, 3081.0, 26998.0, 26689.0, 13843.0, -22277.0, 57072.0, -32655.0, -16485.0, -12766.0], [-32333.0, -27426.0, 21276.0, -37.0, -17026.0, -48263.0, -3740.0, -54504.0, 9566.0, -6585.0, -15782.0], [57374.0, 42639.0, 16015.0, -4583.0, 32157.0, 49947.0, 25546.0, 75801.0, -43710.0, -25280.0, 4282.0], [-43511.0, -4189.0, -57844.0, 2985.0, 12097.0, -34804.0, -12360.0, 9166.0, 9544.0, 14387.0, -21679.0]], [[8656.0, -9695.0, 4321.0, 8946.0, -15807.0, 33474.0, 38300.0, -36072.0, -6631.0, 35543.0, 5847.0], [7131.0, 8299.0, -13389.0, -4700.0, 22695.0, 21190.0, -2103.0, -12452.0, -24472.0, 13084.0, -41436.0], [16960.0, -17567.0, -8037.0, -44061.0, -26636.0, -69494.0, -1844.0, -26528.0, 2031.0, 7835.0, 34539.0], [4277.0, -10916.0, 14206.0, -24192.0, 7692.0, 14008.0, 3039.0, -7183.0, -32049.0, 27135.0, -23335.0], [8583.0, -28031.0, 21712.0, -39107.0, -13677.0, 18359.0, 13424.0, 8324.0, -24600.0, 45139.0, 8966.0]]], [[[-39719.0, 37288.0, -6278.0, 15103.0, -236.0, 18674.0, -9682.0, -32148.0, -23171.0, -9437.0, 18433.0], [48624.0, -14411.0, 16748.0, 11115.0, -20798.0, 18064.0, -9975.0, 40740.0, 1299.0, -18996.0, -14949.0], [-22715.0, -81975.0, -45804.0, -30425.0, -19739.0, 44143.0, 50159.0, 5165.0, 21075.0, 3441.0, -16517.0], [13557.0, 7549.0, 7219.0, 17348.0, -497.0, 22973.0, -7061.0, 3735.0, -16091.0, -22044.0, -13449.0], [42057.0, -29498.0, 28186.0, 6122.0, -4052.0, -5815.0, -6218.0, 46092.0, 11893.0, -2061.0, -37530.0]], [[-2214.0, 21248.0, 27578.0, -9908.0, -3133.0, -13470.0, 34334.0, 48795.0, -3218.0, -28936.0, -8730.0], [-6722.0, 5609.0, 23314.0, 15151.0, 1489.0, 11440.0, 18173.0, 37744.0, 8470.0, -10824.0, 2369.0], [53712.0, 34011.0, -34328.0, -18538.0, 2627.0, 41570.0, 20288.0, -17472.0, -36.0, 16240.0, 19141.0], [17505.0, 5366.0, -2046.0, -27286.0, 9724.0, -26449.0, 4756.0, -18874.0, -29507.0, -12843.0, -16753.0], [38871.0, -9923.0, 17499.0, -16838.0, 5369.0, 3940.0, -6229.0, -52193.0, -25363.0, -309.0, 2719.0]], [[-7185.0, -15073.0, 7063.0, 29897.0, 18235.0, -1369.0, 340.0, 18398.0, 20823.0, -10953.0, 24879.0], [-14156.0, -9434.0, 26649.0, 7786.0, 46330.0, 6130.0, -40342.0, -1357.0, -5335.0, -20164.0, -18353.0], [-7597.0, 18798.0, -10536.0, 28040.0, 23140.0, 13578.0, -31377.0, -18240.0, 23591.0, -35051.0, -9141.0], [2330.0, -8477.0, 11096.0, -3584.0, 10127.0, 7363.0, 889.0, -30824.0, -13571.0, 12415.0, -11080.0], [2192.0, 16421.0, -13570.0, 9728.0, -58288.0, -85508.0, 31298.0, -45135.0, 26958.0, -26792.0, 40035.0]]]]]]);
});
test("matmul (2, 1, 2, 3, 5, 7) x (3, 7, 11, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([3, 7, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 3');
});
test("matmul (2, 1, 2, 3, 5, 7) x (3, 1, 11)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([3, 1, 11]);
    expect(() => matmul(a, b)).toThrow('Expected size for first two dimensions of batch2 tensor to be: [12, 7] but got: [12, 1].');
});
test("matmul (2, 1, 2, 3, 5, 7) x (3, 1, 11, 1)", async () => {
    const a = ones([2, 1, 2, 3, 5, 7]);
    const b = ones([3, 1, 11, 1]);
    expect(() => matmul(a, b)).toThrow('The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2');
});
