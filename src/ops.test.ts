import { ones } from "./factories";
import { tensor } from "./ops_artisanal";

test("sigmoid(1).size == 307", async () => {
    // prime number > 300
    const n = 307;
    const x = ones(n);
    const y = x.sigmoid();
    expect(y.shape[0]).toBe(n);
    const a = await y.toArrayAsync();
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
    expect(a).toEqual([6]);
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
    const a = await y.toArrayAsync();
    expect(a[0]).toBeCloseTo(2);
});

test("mean vector grad", async () => {
    const x = tensor({data:[1, 2, 3, 4], requiresGrad: true});
    const y = x.mean();
    y.backward();
    const a = await x.grad!.toArrayAsync();
    expect(a).toEqual([0.25, 0.25, 0.25, 0.25]);
});

test("sum dim=0/3", async () => {
    const x = tensor([[[-43.0, -78.0, -28.0, -77.0, -74.0], [-32.0, 3.0, 17.0, -40.0, 65.0], [72.0, -33.0, 74.0, 54.0, 8.0]], [[-54.0, -47.0, -23.0, -13.0, -28.0], [39.0, -35.0, -13.0, -56.0, 45.0], [-29.0, 77.0, 86.0, 4.0, -8.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.sum(0);
    expect(y.shape).toEqual([3, 5]);
    expect(await y.toArrayAsync()).toEqual([[-97.0, -125.0, -51.0, -90.0, -102.0], [7.0, -32.0, 4.0, -96.0, 110.0], [43.0, 44.0, 160.0, 58.0, 0.0]]);
});
test("sum dim=1/3", async () => {
    const x = tensor([[[96.0, -17.0, -42.0, 96.0, -58.0], [-49.0, 14.0, 44.0, 38.0, -27.0], [-10.0, -57.0, 48.0, 78.0, 45.0]], [[-59.0, 17.0, 18.0, 22.0, 66.0], [-49.0, 15.0, -3.0, 99.0, -89.0], [-66.0, 42.0, -56.0, -27.0, 17.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.sum(1);
    expect(y.shape).toEqual([2, 5]);
    expect(await y.toArrayAsync()).toEqual([[37.0, -60.0, 50.0, 212.0, -40.0], [-174.0, 74.0, -41.0, 94.0, -6.0]]);
});
test("sum dim=2/3", async () => {
    const x = tensor([[[-23.0, 31.0, 3.0, -38.0, 78.0], [-22.0, 83.0, 84.0, 23.0, 19.0], [-48.0, -85.0, -63.0, 35.0, 67.0]], [[7.0, -4.0, -67.0, -9.0, 10.0], [-89.0, 19.0, -77.0, -25.0, 79.0], [-34.0, -42.0, 8.0, -41.0, -21.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.sum(2);
    expect(y.shape).toEqual([2, 3]);
    expect(await y.toArrayAsync()).toEqual([[51.0, 187.0, -94.0], [-63.0, -93.0, -130.0]]);
});
test("prod dim=0/3", async () => {
    const x = tensor([[[71.0, -10.0, 10.0, 28.0, -35.0], [-69.0, -40.0, 62.0, -23.0, 45.0], [-77.0, 90.0, -77.0, 16.0, 90.0]], [[42.0, 1.0, -2.0, -69.0, -37.0], [-42.0, -55.0, -48.0, -12.0, -49.0], [18.0, -19.0, -65.0, 93.0, -86.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.prod(0);
    expect(y.shape).toEqual([3, 5]);
    expect(await y.toArrayAsync()).toEqual([[2982.0, -10.0, -20.0, -1932.0, 1295.0], [2898.0, 2200.0, -2976.0, 276.0, -2205.0], [-1386.0, -1710.0, 5005.0, 1488.0, -7740.0]]);
});
test("prod dim=1/3", async () => {
    const x = tensor([[[-69.0, 70.0, -95.0, 8.0, 52.0], [-5.0, 48.0, -78.0, 21.0, -42.0], [-67.0, 49.0, 4.0, -22.0, -61.0]], [[-98.0, 35.0, -61.0, 31.0, 4.0], [65.0, 9.0, 87.0, -60.0, -70.0], [93.0, 34.0, 86.0, 2.0, -79.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.prod(1);
    expect(y.shape).toEqual([2, 5]);
    expect(await y.toArrayAsync()).toEqual([[-23115.0, 164640.0, 29640.0, -3696.0, 133224.0], [-592410.0, 10710.0, -456402.0, -3720.0, 22120.0]]);
});
test("prod dim=2/3", async () => {
    const x = tensor([[[-11.0, 73.0, -13.0, 64.0, 80.0], [35.0, -10.0, 52.0, -43.0, 88.0], [77.0, 5.0, -95.0, -60.0, 19.0]], [[14.0, 52.0, -74.0, 85.0, -76.0], [94.0, -83.0, 49.0, -69.0, -62.0], [26.0, 92.0, 38.0, 69.0, -94.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.prod(2);
    expect(y.shape).toEqual([2, 3]);
    expect(await y.toArrayAsync()).toEqual([[53447680.0, 68868800.0, 41695500.0], [348013120.0, -1635470848.0, -589551488.0]]);
});
test("mean dim=0/3", async () => {
    const x = tensor([[[13.0, -66.0, 38.0, 92.0, -77.0], [76.0, 38.0, 15.0, -94.0, -20.0], [20.0, 57.0, -71.0, 8.0, -43.0]], [[70.0, 22.0, 24.0, -39.0, 87.0], [-23.0, 5.0, 4.0, 55.0, 20.0], [20.0, 4.0, 16.0, -52.0, -92.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.mean(0);
    expect(y.shape).toEqual([3, 5]);
    expect(await y.toArrayAsync()).toEqual([[41.5, -22.0, 31.0, 26.5, 5.0], [26.5, 21.5, 9.5, -19.5, 0.0], [20.0, 30.5, -27.5, -22.0, -67.5]]);
});
test("mean dim=1/3", async () => {
    const x = tensor([[[50.0, -94.0, 38.0, 36.0, -84.0], [99.0, 0.0, 66.0, 66.0, -3.0], [-72.0, 50.0, -49.0, 35.0, -7.0]], [[74.0, -33.0, -55.0, 68.0, 9.0], [-15.0, 79.0, 60.0, 59.0, -5.0], [56.0, -35.0, 64.0, 1.0, -48.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.mean(1);
    expect(y.shape).toEqual([2, 5]);
    expect(await y.toArrayAsync()).toEqual([[25.66666603088379, -14.666666984558105, 18.33333396911621, 45.66666793823242, -31.33333396911621], [38.33333206176758, 3.6666667461395264, 23.0, 42.66666793823242, -14.666666984558105]]);
});
test("mean dim=2/3", async () => {
    const x = tensor([[[19.0, 19.0, 97.0, 71.0, -35.0], [81.0, 31.0, 79.0, -13.0, -60.0], [4.0, 87.0, -66.0, 43.0, 87.0]], [[-63.0, 15.0, 55.0, 45.0, -39.0], [-48.0, -42.0, -74.0, 4.0, 79.0], [-97.0, -59.0, 68.0, -20.0, 60.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.mean(2);
    expect(y.shape).toEqual([2, 3]);
    expect(await y.toArrayAsync()).toEqual([[34.20000076293945, 23.600000381469727, 31.0], [2.5999999046325684, -16.200000762939453, -9.600000381469727]]);
});
test("norm dim=0/3", async () => {
    const x = tensor([[[-98.0, -4.0, 22.0, -95.0, 68.0], [39.0, 23.0, 52.0, 15.0, -62.0], [-57.0, 80.0, -6.0, -16.0, 81.0]], [[-40.0, 17.0, 37.0, 58.0, 27.0], [-32.0, -3.0, 40.0, -7.0, -40.0], [6.0, 80.0, -99.0, -43.0, -30.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.norm(0);
    expect(y.shape).toEqual([3, 5]);
    expect(await y.toArrayAsync()).toEqual([[105.84894561767578, 17.464248657226562, 43.046485900878906, 111.3058853149414, 73.16419982910156], [50.447994232177734, 23.194826126098633, 65.6048812866211, 16.552946090698242, 73.7834701538086], [57.314918518066406, 113.1370849609375, 99.18164825439453, 45.880279541015625, 86.37708282470703]]);
});
test("norm dim=1/3", async () => {
    const x = tensor([[[53.0, 96.0, -61.0, -99.0, -39.0], [-94.0, -33.0, 76.0, -32.0, 52.0], [-48.0, 8.0, -97.0, -49.0, -69.0]], [[-73.0, 27.0, -9.0, 39.0, 73.0], [46.0, 18.0, -69.0, -16.0, 4.0], [-89.0, 16.0, -69.0, 7.0, 14.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.norm(1);
    expect(y.shape).toEqual([2, 5]);
    expect(await y.toArrayAsync()).toEqual([[118.10588836669922, 101.82828521728516, 137.49908447265625, 115.00434875488281, 94.79451751708984], [123.95967102050781, 36.18010330200195, 97.9948959350586, 42.731719970703125, 74.43789672851562]]);
});
test("norm dim=2/3", async () => {
    const x = tensor([[[-82.0, -28.0, 41.0, 28.0, 65.0], [83.0, -26.0, -72.0, -7.0, 6.0], [93.0, 73.0, -13.0, 35.0, -33.0]], [[2.0, -64.0, 40.0, -42.0, -62.0], [69.0, -91.0, -76.0, 88.0, 37.0], [77.0, 34.0, -58.0, -85.0, -86.0]]]);
    expect(x.shape).toEqual([2, 3, 5]);
    const y = x.norm(2);
    expect(y.shape).toEqual([2, 3]);
    expect(await y.toArrayAsync()).toEqual([[119.15535736083984, 113.28724670410156, 128.30043029785156], [106.33908081054688, 167.12570190429688, 158.3350830078125]]);
});