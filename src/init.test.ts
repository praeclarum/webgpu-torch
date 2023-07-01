import { empty } from "./factories";
import { tensor } from "./ops_artisanal";

async function testUniformAsync(min: number, max: number) {
    const x = empty([11, 13]);
    x.uniform_(min, max);
    const a = (await x.toArrayAsync()) as number[][];
    let sum = 0.0;
    let count = 0;
    for (let i = 0; i < 11; i++) {
        for (let j = 0; j < 13; j++) {
            expect(a[i][j]).toBeGreaterThanOrEqual(min);
            expect(a[i][j]).toBeLessThanOrEqual(max);
            sum += a[i][j];
            count++;
        }
    }
    const mean = sum / count;
    const variance =
        a.reduce(
            (acc, row) =>
                acc +
                row.reduce((acc2, v) => acc2 + (v - mean) * (v - mean), 0),
            0
        ) / count;
    const std = Math.sqrt(variance);
    expect(mean).toBeCloseTo((min + max) / 2, 1);
    expect(std).toBeCloseTo((max - min) / Math.sqrt(12), 1);
}

test("uniform_ -1 to 1", async () => {
    await testUniformAsync(-1, 1);
});

test("uniform_ 5 to 11", async () => {
    await testUniformAsync(5, 11);
});

test("uniform_ -0.6 to -0.4", async () => {
    await testUniformAsync(-0.6, -0.4);
});
