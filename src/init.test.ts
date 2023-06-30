import { empty } from "./factories";
import { tensor } from "./ops_artisanal";

async function testUniformAsync(min: number, max: number) {
    const x = empty([11, 13]);
    x.uniform_(min, max);
    const a = await x.toArrayAsync() as number[][];
    expect(a.length).toEqual(11);
    expect(a[0].length).toEqual(13);
    for (let i = 0; i < 11; i++) {
        for (let j = 0; j < 13; j++) {
            // expect(a[i][j]).toBeGreaterThanOrEqual(min);
            // expect(a[i][j]).toBeLessThanOrEqual(max);
        }
    }
}

test("uniform_ -1 to 1", async () => {
    await testUniformAsync(-1, 1);
});

test("uniform_ 100 to 1000", async () => {
    await testUniformAsync(100, 1000);
});

test("uniform_ -0.6 to -0.4", async () => {
    await testUniformAsync(-0.6, -0.4);
});
