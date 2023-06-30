import { empty } from "./factories"

test("empty 2d makes a tensor", () => {
    const x = empty([2, 3]);
    expect(x.shape).toEqual([2, 3]);
    expect(x.dtype).toEqual("float32");
});

test("zeros 2d", async () => {
    const x = empty([2, 3]);
    expect(x.shape).toEqual([2, 3]);
    expect(x.dtype).toEqual("float32");
    expect(await x.toArrayAsync()).toEqual([[0, 0, 0], [0, 0, 0]]);
});
