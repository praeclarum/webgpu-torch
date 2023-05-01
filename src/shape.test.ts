import { Shape, Shapeish, shapeishToShape } from "./shape";

test("number to shape", () => {
    expect(shapeishToShape(1)).toEqual([1]);
});

test("shape to shape", () => {
    expect(shapeishToShape([1, 2, 3])).toEqual([1, 2, 3]);
});
