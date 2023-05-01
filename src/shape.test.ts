import { Shape, Shapeish, getShape } from "./shape";

test("number to shape", () => {
    expect(getShape(1)).toEqual([1]);
});

test("shape to shape", () => {
    expect(getShape([1, 2, 3])).toEqual([1, 2, 3]);
});
