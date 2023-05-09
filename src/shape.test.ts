import { Shape, Shapeish, getShape, shapeSize } from "./shape";

test("number to shape", () => {
    expect(getShape(1)).toEqual([1]);
});

test("shape to shape", () => {
    expect(getShape([1, 2, 3])).toEqual([1, 2, 3]);
});

test("size of empty shape is 1", () => {
    expect(shapeSize([])).toBe(1);
});
