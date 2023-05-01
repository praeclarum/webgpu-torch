export type Shape = number[];
export type Shapeish = Shape | number;
export type Strides = number[];

export function getShape(shapeish: Shapeish): Shape {
    if (typeof shapeish === "number") {
        return [shapeish];
    } else {
        return shapeish;
    }
}
