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

export function shapeSize(shape: Shape): number {
    return shape.reduce((a, b) => a * b);
}

export function defaultStrides(shape: Shape): Strides {
    const strides = [];
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

export function shapeGetAxis(shape: Shape, axis: number): number {
    if (axis < 0) {
        axis += shape.length;
    }
    if (axis < 0 || axis >= shape.length) {
        throw new Error(`Axis ${axis} is out of bounds for shape ${shape}`);
    }
    return axis;
}

/*
export function broadcastShape(leftShape: Shape, rightShape: Shape, rightStrides: Strides): {shape: Shape, strides: Strides} {
    const leftRank = leftShape.length;
    const rightRank = rightShape.length;
    const newRank = Math.max(leftRank, rightRank);
    const newShape = new Array(newRank);
    const newStrides = new Array(newRank);
    for (let i = 0; i < newRank; i++) {
        const leftDim = leftShape[leftRank - 1 - i] || 1;
        const rightDim = rightShape[rightRank - 1 - i] || 1;
        if (leftDim !== rightDim && leftDim !== 1 && rightDim !== 1) {
            throw new Error(`Cannot broadcast shapes ${leftShape} and ${rightShape}`);
        }
        newShape[newRank - 1 - i] = Math.max(leftDim, rightDim);
        newStrides[newRank - 1 - i] = rightStrides[rightRank - 1 - i] || 0;
    }
    return {shape: newShape, strides: newStrides};
}
*/
