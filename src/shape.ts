export type Shape = number[];
export type Shapeish = Shape | number;
export type Strides = number[];

export function check(cond: boolean, msgGenerator: () => string) {
    if (!cond) {
        throw new Error(msgGenerator());
    }
}

export interface StridedShape {
    shape: Shape;
    strides: Strides;
}

export function getShape(shapeish: Shapeish): Shape {
    if (typeof shapeish === "number") {
        return [shapeish];
    } else {
        return shapeish;
    }
}

export function shapeSize(shape: Shape): number {
    if (shape.length === 0) {
        return 1;
    } else if (shape.length === 1) {
        return shape[0];
    } else {
        return shape.reduce((a, b) => a * b);
    }
}

export function shapesAreEqual(a: Shape, b: Shape): boolean {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

export function defaultStrides(shape: Shape): Strides {
    const strides = [];
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= Math.max(1, shape[i]);
    }
    return strides;
}

export function contiguousStridedShape(shape: Shape): StridedShape {
    const strides = defaultStrides(shape);
    return { shape, strides };
}

export function stridedShapeIsContiguous(stridedShape: StridedShape): boolean {
    const { shape, strides } = stridedShape;
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (strides[i] !== stride) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

export function broadcastShapes(
    aInput: StridedShape,
    bInput: StridedShape
): {
    output: StridedShape;
    a: StridedShape;
    b: StridedShape;
} {
    const aShape = aInput.shape.slice();
    const bShape = bInput.shape.slice();
    const aStrides = aInput.strides.slice();
    const bStrides = bInput.strides.slice();

    const padFront = (shape: number[], strides: number[], length: number) => {
        while (shape.length < length) {
            shape.unshift(1);
            strides.unshift(0);
        }
    };

    // Pad shapes to the same length by putting 1's in front
    const maxLength = Math.max(aShape.length, bShape.length);
    padFront(aShape, aStrides, maxLength);
    padFront(bShape, bStrides, maxLength);

    const outputShape: number[] = [];
    for (let dim = 0; dim < maxLength; dim++) {
        if (aShape[dim] === 1 || bShape[dim] === 1) {
            outputShape[dim] = Math.max(aShape[dim], bShape[dim]);
            if (aShape[dim] === 1) {
                aStrides[dim] = 0;
            }
            if (bShape[dim] === 1) {
                bStrides[dim] = 0;
            }
        } else if (aShape[dim] === bShape[dim]) {
            outputShape[dim] = aShape[dim];
        } else {
            throw new Error(
                "The size of tensor a (" +
                    aShape[dim] +
                    ") must match the size of tensor b (" +
                    bShape[dim] +
                    ") at non-singleton dimension " +
                    dim
            );
        }
    }

    return {
        output: contiguousStridedShape(outputShape),
        a: { shape: aShape, strides: aStrides },
        b: { shape: bShape, strides: bStrides },
    };
}

export function reshapeBatchedMatmul(tensor: StridedShape): StridedShape {
    const inputShape = tensor.shape;
    const inputStrides = tensor.strides;

    if (inputShape.length < 3) {
        throw new Error("Input tensor must be at least 3D");
    }

    let batchSize = 1;
    for (let i = 0; i < inputShape.length - 2; i++) {
        batchSize *= inputShape[i];
    }

    const newShape = [batchSize].concat(inputShape.slice(inputShape.length - 2));
    const newStrides = [
        inputStrides[inputStrides.length - 3],
        inputStrides[inputStrides.length - 2],
        inputStrides[inputStrides.length - 1],
    ];

    return {
        shape: newShape,
        strides: newStrides,
    };
}

/**
 * If the first argument is 1-dimensional,
 * a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
 * If the second argument is 1-dimensional,
 * a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
 * The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).
 *
 * For example,
 *
 * if input is a (j×1×n×n) tensor and other is a (k×n×n) tensor, out will be a (j×k×n×n) tensor.
 * Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs are broadcastable,
 * and not the matrix dimensions.
 *
 * For example,
 *
 * if input is a (j×1×n×m) tensor and other is a (k×m×p) tensor,
 * these inputs are valid for broadcasting even though the final two dimensions (i.e. the matrix dimensions) are different.
 * out will be a (j×k×n×p) tensor.
 * @param input The first tensor to be multiplied
 * @param other The second tensor to be multiplied
 * @returns The shapes and strides of the inputs and the output
 */
export function broadcastBatchedMatmul(
    input: StridedShape,
    other: StridedShape
): {
    output: StridedShape;
    a: StridedShape;
    b: StridedShape;
} {
    const inputShape = input.shape.slice();
    const inputStrides = input.strides.slice();
    const otherShape = other.shape.slice();
    const otherStrides = other.strides.slice();

    if (inputShape.length < 2 && otherShape.length < 2) {
        throw new Error(
            `Expected at least 2D tensors for matmul broadcast, got ${inputShape} and ${otherShape}`
        );
    }

    const padFront = (shape: number[], strides: number[], length: number) => {
        while (shape.length < length) {
            shape.unshift(1);
            strides.unshift(0);
        }
    };

    let squeezeOutput: number | null = null;
    if (inputShape.length === 1) {
        inputShape.unshift(1);
        inputStrides.unshift(0);
        squeezeOutput = -2;
    } else if (otherShape.length === 1) {
        otherShape.push(1);
        otherStrides.push(0);
        squeezeOutput = -1;
    }

    // Pad shapes to the same length by putting 1's in front
    const maxLength = Math.max(inputShape.length, otherShape.length);
    padFront(inputShape, inputStrides, maxLength);
    padFront(otherShape, otherStrides, maxLength);

    const outputShape: number[] = [];
    for (let dim = 0; dim < inputShape.length - 2; dim++) {
        if (inputShape[dim] === 1 || otherShape[dim] === 1) {
            outputShape[dim] = Math.max(inputShape[dim], otherShape[dim]);
            if (inputShape[dim] === 1) {
                inputStrides[dim] = 0;
            }
            if (otherShape[dim] === 1) {
                otherStrides[dim] = 0;
            }
        } else if (inputShape[dim] === otherShape[dim]) {
            outputShape[dim] = inputShape[dim];
        } else {
            throw new Error(
                "The size of tensor a (" +
                    inputShape[dim] +
                    ") must match the size of tensor b (" +
                    otherShape[dim] +
                    ") at non-singleton dimension " +
                    dim
            );
        }
    }
    outputShape.push(inputShape[inputShape.length - 2]);
    outputShape.push(otherShape[otherShape.length - 1]);
    if (squeezeOutput !== null) {
        outputShape.splice(squeezeOutput, 1);
    }

    return {
        output: contiguousStridedShape(outputShape),
        a: { shape: inputShape, strides: inputStrides },
        b: { shape: otherShape, strides: otherStrides },
    };
}

export function validateIdx(rank: number, idx: number): void {
    check(
        (idx >= 0 && idx < rank) || (idx === 0),
        () => `index ${idx} is out of bounds for dimension ${rank}`,
    );
}
export function validateDimLength(length: number): void {
    check(length >= 0, () => `dimension length ${length} must be non-negative`);
}

export function canonicalizeDim(rank: number, idx: number, wrapScalar: boolean = true): number {
    if (rank < 0) {
        throw new Error(`Rank must be non-negative, got ${rank}`);
    }
    if (rank === 0) {
        if (!wrapScalar) {
            throw new Error(`Dimension specified as ${idx} but tensor has no dimensions`);
        }
        rank = 1;
    }
    if (idx >0 && idx < rank) {
        return idx;
    }
    let _idx: number;
    if (idx < 0) {
        _idx = idx + rank;
    }
    else {
        _idx = idx;
    }
    if (_idx < 0 || _idx >= rank) {
        throw new Error(`Dimension out of range (expected to be in range of [${-rank}, ${rank}), but got ${idx})`);
    }
    return _idx;
}
