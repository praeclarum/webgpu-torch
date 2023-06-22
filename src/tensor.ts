import type { Device, Deviceish } from "./device";
import { getDevice } from "./devices";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { ones } from "./factories";
import { ATypedArray, Dtype, getDtype } from "./dtype";
import {
    TensorArrayData,
    UntypedStorage,
    newTypedArrayFromArray,
} from "./storage";
import { type GradientFunction, type GradientContext, isGradEnabled } from "./autograd";
import { shaderTypeToDtype, type KernelConfigInput, type KernelParamsInput } from "./kernel";
import * as ops from "./ops_opgen";
import * as aops from "./ops_artisanal";
import { TensorBase } from "./tensor_base";
import { GraphNode, SourceNode, ComputedNode, GraphNodeOutputRef, GraphNodeOutputSpec } from "./graph";

export type MemoryFormat = "contiguousFormat" | "preserveFormat";

export type TensorData = TensorArrayData | ATypedArray | UntypedStorage;

export type TensorSpec = {
    data: TensorData;
    dtype?: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
    shape?: Shape;
    strides?: Strides;
};

export class Tensor extends TensorBase {
    private _device: Device;

    // private _storage: UntypedStorage;
    private _dtype: Dtype;
    private _shape: Shape;
    private _strides: Strides;

    private _requiresGrad: boolean = false;
    private _gradFunc: GradientFunction | null;
    private _gradCtx: GradientContext | null;
    public grad: Tensor | null = null;

    private _node: GraphNodeOutputRef;

    get node(): GraphNodeOutputRef {
        return this._node;
    }
    get storage(): UntypedStorage {
        // return this._storage;
        return this.node.node.storages[this.node.outputIndex];
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get shape(): Shape {
        return this._shape;
    }
    get strides(): Strides {
        return this._strides;
    }
    get device(): Device {
        return this._device;
    }
    get isContiguous(): boolean {
        let strides = this.strides;
        let shape = this.shape;
        let offset = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== offset) {
                return false;
            }
            offset *= shape[i];
        }
        return true;
    }
    private get isScalar(): boolean {
        return (
            this.shape.length === 0 ||
            (this.shape.length === 1 && this.shape[0] === 1)
        );
    }

    get requiresGrad(): boolean {
        return this._requiresGrad;
    }
    set requiresGrad(value: boolean) {
        if (this._gradFunc) {
            throw new Error(
                "You can only change requiresGrad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use valueNoGrad = value.detach()."
            );
        }
        this._requiresGrad = value;
    }
    get gradFunc(): GradientFunction | null {
        return this._gradFunc;
    }

    constructor(node: GraphNodeOutputRef)
    constructor(spec: TensorSpec)
    constructor(
        array: TensorData,
        dtype?: Dtype,
        device?: Deviceish,
        requiresGrad?: boolean)
    constructor(
        arrayOrSpec: TensorData | TensorSpec | GraphNodeOutputRef,
        dtype?: Dtype,
        device?: Deviceish,
        requiresGrad?: boolean
    ) {
        super();
        let d = getDevice(device);
        let dt = getDtype(dtype);
        if (arrayOrSpec === null) {
            throw new Error("Cannot create tensor from null");
        } else if (arrayOrSpec instanceof Array) {
            const array = newTypedArrayFromArray(arrayOrSpec, dt, d);
            // this._storage = array.storage;
            this._dtype = dt;
            this._shape = array.shape;
            this._strides = array.strides;
            this._node = {
                node: new SourceNode(array.storage, this._dtype, this._shape, this._strides),
                outputIndex: 0
            };
        } else if (arrayOrSpec.hasOwnProperty("node") && (arrayOrSpec as any).node instanceof GraphNode) {
            const noder = arrayOrSpec as GraphNodeOutputRef;
            const spec = noder.node.outputs[noder.outputIndex];
            this._dtype = spec.dtype;
            this._shape = spec.shape;
            this._strides = spec.strides;
            this._node = noder;
        } else if (arrayOrSpec.hasOwnProperty("data")) {
            const jdata = arrayOrSpec as TensorSpec;
            d = jdata.device ? getDevice(jdata.device) : d;
            dt = jdata.dtype ? getDtype(jdata.dtype) : dt;
            requiresGrad = requiresGrad || jdata.requiresGrad;
            let storage: UntypedStorage;
            if (jdata.data instanceof Array) {
                const array = newTypedArrayFromArray(jdata.data, dt, d);
                storage = array.storage;
                this._dtype = dt;
                this._shape = array.shape;
                this._strides = array.strides;
            } else if (jdata.data instanceof UntypedStorage) {
                storage = jdata.data;
                this._dtype = dt;
                if (jdata.shape === undefined && jdata.strides === undefined) {
                    throw new Error("Cannot create tensor from storage without also specifying the shape and strides.");
                }
                this._shape = jdata.shape || [];
                this._strides = jdata.strides || defaultStrides(this._shape);
            } else {
                throw new Error("Cannot create tensor from json data " + jdata);
            }
            this._node = {
                node: new SourceNode(storage, this._dtype, this._shape, this._strides),
                outputIndex: 0
            };
        } else {
            throw new Error(
                "Invalid data type for Tensor constructor. Expected an array of values or a json object with a 'data' property."
            );
        }
        this._device = d;
        this._requiresGrad = requiresGrad || false;
        this._gradFunc = null;
        this._gradCtx = null;
        this.grad = null;
    }
    withShape(shape: Shape, strides: Strides): Tensor {
        if (shapeSize(shape) != shapeSize(this.shape)) {
            throw new Error(
                `Cannot reshape tensor of size ${this.shape} to ${shape}`
            );
        }
        return new Tensor({
            data: this.storage,
            dtype: this.dtype,
            device: this.device,
            requiresGrad: this.requiresGrad,
            shape,
            strides,
        });
    }

    get [Symbol.toStringTag]() {
        return "Tensor";
    }
    toString(): string {
        let rg = this.requiresGrad ? ", requiresGrad=true" : "";
        if (this._gradFunc) {
            rg = ", gradFunc";
        }
        return `tensor([${this.shape}], ${this.dtype}${rg})`;
    }
    async toArrayAsync(): Promise<TensorArrayData> {
        const data = await this.storage.toTypedArrayAsync(this.dtype);
        const shape = this.shape;
        const strides = this.strides;
    
        if (shape.length == 0 || (shape.length == 1 && shape[0] == 1)) {
            return [data[0]];
        }
    
        const index: number[] = [];
        return readArray(index);
    
        function readArray(index: number[]): TensorArrayData {
            const dim = index.length;
    
            if (dim == shape.length - 1) {
                const offset = calculateOffset(index);
                const length = shape[dim];
                const subarray = data.subarray(offset, offset + length);
                return Array.from(subarray);
            } else {
                const result: TensorArrayData = [];
                for (let i = 0; i < shape[dim]; i++) {
                    index.push(i);
                    result.push(readArray(index));
                    index.pop();
                }
                return result;
            }
        }
    
        function calculateOffset(index: number[]): number {
            let offset = 0;
            for (let i = 0; i < index.length; i++) {
                offset += index[i] * strides[i];
            }
            return offset;
        }
    }

    runKernelInplace(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        ...additionalInputs: Tensor[]
    ): Tensor {
        if (this.requiresGrad && isGradEnabled()) {
            throw new Error(`A tensor that requires a gradient cannot be used in an in-place operation`);
        }
        const kernel = this.device.getKernel(name, config);
        const inputBuffers = additionalInputs.map((t) => t.storage);
        kernel.run(inputBuffers, params, [this.storage]);
        return this;
    }
    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputs: Shape[] | Tensor[],
        ...additionalInputs: Tensor[]
    ): Tensor[] {
        if (outputs.length === 0) {
            throw new Error(`Cannot run kernel "${name}" without any outputs`);
        }
        const d = this.device;
        const kernel = d.getKernel(name, config);
        const outputsAreTensors = outputs[0] instanceof Tensor;
        const lazy = true;
        if (lazy) {
            const inputRefs = [this._node, ...additionalInputs.map((t) => t._node)];
            const outputSpecs: GraphNodeOutputSpec[] = [];
            for (let i = 0; i < outputs.length; i++) {
                const output = outputs[i];
                if (output instanceof Tensor) {
                    outputSpecs.push({
                        shape: output.shape,
                        dtype: output.dtype,
                        strides: output.strides,
                    });
                }
                else {
                    outputSpecs.push({
                        shape: output,
                        dtype: shaderTypeToDtype(kernel.spec.outputs[i].shaderType),
                        strides: defaultStrides(output),
                    });
                }
            }
            const node = new ComputedNode(kernel, inputRefs, params, outputSpecs);
            return outputs.map((output, i) => {
                const ref = {
                    node: node,
                    outputIndex: i,
                };
                return new Tensor(ref);
            });
        }
        else {
            const inputStorages = [
                this.storage,
                ...additionalInputs.map((t) =>t.storage),
            ];
            const providedOutputStorages = outputsAreTensors ? (outputs as Tensor[]).map((t) =>
                t.storage) : undefined;
            const outputStorages = kernel.run(inputStorages, params, providedOutputStorages);
            if (outputsAreTensors) {
                return outputs as Tensor[];
            }
            else {
                if (outputStorages.length !== outputs.length) {
                    throw new Error(
                        `Expected ${outputs.length} output buffers (given the provided outputShapes to runKernel), but got ${outputStorages.length} output buffers when running the kernel "${name}".`
                    );
                }
                return outputStorages.map(
                    (outputBuffer, i) => {
                        return new Tensor({
                            data: outputBuffer as UntypedStorage,
                            dtype: this.dtype,
                            shape: (outputs as Shape[])[i],
                            strides: defaultStrides((outputs as Shape[])[i]),
                            device: this.device,
                        });
                    }
                );
            }
        }
    }

    detach(): Tensor {
        if (this._requiresGrad || this._gradFunc) {
            return new Tensor({
                data: this.storage,
                dtype: this.dtype,
                requiresGrad: false,
                shape: this.shape,
                strides: this.strides,
                device: this.device,
            });
        }
        return this;
    }

    setGradientFunction(
        ctx: GradientContext,
        gradFunc: GradientFunction
    ): void {
        this._gradFunc = gradFunc;
        this._gradCtx = ctx;
        this._requiresGrad = true;
    }

    backward(gradient?: Tensor): void {
        if (!this.requiresGrad) {
            throw new Error(`Cannot call backward on a tensor that does not require gradients`);
        }
        let grad: Tensor;
        if (gradient) {
            grad = gradient;
        } else {
            if (!this.isScalar) {
                throw new Error(
                    `Gradient can only be implicitly created for scalar outputs`
                );
            }
            grad = ones(1);
        }
        if (this.grad) {
            this.grad.add_(grad);
        } else {
            this.grad = grad;
        }
        if (!this._gradFunc || !this._gradCtx) {
            return;
        }
        const grads = this._gradFunc(this._gradCtx, grad);
        const inputs = this._gradCtx.inputsWithGradient;
        for (let i = 0; i < inputs.length; i++) {
            const input = inputs[i];
            if (input === null) {
                continue;
            }
            const grad = grads[i];
            if (grad) {
                input.backward(grad);
            } else {
                throw new Error(
                    `Gradient function did not return a gradient for input #${i} (out of ${inputs.length}). ${grads.length} gradients were returned.`
                );
            }
        }
    }

    expand(shape: Shape): Tensor {
        const newShape = shape.slice();
        const newStrides = Array(newShape.length).fill(0);
        // Update newStrides based on the current strides
        // so that the expansion happens
        // in the correct direction
        let j = newShape.length - 1;
        let thisShape = this.shape;
        let thisStrides = this.strides;
        for (let i = thisShape.length - 1; i >= 0; i--) {
            if (thisShape[i] === 1) {
                newStrides[j] = 0;
            } else {
                newStrides[j] = thisStrides[i];
                j--;
            }
            if (newShape[j] === -1) {
                newShape[j] = thisShape[i];
            }
        }
        // console.log("EXPAND", this.shape, this.strides, shape, newShape, newStrides);
        return this.withShape(newShape, newStrides);
    }
    mm(other: Tensor): Tensor {
        return aops.mm(this, other);
    }
    t(): Tensor {
        return aops.t(this);
    }
    zero_(): Tensor {
        throw new Error("Tensor zero_ is not implemented");
    }

    // Codegen marker
    /**
    * ![Plot of abs and its gradient](../../plots/abs.svg)
    *
    * Calculates:
    * ```js
    * output = abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
    * ```
    *
    * @returns the output tensor
    */
    abs(output?: Tensor): Tensor {
        return ops.abs(this, output);
    }
    /**
    * Alias for `abs`.
    *
    * ![Plot of abs and its gradient](../../plots/abs.svg)
    *
    * Calculates:
    * ```js
    * output = abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
    * ```
    *
    * @returns the output tensor
    */
    absolute(output?: Tensor): Tensor {
        return ops.abs(this);
    }
    /**
    * ![Plot of abs and its gradient](../../plots/abs.svg)
    *
    * Calculates:
    * ```js
    * output = abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
    * ```
    *
    * @returns the output tensor
    */
    abs_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("abs_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of acos and its gradient](../../plots/acos.svg)
    *
    * Calculates:
    * ```js
    * output = acos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    acos(output?: Tensor): Tensor {
        return ops.acos(this, output);
    }
    /**
    * Alias for `acos`.
    *
    * ![Plot of acos and its gradient](../../plots/acos.svg)
    *
    * Calculates:
    * ```js
    * output = acos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    arccos(output?: Tensor): Tensor {
        return ops.acos(this);
    }
    /**
    * ![Plot of acos and its gradient](../../plots/acos.svg)
    *
    * Calculates:
    * ```js
    * output = acos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    acos_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acos_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of acosh and its gradient](../../plots/acosh.svg)
    *
    * Calculates:
    * ```js
    * output = acosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input - 1)
    * ```
    *
    * @returns the output tensor
    */
    acosh(output?: Tensor): Tensor {
        return ops.acosh(this, output);
    }
    /**
    * Alias for `acosh`.
    *
    * ![Plot of acosh and its gradient](../../plots/acosh.svg)
    *
    * Calculates:
    * ```js
    * output = acosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input - 1)
    * ```
    *
    * @returns the output tensor
    */
    arccosh(output?: Tensor): Tensor {
        return ops.acosh(this);
    }
    /**
    * ![Plot of acosh and its gradient](../../plots/acosh.svg)
    *
    * Calculates:
    * ```js
    * output = acosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input - 1)
    * ```
    *
    * @returns the output tensor
    */
    acosh_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acosh_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input + other * alpha
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    add(other: number | Tensor, alpha?: number): Tensor {
        return ops.add(this, other, alpha);
    }
    /**
    * Calculates:
    * ```js
    * output = input + other * alpha
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    add_(other: number | Tensor, alpha?: number): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
                alpha: alpha || 1.0,
            };
            return this.runKernelInplace("add_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
                alpha: alpha || 1.0,
            };
            return this.runKernelInplace("add_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of asin and its gradient](../../plots/asin.svg)
    *
    * Calculates:
    * ```js
    * output = asin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    asin(output?: Tensor): Tensor {
        return ops.asin(this, output);
    }
    /**
    * Alias for `asin`.
    *
    * ![Plot of asin and its gradient](../../plots/asin.svg)
    *
    * Calculates:
    * ```js
    * output = asin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    arcsin(output?: Tensor): Tensor {
        return ops.asin(this);
    }
    /**
    * ![Plot of asin and its gradient](../../plots/asin.svg)
    *
    * Calculates:
    * ```js
    * output = asin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    asin_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asin_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of asinh and its gradient](../../plots/asinh.svg)
    *
    * Calculates:
    * ```js
    * output = asinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    asinh(output?: Tensor): Tensor {
        return ops.asinh(this, output);
    }
    /**
    * Alias for `asinh`.
    *
    * ![Plot of asinh and its gradient](../../plots/asinh.svg)
    *
    * Calculates:
    * ```js
    * output = asinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    arcsinh(output?: Tensor): Tensor {
        return ops.asinh(this);
    }
    /**
    * ![Plot of asinh and its gradient](../../plots/asinh.svg)
    *
    * Calculates:
    * ```js
    * output = asinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    asinh_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asinh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of atan and its gradient](../../plots/atan.svg)
    *
    * Calculates:
    * ```js
    * output = atan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    atan(output?: Tensor): Tensor {
        return ops.atan(this, output);
    }
    /**
    * Alias for `atan`.
    *
    * ![Plot of atan and its gradient](../../plots/atan.svg)
    *
    * Calculates:
    * ```js
    * output = atan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    arctan(output?: Tensor): Tensor {
        return ops.atan(this);
    }
    /**
    * ![Plot of atan and its gradient](../../plots/atan.svg)
    *
    * Calculates:
    * ```js
    * output = atan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    atan_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = atan2(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    atan2(other: number | Tensor): Tensor {
        return ops.atan2(this, other);
    }
    /**
    * Alias for `atan2`.
    *
    * Calculates:
    * ```js
    * output = atan2(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    arctan2(other: number | Tensor): Tensor {
        return ops.atan2(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = atan2(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    atan2_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("atan2_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("atan2_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of ceil and its gradient](../../plots/ceil.svg)
    *
    * Calculates:
    * ```js
    * output = ceil(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    ceil(output?: Tensor): Tensor {
        return ops.ceil(this, output);
    }
    /**
    * ![Plot of ceil and its gradient](../../plots/ceil.svg)
    *
    * Calculates:
    * ```js
    * output = ceil(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    ceil_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ceil_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = other >= 0 ? abs(input) : -abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * var dir = other >= 0 ? (input >= 0 ? 1.0 : -1.0) : (input >= 0 ? -1.0 : 1.0); inputGrad = input == 0.0 ? 0.0 : outputGrad * dir; otherGrad = 0
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    copysign(other: number | Tensor): Tensor {
        return ops.copysign(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = other >= 0 ? abs(input) : -abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * var dir = other >= 0 ? (input >= 0 ? 1.0 : -1.0) : (input >= 0 ? -1.0 : 1.0); inputGrad = input == 0.0 ? 0.0 : outputGrad * dir; otherGrad = 0
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    copysign_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("copysign_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("copysign_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of cos and its gradient](../../plots/cos.svg)
    *
    * Calculates:
    * ```js
    * output = cos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad * sin(input)
    * ```
    *
    * @returns the output tensor
    */
    cos(output?: Tensor): Tensor {
        return ops.cos(this, output);
    }
    /**
    * ![Plot of cos and its gradient](../../plots/cos.svg)
    *
    * Calculates:
    * ```js
    * output = cos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad * sin(input)
    * ```
    *
    * @returns the output tensor
    */
    cos_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cos_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of cosh and its gradient](../../plots/cosh.svg)
    *
    * Calculates:
    * ```js
    * output = cosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * sinh(input)
    * ```
    *
    * @returns the output tensor
    */
    cosh(output?: Tensor): Tensor {
        return ops.cosh(this, output);
    }
    /**
    * ![Plot of cosh and its gradient](../../plots/cosh.svg)
    *
    * Calculates:
    * ```js
    * output = cosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * sinh(input)
    * ```
    *
    * @returns the output tensor
    */
    cosh_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cosh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of deg2rad and its gradient](../../plots/deg2rad.svg)
    *
    * Calculates:
    * ```js
    * output = input * 0.017453292519943295
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 0.017453292519943295
    * ```
    *
    * @returns the output tensor
    */
    deg2rad(output?: Tensor): Tensor {
        return ops.deg2rad(this, output);
    }
    /**
    * ![Plot of deg2rad and its gradient](../../plots/deg2rad.svg)
    *
    * Calculates:
    * ```js
    * output = input * 0.017453292519943295
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 0.017453292519943295
    * ```
    *
    * @returns the output tensor
    */
    deg2rad_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("deg2rad_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input / other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    div(other: number | Tensor): Tensor {
        return ops.div(this, other);
    }
    /**
    * Alias for `div`.
    *
    * Calculates:
    * ```js
    * output = input / other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    divide(other: number | Tensor): Tensor {
        return ops.div(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input / other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    div_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("div_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("div_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of exp and its gradient](../../plots/exp.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    exp(output?: Tensor): Tensor {
        return ops.exp(this, output);
    }
    /**
    * ![Plot of exp and its gradient](../../plots/exp.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    exp_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of exp2 and its gradient](../../plots/exp2.svg)
    *
    * Calculates:
    * ```js
    * output = exp2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp2(input) * 0.6931471805599453
    * ```
    *
    * @returns the output tensor
    */
    exp2(output?: Tensor): Tensor {
        return ops.exp2(this, output);
    }
    /**
    * ![Plot of exp2 and its gradient](../../plots/exp2.svg)
    *
    * Calculates:
    * ```js
    * output = exp2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp2(input) * 0.6931471805599453
    * ```
    *
    * @returns the output tensor
    */
    exp2_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp2_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of expm1 and its gradient](../../plots/expm1.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input) - 1.0
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    expm1(output?: Tensor): Tensor {
        return ops.expm1(this, output);
    }
    /**
    * ![Plot of expm1 and its gradient](../../plots/expm1.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input) - 1.0
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    expm1_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("expm1_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of floor and its gradient](../../plots/floor.svg)
    *
    * Calculates:
    * ```js
    * output = floor(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    floor(output?: Tensor): Tensor {
        return ops.floor(this, output);
    }
    /**
    * ![Plot of floor and its gradient](../../plots/floor.svg)
    *
    * Calculates:
    * ```js
    * output = floor(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    floor_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of frac and its gradient](../../plots/frac.svg)
    *
    * Calculates:
    * ```js
    * output = input >= 0.0 ? fract(input) : -fract(-input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    frac(output?: Tensor): Tensor {
        return ops.frac(this, output);
    }
    /**
    * ![Plot of frac and its gradient](../../plots/frac.svg)
    *
    * Calculates:
    * ```js
    * output = input >= 0.0 ? fract(input) : -fract(-input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    frac_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("frac_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = sqrt(input * input + other * other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * input / sqrt(input * input + other * other); otherGrad = outputGrad * other / sqrt(input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    hypot(other: number | Tensor): Tensor {
        return ops.hypot(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = sqrt(input * input + other * other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * input / sqrt(input * input + other * other); otherGrad = outputGrad * other / sqrt(input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    hypot_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("hypot_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("hypot_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * Calculates:
    * ```js
    * output = input * pow(2.0, other)
    * ```
    *
    * Gradient:
    * ```js
    * var out = pow(2.0, other); inputGrad = outputGrad * out; otherGrad = outputGrad * input * out * 0.6931471805599453
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    ldexp(other: number | Tensor): Tensor {
        return ops.ldexp(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input * pow(2.0, other)
    * ```
    *
    * Gradient:
    * ```js
    * var out = pow(2.0, other); inputGrad = outputGrad * out; otherGrad = outputGrad * input * out * 0.6931471805599453
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    ldexp_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("ldexp_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("ldexp_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of log and its gradient](../../plots/log.svg)
    *
    * Calculates:
    * ```js
    * output = log(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / input
    * ```
    *
    * @returns the output tensor
    */
    log(output?: Tensor): Tensor {
        return ops.log(this, output);
    }
    /**
    * ![Plot of log and its gradient](../../plots/log.svg)
    *
    * Calculates:
    * ```js
    * output = log(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / input
    * ```
    *
    * @returns the output tensor
    */
    log_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of log10 and its gradient](../../plots/log10.svg)
    *
    * Calculates:
    * ```js
    * output = log(input) * 0.4342944819032518
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 2.302585092994046)
    * ```
    *
    * @returns the output tensor
    */
    log10(output?: Tensor): Tensor {
        return ops.log10(this, output);
    }
    /**
    * ![Plot of log10 and its gradient](../../plots/log10.svg)
    *
    * Calculates:
    * ```js
    * output = log(input) * 0.4342944819032518
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 2.302585092994046)
    * ```
    *
    * @returns the output tensor
    */
    log10_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log10_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of log1p and its gradient](../../plots/log1p.svg)
    *
    * Calculates:
    * ```js
    * output = log(input + 1.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input + 1.0)
    * ```
    *
    * @returns the output tensor
    */
    log1p(output?: Tensor): Tensor {
        return ops.log1p(this, output);
    }
    /**
    * ![Plot of log1p and its gradient](../../plots/log1p.svg)
    *
    * Calculates:
    * ```js
    * output = log(input + 1.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input + 1.0)
    * ```
    *
    * @returns the output tensor
    */
    log1p_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log1p_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of log2 and its gradient](../../plots/log2.svg)
    *
    * Calculates:
    * ```js
    * output = log2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 0.6931471805599453)
    * ```
    *
    * @returns the output tensor
    */
    log2(output?: Tensor): Tensor {
        return ops.log2(this, output);
    }
    /**
    * ![Plot of log2 and its gradient](../../plots/log2.svg)
    *
    * Calculates:
    * ```js
    * output = log2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 0.6931471805599453)
    * ```
    *
    * @returns the output tensor
    */
    log2_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log2_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = log(exp(input) + exp(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp(input); var eoth = exp(other); var addeinv = outputGrad/(ein + eoth); inputGrad = addeinv * ein; otherGrad = addeinv * eoth
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp(other: number | Tensor): Tensor {
        return ops.logaddexp(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = log(exp(input) + exp(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp(input); var eoth = exp(other); var addeinv = outputGrad/(ein + eoth); inputGrad = addeinv * ein; otherGrad = addeinv * eoth
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("logaddexp_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("logaddexp_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * Calculates:
    * ```js
    * output = log2(exp2(input) + exp2(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp2(input); var eoth = exp2(other); var sum_ein_eoth = ein + eoth; inputGrad = outputGrad * (ein / sum_ein_eoth); otherGrad = outputGrad * (eoth / sum_ein_eoth );
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp2(other: number | Tensor): Tensor {
        return ops.logaddexp2(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = log2(exp2(input) + exp2(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp2(input); var eoth = exp2(other); var sum_ein_eoth = ein + eoth; inputGrad = outputGrad * (ein / sum_ein_eoth); otherGrad = outputGrad * (eoth / sum_ein_eoth );
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp2_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("logaddexp2_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("logaddexp2_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * Calculates:
    * ```js
    * output = input * other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other; otherGrad = outputGrad * input
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    mul(other: number | Tensor): Tensor {
        return ops.mul(this, other);
    }
    /**
    * Alias for `mul`.
    *
    * Calculates:
    * ```js
    * output = input * other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other; otherGrad = outputGrad * input
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    multiply(other: number | Tensor): Tensor {
        return ops.mul(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input * other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other; otherGrad = outputGrad * input
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    mul_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("mul_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("mul_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of neg and its gradient](../../plots/neg.svg)
    *
    * Calculates:
    * ```js
    * output = -input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad
    * ```
    *
    * @returns the output tensor
    */
    neg(output?: Tensor): Tensor {
        return ops.neg(this, output);
    }
    /**
    * Alias for `neg`.
    *
    * ![Plot of neg and its gradient](../../plots/neg.svg)
    *
    * Calculates:
    * ```js
    * output = -input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad
    * ```
    *
    * @returns the output tensor
    */
    negative(output?: Tensor): Tensor {
        return ops.neg(this);
    }
    /**
    * ![Plot of neg and its gradient](../../plots/neg.svg)
    *
    * Calculates:
    * ```js
    * output = -input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad
    * ```
    *
    * @returns the output tensor
    */
    neg_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("neg_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of positive and its gradient](../../plots/positive.svg)
    *
    * Calculates:
    * ```js
    * output = input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    positive(output?: Tensor): Tensor {
        return ops.positive(this, output);
    }
    /**
    * ![Plot of positive and its gradient](../../plots/positive.svg)
    *
    * Calculates:
    * ```js
    * output = input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    positive_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("positive_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = pow(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other * pow(input, other - 1.0); otherGrad = outputGrad * pow(input, other) * log(input)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    pow(other: number | Tensor): Tensor {
        return ops.pow(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = pow(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other * pow(input, other - 1.0); otherGrad = outputGrad * pow(input, other) * log(input)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    pow_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("pow_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("pow_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of rad2deg and its gradient](../../plots/rad2deg.svg)
    *
    * Calculates:
    * ```js
    * output = input * 57.29577951308232
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 57.29577951308232
    * ```
    *
    * @returns the output tensor
    */
    rad2deg(output?: Tensor): Tensor {
        return ops.rad2deg(this, output);
    }
    /**
    * ![Plot of rad2deg and its gradient](../../plots/rad2deg.svg)
    *
    * Calculates:
    * ```js
    * output = input * 57.29577951308232
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 57.29577951308232
    * ```
    *
    * @returns the output tensor
    */
    rad2deg_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rad2deg_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of reciprocal and its gradient](../../plots/reciprocal.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (input * input)
    * ```
    *
    * @returns the output tensor
    */
    reciprocal(output?: Tensor): Tensor {
        return ops.reciprocal(this, output);
    }
    /**
    * ![Plot of reciprocal and its gradient](../../plots/reciprocal.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (input * input)
    * ```
    *
    * @returns the output tensor
    */
    reciprocal_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("reciprocal_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of relu and its gradient](../../plots/relu.svg)
    *
    * Calculates:
    * ```js
    * output = max(input, 0.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input > 0.0 ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    relu(output?: Tensor): Tensor {
        return ops.relu(this, output);
    }
    /**
    * ![Plot of relu and its gradient](../../plots/relu.svg)
    *
    * Calculates:
    * ```js
    * output = max(input, 0.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input > 0.0 ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    relu_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("relu_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of round and its gradient](../../plots/round.svg)
    *
    * Calculates:
    * ```js
    * output = round(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    round(output?: Tensor): Tensor {
        return ops.round(this, output);
    }
    /**
    * ![Plot of round and its gradient](../../plots/round.svg)
    *
    * Calculates:
    * ```js
    * output = round(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    round_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("round_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of rsqrt and its gradient](../../plots/rsqrt.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (2.0 * sqrt(input) * input)
    * ```
    *
    * @returns the output tensor
    */
    rsqrt(output?: Tensor): Tensor {
        return ops.rsqrt(this, output);
    }
    /**
    * ![Plot of rsqrt and its gradient](../../plots/rsqrt.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (2.0 * sqrt(input) * input)
    * ```
    *
    * @returns the output tensor
    */
    rsqrt_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rsqrt_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sigmoid and its gradient](../../plots/sigmoid.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)
    * ```
    *
    * @returns the output tensor
    */
    sigmoid(output?: Tensor): Tensor {
        return ops.sigmoid(this, output);
    }
    /**
    * ![Plot of sigmoid and its gradient](../../plots/sigmoid.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)
    * ```
    *
    * @returns the output tensor
    */
    sigmoid_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sigmoid_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sign and its gradient](../../plots/sign.svg)
    *
    * Calculates:
    * ```js
    * output = sign(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    sign(output?: Tensor): Tensor {
        return ops.sign(this, output);
    }
    /**
    * ![Plot of sign and its gradient](../../plots/sign.svg)
    *
    * Calculates:
    * ```js
    * output = sign(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    sign_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sign_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of silu and its gradient](../../plots/silu.svg)
    *
    * Calculates:
    * ```js
    * output = input / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
    * ```
    *
    * @returns the output tensor
    */
    silu(output?: Tensor): Tensor {
        return ops.silu(this, output);
    }
    /**
    * ![Plot of silu and its gradient](../../plots/silu.svg)
    *
    * Calculates:
    * ```js
    * output = input / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
    * ```
    *
    * @returns the output tensor
    */
    silu_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("silu_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sin and its gradient](../../plots/sin.svg)
    *
    * Calculates:
    * ```js
    * output = sin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cos(input)
    * ```
    *
    * @returns the output tensor
    */
    sin(output?: Tensor): Tensor {
        return ops.sin(this, output);
    }
    /**
    * ![Plot of sin and its gradient](../../plots/sin.svg)
    *
    * Calculates:
    * ```js
    * output = sin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cos(input)
    * ```
    *
    * @returns the output tensor
    */
    sin_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sin_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sinc and its gradient](../../plots/sinc.svg)
    *
    * Calculates:
    * ```js
    * var inpi = input * 3.141592653589793; output = input == 0.0 ? 1.0 : sin(inpi) / inpi
    * ```
    *
    * Gradient:
    * ```js
    * var inpi = input * 3.141592653589793; inputGrad = input == 0.0 ? 0.0 : (outputGrad * 3.141592653589793 * (inpi*cos(inpi) - sin(inpi)) / (inpi*inpi))
    * ```
    *
    * @returns the output tensor
    */
    sinc(output?: Tensor): Tensor {
        return ops.sinc(this, output);
    }
    /**
    * ![Plot of sinc and its gradient](../../plots/sinc.svg)
    *
    * Calculates:
    * ```js
    * var inpi = input * 3.141592653589793; output = input == 0.0 ? 1.0 : sin(inpi) / inpi
    * ```
    *
    * Gradient:
    * ```js
    * var inpi = input * 3.141592653589793; inputGrad = input == 0.0 ? 0.0 : (outputGrad * 3.141592653589793 * (inpi*cos(inpi) - sin(inpi)) / (inpi*inpi))
    * ```
    *
    * @returns the output tensor
    */
    sinc_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinc_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sinh and its gradient](../../plots/sinh.svg)
    *
    * Calculates:
    * ```js
    * output = sinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cosh(input)
    * ```
    *
    * @returns the output tensor
    */
    sinh(output?: Tensor): Tensor {
        return ops.sinh(this, output);
    }
    /**
    * ![Plot of sinh and its gradient](../../plots/sinh.svg)
    *
    * Calculates:
    * ```js
    * output = sinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cosh(input)
    * ```
    *
    * @returns the output tensor
    */
    sinh_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sqrt and its gradient](../../plots/sqrt.svg)
    *
    * Calculates:
    * ```js
    * output = sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (2.0 * sqrt(input))
    * ```
    *
    * @returns the output tensor
    */
    sqrt(output?: Tensor): Tensor {
        return ops.sqrt(this, output);
    }
    /**
    * ![Plot of sqrt and its gradient](../../plots/sqrt.svg)
    *
    * Calculates:
    * ```js
    * output = sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (2.0 * sqrt(input))
    * ```
    *
    * @returns the output tensor
    */
    sqrt_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sqrt_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of square and its gradient](../../plots/square.svg)
    *
    * Calculates:
    * ```js
    * output = input * input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 2.0 * input
    * ```
    *
    * @returns the output tensor
    */
    square(output?: Tensor): Tensor {
        return ops.square(this, output);
    }
    /**
    * ![Plot of square and its gradient](../../plots/square.svg)
    *
    * Calculates:
    * ```js
    * output = input * input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 2.0 * input
    * ```
    *
    * @returns the output tensor
    */
    square_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("square_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input - other * alpha
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = -outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    sub(other: number | Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    /**
    * Alias for `sub`.
    *
    * Calculates:
    * ```js
    * output = input - other * alpha
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = -outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    subtract(other: number | Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    /**
    * Calculates:
    * ```js
    * output = input - other * alpha
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = -outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    sub_(other: number | Tensor, alpha?: number): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
                alpha: alpha || 1.0,
            };
            return this.runKernelInplace("sub_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
                alpha: alpha || 1.0,
            };
            return this.runKernelInplace("sub_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * ![Plot of tan and its gradient](../../plots/tan.svg)
    *
    * Calculates:
    * ```js
    * output = tan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (cos(input) * cos(input))
    * ```
    *
    * @returns the output tensor
    */
    tan(output?: Tensor): Tensor {
        return ops.tan(this, output);
    }
    /**
    * ![Plot of tan and its gradient](../../plots/tan.svg)
    *
    * Calculates:
    * ```js
    * output = tan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (cos(input) * cos(input))
    * ```
    *
    * @returns the output tensor
    */
    tan_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tan_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of tanh and its gradient](../../plots/tanh.svg)
    *
    * Calculates:
    * ```js
    * output = tanh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))
    * ```
    *
    * @returns the output tensor
    */
    tanh(output?: Tensor): Tensor {
        return ops.tanh(this, output);
    }
    /**
    * ![Plot of tanh and its gradient](../../plots/tanh.svg)
    *
    * Calculates:
    * ```js
    * output = tanh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))
    * ```
    *
    * @returns the output tensor
    */
    tanh_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tanh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of trunc and its gradient](../../plots/trunc.svg)
    *
    * Calculates:
    * ```js
    * output = trunc(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    trunc(output?: Tensor): Tensor {
        return ops.trunc(this, output);
    }
    /**
    * Alias for `trunc`.
    *
    * ![Plot of trunc and its gradient](../../plots/trunc.svg)
    *
    * Calculates:
    * ```js
    * output = trunc(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    fix(output?: Tensor): Tensor {
        return ops.trunc(this);
    }
    /**
    * ![Plot of trunc and its gradient](../../plots/trunc.svg)
    *
    * Calculates:
    * ```js
    * output = trunc(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    trunc_(output?: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("trunc_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input == 0.0 ? 0.0 : input * log(other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other); otherGrad = input == 0.0 ? 0.0 : outputGrad * (input / other);
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    xlogy(other: number | Tensor): Tensor {
        return ops.xlogy(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input == 0.0 ? 0.0 : input * log(other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other); otherGrad = input == 0.0 ? 0.0 : outputGrad * (input / other);
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    xlogy_(other: number | Tensor): Tensor {
        if (typeof other === "number") {
            const params = {
                size: shapeSize(this.shape),
                other: other,
            };
            return this.runKernelInplace("xlogy_scalar_", { dtype: this.dtype }, params);
        } else {
            const params = {
                size: shapeSize(this.shape),
            };
            return this.runKernelInplace("xlogy_", { dtype: this.dtype }, params, other);
        }
    }
    /**
    * Calculates:
    * ```js
    * output = output && input
    * ```
    *
    * with an initial value of `output = 1`.
    *
    * Gradient:
    * ```js
    * inputGrad = output ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    all(dim?: number, keepdim?: boolean): Tensor {
        return ops.all(this, dim, keepdim);
    }
    /**
    * Calculates:
    * ```js
    * output = output || input
    * ```
    *
    * with an initial value of `output = 0`.
    *
    * Gradient:
    * ```js
    * inputGrad = output ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    any(dim?: number, keepdim?: boolean): Tensor {
        return ops.any(this, dim, keepdim);
    }
    /**
    * Calculates:
    * ```js
    * output = output + input
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / f32(inputSize)
    * ```
    *
    * @returns the output tensor
    */
    mean(dim?: number, keepdim?: boolean): Tensor {
        return ops.mean(this, dim, keepdim);
    }
    /**
    * Calculates:
    * ```js
    * output = output + input * input
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * input / output
    * ```
    *
    * @returns the output tensor
    */
    norm(dim?: number, keepdim?: boolean): Tensor {
        return ops.norm(this, dim, keepdim);
    }
    /**
    * Calculates:
    * ```js
    * output = output * input
    * ```
    *
    * with an initial value of `output = 1.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * output / input
    * ```
    *
    * @returns the output tensor
    */
    prod(dim?: number, keepdim?: boolean): Tensor {
        return ops.prod(this, dim, keepdim);
    }
    /**
    * Calculates:
    * ```js
    * output = output + input
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    sum(dim?: number, keepdim?: boolean): Tensor {
        return ops.sum(this, dim, keepdim);
    }
    /**
    * Calculates:
    * ```js
    * output = output + (input != 0)
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * @returns the output tensor
    */
    countNonzero(dim?: number, keepdim?: boolean): Tensor {
        return ops.countNonzero(this, dim, keepdim);
    }
    // End codegen marker
}
