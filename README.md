# webgpu-torch

Tensor computation and autograd with WebGPU acceleration inspired by pytorch.

[![npm version](https://badge.fury.io/js/webgpu-torch.svg)](https://www.npmjs.com/package/webgpu-torch)

Homepage: https://praeclarum.org/webgpu-torch

## Installation

Webgpu-torch has no dependencies so you can just include it in your web page.

```html
<script src="https://cdn.jsdelivr.net/npm/webgpu-torch@latest/torch.js"></script>
```

You can also include it using npm:

```bash
npm i webgpu-torch
```

## Usage

If you want to use WebGPU tensors, you must first initialize the library with:
    
```js
if (!await torch.initWebGPUAsync()) {
    console.warn(`WebGPU is not supported.`);
}
```

It's an async function and will really do want to await it before doing anything else.
All it does is query the `navigator` object for a valid `GPUDevice`.
Sadly, that's an async operation.

### GPU Accelerated Tensors

```js
// Create a tensor
const a = torch.tensor([[1, 2, 3], [4, 5, 6]]);

// Create another tensor
const b = torch.tensor([[7, 8, 9], [10, 11, 12]]);

// Add them
const c = a.add(b);
```

Tensors use WebGPU memory (`GPUBuffers`) to store their data.
When we want to read values from the tensor we have to *map* it to the CPU address space.
This can be accomplished at a low level with `await a.storage.mapReadAsync()` or at a high level with `await a.toArrayAsync()`. Most functions in this library present a synchronous interface, but they are all asynchronous under the hood. Mapping the data to the CPU address space is the only visibly asynchronous operation in the library.

```js
const floatArray = await c.toArrayAsync();
console.log(floatArray);
```


### Autograd Support

Math is fun, but it's even more fun when you do it backwards.

```js
// Create a tensor
const a = torch.tensor({data: [[1, 2, 3], [4, 5, 6]], requiresGrad: true});

// Create another tensor
const b = torch.tensor({data: [[7, 8, 9], [10, 11, 12]], requiresGrad: true});

// Add them
const c = a.add(b);

// Differentiate them
c.backward();
```

After this code executes, there will be gradient tensor values in `a.grad`, `b.grad`, and `c.grad`.


## API

Although this library was inspired by pytorch, it is not a clone and was written from scratch.
Its API surface is therefore not 100% compatible with pytorch, but I prioritize making it as similar as possible.

### Fundamental Types

* `Device` is an abstraction over CPUs and GPUs allowing you to specify where tensors are allocated and executed.
* `Dtype` is the data type of tensors and are specified as strings. Currently only `"float32"` is supported.
* `Shape` is an array of integers that specifies the size of each dimension of a tensor. For example, `[32, 3, 128, 256]` would be 32 batched 256x128 RGB images.
* `Tensor` is a multi-dimensional array of data. It has a `device`, a `dtype`, a `shape`, and `storage` properties. It can be created in a variety of ways.
    * **Directly** with `torch.tensor(array)` or `new torch.Tensor(array)`
    * **From factory functions** like `torch.zeros(shape)` or `torch.ones(shape)`
    * **From operations** like `a.add(b)` or `a.mm(b)`
    * **From a gradient calculation** like `a.add(b).backward()`
* `AutoFunction` is the base class for all autograd functions. It has a `forward` method that computes the output tensor and a `backward` method that computes the gradients of the inputs. They live in the `torch.functions` object. Functions should be called using their `apply` method.
* `Kernel` is basic operation that can be executed on the GPU.

### Tensor Operations

You have your basic unary operations like `abs` that can be called from a global function or on the tensor directly:

```js
const a = torch.tensor([[-1, 2, -3], [4, -5, 6]]);
const abs = torch.abs(a);
const abs2 = a.abs();
```

Your binary operations like `add` can be called in the same way:

```js
const b = torch.tensor([[7, -8, 9], [-10, 11, -12]]);
const sum = torch.add(a, b);
const sum2 = a.add(b);
```

I'm working on documenting the full list. For now, checkout the file [op_table.ts](src/op_table.ts) for a list of most of the operations.


## TODO

Here are the big components of the library:

- [x] GPU Tensors
- [x] GPU Kernels
- [x] Autograd Functions
- [ ] Datatypes beyond float32
- [ ] Optimizers (SGD and Adam)
- [x] Modules
- [ ] Save and restore (ONNX, safetensors)

In terms of supported operations, there's still a bit of work to be done:

- [x] Basic math
- [x] Convolution
- [ ] Indexing
- [ ] Broadcasting
- [x] Reductions
- [ ] Imaging


## Acknowledgements

I want to thank the Torch7 Lua environment for getting me into neural networks.

I want to thank the pytorch team for inspiring me.

I want to thank the webgpu teams at all the browser vendors for making this possible.
