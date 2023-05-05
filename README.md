# webgpu-torch

Tensor computation and autograd with WebGPU acceleration inspired by pytorch.

[![npm version](https://badge.fury.io/js/webgpu-torch.svg)](https://www.npmjs.com/package/webgpu-torch)

Site: https://praeclarum.org/webgpu-torch/

## Installation

Webgpu-torch has no dependencies so you can just include it in your web page.

```html
<script src="scripts/torch.js"></script>
```

You can also include it using npm:

```bash
npm i webgpu-torch
```

## Usage

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
const a = torch.tensor([[1, 2, 3], [4, 5, 6]]);

// Create another tensor
const b = torch.tensor([[7, 8, 9], [10, 11, 12]]);

// Add them
const c = a.add(b);

// Differentiate them
c.backward();
```

After this code executes, there will be gradient tensor values in `a.grad`, `b.grad`, and `c.grad`.


## TODO

- [x] Tensors
- [x] Autograd
- [ ] Optimizers
- [ ] Convolution
- [ ] Indexing
- [ ] Broadcasting
- [ ] Reductions
- [ ] Imaging


## Acknowledgements

I want to thank the Torch7 Lua environment for getting me into neural networks.

I want to thank the pytorch team for inspiring me.

I want to thank the webgpu teams at all the browser vendors for making this possible.
