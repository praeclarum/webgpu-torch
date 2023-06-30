import { Tensor } from "./tensor";

export type StateDict = { [key: string]: Tensor };

/**
 * Base class for implementing a module, which is a reusable portion of a neural network.
 * Modules can contain other modules, allowing for a tree-like structure.
 * 
 * To add a submodule to a module, assign it as a property of the module in the constructor.
 * Its name will be the name of the property.
 * 
 * Modules can also contain `Parameter` objects, which are tensors that are automatically
 * updated by optimizers and are saved when calling `saveDict()`.
 * To add a parameter to a module, assign it as a property of the module in the constructor.
 * 
 * Module subtypes implement a `forward` function in order to define the computation of the module.
 * 
 * ```
 * class Model extends torch.nn.Module {
 *     conv1: torch.nn.Conv2d;
 *     conv2: torch.nn.Conv2d;
 *     constructor() {
 *         super();
 *         this.conv1 = torch.nn.Conv2d(1, 20, 5);
 *         this.conv2 = torch.nn.Conv2d(20, 20, 5);
 *     }
 *     forward(input: torch.Tensor): torch.Tensor {
 *         let output = this.conv1.forward(input);
 *         output = output.relu();
 *         output = this.conv2.forward(output);
 *         output = output.relu();
 *         return output;
 *     }
 * }
 * ```
 */
export class Module {
    private _children: [string|number, Module][] | null = null;
    /**
     * Returns the immediate children (submodules) of this module along with their names.
     */
    get namedChildren(): [string|number, Module][] {
        if (this._children === null) {
            this._children = [];
            for (const key in this) {
                const value = (this as any)[key];
                if (value instanceof Module) {
                    this._children.push([key, value]);
                }
            }
        }
        return this._children;
    }
    /**
     * Returns the immediate children (submodules) of this module.
     */
    get children(): Module[] {
        return this.namedChildren.map(([_, value]) => value);
    }

    private _parameters: [string, Parameter][] | null = null;
    private get immediateParameters(): [string, Parameter][] {
        if (this._parameters === null) {
            this._parameters = [];
            for (const key in this) {
                const value = (this as any)[key];
                if (value instanceof Parameter) {
                    this._parameters.push([key, value]);
                }
            }
        }
        return this._parameters;
    }

    private _buffers: [string, Tensor | null][] = [];
    private _nonPersistentBuffersSet: Set<string> = new Set<string>();

    private _training: boolean = true;
    get training(): boolean {
        return this._training;
    }

    get [Symbol.toStringTag]() {
        return "Module";
    }

    /**
     * Produces a multiline readable string of the module and its descendants.
     * @returns A string representation of the module and its descendants.
     */
    toString(): string {
        const childLines: string[] = [];
        for (const [name, module] of this.namedChildren) {
            const modLines = module.toString().split("\n");
            const modStr =
                modLines.length === 1
                    ? modLines[0]
                    : modLines[0] +
                      "\n" +
                      modLines
                          .splice(1)
                          .map((line) => `  ${line}`)
                          .join("\n");
            childLines.push(`(${name}): ${modStr}`);
        }
        const lines = childLines;
        let mainStr = this.constructor.name + "(";
        if (lines.length > 0) {
            mainStr += `\n  ${lines.join("\n  ")}\n`;
        }
        mainStr += ")";
        return mainStr;
    }

    /**
     * Adds a child module of the current module.
     * 
     * The module can be accessed as a property using the given name.
     * @param name name of the child module
     * @param module child module to be added
     */
    addModule(name: string|number, module: Module): void {
        if ((this as any)[name] !== undefined) {
            throw new Error(`Module already has a child named ${name}`);
        }
        if (typeof name === "string" && name.indexOf(".") !== -1) {
            throw new Error(`Module name cannot contain "."`);
        }
        (this as any)[name] = module;
        this._children = null;
    }
    /**
     * Returns this module and its descendants' along with their prefixed names.
     * @param memo is a set of modules used to avoid double counting.
     * @param prefix is prependended to the names of the modules.
     * @param removeDuplicate is a boolean indicating whether to remove duplicate modules.
     * @returns a generator of [prefixed name, module] pairs.
     */
    *namedModules(
        memo?: Set<Module>,
        prefix: string = "",
        removeDuplicate: boolean = true
    ): Generator<[string|number, Module]> {
        memo = memo || new Set<Module>();
        if (!memo.has(this)) {
            if (removeDuplicate) {
                memo.add(this);
            }
            yield [prefix, this];
            for (const [name, module] of this.namedChildren) {
                if (!module) {
                    continue;
                }
                const submodulePrefix = prefix ? `${prefix}.${name}` : `${name}`;
                yield* module.namedModules(
                    memo,
                    submodulePrefix,
                    removeDuplicate
                );
            }
        }
    }
    /**
     * Returns this module and its descendants.
     * @returns a generator of modules
     */
    *modules(): Generator<Module> {
        for (const [_, module] of this.namedModules()) {
            yield module;
        }
    }

    private *_named_members<T>(
        get_members_fn: (m: Module) => [string, T][],
        prefix = "",
        recurse = true,
        remove_duplicate: boolean = true
    ): Generator<[string, T]> {
        const memo = new Set<T>();
        const modules: Generator<[string|number, Module]> | [string|number, Module][] =
            recurse
                ? this.namedModules(undefined, prefix, remove_duplicate)
                : [[prefix, this]];
        for (var [module_prefix, module] of modules) {
            const members = get_members_fn(module);
            for (var [k, v] of members) {
                if (v == null || memo.has(v)) continue;
                if (remove_duplicate) memo.add(v);
                const name = module_prefix ? `${module_prefix}.${k}` : k;
                yield [name, v];
            }
        }
    }
    /**
     * Gets this module and its descendants' (if `recurse = true`) parameters along with their prefixed names.
     * @param prefix is prependended to the names of the parameters
     * @param recurse whether to include submodule parameters
     * @param removeDuplicate whether to remove duplicate parameters
     * @returns a generator of [prefixed name, parameter] pairs
     */
    namedParameters(
        prefix: string = "",
        recurse: boolean = true,
        removeDuplicate: boolean = true
    ): Generator<[string, Parameter]> {
        return this._named_members(
            (m) => m.immediateParameters,
            prefix,
            recurse,
            removeDuplicate
        );
    }
    *parameters(): Generator<Parameter> {
        for (const [_, parameter] of this.namedParameters()) {
            yield parameter;
        }
    }

    registerBuffer(
        name: string,
        tensor: Tensor | null,
        persistent: boolean = true
    ): void {
        if (!name) {
            throw new Error("Buffer name must not be empty.");
        }
        if (name.indexOf(".") !== -1) {
            throw new Error("Buffer name must not contain a period.");
        }
        if (this._buffers.some(([n, _]) => n === name)) {
            throw new Error(`Buffer ${name} already registered.`);
        }
        this._buffers.push([name, tensor]);
        if (persistent) {
            this._nonPersistentBuffersSet.delete(name);
        } else {
            this._nonPersistentBuffersSet.add(name);
        }
    }
    namedBuffers(
        prefix: string = "",
        recurse: boolean = true,
        removeDuplicate: boolean = true
    ): Generator<[string, Tensor | null]> {
        return this._named_members(
            (m) => m._buffers,
            prefix,
            recurse,
            removeDuplicate
        );
    }
    *buffers(): Generator<Tensor | null> {
        for (const [_, buffer] of this.namedBuffers()) {
            yield buffer;
        }
    }
    getBuffer(name: string): Tensor | null {
        const buffer = this._buffers.find(([n, _]) => n === name);
        return buffer ? buffer[1] : null;
    }

    /**
     * Sets the module in training mode.
     * @param mode whether to set training mode (`true`) or evaluation mode (`false`).
     * @returns this module
     */
    train(mode: boolean = true): ThisType<Module> {
        this._training = mode;
        for (const module of this.children) {
            module.train(mode);
        }
        return this;
    }
    /**
     * Sets the module in evaluation mode.
     * @returns this module
     */
    eval(): ThisType<Module> {
        return this.train(false);
    }

    /**
     * Change if autograd should record operations on parameters in this module.
     * @param requiresGrad whether to enable gradient calculation for parameters in this module.
     * @returns this module
     */
    requiresGrad(requiresGrad = true): ThisType<Module> {
        for (const parameter of this.parameters()) {
            parameter.requiresGrad = requiresGrad;
        }
        return this;
    }
    /**
     * Zeros out the gradients of all parameters. That can be accomplished either by
     * setting the `grad` property to `null` (`setToNull=true`) or
     * by filling the gradient tensor with zeros (`setToNull=false`).
     * @param setToNull whether to set gradients to `null` (`true`) or to zero tensors (`false`).
     * @returns this module
     */
    zeroGrad(setToNull: boolean = true): ThisType<Module> {
        for (const parameter of this.parameters()) {
            if (parameter.grad) {
                if (setToNull) {
                    parameter.grad = null;
                } else {
                    if (parameter.grad.gradFunc) {
                        parameter.grad.detach();
                    } else {
                        parameter.grad.requiresGrad = false;
                    }
                    parameter.grad.zero_();
                    throw new Error(
                        "Not implemented: Cannot set gradients to 0. Use setToNull=true instead."
                    );
                }
            }
        }
        return this;
    }

    /**
     * Returns the state dictionary of the module and its descendants.
     * @param destination An optional state dictionary to update with the module's state.
     * @param prefix A string to prepend to the names of the state entries (default is an empty string).
     * @param keepVars A boolean flag, if true keeps the tensors attached to autograd (default is true).
     * @returns The updated state dictionary containing the module's state.
     */
    stateDict(
        destination?: StateDict,
        prefix: string = "",
        keepVars: boolean = true
    ): StateDict {
        destination = destination || {};
        this._saveToStateDict(destination, prefix, keepVars);
        for (const [name, module] of this.namedChildren) {
            if (module) {
                module.stateDict(destination, prefix + name + ".", keepVars);
            }
        }
        return destination;
    }
    private _saveToStateDict(
        destination: StateDict,
        prefix: string,
        keepVars: boolean
    ): void {
        for (const [name, parameter] of this.namedParameters()) {
            if (parameter) {
                destination[prefix + name] = keepVars
                    ? parameter
                    : parameter.detach();
            }
        }
        for (const [name, buffer] of this.namedBuffers()) {
            if (buffer && !this._nonPersistentBuffersSet.has(name)) {
                destination[prefix + name] = keepVars
                    ? buffer
                    : buffer.detach();
            }
        }
    }
    /**
     * Loads the state of the module and its descendants from the given state dictionary.
     * @param stateDict The state dictionary containing the state of the module to load.
     */
    loadStateDict(stateDict: StateDict): void {
        function load(
            module: Module,
            localStateDict: StateDict,
            prefix: string
        ) {
            module._loadFromStateDict(localStateDict, prefix);
            for (const [name, child] of module.namedChildren) {
                if (child) {
                    const childPrefix = prefix + name + ".";
                    const childStateDict: StateDict = {};
                    for (const [k, v] of Object.entries(localStateDict)) {
                        if (k.startsWith(childPrefix)) {
                            childStateDict[k] = v;
                        }
                    }
                    load(child, childStateDict, childPrefix);
                }
            }
        }
        load(this, stateDict, "");
    }
    private _loadFromStateDict(stateDict: StateDict, prefix: string): void {
        throw new Error("State dict loading not implemented");
    }
}

export class Parameter extends Tensor {
    constructor(data: Tensor, requiresGrad: boolean = true) {
        data = data; // || empty(0);
        super({
            data: data.storage,
            dtype: data.dtype,
            shape: data.shape,
            strides: data.strides,
            requiresGrad,
            device: data.device,
        });
    }
}

/**
 * An abstraction for modules that accept child modules as arguments.
 */
export class Container extends Module {
}

/**
 * Hold submodules in a list.
 */
export class ModuleList extends Module {
    get length() {
        return this.children.length;
    }
    [index: number]: Module;
    [Symbol.iterator]() {
        return this.children[Symbol.iterator]();
    }
    constructor(modules: Module[] = []) {
        super();
        for (const [i, module] of modules.entries()) {
            this.addModule(i, module);
        }
    }
    push(module: Module) {
        this.addModule(this.children.length, module);
    }
}

/**
 * A sequential container.
 * 
 * Modules will be added to it in the order they are passed in the constructor.
 * 
 * The `forward()` method of Sequential accepts any input and forwards it to the first module.
 * It then forwards the output of that module to the second module, and so on,
 * finally returning the output of the last module.
 */
export class Sequential extends Container {
    get length() {
        return this.children.length;
    }
    [index: number]: Module;
    [Symbol.iterator]() {
        return this.children[Symbol.iterator]();
    }
    constructor(...modules: Module[]) {
        super();
        for (const module of modules) {
            this.push(module);
        }
    }
    push(module: Module) {
        this.addModule(this.children.length, module);
    }
    forward(input: Tensor): Tensor {
        for (const module of this.children) {
            input = (module as any).forward(input);
        }
        return input;
    }
}
