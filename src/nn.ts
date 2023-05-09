import { Tensor } from "./tensor";

type ModuleMember = Buffer | Module | Parameter;

export class Module {
    private _children: [string, Module][] | null = null;
    get namedChildren(): [string, Module][] {
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

    private _buffers: [string, Tensor|null][] = [];
    private _nonPersistentBuffersSet: Set<string> = new Set<string>();

    get [Symbol.toStringTag]() {
        return "Module";
    }

    constructor() {}

    *namedModules(
        memo?: Set<Module>,
        prefix: string = "",
        removeDuplicate: boolean = true
    ): Generator<[string, Module]> {
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
                const submodulePrefix = prefix ? `${prefix}.${name}` : name;
                yield* module.namedModules(
                    memo,
                    submodulePrefix,
                    removeDuplicate
                );
            }
        }
    }
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
        const modules: Generator<[string, Module]> | [string, Module][] =
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

    registerBuffer(name: string, tensor: Tensor|null, persistent: boolean = true): void {
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
        }
        else {
            this._nonPersistentBuffersSet.add(name);
        }
    }
    namedBuffers(prefix: string = "", recurse: boolean = true, removeDuplicate: boolean = true): Generator<[string, Tensor|null]> {
        return this._named_members(
            (m) => m._buffers,
            prefix,
            recurse,
            removeDuplicate
        );
    }
    *buffers(): Generator<Tensor|null> {
        for (const [_, buffer] of this.namedBuffers()) {
            yield buffer;
        }
    }
}

export class Sequential extends Module {}

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

export class Conv2d extends Module {
    private _inChannels: number;
    private _outChannels: number;
    get inChannels(): number {
        return this._inChannels;
    }
    get outChannels(): number {
        return this._outChannels;
    }
    constructor(inChannels: number, outChannels: number) {
        super();
        // Set properties directly on 'this'
        this._inChannels = inChannels;
        this._outChannels = outChannels;
    }
}
