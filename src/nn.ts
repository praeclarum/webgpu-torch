export type ModuleAttribute = Module;

export class Module {
    private _children: [string, Module][] | null = null;

    get namedChildren(): [string, Module][] {
        if (this._children === null) {
            return this.findChildren();
        }
        return this._children;
    }
    get children(): Module[] {
        return this.namedChildren.map(([_, value]) => value);
    }
    get [Symbol.toStringTag]() {
        return "Module";
    }

    constructor() {}

    private findChildren(): [string, Module][] {
        this._children = [];
        for (const key in this) {
            const value = (this as any)[key];
            if (value instanceof Module) {
                this._children.push([key, value]);
            }
        }
        return this._children;
    }

    *namedModules(memo?: Set<Module>, prefix: string = "", removeDuplicate: boolean = true): Generator<[string, Module]> {
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
                yield* module.namedModules(memo, submodulePrefix, removeDuplicate);
            }
        }            
    }
    *modules(): Generator<Module> {
        for (const [_, module] of this.namedModules()) {
            yield module;
        }
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
