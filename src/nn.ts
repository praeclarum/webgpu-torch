export type ModuleAttribute = Module;

export class Module {
    private _children: Module[] | null = null;

    get children(): Module[] {
        if (this._children === null) {
            return this.findChildren();
        }
        return this._children;
    }
    get [Symbol.toStringTag]() {
        return "Module";
    }

    constructor() {}

    private findChildren(): ModuleAttribute[] {
        this._children = [];
        for (const key in this) {
            const value = (this as any)[key];
            if (value instanceof Module) {
                this._children.push(value);
            }
        }
        return this._children;
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
