import { Tensor } from "./tensor";

export class RequiredParameter {}
export const required = new RequiredParameter();

export type ParameterGroup = {
    [key: string]: Tensor[] | number | Boolean | null | RequiredParameter;
};

/** Base class of all optimizers. */
export abstract class Optimizer {
    defaults: { [key: string]: number | Boolean | null | RequiredParameter };
    paramGroups: ParameterGroup[] = [];
    constructor(
        params: ParameterGroup | Tensor[],
        defaults: { [key: string]: number | Boolean | null | RequiredParameter }
    ) {
        this.defaults = defaults;
        let paramGroups: ParameterGroup[];
        if (params.length === 0) {
            throw new Error("Optimizer got an empty parameter list");
        }
        if (Array.isArray(params)) {
            paramGroups = [{ params: params }];
        } else {
            paramGroups = [params];
        }
        for (const paramGroup of paramGroups) {
            this.addParamGroup(paramGroup);
        }
    }
    addParamGroup(paramGroup: ParameterGroup) {
        // Make sure defaults are set
        for (const key in this.defaults) {
            if (!(key in paramGroup)) {
                const def = this.defaults[key];
                if (def === required) {
                    throw new Error(
                        `Parameter group didn't specify a value of ${key}`
                    );
                }
                paramGroup[key] = def;
            }
        }
        this.paramGroups.push(paramGroup);
    }
    abstract step(closure?: () => Tensor): Tensor | null;
}

export class SGD extends Optimizer {
    constructor(
        params: ParameterGroup | Tensor[],
        lr: number | Tensor | RequiredParameter = required,
        momentum: number = 0,
        dampening: number = 0,
        weightDecay: number = 0,
        nesterov: Boolean = false,
        maximize: Boolean = false,
        foreach?: Boolean,
        differentiable: Boolean = false
    ) {
        super(params, {
            lr,
            momentum,
            dampening,
            weightDecay,
            nesterov,
            maximize,
            foreach: foreach === undefined ? false : foreach,
            differentiable,
        });
    }
    step(closure?: (() => Tensor) | undefined): Tensor | null {
        throw new Error("Method not implemented.");
    }
}
