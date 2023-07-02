import { Tensor } from "./tensor";
import { enableGrad, noGrad } from "./autograd";
import { clone } from "./ops_artisanal";

export class RequiredParameter {}
export const required = new RequiredParameter();

export type ParameterGroup = {
    [key: string]: Tensor[] | number | Boolean | null | RequiredParameter;
};

/** Base class of all optimizers. */
export abstract class Optimizer {
    defaults: { [key: string]: number | Boolean | null | RequiredParameter };
    paramGroups: ParameterGroup[] = [];
    state: Map<Tensor, { [key: string]: Tensor | null }> = new Map();
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
    zeroGrad(setToNull: boolean = true) {
        for (const group of this.paramGroups) {
            for (const param of group.params as Tensor[]) {
                if (param.grad !== null) {
                    if (setToNull) {
                        param.grad = null;
                    } else {
                        param.grad.zero_();
                    }
                }
            }
        }
    }
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
        return noGrad(() => {
            let loss: Tensor | null = null;
            if (closure !== undefined) {
                enableGrad(() => {
                    loss = closure();
                });
            }
            for (let group of this.paramGroups) {
                const params_with_grad: Tensor[] = [];
                const d_p_list: Tensor[] = [];
                const momentum_buffer_list: (Tensor|null)[] = [];

                const has_sparse_grad = this.initGroup(
                    group,
                    params_with_grad,
                    d_p_list,
                    momentum_buffer_list
                );

                sgd(
                    params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    has_sparse_grad,
                    group["foreach"] as Boolean | null,
                    group["weightDecay"] as number,
                    group["momentum"] as number,
                    group["lr"] as number,
                    group["dampening"] as number,
                    group["nesterov"] as Boolean,
                    group["maximize"] as Boolean
                );

                // Update state['momentum_buffer']
                for (let i in params_with_grad) {
                    const p = params_with_grad[i];
                    const momentum_buffer = momentum_buffer_list[i];
                    if (momentum_buffer === null) continue;
                    const state = this.state.get(p);
                    if (state === undefined) {
                        this.state.set(p, { "momentum_buffer": momentum_buffer });
                    }
                    else {
                        state["momentum_buffer"] = momentum_buffer;
                    }
                }
            }
            return loss;
        });
    }
    initGroup(
        group: ParameterGroup,
        params_with_grad: Tensor[],
        d_p_list: Tensor[],
        momentum_buffer_list: (Tensor | null)[]
    ): Boolean {
        let has_sparse_grad = false;
        for (const p of group["params"] as Tensor[]) {
            if (p.grad !== null) {
                params_with_grad.push(p);
                d_p_list.push(p.grad);
            }
            const state = this.state.get(p);
            if (state === undefined || state["momentum_buffer"] === undefined) {
                momentum_buffer_list.push(null);
                this.state.set(p, { "momentum_buffer": null });
            }
            else {
                momentum_buffer_list.push(state["momentum_buffer"]);
            }
        }
        return has_sparse_grad;
    }
}

function sgd(
    params: Tensor[],
    d_p_list: Tensor[],
    momentum_buffer_list: (Tensor | null)[],
    has_sparse_grad: Boolean | null = null,
    foreach: Boolean | null = null,
    weightDecay: number,
    momentum: number,
    lr: number,
    dampening: number,
    nesterov: Boolean,
    maximize: Boolean
) {
    for (let i in params) {
        const param = params[i];
        let d_p = maximize ?
            d_p_list[i].neg() :
            d_p_list[i];
        if (weightDecay !== 0) {
            d_p = d_p.add(param, weightDecay);
        }
        if (momentum !== 0) {
            let buf = momentum_buffer_list[i];
            if (buf === null) {
                buf = clone(d_p).detach();
                momentum_buffer_list[i] = buf;
            }
            else {
                buf.mul_(momentum).add_(d_p, 1 - dampening);
            }
            if (nesterov) {
                d_p = d_p.add(buf, momentum);
            }
            else {
                d_p = buf;
            }
        }
        param.add_(d_p, -lr);
    }
}
