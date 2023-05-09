import { zeros } from './factories';
import * as nn from './nn';

class A extends nn.Module {
    b: B;
    d: D;
    p1: nn.Parameter;
    constructor() {
        super();
        this.b = new B();
        this.d = new D();
        this.p1 = new nn.Parameter(zeros([1, 2, 3]));
        this.registerBuffer("buf1", zeros([100, 200, 300]));
    }
}

class B extends nn.Module {
    x1: X;
    p2: nn.Parameter;
    constructor() {
        super();
        this.x1 = new X();
        this.p2 = new nn.Parameter(zeros([4, 5, 6]));
    }
}

class D extends B {
    x2: X;
    constructor() {
        super();
        this.x2 = new X();
    }
}

class X extends nn.Module {    
}

test("derived module has parent immediate children", () => {
    const m = new D();
    const children = m.children
    expect(children.length).toBe(2);
    expect(children[0]).toBe(m.x1);
    expect(children[1]).toBe(m.x2);
});

test("parent module can iterate over all named children modules", () => {
    const m = new A();
    const children = Array.from(m.namedModules());
    expect(children.length).toBe(6);
    expect(children[0][0]).toBe("");
    expect(children[1][0]).toBe("b");
    expect(children[2][0]).toBe("b.x1");
    expect(children[3][0]).toBe("d");
    expect(children[4][0]).toBe("d.x1");
    expect(children[5][0]).toBe("d.x2");
    expect(children[5][1]).toBe(m.d.x2);
});

test("parent module can iterate over all unnamed children modules", () => {
    const m = new A();
    const children = Array.from(m.modules());
    expect(children.length).toBe(6);
    expect(children[5]).toBe(m.d.x2);
});

test("all parameters", () => {
    const m = new A();
    const parameters = Array.from(m.namedParameters());
    expect(parameters.length).toBe(3);
    expect(parameters[0][0]).toBe("p1");
    expect(parameters[1][0]).toBe("b.p2");
    expect(parameters[2][0]).toBe("d.p2");
});

test("registered named buffers exist", () => {
    const m = new A();
    const buffers = Array.from(m.namedBuffers());
    expect(buffers.length).toBe(1);
    expect(buffers[0][0]).toBe("buf1");
});

test("registered unnamed buffers exist", () => {
    const m = new A();
    const buffers = Array.from(m.buffers());
    expect(buffers.length).toBe(1);
});

test("toString is formatted correctly", () => {
    const m = new A();
    expect(m.toString()).toBe(`A(
  (b): B(
    (x1): X()
  )
  (d): D(
    (x1): X()
    (x2): X()
  )
)`);
});

test("toggle train mode", () => {
    const m = new A();
    expect(m.training).toBe(true);
    m.train(false);
    expect(m.training).toBe(false);
    m.train();
    expect(m.b.training).toBe(true);
    expect(m.b.x1.training).toBe(true);
    expect(m.d.training).toBe(true);
    expect(m.d.x1.training).toBe(true);
    expect(m.d.x2.training).toBe(true);
    m.eval();
    expect(m.training).toBe(false);
    expect(m.b.training).toBe(false);
    expect(m.b.x1.training).toBe(false);
    expect(m.d.training).toBe(false);
    expect(m.d.x1.training).toBe(false);
    expect(m.d.x2.training).toBe(false);
});

test("toggle requiresGrad", () => {
    const m = new A();
    expect(m.p1.requiresGrad).toBe(true);
    expect(m.b.p2.requiresGrad).toBe(true);
    expect(m.d.p2.requiresGrad).toBe(true);
    m.requiresGrad(false);
    expect(m.p1.requiresGrad).toBe(false);
    expect(m.b.p2.requiresGrad).toBe(false);
    expect(m.d.p2.requiresGrad).toBe(false);
    m.requiresGrad();
    expect(m.p1.requiresGrad).toBe(true);
    expect(m.b.p2.requiresGrad).toBe(true);
    expect(m.d.p2.requiresGrad).toBe(true);
});

test("zero grad with no grads", () => {
    const m = new A();
    m.zeroGrad();
    expect(m.p1.grad).toBe(null);
    expect(m.b.p2.grad).toBe(null);
    expect(m.d.p2.grad).toBe(null);
});

test("Conv2d can set inChannels and outChannels", () => {
    const conv2d = new nn.Conv2d(3, 4);
    expect(conv2d.inChannels).toBe(3);
    expect(conv2d.outChannels).toBe(4);
});
