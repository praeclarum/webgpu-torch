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

test("parent module can iterate over all children modules", () => {
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

test("all parameters", () => {
    const m = new A();
    const parameters = Array.from(m.namedParameters());
    expect(parameters.length).toBe(3);
    expect(parameters[0][0]).toBe("p1");
    expect(parameters[1][0]).toBe("b.p2");
    expect(parameters[2][0]).toBe("d.p2");
});

test("registered buffers exist", () => {
    const m = new A();
    const buffers = Array.from(m.namedBuffers());
    expect(buffers.length).toBe(1);
    expect(buffers[0][0]).toBe("buf1");
});

test("Conv2d can set inChannels and outChannels", () => {
    const conv2d = new nn.Conv2d(3, 4);
    expect(conv2d.inChannels).toBe(3);
    expect(conv2d.outChannels).toBe(4);
});
