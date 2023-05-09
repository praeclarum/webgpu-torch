import * as nn from './nn';

class A extends nn.Module {
    b: B
    d: D
    constructor() {
        super();
        this.b = new B();
        this.d = new D();
    }
}

class B extends nn.Module {
    x1: X
    constructor() {
        super();
        this.x1 = new X();
    }
}

class D extends B {
    x2: X
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

test("Conv2d can set inChannels and outChannels", () => {
    const conv2d = new nn.Conv2d(3, 4);
    expect(conv2d.inChannels).toBe(3);
    expect(conv2d.outChannels).toBe(4);
});
