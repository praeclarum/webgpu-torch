import { BufferHeap, HeapBuffer } from "./storage";

test("buffer heap retains buffer", () => {
    const a = new ArrayBuffer(128);
    const heap = new BufferHeap(a, 128, 0);
    expect(heap.buffer).toBe(a);
});

test("buffer heap with non power of two size fails", () => {
    expect(() => {
        new BufferHeap(null, 100, 0);
    }).toThrow();
});

test("buffer heap with power of two size succeeds", () => {
    const heap = new BufferHeap(null, 128, 0);
    expect(heap.size).toBe(128);
});

test("buffer heap alloc 16", () => {
    const heap = new BufferHeap(null, 128, 0);
    const buffer = heap.alloc(16);
    expect(buffer).not.toBeNull();
    expect(buffer!.byteSize).toBe(16);
});

test("buffer heap alloc larger than min order", () => {
    const heap = new BufferHeap(null, 128, 8);
    const buffer = heap.alloc(16);
    expect(buffer).toBeNull();
});

test("buffer heap too much", () => {
    const heap = new BufferHeap(null, 128, 0);
    for (let i = 0; i < 8; i++) {
        const buffer = heap.alloc(16);
        expect(buffer).not.toBeNull();
    }
    const badBuffer = heap.alloc(16);
    expect(badBuffer).toBeNull();
});

test("buffer heap free from too much", () => {
    const heap = new BufferHeap(null, 128, 0);
    const buffers: HeapBuffer<null>[] = [];
    for (let i = 0; i < 16; i++) {
        const buffer = heap.alloc(8);
        expect(buffer).not.toBeNull();
        buffers.push(buffer!);
    }
    const badBuffer = heap.alloc(16);
    expect(badBuffer).toBeNull();
    heap.free(buffers[1]);
    heap.free(buffers[0]);
    const goodBuffer = heap.alloc(16);
    expect(goodBuffer).not.toBeNull();
});
