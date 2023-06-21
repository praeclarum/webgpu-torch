import { BufferHeap, HeapBuffer } from "./storage";

test("buffer heap with non power of two size fails", () => {
    expect(() => {
        new BufferHeap(null, 100, 0);
    }).toThrow();
});

test("buffer heap with power of two size succeeds", () => {
    const heap = new BufferHeap(null, 128, 0);
    expect(heap.size).toBe(128);
});
