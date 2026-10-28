public class ResizableQueue {
    private int[] data = new int[4];
    private int front = 0, rear = 0, size = 0;
    public void enqueue(int x) {
        if (size == data.length) resize();
        data[rear] = x; rear = (rear + 1) % data.length; size++;
    }
    public int dequeue() { int val = data[front]; front = (front + 1) % data.length; size--; return val; }
    private void resize() {
        int[] next = new int[data.length * 2];
        for (int i = 0; i < size; i++) next[i] = data[(front + i) % data.length];
        data = next; front = 0; rear = size;
    }
    public boolean isEmpty() { return size == 0; }
}
