public class ArrayQueue {
    private int[] data = new int[100];
    private int front = 0, rear = 0, size = 0;
    public void enqueue(int x) { data[rear] = x; rear = (rear + 1) % data.length; size++; }
    public int dequeue() { int val = data[front]; front = (front + 1) % data.length; size--; return val; }
    public boolean isEmpty() { return size == 0; }
}
