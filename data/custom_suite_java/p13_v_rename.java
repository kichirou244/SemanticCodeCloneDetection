public class CircularQueue {
    // Hang doi vong cai dat bang mang tinh
    private int[] buffer = new int[100];
    private int head = 0, tail = 0, count = 0;
    public void enqueue(int item) { buffer[tail] = item; tail = (tail + 1) % buffer.length; count++; }
    public int dequeue() { int item = buffer[head]; head = (head + 1) % buffer.length; count--; return item; }
    public boolean isEmpty() { return count == 0; }
}
