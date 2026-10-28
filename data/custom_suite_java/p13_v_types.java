public class GenericArrayQueue<T> {
    private Object[] data = new Object[100];
    private int front = 0, rear = 0, size = 0;
    public void enqueue(T x) { data[rear] = x; rear = (rear + 1) % data.length; size++; }
    @SuppressWarnings("unchecked")
    public T dequeue() { T val = (T) data[front]; front = (front + 1) % data.length; size--; return val; }
    public boolean isEmpty() { return size == 0; }
}
