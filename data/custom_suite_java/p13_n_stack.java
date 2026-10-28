public class ArrayStack {
    private int[] data = new int[100];
    private int top = -1;
    public void enqueue(int x) { data[++top] = x; } // goi enqueue nhung hanh vi la push
    public int dequeue() { return data[top--]; }     // hanh vi la pop
    public boolean isEmpty() { return top == -1; }
}
