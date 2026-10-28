public class LinkedQueue {
    private static class Node { int val; Node next; Node(int v) { val = v; } }
    private Node head, tail;
    public void enqueue(int x) {
        Node n = new Node(x);
        if (tail == null) { head = tail = n; }
        else { tail.next = n; tail = n; }
    }
    public int dequeue() {
        int val = head.val;
        head = head.next;
        if (head == null) tail = null;
        return val;
    }
    public boolean isEmpty() { return head == null; }
}
