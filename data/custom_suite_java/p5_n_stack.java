public class LinkedStack {
    private static class Node {
        int val;
        Node next;
        Node(int val, Node next) {
            this.val = val;
            this.next = next;
        }
    }

    private Node top;

    public void push(int val) {
        top = new Node(val, top);
    }

    public int pop() {
        if (top == null) {
            throw new java.util.EmptyStackException();
        }
        int value = top.val;
        top = top.next;
        return value;
    }

    public int peek() {
        if (top == null) {
            throw new java.util.EmptyStackException();
        }
        return top.val;
    }
}
