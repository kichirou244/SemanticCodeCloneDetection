public class SinglyList {
    private static class Node {
        int val;
        Node next;
        Node(int val) { this.val = val; }
    }

    private Node head;

    public void add(int val) {
        Node newNode = new Node(val);
        if (head == null) {
            head = newNode;
            return;
        }
        Node curr = head;
        while (curr.next != null) {
            curr = curr.next;
        }
        curr.next = newNode;
    }

    public boolean contains(int target) {
        Node curr = head;
        while (curr != null) {
            if (curr.val == target) return true;
            curr = curr.next;
        }
        return false;
    }

    public int sum() {
        int total = 0;
        Node curr = head;
        while (curr != null) {
            total += curr.val;
            curr = curr.next;
        }
        return total;
    }
}
