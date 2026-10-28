public class SentinelLinkedList {
    private static class Node {
        int val;
        Node next;
        Node(int val) { this.val = val; }
    }

    private final Node dummy = new Node(0);

    public void add(int val) {
        Node curr = dummy;
        while (curr.next != null) {
            curr = curr.next;
        }
        curr.next = new Node(val);
    }

    public boolean contains(int target) {
        Node curr = dummy.next;
        while (curr != null) {
            if (curr.val == target) return true;
            curr = curr.next;
        }
        return false;
    }

    public int sum() {
        int total = 0;
        Node curr = dummy.next;
        while (curr != null) {
            total += curr.val;
            curr = curr.next;
        }
        return total;
    }
}
