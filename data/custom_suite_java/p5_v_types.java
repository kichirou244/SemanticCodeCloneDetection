public class GenericLinkedList<T extends Number> {
    private static class GenericNode<T> {
        T data;
        GenericNode<T> next;
        GenericNode(T data) { this.data = data; }
    }

    private GenericNode<T> root;

    public void add(T val) {
        GenericNode<T> newNode = new GenericNode<>(val);
        if (root == null) {
            root = newNode;
            return;
        }
        GenericNode<T> temp = root;
        while (temp.next != null) {
            temp = temp.next;
        }
        temp.next = newNode;
    }

    public boolean contains(T target) {
        GenericNode<T> temp = root;
        while (temp != null) {
            if (temp.data != null && temp.data.equals(target)) return true;
            temp = temp.next;
        }
        return false;
    }

    public double sum() {
        double total = 0.0;
        GenericNode<T> temp = root;
        while (temp != null) {
            if (temp.data != null) total += temp.data.doubleValue();
            temp = temp.next;
        }
        return total;
    }
}
