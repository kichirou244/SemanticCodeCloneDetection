public class LinkedContainer {
    private static class ElementNode {
        int element;
        ElementNode nextNode;
        ElementNode(int elem) { this.element = elem; }
    }

    private ElementNode firstNode;

    public void append(int item) {
        ElementNode created = new ElementNode(item);
        if (firstNode == null) {
            firstNode = created;
            return;
        }
        ElementNode pointer = firstNode;
        while (pointer.nextNode != null) {
            pointer = pointer.nextNode;
        }
        pointer.nextNode = created;
    }

    public boolean hasElement(int query) {
        ElementNode pointer = firstNode;
        while (pointer != null) {
            if (pointer.element == query) return true;
            pointer = pointer.nextNode;
        }
        return false;
    }

    public int calculateSum() {
        int accumulator = 0;
        ElementNode pointer = firstNode;
        while (pointer != null) {
            accumulator += pointer.element;
            pointer = pointer.nextNode;
        }
        return accumulator;
    }
}
