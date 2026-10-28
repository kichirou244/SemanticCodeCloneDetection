struct Element { int data; Element* nextNode; Element(int d): data(d), nextNode(nullptr) {} };
class Container {
    Element* first = nullptr;
public:
    void append(int item) {
        Element* e = new Element(item);
        if (!first) { first = e; return; }
        Element* p = first;
        while (p->nextNode) p = p->nextNode;
        p->nextNode = e;
    }
    bool hasElement(int q) {
        Element* p = first;
        while (p) { if (p->data == q) return true; p = p->nextNode; }
        return false;
    }
    int calculateSum() {
        int total = 0; Element* p = first;
        while (p) { total += p->data; p = p->nextNode; }
        return total;
    }
};
