struct Node { int val; Node* next; Node(int v): val(v), next(nullptr) {} };
class SentinelList {
    Node dummy;
public:
    SentinelList(): dummy(0) {}
    void add(int v) {
        Node* cur = &dummy;
        while (cur->next) cur = cur->next;
        cur->next = new Node(v);
    }
    bool contains(int t) {
        Node* cur = dummy.next;
        while (cur) { if (cur->val == t) return true; cur = cur->next; }
        return false;
    }
    int sum() {
        int total = 0; Node* cur = dummy.next;
        while (cur) { total += cur->val; cur = cur->next; }
        return total;
    }
};
