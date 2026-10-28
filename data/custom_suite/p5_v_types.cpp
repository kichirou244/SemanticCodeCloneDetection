template<typename T>
struct Node { T val; Node* next; Node(T v): val(v), next(nullptr) {} };
template<typename T>
class GenericList {
    Node<T>* head = nullptr;
public:
    void add(T v) {
        Node<T>* n = new Node<T>(v);
        if (!head) { head = n; return; }
        Node<T>* cur = head;
        while (cur->next) cur = cur->next;
        cur->next = n;
    }
    bool contains(T t) {
        Node<T>* cur = head;
        while (cur) { if (cur->val == t) return true; cur = cur->next; }
        return false;
    }
    T sum() {
        T total = 0; Node<T>* cur = head;
        while (cur) { total += cur->val; cur = cur->next; }
        return total;
    }
};
