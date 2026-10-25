#include <bits/stdc++.h>
using namespace std;

struct Node {
    int val;
    Node *next;
};

class IntList {
private:
    Node *head;
public:
    IntList() {
        head = NULL;
    }
    ~IntList() {
        clear();
    }
    void insert(int v) {
        Node *node = new Node;
        node->val = v;
        node->next = NULL;
        if (head == NULL || v <= head->val) {
            node->next = head;
            head = node;
            return;
        }
        Node *p = head;
        while (p->next != NULL && p->next->val <= v) {
            p = p->next;
        }
        node->next = p->next;
        p->next = node;
    }
    int dedup() {
        int removed = 0;
        Node *p = head;
        while (p != NULL && p->next != NULL) {
            if (p->val == p->next->val) {
                Node *dup = p->next;
                p->next = dup->next;
                delete dup;
                removed = removed + 1;
            } else {
                p = p->next;
            }
        }
        return removed;
    }
    long long total() const {
        long long sum = 0;
        for (Node *p = head; p != NULL; p = p->next) {
            sum = sum + p->val;
        }
        return sum;
    }
    void show() const {
        for (Node *p = head; p != NULL; p = p->next) {
            cout << p->val << " ";
        }
        cout << endl;
    }
    void clear() {
        while (head != NULL) {
            Node *nxt = head->next;
            delete head;
            head = nxt;
        }
    }
};

int main() {
    int n;
    cin >> n;
    IntList list;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        list.insert(x);
    }
    list.show();
    cout << "removed " << list.dedup() << endl;
    list.show();
    cout << "sum " << list.total() << endl;
    return 0;
}
