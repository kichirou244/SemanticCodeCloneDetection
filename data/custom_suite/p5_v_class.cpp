#include <bits/stdc++.h>
using namespace std;

class IntList {
public:
    struct Node {
        int val;
        Node *next;
    };

    IntList() {
        head = NULL;
    }

    void pushBack(int v) {
        Node *node = new Node;
        node->val = v;
        node->next = NULL;
        if (head == NULL) {
            head = node;
            return;
        }
        Node *p = head;
        while (p->next != NULL) {
            p = p->next;
        }
        p->next = node;
    }

    long long sum() {
        long long total = 0;
        for (Node *p = head; p; p = p->next) {
            total = total + p->val;
        }
        return total;
    }

    void print() {
        for (Node *p = head; p; p = p->next) {
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

private:
    Node *head;
};

int main() {
    int n;
    cin >> n;
    IntList list;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        list.pushBack(x);
    }
    list.print();
    cout << list.sum() << endl;
    list.clear();
    return 0;
}
