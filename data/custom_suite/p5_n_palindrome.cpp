#include <bits/stdc++.h>
using namespace std;

class Stack {
public:
    struct Node {
        int val;
        Node *next;
    };

    Stack() {
        top = NULL;
    }

    void push(int v) {
        Node *p = new Node;
        p->val = v;
        p->next = top;
        top = p;
    }

    int pop() {
        int v = top->val;
        Node *p = top;
        top = top->next;
        delete p;
        return v;
    }

    bool empty() {
        return top == NULL;
    }

    void clear() {
        while (top != NULL) {
            Node *nxt = top->next;
            delete top;
            top = nxt;
        }
    }

private:
    Node *top;
};

int main() {
    int n;
    cin >> n;
    int a[1000];
    Stack st;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        st.push(a[i]);
    }
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (st.pop() != a[i]) {
            ok = false;
        }
    }
    if (ok) {
        cout << "YES" << endl;
    } else {
        cout << "NO" << endl;
    }
    st.clear();
    return 0;
}
