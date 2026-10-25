#include <bits/stdc++.h>
using namespace std;

struct Node {
    int val;
    Node *next;
};

Node* pushBack(Node *head, int v) {
    Node *node = new Node;
    node->val = v;
    node->next = NULL;
    if (head == NULL) {
        return node;
    }
    Node *p = head;
    while (p->next != NULL) {
        p = p->next;
    }
    p->next = node;
    return head;
}

int removeGreater(Node *&head, int limit) {
    int removed = 0;
    while (head != NULL && head->val > limit) {
        Node *old = head;
        head = head->next;
        delete old;
        removed = removed + 1;
    }
    Node *p = head;
    while (p != NULL && p->next != NULL) {
        if (p->next->val > limit) {
            Node *old = p->next;
            p->next = old->next;
            delete old;
            removed = removed + 1;
        } else {
            p = p->next;
        }
    }
    return removed;
}

int countOdd(Node *head) {
    int odd = 0;
    for (Node *p = head; p != NULL; p = p->next) {
        if (p->val % 2 != 0) {
            odd = odd + 1;
        }
    }
    return odd;
}

void printList(Node *head) {
    for (Node *p = head; p != NULL; p = p->next) {
        cout << p->val << " ";
    }
    cout << endl;
}

void freeList(Node *head) {
    while (head != NULL) {
        Node *nxt = head->next;
        delete head;
        head = nxt;
    }
}

int main() {
    int n;
    cin >> n;
    Node *head = NULL;
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        head = pushBack(head, x);
    }
    printList(head);
    int limit;
    cin >> limit;
    cout << "removed " << removeGreater(head, limit) << endl;
    printList(head);
    cout << "odd " << countOdd(head) << endl;
    freeList(head);
    return 0;
}
