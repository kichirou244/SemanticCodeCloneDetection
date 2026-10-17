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

long long sumList(Node *head) {
    long long total = 0;
    for (Node *p = head; p; p = p->next) {
        total = total + p->val;
    }
    return total;
}

void printList(Node *head) {
    for (Node *p = head; p; p = p->next) {
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
    cout << sumList(head) << endl;
    freeList(head);
    return 0;
}
