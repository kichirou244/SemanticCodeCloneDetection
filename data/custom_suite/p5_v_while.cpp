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
    Node *cur = head;
    while (cur->next != NULL) {
        cur = cur->next;
    }
    cur->next = node;
    return head;
}

long long sumList(Node *head) {
    long long acc = 0;
    Node *cur = head;
    while (cur != NULL) {
        acc = acc + cur->val;
        cur = cur->next;
    }
    return acc;
}

void printList(Node *head) {
    Node *cur = head;
    while (cur != NULL) {
        cout << cur->val << " ";
        cur = cur->next;
    }
    cout << endl;
}

void freeList(Node *head) {
    while (head != NULL) {
        Node *tmp = head->next;
        delete head;
        head = tmp;
    }
}

int main() {
    int n;
    cin >> n;
    Node *head = NULL;
    int i = 0;
    while (i < n) {
        int x;
        cin >> x;
        head = pushBack(head, x);
        i++;
    }
    printList(head);
    cout << sumList(head) << endl;
    freeList(head);
    return 0;
}
