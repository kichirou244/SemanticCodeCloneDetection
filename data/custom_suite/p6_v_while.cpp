#include <bits/stdc++.h>
using namespace std;

struct Node {
    int val;
    Node *next;
};

Node* insertSorted(Node *head, int v) {
    Node *fresh = new Node;
    fresh->val = v;
    fresh->next = NULL;
    if (head == NULL || v <= head->val) {
        fresh->next = head;
        return fresh;
    }
    Node *cur = head;
    for (; cur->next != NULL; cur = cur->next) {
        if (cur->next->val > v) {
            break;
        }
    }
    fresh->next = cur->next;
    cur->next = fresh;
    return head;
}

int removeDuplicates(Node *head) {
    int cnt = 0;
    Node *cur = head;
    while (cur != NULL) {
        if (cur->next == NULL) {
            break;
        }
        if (cur->val != cur->next->val) {
            cur = cur->next;
            continue;
        }
        Node *dup = cur->next;
        cur->next = dup->next;
        delete dup;
        cnt = cnt + 1;
    }
    return cnt;
}

long long sumList(Node *head) {
    long long acc = 0;
    Node *cur = head;
    do {
        if (cur == NULL) {
            break;
        }
        acc = acc + cur->val;
        cur = cur->next;
    } while (cur != NULL);
    return acc;
}

void printList(Node *head) {
    for (Node *cur = head; cur != NULL; cur = cur->next) {
        cout << cur->val << " ";
    }
    cout << endl;
}

void freeList(Node *head) {
    Node *cur = head;
    while (cur != NULL) {
        Node *nxt = cur->next;
        delete cur;
        cur = nxt;
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
        head = insertSorted(head, x);
        i = i + 1;
    }
    printList(head);
    cout << "removed " << removeDuplicates(head) << endl;
    printList(head);
    cout << "sum " << sumList(head) << endl;
    freeList(head);
    return 0;
}
