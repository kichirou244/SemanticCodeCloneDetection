#include <cstdio>

struct Node {
    int val;
    Node *next;
};

Node* insertSorted(Node *head, int v) {
    Node *node = new Node;
    node->val = v;
    node->next = NULL;
    if (head == NULL || v <= head->val) {
        node->next = head;
        return node;
    }
    Node *p = head;
    while (p->next != NULL && p->next->val <= v) {
        p = p->next;
    }
    node->next = p->next;
    p->next = node;
    return head;
}

int removeDuplicates(Node *head) {
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

long long sumList(Node *head) {
    long long total = 0;
    Node *p = head;
    while (p != NULL) {
        total = total + p->val;
        p = p->next;
    }
    return total;
}

void printList(Node *head) {
    Node *p = head;
    while (p != NULL) {
        printf("%d ", p->val);
        p = p->next;
    }
    printf("\n");
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
    scanf("%d", &n);
    Node *head = NULL;
    for (int i = 0; i < n; i++) {
        int x;
        scanf("%d", &x);
        head = insertSorted(head, x);
    }
    printList(head);
    printf("removed %d\n", removeDuplicates(head));
    printList(head);
    printf("sum %lld\n", sumList(head));
    freeList(head);
    return 0;
}
