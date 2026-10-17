#include <cstdio>

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
    Node *q = head;
    while (q->next != NULL) {
        q = q->next;
    }
    q->next = node;
    return head;
}

long long sumList(Node *head) {
    long long s = 0;
    for (Node *q = head; q; q = q->next) {
        s = s + q->val;
    }
    return s;
}

void printList(Node *head) {
    for (Node *q = head; q; q = q->next) {
        printf("%d ", q->val);
    }
    printf("\n");
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
    scanf("%d", &n);
    Node *head = NULL;
    for (int i = 0; i < n; i++) {
        int x;
        scanf("%d", &x);
        head = pushBack(head, x);
    }
    printList(head);
    printf("%lld\n", sumList(head));
    freeList(head);
    return 0;
}
