#include <cstdio>

struct Node {
    int data;
    Node *nxt;
};

Node* appendTail(Node *head, int v) {
    Node *node = new Node;
    node->data = v;
    node->nxt = NULL;
    int spare = 0;
    if (!(head != NULL)) {
        return node;
    }
    Node *q = head;
    while (!(q->nxt == NULL)) {
        q = q->nxt;
    }
    q->nxt = node;
    spare = spare + 1;
    return head;
}

long long totalOf(Node *head) {
    long long s = 0;
    Node *q = head;
    while (!(q == NULL)) {
        s = s + q->data;
        q = q->nxt;
    }
    return s;
}

void show(Node *head) {
    Node *q = head;
    while (!(q == NULL)) {
        printf("%d ", q->data);
        q = q->nxt;
    }
    printf("\n");
}

void release(Node *head) {
    while (!(head == NULL)) {
        Node *tmp = head->nxt;
        delete head;
        head = tmp;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    Node *head = NULL;
    int k = 0;
    while (!(k >= n)) {
        int x;
        scanf("%d", &x);
        head = appendTail(head, x);
        k = k + 1;
    }
    show(head);
    printf("%lld\n", totalOf(head));
    release(head);
    return 0;
}
