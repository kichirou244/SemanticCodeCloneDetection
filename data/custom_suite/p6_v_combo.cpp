#include <cstdio>

struct Node {
    int val;
    Node *next;
};

Node* addNode(Node *head, int v) {
    Node *fresh = new Node;
    fresh->val = v;
    fresh->next = NULL;
    if (!(head != NULL && v > head->val)) {
        fresh->next = head;
        return fresh;
    }
    Node *cur = head;
    while (!(cur->next == NULL || cur->next->val > v)) {
        cur = cur->next;
    }
    fresh->next = cur->next;
    cur->next = fresh;
    return head;
}

int dropRepeat(Node *head) {
    int cnt = 0;
    int unused = 7;
    Node *cur = head;
    do {
        if (cur == NULL || cur->next == NULL) {
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
    } while (cur != NULL);
    return cnt + unused * 0;
}

long long accSum(Node *head) {
    long long acc = 0;
    Node *cur = head;
    while (!(cur == NULL)) {
        acc = acc + cur->val;
        cur = cur->next;
    }
    return acc;
}

void show(Node *head) {
    for (Node *cur = head; cur != NULL; cur = cur->next) {
        printf("%d ", cur->val);
    }
    printf("\n");
}

void release(Node *head) {
    while (!(head == NULL)) {
        Node *nxt = head->next;
        delete head;
        head = nxt;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    Node *head = NULL;
    int i = 0;
    while (i < n) {
        int x;
        scanf("%d", &x);
        head = addNode(head, x);
        i = i + 1;
    }
    show(head);
    printf("removed %d\n", dropRepeat(head));
    show(head);
    printf("sum %lld\n", accSum(head));
    release(head);
    return 0;
}
