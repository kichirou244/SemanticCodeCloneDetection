"""Sinh day du 15 bai toan (90 file C/C++) va manifest pairs.jsonl cho data/custom_suite/
Ti le chuan 1:5 (120 Clones : 600 Non-clones = 720 cap danh gia).
"""

from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "custom_suite"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# P1: Bubble Sort
P1 = {
    "p1_base_bubblesort.cpp": """#include <iostream>
using namespace std;
void sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}""",
    "p1_v_rename.cpp": """#include <iostream>
using namespace std;
void sort(int numbers[], int length) {
    for (int step = 0; step < length - 1; step++) {
        for (int idx = 0; idx < length - step - 1; idx++) {
            if (numbers[idx] > numbers[idx + 1]) {
                int swapVal = numbers[idx];
                numbers[idx] = numbers[idx + 1];
                numbers[idx + 1] = swapVal;
            }
        }
    }
}""",
    "p1_v_types.cpp": """#include <vector>
using namespace std;
void sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}""",
    "p1_v_while.cpp": """void sort(int arr[], int n) {
    int i = 0;
    bool swapped = true;
    while (i < n - 1 && swapped) {
        swapped = false;
        int j = 0;
        while (j < n - i - 1) {
            if (arr[j] > arr[j + 1]) {
                int t = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = t;
                swapped = true;
            }
            j++;
        }
        i++;
    }
}""",
    "p1_v_quicksort.cpp": """void quickSort(int a[], int low, int high) {
    if (low < high) {
        int pivot = a[high], i = low - 1;
        for (int j = low; j < high; j++) {
            if (a[j] <= pivot) {
                i++;
                int temp = a[i]; a[i] = a[j]; a[j] = temp;
            }
        }
        int temp = a[i + 1]; a[i + 1] = a[high]; a[high] = temp;
        int p = i + 1;
        quickSort(a, low, p - 1);
        quickSort(a, p + 1, high);
    }
}
void sort(int arr[], int n) {
    quickSort(arr, 0, n - 1);
}""",
    "p1_n_reverse.cpp": """void reverse(int arr[], int n) {
    for (int i = 0; i < n / 2; i++) {
        int j = n - 1 - i;
        if (i < j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
}""",
}

# P2: GCD
P2 = {
    "p2_base_gcd.cpp": """#include <cmath>
int gcd(int a, int b) {
    while (b != 0) {
        int rem = a % b;
        a = b;
        b = rem;
    }
    return std::abs(a);
}""",
    "p2_v_rename.cpp": """#include <cmath>
int gcd(int num1, int num2) {
    while (num2 != 0) {
        int remainder = num1 % num2;
        num1 = num2;
        num2 = remainder;
    }
    return std::abs(num1);
}""",
    "p2_v_types.cpp": """#include <cmath>
long long gcd(long long a, long long b) {
    while (b != 0) {
        long long rem = a % b;
        a = b;
        b = rem;
    }
    return std::abs(a);
}""",
    "p2_v_recursion.cpp": """#include <cmath>
int gcd(int a, int b) {
    if (b == 0) return std::abs(a);
    return gcd(b, a % b);
}""",
    "p2_v_binary_gcd.cpp": """#include <cmath>
int gcd(int u, int v) {
    if (u == 0) return std::abs(v);
    if (v == 0) return std::abs(u);
    u = std::abs(u); v = std::abs(v);
    int shift = 0;
    while (((u | v) & 1) == 0) { u >>= 1; v >>= 1; shift++; }
    while ((u & 1) == 0) u >>= 1;
    do {
        while ((v & 1) == 0) v >>= 1;
        if (u > v) { int temp = u; u = v; v = temp; }
        v -= u;
    } while (v != 0);
    return u << shift;
}""",
    "p2_n_digitsum.cpp": """#include <cmath>
int sumDigits(int n) {
    int sum = 0;
    n = std::abs(n);
    while (n != 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}""",
}

# P3: Graph BFS / DFS
P3 = {
    "p3_base_bfs.cpp": """#include <queue>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);
    while (!q.empty()) {
        int curr = q.front(); q.pop();
        if (curr == target) return true;
        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[curr][neighbor] == 1 && !visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    return false;
}""",
    "p3_v_rename.cpp": """#include <queue>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& adjMat, int src, int dest) {
    int total = adjMat.size();
    vector<bool> seen(total, false);
    queue<int> worklist;
    seen[src] = true;
    worklist.push(src);
    while (!worklist.empty()) {
        int node = worklist.front(); worklist.pop();
        if (node == dest) return true;
        for (int adj = 0; adj < total; adj++) {
            if (adjMat[node][adj] == 1 && !seen[adj]) {
                seen[adj] = true;
                worklist.push(adj);
            }
        }
    }
    return false;
}""",
    "p3_v_types.cpp": """#include <deque>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<char> visited(n, 0);
    deque<int> q;
    visited[start] = 1;
    q.push_back(start);
    while (!q.empty()) {
        int curr = q.front(); q.pop_front();
        if (curr == target) return true;
        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[curr][neighbor] == 1 && !visited[neighbor]) {
                visited[neighbor] = 1;
                q.push_back(neighbor);
            }
        }
    }
    return false;
}""",
    "p3_v_levels.cpp": """#include <queue>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<int> dist(n, -1);
    queue<int> q;
    dist[start] = 0;
    q.push(start);
    while (!q.empty()) {
        int curr = q.front(); q.pop();
        if (curr == target) return true;
        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[curr][neighbor] == 1 && dist[neighbor] == -1) {
                dist[neighbor] = dist[curr] + 1;
                q.push(neighbor);
            }
        }
    }
    return dist[target] != -1;
}""",
    "p3_v_dfs.cpp": """#include <vector>
using namespace std;
bool dfs(const vector<vector<int>>& g, int curr, int target, vector<bool>& visited) {
    if (curr == target) return true;
    visited[curr] = true;
    for (int next = 0; next < (int)g.size(); next++) {
        if (g[curr][next] == 1 && !visited[next]) {
            if (dfs(g, next, target, visited)) return true;
        }
    }
    return false;
}
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    vector<bool> visited(graph.size(), false);
    return dfs(graph, start, target, visited);
}""",
    "p3_n_bipartite.cpp": """#include <queue>
#include <vector>
using namespace std;
bool isBipartite(const vector<vector<int>>& graph, int start, int dummy) {
    int n = graph.size();
    vector<int> color(n, 0);
    queue<int> q;
    color[start] = 1;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v = 0; v < n; v++) {
            if (graph[u][v] == 1) {
                if (color[v] == 0) { color[v] = -color[u]; q.push(v); }
                else if (color[v] == color[u]) return false;
            }
        }
    }
    return true;
}""",
}

# P4: Palindrome
P4 = {
    "p4_base_palindrome.cpp": """#include <string>
using namespace std;
bool isPalindrome(const string& s) {
    int left = 0, right = (int)s.length() - 1;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left++; right--;
    }
    return true;
}""",
    "p4_v_rename.cpp": """#include <string>
using namespace std;
bool isPalindrome(const string& text) {
    int startIdx = 0, endIdx = (int)text.length() - 1;
    while (startIdx < endIdx) {
        if (text[startIdx] != text[endIdx]) return false;
        startIdx++; endIdx--;
    }
    return true;
}""",
    "p4_v_types.cpp": """#include <cstring>
bool isPalindrome(const char* s) {
    int left = 0, right = (int)strlen(s) - 1;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left++; right--;
    }
    return true;
}""",
    "p4_v_clean.cpp": """#include <string>
#include <cctype>
using namespace std;
bool isPalindrome(const string& s) {
    int i = 0, j = (int)s.length() - 1;
    while (i < j) {
        while (i < j && !isalnum(s[i])) i++;
        while (i < j && !isalnum(s[j])) j--;
        if (tolower(s[i]) != tolower(s[j])) return false;
        i++; j--;
    }
    return true;
}""",
    "p4_v_reverse.cpp": """#include <string>
#include <algorithm>
using namespace std;
bool isPalindrome(const string& s) {
    string rev = s;
    reverse(rev.begin(), rev.end());
    return s == rev;
}""",
    "p4_n_reverse_string.cpp": """#include <string>
using namespace std;
string reverseString(string s) {
    int left = 0, right = (int)s.length() - 1;
    while (left < right) {
        char temp = s[left]; s[left] = s[right]; s[right] = temp;
        left++; right--;
    }
    return s;
}""",
}

# P5: Linked List
P5 = {
    "p5_base_list.cpp": """struct Node { int val; Node* next; Node(int v): val(v), next(nullptr) {} };
class List {
    Node* head = nullptr;
public:
    void add(int v) {
        Node* n = new Node(v);
        if (!head) { head = n; return; }
        Node* cur = head;
        while (cur->next) cur = cur->next;
        cur->next = n;
    }
    bool contains(int t) {
        Node* cur = head;
        while (cur) { if (cur->val == t) return true; cur = cur->next; }
        return false;
    }
    int sum() {
        int total = 0; Node* cur = head;
        while (cur) { total += cur->val; cur = cur->next; }
        return total;
    }
};""",
    "p5_v_rename.cpp": """struct Element { int data; Element* nextNode; Element(int d): data(d), nextNode(nullptr) {} };
class Container {
    Element* first = nullptr;
public:
    void append(int item) {
        Element* e = new Element(item);
        if (!first) { first = e; return; }
        Element* p = first;
        while (p->nextNode) p = p->nextNode;
        p->nextNode = e;
    }
    bool hasElement(int q) {
        Element* p = first;
        while (p) { if (p->data == q) return true; p = p->nextNode; }
        return false;
    }
    int calculateSum() {
        int total = 0; Element* p = first;
        while (p) { total += p->data; p = p->nextNode; }
        return total;
    }
};""",
    "p5_v_types.cpp": """template<typename T>
struct Node { T val; Node* next; Node(T v): val(v), next(nullptr) {} };
template<typename T>
class GenericList {
    Node<T>* head = nullptr;
public:
    void add(T v) {
        Node<T>* n = new Node<T>(v);
        if (!head) { head = n; return; }
        Node<T>* cur = head;
        while (cur->next) cur = cur->next;
        cur->next = n;
    }
    bool contains(T t) {
        Node<T>* cur = head;
        while (cur) { if (cur->val == t) return true; cur = cur->next; }
        return false;
    }
    T sum() {
        T total = 0; Node<T>* cur = head;
        while (cur) { total += cur->val; cur = cur->next; }
        return total;
    }
};""",
    "p5_v_while.cpp": """struct Node { int val; Node* next; Node(int v): val(v), next(nullptr) {} };
class SentinelList {
    Node dummy;
public:
    SentinelList(): dummy(0) {}
    void add(int v) {
        Node* cur = &dummy;
        while (cur->next) cur = cur->next;
        cur->next = new Node(v);
    }
    bool contains(int t) {
        Node* cur = dummy.next;
        while (cur) { if (cur->val == t) return true; cur = cur->next; }
        return false;
    }
    int sum() {
        int total = 0; Node* cur = dummy.next;
        while (cur) { total += cur->val; cur = cur->next; }
        return total;
    }
};""",
    "p5_v_dynamic_array.cpp": """#include <vector>
using namespace std;
class DynamicList {
    vector<int> data;
public:
    void add(int v) { data.push_back(v); }
    bool contains(int t) {
        for (int x : data) if (x == t) return true;
        return false;
    }
    int sum() {
        int s = 0; for (int x : data) s += x;
        return s;
    }
};""",
    "p5_n_stack.cpp": """struct Node { int val; Node* next; Node(int v, Node* n): val(v), next(n) {} };
class LinkedStack {
    Node* top = nullptr;
public:
    void push(int v) { top = new Node(v, top); }
    int pop() {
        int v = top->val; Node* old = top; top = top->next; delete old;
        return v;
    }
    int peek() { return top->val; }
};""",
}

# P6: Binary Search
P6 = {
    "p6_base_bsearch.cpp": """int search(int arr[], int n, int target) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}""",
    "p6_v_rename.cpp": """int search(int numbers[], int count, int key) {
    int leftIdx = 0, rightIdx = count - 1;
    while (leftIdx <= rightIdx) {
        int middle = leftIdx + (rightIdx - leftIdx) / 2;
        if (numbers[middle] == key) return middle;
        if (numbers[middle] < key) leftIdx = middle + 1;
        else rightIdx = middle - 1;
    }
    return -1;
}""",
    "p6_v_types.cpp": """#include <vector>
using namespace std;
int search(const vector<long long>& arr, long long target) {
    int low = 0, high = (int)arr.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}""",
    "p6_v_recursion.cpp": """int bsearch(int arr[], int target, int low, int high) {
    if (low > high) return -1;
    int mid = low + (high - low) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] < target) return bsearch(arr, target, mid + 1, high);
    return bsearch(arr, target, low, mid - 1);
}
int search(int arr[], int n, int target) {
    return bsearch(arr, target, 0, n - 1);
}""",
    "p6_v_exponential.cpp": """#include <algorithm>
using namespace std;
int search(int arr[], int n, int target) {
    if (n == 0) return -1;
    if (arr[0] == target) return 0;
    int i = 1;
    while (i < n && arr[i] <= target) i *= 2;
    int low = i / 2, high = min(i, n - 1);
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}""",
    "p6_n_linear_count.cpp": """int countOccurrences(int arr[], int n, int target) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) count++;
    }
    return count;
}""",
}

# P7: Fibonacci
P7 = {
    "p7_base_fib.cpp": """long long getFib(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long temp = a + b; a = b; b = temp;
    }
    return b;
}""",
    "p7_v_rename.cpp": """long long getFib(int num) {
    if (num <= 1) return num;
    long long first = 0, second = 1;
    for (int step = 2; step <= num; step++) {
        long long nextVal = first + second; first = second; second = nextVal;
    }
    return second;
}""",
    "p7_v_types.cpp": """double getFib(int n) {
    if (n <= 1) return (double)n;
    double a = 0.0, b = 1.0;
    for (int i = 2; i <= n; i++) {
        double temp = a + b; a = b; b = temp;
    }
    return b;
}""",
    "p7_v_recursion.cpp": """#include <vector>
using namespace std;
long long fibMemo(int n, vector<long long>& memo) {
    if (n <= 1) return n;
    if (memo[n] != 0) return memo[n];
    return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
}
long long getFib(int n) {
    vector<long long> memo(n + 2, 0);
    return fibMemo(n, memo);
}""",
    "p7_v_matrix.cpp": """void multiply(long long F[2][2], long long M[2][2]) {
    long long x = F[0][0]*M[0][0] + F[0][1]*M[1][0];
    long long y = F[0][0]*M[0][1] + F[0][1]*M[1][1];
    long long z = F[1][0]*M[0][0] + F[1][1]*M[1][0];
    long long w = F[1][0]*M[0][1] + F[1][1]*M[1][1];
    F[0][0] = x; F[0][1] = y; F[1][0] = z; F[1][1] = w;
}
void power(long long F[2][2], int n) {
    if (n == 0 || n == 1) return;
    long long M[2][2] = {{1, 1}, {1, 0}};
    power(F, n / 2);
    multiply(F, F);
    if (n % 2 != 0) multiply(F, M);
}
long long getFib(int n) {
    if (n <= 1) return n;
    long long F[2][2] = {{1, 1}, {1, 0}};
    power(F, n - 1);
    return F[0][0];
}""",
    "p7_n_factorial.cpp": """long long getFactorial(int n) {
    if (n <= 1) return 1;
    long long res = 1;
    for (int i = 2; i <= n; i++) res *= i;
    return res;
}""",
}

# P8: Prime Check
P8 = {
    "p8_base_prime.cpp": """bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}""",
    "p8_v_rename.cpp": """bool isPrime(int number) {
    if (number <= 1) return false;
    if (number <= 3) return true;
    if (number % 2 == 0 || number % 3 == 0) return false;
    for (int factor = 5; factor * factor <= number; factor += 6) {
        if (number % factor == 0 || number % (factor + 2) == 0) return false;
    }
    return true;
}""",
    "p8_v_types.cpp": """bool isPrime(long long n) {
    if (n <= 1LL) return false;
    if (n <= 3LL) return true;
    if (n % 2LL == 0 || n % 3LL == 0) return false;
    for (long long i = 5LL; i * i <= n; i += 6LL) {
        if (n % i == 0 || n % (i + 2LL) == 0) return false;
    }
    return true;
}""",
    "p8_v_while.cpp": """bool isPrime(int n) {
    if (n <= 1) return false;
    int d = 2;
    while (d * d <= n) {
        if (n % d == 0) return false;
        d++;
    }
    return true;
}""",
    "p8_v_sieve.cpp": """#include <cmath>
bool isPrime(int n) {
    if (n < 2) return false;
    int limit = (int)sqrt(n);
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0) return false;
    }
    return true;
}""",
    "p8_n_square.cpp": """#include <cmath>
bool isPerfectSquare(int n) {
    if (n < 0) return false;
    int root = (int)sqrt(n);
    return root * root == n;
}""",
}

# P9: Matrix Multiplication
P9 = {
    "p9_base_matmul.cpp": """#include <vector>
using namespace std;
vector<vector<int>> multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int r1 = A.size(), c1 = A[0].size(), c2 = B[0].size();
    vector<vector<int>> C(r1, vector<int>(c2, 0));
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}""",
    "p9_v_rename.cpp": """#include <vector>
using namespace std;
vector<vector<int>> multiply(const vector<vector<int>>& firstMat, const vector<vector<int>>& secondMat) {
    int rowsA = firstMat.size(), colsA = firstMat[0].size(), colsB = secondMat[0].size();
    vector<vector<int>> res(rowsA, vector<int>(colsB, 0));
    for (int r = 0; r < rowsA; r++) {
        for (int c = 0; c < colsB; c++) {
            for (int p = 0; p < colsA; p++) {
                res[r][c] += firstMat[r][p] * secondMat[p][c];
            }
        }
    }
    return res;
}""",
    "p9_v_types.cpp": """#include <vector>
using namespace std;
vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int r1 = A.size(), c1 = A[0].size(), c2 = B[0].size();
    vector<vector<double>> C(r1, vector<double>(c2, 0.0));
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}""",
    "p9_v_while.cpp": """#include <vector>
using namespace std;
vector<vector<int>> multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int r1 = A.size(), c1 = A[0].size(), c2 = B[0].size();
    vector<vector<int>> C(r1, vector<int>(c2, 0));
    int i = 0;
    while (i < r1) {
        int j = 0;
        while (j < c2) {
            int k = 0;
            while (k < c1) { C[i][j] += A[i][k] * B[k][j]; k++; }
            j++;
        }
        i++;
    }
    return C;
}""",
    "p9_v_flat.cpp": """#include <vector>
using namespace std;
vector<int> multiply(const vector<int>& A, const vector<int>& B, int r1, int c1, int c2) {
    vector<int> C(r1 * c2, 0);
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            int sum = 0;
            for (int k = 0; k < c1; k++) sum += A[i * c1 + k] * B[k * c2 + j];
            C[i * c2 + j] = sum;
        }
    }
    return C;
}""",
    "p9_n_transpose.cpp": """#include <vector>
using namespace std;
vector<vector<int>> transpose(const vector<vector<int>>& A) {
    int rows = A.size(), cols = A[0].size();
    vector<vector<int>> T(cols, vector<int>(rows, 0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) T[j][i] = A[i][j];
    }
    return T;
}""",
}

# P10: Anagram
P10 = {
    "p10_base_anagram.cpp": """#include <string>
#include <algorithm>
using namespace std;
bool isAnagram(string s1, string s2) {
    if (s1.length() != s2.length()) return false;
    sort(s1.begin(), s1.end());
    sort(s2.begin(), s2.end());
    return s1 == s2;
}""",
    "p10_v_rename.cpp": """#include <string>
#include <algorithm>
using namespace std;
bool isAnagram(string firstStr, string secondStr) {
    if (firstStr.length() != secondStr.length()) return false;
    sort(firstStr.begin(), firstStr.end());
    sort(secondStr.begin(), secondStr.end());
    return firstStr == secondStr;
}""",
    "p10_v_types.cpp": """#include <string>
#include <algorithm>
#include <cctype>
using namespace std;
bool isAnagram(string s1, string s2) {
    for (char& c : s1) c = tolower(c);
    for (char& c : s2) c = tolower(c);
    if (s1.length() != s2.length()) return false;
    sort(s1.begin(), s1.end());
    sort(s2.begin(), s2.end());
    return s1 == s2;
}""",
    "p10_v_clean.cpp": """#include <string>
#include <algorithm>
using namespace std;
bool isAnagram(string s1, string s2) {
    s1.erase(remove_if(s1.begin(), s1.end(), ::isspace), s1.end());
    s2.erase(remove_if(s2.begin(), s2.end(), ::isspace), s2.end());
    if (s1.length() != s2.length()) return false;
    sort(s1.begin(), s1.end());
    sort(s2.begin(), s2.end());
    return s1 == s2;
}""",
    "p10_v_frequency.cpp": """#include <string>
#include <vector>
using namespace std;
bool isAnagram(const string& s1, const string& s2) {
    if (s1.length() != s2.length()) return false;
    vector<int> counts(256, 0);
    for (int i = 0; i < (int)s1.length(); i++) {
        counts[(unsigned char)s1[i]]++;
        counts[(unsigned char)s2[i]]--;
    }
    for (int c : counts) if (c != 0) return false;
    return true;
}""",
    "p10_n_substring.cpp": """#include <string>
using namespace std;
bool isSubstring(const string& s1, const string& s2) {
    return s1.find(s2) != string::npos;
}""",
}

# P11: Kadane
P11 = {
    "p11_base_kadane.cpp": """#include <vector>
#include <algorithm>
using namespace std;
int maxSubArray(const vector<int>& nums) {
    int maxSoFar = nums[0], currMax = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
        currMax = max(nums[i], currMax + nums[i]);
        maxSoFar = max(maxSoFar, currMax);
    }
    return maxSoFar;
}""",
    "p11_v_rename.cpp": """#include <vector>
#include <algorithm>
using namespace std;
int maxSubArray(const vector<int>& numbers) {
    int bestSum = numbers[0], runningSum = numbers[0];
    for (size_t idx = 1; idx < numbers.size(); idx++) {
        runningSum = max(numbers[idx], runningSum + numbers[idx]);
        bestSum = max(bestSum, runningSum);
    }
    return bestSum;
}""",
    "p11_v_types.cpp": """#include <vector>
#include <algorithm>
using namespace std;
long long maxSubArray(const vector<long long>& nums) {
    long long maxSoFar = nums[0], currMax = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
        currMax = max(nums[i], currMax + nums[i]);
        maxSoFar = max(maxSoFar, currMax);
    }
    return maxSoFar;
}""",
    "p11_v_while.cpp": """#include <vector>
using namespace std;
int maxSubArray(const vector<int>& nums) {
    int maxSoFar = nums[0], currMax = nums[0];
    size_t i = 1;
    while (i < nums.size()) {
        if (currMax < 0) currMax = nums[i];
        else currMax += nums[i];
        if (currMax > maxSoFar) maxSoFar = currMax;
        i++;
    }
    return maxSoFar;
}""",
    "p11_v_divide_conquer.cpp": """#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
int crossSum(const vector<int>& nums, int l, int m, int r) {
    int leftSum = INT_MIN, sum = 0;
    for (int i = m; i >= l; i--) { sum += nums[i]; leftSum = max(leftSum, sum); }
    int rightSum = INT_MIN; sum = 0;
    for (int i = m + 1; i <= r; i++) { sum += nums[i]; rightSum = max(rightSum, sum); }
    return leftSum + rightSum;
}
int solve(const vector<int>& nums, int l, int r) {
    if (l == r) return nums[l];
    int m = l + (r - l) / 2;
    int leftMax = solve(nums, l, m);
    int rightMax = solve(nums, m + 1, r);
    int crossMax = crossSum(nums, l, m, r);
    return max({leftMax, rightMax, crossMax});
}
int maxSubArray(const vector<int>& nums) {
    return solve(nums, 0, (int)nums.size() - 1);
}""",
    "p11_n_min_subarray.cpp": """#include <vector>
#include <algorithm>
using namespace std;
int minSubArray(const vector<int>& nums) {
    int minSoFar = nums[0], currMin = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
        currMin = min(nums[i], currMin + nums[i]);
        minSoFar = min(minSoFar, currMin);
    }
    return minSoFar;
}""",
}

# P12: Merge Sorted
P12 = {
    "p12_base_merge.cpp": """#include <vector>
using namespace std;
vector<int> merge(const vector<int>& a, const vector<int>& b) {
    vector<int> res(a.size() + b.size());
    size_t i = 0, j = 0, k = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) res[k++] = a[i++];
        else res[k++] = b[j++];
    }
    while (i < a.size()) res[k++] = a[i++];
    while (j < b.size()) res[k++] = b[j++];
    return res;
}""",
    "p12_v_rename.cpp": """#include <vector>
using namespace std;
vector<int> merge(const vector<int>& leftArr, const vector<int>& rightArr) {
    vector<int> merged(leftArr.size() + rightArr.size());
    size_t p1 = 0, p2 = 0, idx = 0;
    while (p1 < leftArr.size() && p2 < rightArr.size()) {
        if (leftArr[p1] <= rightArr[p2]) merged[idx++] = leftArr[p1++];
        else merged[idx++] = rightArr[p2++];
    }
    while (p1 < leftArr.size()) merged[idx++] = leftArr[p1++];
    while (p2 < rightArr.size()) merged[idx++] = rightArr[p2++];
    return merged;
}""",
    "p12_v_types.cpp": """#include <vector>
using namespace std;
vector<long long> merge(const vector<long long>& a, const vector<long long>& b) {
    vector<long long> res(a.size() + b.size());
    size_t i = 0, j = 0, k = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) res[k++] = a[i++];
        else res[k++] = b[j++];
    }
    while (i < a.size()) res[k++] = a[i++];
    while (j < b.size()) res[k++] = b[j++];
    return res;
}""",
    "p12_v_while.cpp": """#include <vector>
using namespace std;
vector<int> merge(const vector<int>& a, const vector<int>& b) {
    size_t n = a.size(), m = b.size();
    vector<int> res(n + m);
    size_t i = 0, j = 0;
    for (size_t k = 0; k < n + m; k++) {
        if (i >= n) res[k] = b[j++];
        else if (j >= m) res[k] = a[i++];
        else if (a[i] <= b[j]) res[k] = a[i++];
        else res[k] = b[j++];
    }
    return res;
}""",
    "p12_v_queue.cpp": """#include <vector>
#include <queue>
using namespace std;
vector<int> merge(const vector<int>& a, const vector<int>& b) {
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int x : a) pq.push(x);
    for (int y : b) pq.push(y);
    vector<int> res;
    while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
    return res;
}""",
    "p12_n_intersection.cpp": """#include <vector>
using namespace std;
vector<int> intersect(const vector<int>& a, const vector<int>& b) {
    vector<int> common;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { common.push_back(a[i]); i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    return common;
}""",
}

# P13: Queue Data Structure
P13 = {
    "p13_base_queue.cpp": """class ArrayQueue {
    int data[100];
    int front = 0, rear = 0, count = 0;
public:
    void enqueue(int x) { data[rear] = x; rear = (rear + 1) % 100; count++; }
    int dequeue() { int val = data[front]; front = (front + 1) % 100; count--; return val; }
    bool isEmpty() { return count == 0; }
};""",
    "p13_v_rename.cpp": """class CircularQueue {
    int buffer[100];
    int head = 0, tail = 0, size = 0;
public:
    void enqueue(int item) { buffer[tail] = item; tail = (tail + 1) % 100; size++; }
    int dequeue() { int item = buffer[head]; head = (head + 1) % 100; size--; return item; }
    bool isEmpty() { return size == 0; }
};""",
    "p13_v_types.cpp": """template<typename T>
class GenericQueue {
    T data[100];
    int front = 0, rear = 0, count = 0;
public:
    void enqueue(T x) { data[rear] = x; rear = (rear + 1) % 100; count++; }
    T dequeue() { T val = data[front]; front = (front + 1) % 100; count--; return val; }
    bool isEmpty() { return count == 0; }
};""",
    "p13_v_while.cpp": """#include <vector>
using namespace std;
class ResizableQueue {
    vector<int> data = vector<int>(4);
    int front = 0, rear = 0, size = 0;
public:
    void enqueue(int x) {
        if (size == (int)data.size()) {
            vector<int> next(data.size() * 2);
            for (int i = 0; i < size; i++) next[i] = data[(front + i) % data.size()];
            data = next; front = 0; rear = size;
        }
        data[rear] = x; rear = (rear + 1) % data.size(); size++;
    }
    int dequeue() { int val = data[front]; front = (front + 1) % data.size(); size--; return val; }
    bool isEmpty() { return size == 0; }
};""",
    "p13_v_linked_queue.cpp": """struct Node { int val; Node* next; Node(int v): val(v), next(nullptr) {} };
class LinkedQueue {
    Node* head = nullptr; Node* tail = nullptr;
public:
    void enqueue(int x) {
        Node* n = new Node(x);
        if (!tail) { head = tail = n; }
        else { tail->next = n; tail = n; }
    }
    int dequeue() {
        int v = head->val; Node* old = head; head = head->next;
        if (!head) tail = nullptr;
        delete old; return v;
    }
    bool isEmpty() { return head == nullptr; }
};""",
    "p13_n_stack.cpp": """class ArrayStack {
    int data[100];
    int top = -1;
public:
    void enqueue(int x) { data[++top] = x; } // hanh vi la push
    int dequeue() { return data[top--]; }    // hanh vi la pop
    bool isEmpty() { return top == -1; }
};""",
}

# P14: Power Function
P14 = {
    "p14_base_power.cpp": """double power(double x, int n) {
    if (n == 0) return 1.0;
    long long N = n;
    if (N < 0) { x = 1.0 / x; N = -N; }
    double ans = 1.0;
    for (long long i = 0; i < N; i++) ans *= x;
    return ans;
}""",
    "p14_v_rename.cpp": """double power(double base, int exponent) {
    if (exponent == 0) return 1.0;
    long long exp = exponent;
    if (exp < 0) { base = 1.0 / base; exp = -exp; }
    double result = 1.0;
    for (long long step = 0; step < exp; step++) result *= base;
    return result;
}""",
    "p14_v_types.cpp": """long long power(long long base, int exp) {
    long long res = 1LL;
    for (int i = 0; i < exp; i++) res *= base;
    return res;
}""",
    "p14_v_while.cpp": """double power(double x, int n) {
    if (n == 0) return 1.0;
    long long N = n;
    if (N < 0) { x = 1.0 / x; N = -N; }
    double ans = 1.0;
    while (N > 0) { ans *= x; N--; }
    return ans;
}""",
    "p14_v_binary_exp.cpp": """double power(double x, int n) {
    long long N = n;
    if (N < 0) { x = 1.0 / x; N = -N; }
    double ans = 1.0, current = x;
    for (long long i = N; i > 0; i /= 2) {
        if (i % 2 == 1) ans *= current;
        current *= current;
    }
    return ans;
}""",
    "p14_n_sqrt.cpp": """double power(double x, int dummy) { // Newton sqrt
    if (x < 0) return -1.0;
    double guess = x / 2.0;
    for (int i = 0; i < 20; i++) guess = (guess + x / guess) / 2.0;
    return guess;
}""",
}

# P15: Longest Common Prefix
P15 = {
    "p15_base_lcp.cpp": """#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    string prefix = strs[0];
    for (size_t i = 1; i < strs.size(); i++) {
        while (strs[i].find(prefix) != 0) {
            prefix = prefix.substr(0, prefix.length() - 1);
            if (prefix.empty()) return "";
        }
    }
    return prefix;
}""",
    "p15_v_rename.cpp": """#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& words) {
    if (words.empty()) return "";
    string commonPref = words[0];
    for (size_t idx = 1; idx < words.size(); idx++) {
        while (words[idx].find(commonPref) != 0) {
            commonPref = commonPref.substr(0, commonPref.length() - 1);
            if (commonPref.empty()) return "";
        }
    }
    return commonPref;
}""",
    "p15_v_types.cpp": """#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    string prefix = strs[0];
    for (size_t i = 1; i < strs.size(); i++) {
        while (strs[i].rfind(prefix, 0) != 0) {
            prefix.pop_back();
            if (prefix.empty()) return "";
        }
    }
    return prefix;
}""",
    "p15_v_while.cpp": """#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    string prefix = strs[0];
    size_t i = 1;
    while (i < strs.size()) {
        while (strs[i].find(prefix) != 0) {
            prefix = prefix.substr(0, prefix.length() - 1);
            if (prefix.empty()) return "";
        }
        i++;
    }
    return prefix;
}""",
    "p15_v_vertical.cpp": """#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    for (size_t i = 0; i < strs[0].length(); i++) {
        char c = strs[0][i];
        for (size_t j = 1; j < strs.size(); j++) {
            if (i == strs[j].length() || strs[j][i] != c) {
                return strs[0].substr(0, i);
            }
        }
    }
    return strs[0];
}""",
    "p15_n_longest_string.cpp": """#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) { // chuoi dai nhat
    if (strs.empty()) return "";
    string longest = strs[0];
    for (size_t i = 1; i < strs.size(); i++) {
        if (strs[i].length() > longest.length()) longest = strs[i];
    }
    return longest;
}""",
}

ALL_DICTS = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15]


def write_all_files():
    for d in ALL_DICTS:
        for fname, content in d.items():
            path = DATA_DIR / fname
            path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Tong so file C/C++ trong {DATA_DIR}: {len(list(DATA_DIR.glob('*.cpp')))}")


def generate_manifest_1_to_5():
    problems = [f"p{i}" for i in range(1, 16)]
    clones = []
    nonclone_traps = []
    nonclone_cross = []

    prob_files: dict[str, dict[str, str]] = {}
    for p in problems:
        flist = list(DATA_DIR.glob(f"{p}_*.cpp"))
        base = next(f for f in flist if "_base_" in f.name).name
        v_t1 = next(f for f in flist if "_v_rename" in f.name).name
        v_t2 = next(f for f in flist if "_v_types" in f.name).name
        v_t3 = next(f for f in flist if "_v_while" in f.name or "_v_recursion" in f.name or "_v_levels" in f.name or "_v_clean" in f.name).name
        v_t4 = next(f for f in flist if f.name not in [base, v_t1, v_t2, v_t3] and "_n_" not in f.name).name
        n_hard = next(f for f in flist if "_n_" in f.name).name
        prob_files[p] = {
            "base": base,
            "v_t1": v_t1,
            "v_t2": v_t2,
            "v_t3": v_t3,
            "v_t4": v_t4,
            "n_hard": n_hard,
        }

    # Sinh toan bo cap clone
    for p, fs in prob_files.items():
        clones.append({"pair_id": f"{p}_t1", "a": fs["base"], "b": fs["v_t1"], "label": 1, "category": "clone_type1", "note": f"{p} rename"})
        clones.append({"pair_id": f"{p}_t2", "a": fs["base"], "b": fs["v_t2"], "label": 1, "category": "clone_type2", "note": f"{p} types"})
        clones.append({"pair_id": f"{p}_t1_t2", "a": fs["v_t1"], "b": fs["v_t2"], "label": 1, "category": "clone_type2", "note": f"{p} t1 vs t2"})
        clones.append({"pair_id": f"{p}_t3", "a": fs["base"], "b": fs["v_t3"], "label": 1, "category": "clone_type3", "note": f"{p} control"})
        clones.append({"pair_id": f"{p}_t1_t3", "a": fs["v_t1"], "b": fs["v_t3"], "label": 1, "category": "clone_type3", "note": f"{p} t1 vs t3"})
        clones.append({"pair_id": f"{p}_t4", "a": fs["base"], "b": fs["v_t4"], "label": 1, "category": "clone_type4", "note": f"{p} semantic different algorithm"})
        clones.append({"pair_id": f"{p}_t2_t4", "a": fs["v_t2"], "b": fs["v_t4"], "label": 1, "category": "clone_type4", "note": f"{p} t2 vs t4"})
        clones.append({"pair_id": f"{p}_t3_t4", "a": fs["v_t3"], "b": fs["v_t4"], "label": 1, "category": "clone_type4", "note": f"{p} t3 vs t4"})

    # Sinh non-clone trap (cung skeleton)
    for p, fs in prob_files.items():
        for variant_key in ["base", "v_t1", "v_t2", "v_t3", "v_t4"]:
            nonclone_traps.append({
                "pair_id": f"{p}_trap_{variant_key}",
                "a": fs[variant_key],
                "b": fs["n_hard"],
                "label": 0,
                "category": "nonclone_trap",
                "note": f"{p} {variant_key} vs skeleton trap",
            })

    # Sinh non-clone cross (khac bai)
    for i in range(len(problems)):
        for j in range(i + 1, len(problems)):
            pi, pj = problems[i], problems[j]
            fi, fj = prob_files[pi], prob_files[pj]
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_bb", "a": fi["base"], "b": fj["base"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t1", "a": fi["v_t1"], "b": fj["v_t1"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t2", "a": fi["v_t2"], "b": fj["v_t2"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t3", "a": fi["v_t3"], "b": fj["v_t3"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t4", "a": fi["v_t4"], "b": fj["v_t4"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_nh", "a": fi["n_hard"], "b": fj["n_hard"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})

    random.seed(42)
    # Target: EXACT 120 Clones : 600 Non-clones (Total = 720, Ratio 1:5)
    random.shuffle(clones)
    selected_clones = clones[:120]

    random.shuffle(nonclone_traps)
    random.shuffle(nonclone_cross)
    # Lay 75 traps va 525 cross = 600 non-clones
    selected_traps = nonclone_traps[:75]
    selected_cross = nonclone_cross[:525]
    selected_nonclones = selected_traps + selected_cross

    final_pairs = selected_clones + selected_nonclones
    random.shuffle(final_pairs)

    manifest_path = DATA_DIR / "pairs.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Manifest {manifest_path}:")
    print(f"  - Tong so cap: {len(final_pairs)}")
    print(f"  - Clones (label 1): {len(selected_clones)}")
    print(f"  - Non-clones (label 0): {len(selected_nonclones)}")
    print(f"  - Ti le Clone : Non-clone = {len(selected_clones)} : {len(selected_nonclones)} = 1 : {len(selected_nonclones)//len(selected_clones)}")


def main():
    write_all_files()
    generate_manifest_1_to_5()


if __name__ == "__main__":
    main()
