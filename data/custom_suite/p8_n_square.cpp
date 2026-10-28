#include <cmath>
bool isPerfectSquare(int n) {
    if (n < 0) return false;
    int root = (int)sqrt(n);
    return root * root == n;
}
