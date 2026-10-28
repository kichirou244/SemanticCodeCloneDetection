#include <cmath>
bool isPrime(int n) {
    if (n < 2) return false;
    int limit = (int)sqrt(n);
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0) return false;
    }
    return true;
}
