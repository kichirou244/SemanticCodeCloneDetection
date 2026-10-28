#include <cmath>
int sumDigits(int n) {
    int sum = 0;
    n = std::abs(n);
    while (n != 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}
