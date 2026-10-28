#include <cmath>
int gcd(int num1, int num2) {
    while (num2 != 0) {
        int remainder = num1 % num2;
        num1 = num2;
        num2 = remainder;
    }
    return std::abs(num1);
}
