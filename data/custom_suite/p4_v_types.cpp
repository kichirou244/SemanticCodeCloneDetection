#include <cstring>
bool isPalindrome(const char* s) {
    int left = 0, right = (int)strlen(s) - 1;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left++; right--;
    }
    return true;
}
