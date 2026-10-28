#include <string>
using namespace std;
bool isPalindrome(const string& text) {
    int startIdx = 0, endIdx = (int)text.length() - 1;
    while (startIdx < endIdx) {
        if (text[startIdx] != text[endIdx]) return false;
        startIdx++; endIdx--;
    }
    return true;
}
