#include <string>
using namespace std;
string reverseString(string s) {
    int left = 0, right = (int)s.length() - 1;
    while (left < right) {
        char temp = s[left]; s[left] = s[right]; s[right] = temp;
        left++; right--;
    }
    return s;
}
