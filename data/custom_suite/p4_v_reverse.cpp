#include <string>
#include <algorithm>
using namespace std;
bool isPalindrome(const string& s) {
    string rev = s;
    reverse(rev.begin(), rev.end());
    return s == rev;
}
