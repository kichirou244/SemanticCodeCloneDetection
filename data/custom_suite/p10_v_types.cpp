#include <string>
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
}
