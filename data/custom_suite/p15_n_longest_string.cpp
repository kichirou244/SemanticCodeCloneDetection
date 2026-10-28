#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) { // chuoi dai nhat
    if (strs.empty()) return "";
    string longest = strs[0];
    for (size_t i = 1; i < strs.size(); i++) {
        if (strs[i].length() > longest.length()) longest = strs[i];
    }
    return longest;
}
