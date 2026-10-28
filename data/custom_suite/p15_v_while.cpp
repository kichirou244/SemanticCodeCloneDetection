#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    string prefix = strs[0];
    size_t i = 1;
    while (i < strs.size()) {
        while (strs[i].find(prefix) != 0) {
            prefix = prefix.substr(0, prefix.length() - 1);
            if (prefix.empty()) return "";
        }
        i++;
    }
    return prefix;
}
