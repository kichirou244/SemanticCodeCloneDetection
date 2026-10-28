#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    string prefix = strs[0];
    for (size_t i = 1; i < strs.size(); i++) {
        while (strs[i].find(prefix) != 0) {
            prefix = prefix.substr(0, prefix.length() - 1);
            if (prefix.empty()) return "";
        }
    }
    return prefix;
}
