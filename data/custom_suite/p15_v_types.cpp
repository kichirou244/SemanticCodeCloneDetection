#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    string prefix = strs[0];
    for (size_t i = 1; i < strs.size(); i++) {
        while (strs[i].rfind(prefix, 0) != 0) {
            prefix.pop_back();
            if (prefix.empty()) return "";
        }
    }
    return prefix;
}
