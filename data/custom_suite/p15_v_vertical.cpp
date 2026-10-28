#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& strs) {
    if (strs.empty()) return "";
    for (size_t i = 0; i < strs[0].length(); i++) {
        char c = strs[0][i];
        for (size_t j = 1; j < strs.size(); j++) {
            if (i == strs[j].length() || strs[j][i] != c) {
                return strs[0].substr(0, i);
            }
        }
    }
    return strs[0];
}
