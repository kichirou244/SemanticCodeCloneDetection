#include <string>
#include <vector>
using namespace std;
string longestCommonPrefix(const vector<string>& words) {
    if (words.empty()) return "";
    string commonPref = words[0];
    for (size_t idx = 1; idx < words.size(); idx++) {
        while (words[idx].find(commonPref) != 0) {
            commonPref = commonPref.substr(0, commonPref.length() - 1);
            if (commonPref.empty()) return "";
        }
    }
    return commonPref;
}
