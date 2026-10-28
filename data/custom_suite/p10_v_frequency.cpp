#include <string>
#include <vector>
using namespace std;
bool isAnagram(const string& s1, const string& s2) {
    if (s1.length() != s2.length()) return false;
    vector<int> counts(256, 0);
    for (int i = 0; i < (int)s1.length(); i++) {
        counts[(unsigned char)s1[i]]++;
        counts[(unsigned char)s2[i]]--;
    }
    for (int c : counts) if (c != 0) return false;
    return true;
}
