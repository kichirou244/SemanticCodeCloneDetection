#include <string>
using namespace std;
bool isSubstring(const string& s1, const string& s2) {
    return s1.find(s2) != string::npos;
}
