#include <string>
#include <algorithm>
using namespace std;
bool isAnagram(string firstStr, string secondStr) {
    if (firstStr.length() != secondStr.length()) return false;
    sort(firstStr.begin(), firstStr.end());
    sort(secondStr.begin(), secondStr.end());
    return firstStr == secondStr;
}
