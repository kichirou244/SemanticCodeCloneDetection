#include <bits/stdc++.h>
using namespace std;
int main()
{
	long long a, b;
	cin >> a >> b;
	if (a > 0 and b > 0)
	{
		long long chenhlech;
		chenhlech = abs(a - b);
		if (a > b)
		{
			cout << "deficit"<<endl;
		}
		else if (b > a)
		{
			cout << "profit"<<endl;
		}
		else if (a == b)
		{
			cout << "break-even"<<endl;
		}
		cout << chenhlech;
	}
}