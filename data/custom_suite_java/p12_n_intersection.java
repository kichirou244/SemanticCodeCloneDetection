import java.util.ArrayList;
import java.util.List;

public class ArrayIntersection {
    public static int[] intersect(int[] a, int[] b) {
        List<Integer> common = new ArrayList<>();
        int i = 0, j = 0;
        while (i < a.length && j < b.length) {
            if (a[i] == b[j]) { common.add(a[i]); i++; j++; }
            else if (a[i] < b[j]) i++;
            else j++;
        }
        int[] res = new int[common.size()];
        for (int k = 0; k < res.length; k++) res[k] = common.get(k);
        return res;
    }
}
