public class MatrixMulWhile {
    public static int[][] multiply(int[][] A, int[][] B) {
        int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
        int[][] C = new int[r1][c2];
        int i = 0;
        while (i < r1) {
            int j = 0;
            while (j < c2) {
                int k = 0;
                while (k < c1) {
                    C[i][j] += A[i][k] * B[k][j];
                    k++;
                }
                j++;
            }
            i++;
        }
        return C;
    }
}
