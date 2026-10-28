public class MatrixTranspose {
    public static int[][] transpose(int[][] A) {
        int rows = A.length, cols = A[0].length;
        int[][] T = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }
}
