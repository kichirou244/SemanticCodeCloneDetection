public class MatrixMultiplicationDouble {
    public static double[][] multiply(double[][] A, double[][] B) {
        int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
        double[][] C = new double[r1][c2];
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
}
