package test;

import math.Matrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {

    private Matrix m1; // 2x2
    private Matrix m2; // 2x2
    private Matrix m3; // 2x3

    @BeforeEach
    void setUp() {
        m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
        m2 = new Matrix(new double[][]{{5, 6}, {7, 8}});
        m3 = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
    }

    @Test
    @DisplayName("Constructor should create a matrix with given dimensions and zero values")
    void testConstructorWithDimensions() {
        Matrix m = new Matrix(2, 3);
        assertEquals(2, m.rows());
        assertEquals(3, m.cols());
        assertEquals(0.0, m.get(1, 2));
    }

    @Test
    @DisplayName("Constructor should create a matrix from a 2D array")
    void testConstructorWithData() {
        double[][] data = {{1.1, 2.2}, {3.3, 4.4}};
        Matrix m = new Matrix(data);
        assertEquals(2, m.rows());
        assertEquals(2, m.cols());
        assertEquals(3.3, m.get(1, 0));
        // Ensure data is copied, not referenced
        data[1][0] = 9.9;
        assertEquals(3.3, m.get(1, 0));
    }

    @Test
    @DisplayName("Rand should create a matrix with random values")
    void testRand() {
        Matrix r1 = Matrix.Rand(2, 2, 123);
        Matrix r2 = Matrix.Rand(2, 2, 123);
        assertEquals(r1, r2, "Matrices with the same seed should be equal");
    }

    @Test
    @DisplayName("apply should apply a function to all elements")
    void testApply() {
        Matrix result = m1.apply(x -> x * 2);
        Matrix expected = new Matrix(new double[][]{{2, 4}, {6, 8}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("mult by scalar should multiply all elements by the scalar")
    void testMultScalar() {
        Matrix result = m1.mult(2.5);
        Matrix expected = new Matrix(new double[][]{{2.5, 5.0}, {7.5, 10.0}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("add scalar should add a scalar to all elements")
    void testAddScalar() {
        Matrix result = m1.add(10);
        Matrix expected = new Matrix(new double[][]{{11, 12}, {13, 14}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("subFromScalar should subtract each element from a scalar")
    void testSubFromScalar() {
        Matrix result = m1.subFromScalar(10);
        Matrix expected = new Matrix(new double[][]{{9, 8}, {7, 6}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("add should perform element-wise addition")
    void testAddMatrix() {
        Matrix result = m1.add(m2);
        Matrix expected = new Matrix(new double[][]{{6, 8}, {10, 12}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("add should throw exception for incompatible sizes")
    void testAddMatrixIncompatible() {
        assertThrows(IllegalArgumentException.class, () -> m1.add(m3));
    }

    @Test
    @DisplayName("mult should perform element-wise multiplication")
    void testMultMatrix() {
        Matrix result = m1.mult(m2);
        Matrix expected = new Matrix(new double[][]{{5, 12}, {21, 32}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("sub should perform element-wise subtraction")
    void testSubMatrix() {
        Matrix result = m1.sub(m2);
        Matrix expected = new Matrix(new double[][]{{-4, -4}, {-4, -4}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("sum should return the sum of all elements")
    void testSum() {
        assertEquals(10.0, m1.sum(), 1e-9);
        assertEquals(26.0, m2.sum(), 1e-9);
    }

    @Test
    @DisplayName("dot product should be calculated correctly")
    void testDot() {
        // m1 (2x2) dot m3 (2x3) -> 2x3
        Matrix result = m1.dot(m3);
        Matrix expected = new Matrix(new double[][]{{9, 12, 15}, {19, 26, 33}});
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("dot product should throw exception for incompatible sizes")
    void testDotIncompatible() {
        // m3 (2x3) dot m1 (2x2)
        assertThrows(IllegalArgumentException.class, () -> m3.dot(m1));
    }

    @Test
    @DisplayName("transpose should swap rows and columns")
    void testTranspose() {
        Matrix result = m3.transpose();
        Matrix expected = new Matrix(new double[][]{{1, 4}, {2, 5}, {3, 6}});
        assertEquals(expected, result);
        // Transposing twice should return the original matrix
        assertEquals(m3, result.transpose());
    }

    @Test
    @DisplayName("equals should compare matrices correctly")
    void testEquals() {
        Matrix sameAsM1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix different = new Matrix(new double[][]{{9, 9}, {9, 9}});

        assertTrue(m1.equals(sameAsM1));
        assertFalse(m1.equals(different));
        assertFalse(m1.equals(m2));
        assertFalse(m1.equals(new Object()));
    }

    @Test
    @DisplayName("hashCode should be consistent with equals")
    void testHashCode() {
        Matrix sameAsM1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertEquals(m1.hashCode(), sameAsM1.hashCode());
        assertNotEquals(m1.hashCode(), m2.hashCode());
    }
}