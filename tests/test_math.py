import unittest
import sys
import os

# Adjust path to import app services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.math_service import MathService

class TestMathOperations(unittest.TestCase):
    def test_arithmetic(self):
        """Test standard addition, multiplication and divisions"""
        res = MathService.solve_problem("arithmetic", "25 + 47")
        self.assertEqual(res["result_text"], "25 + 47 = 72")
        self.assertIn("equals 72", res["result_spoken"])

    def test_scientific_trig(self):
        """Test trigonometric evaluation in radians and degrees conversion"""
        # sin(pi/2) = 1
        res = MathService.solve_problem("scientific", "sin(pi/2)")
        self.assertEqual(res["result_text"], "sin(pi/2) = 1")
        
        # sqrt(256) = 16
        res_sqrt = MathService.solve_problem("scientific", "sqrt(256)")
        self.assertIn("16", res_sqrt["result_text"])

    def test_algebra_solver(self):
        """Test quadratic equations solver x^2 - 5x + 6 = 0"""
        res = MathService.solve_problem("solve", "x**2 - 5*x + 6", {"wrt": "x"})
        self.assertTrue("2" in res["result_text"] and "3" in res["result_text"])
        self.assertIn("solutions for x are 2 or 3", res["result_spoken"])

    def test_differentiation(self):
        """Test derivative of x^3 + 2x^2 + 5 -> 3x^2 + 4x"""
        res = MathService.solve_problem("differentiate", "x**3 + 2*x**2 + 5", {"wrt": "x", "order": 1})
        self.assertEqual(res["result_text"], "3*x**2 + 4*x")
        self.assertIn("3 times x squared plus 4 times x", res["result_spoken"])

    def test_integration_indefinite(self):
        """Test integration of x^2 + 2x + 1 -> x^3/3 + x^2 + x"""
        res = MathService.solve_problem("integrate", "x**2 + 2*x + 1", {"wrt": "x"})
        self.assertIn("x**3/3 + x**2 + x + C", res["result_text"])
        self.assertIn("plus constant C", res["result_spoken"])

    def test_integration_definite(self):
        """Test integration of x from 0 to 2 -> x^2/2 evaluated -> 2"""
        res = MathService.solve_problem("integrate", "x", {"wrt": "x", "lower_limit": "0", "upper_limit": "2"})
        self.assertEqual(res["result_text"], "2")
        self.assertIn("from 0 to 2 equals 2", res["result_spoken"])

    def test_limits(self):
        """Test limit of sin(x)/x as x approaches 0 -> 1"""
        res = MathService.solve_problem("limit", "sin(x)/x", {"wrt": "x", "approaches": "0"})
        self.assertEqual(res["result_text"], "1")
        self.assertIn("approaches 0 equals 1", res["result_spoken"])

    def test_matrix_determinant(self):
        """Test 2x2 matrix determinant"""
        res = MathService.solve_problem("matrix", "[[1, 2], [3, 4]]", {"operation": "determinant"})
        self.assertEqual(res["result_text"], "det(A) = -2")

    def test_matrix_inverse(self):
        """Test 2x2 matrix inversion"""
        res = MathService.solve_problem("matrix", "[[1, 2], [3, 4]]", {"operation": "inverse"})
        self.assertIn("[[-2, 1]", res["result_text"])

    def test_statistics_mean_std(self):
        """Test dataset mean and standard deviations calculations"""
        # Mean of 5, 10, 15, 20 is 12.5
        res_mean = MathService.solve_problem("statistics", "[5, 10, 15, 20]", {"operation": "mean"})
        self.assertEqual(res_mean["result_text"], "Mean: 12.5")

        # Median of 5, 10, 15, 20 is 12.5
        res_med = MathService.solve_problem("statistics", "[5, 10, 15, 20]", {"operation": "median"})
        self.assertEqual(res_med["result_text"], "Median: 12.5")

if __name__ == "__main__":
    unittest.main()
