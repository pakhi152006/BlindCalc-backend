import sympy as sp
import numpy as np
import scipy.stats as stats
import pandas as pd
import re
import math
from typing import Dict, Any, Union, List

from app.config import logger

def clean_mathematical_text(text: str) -> str:
    """
    Cleans general math symbols for better voice articulation.
    """
    # Replace standard symbols for TTS audio friendliness
    spoken = text
    spoken = spoken.replace("**2", " squared")
    spoken = spoken.replace("**3", " cubed")
    spoken = re.sub(r'\*\*(\d+)', r' to the power of \1', spoken)
    spoken = spoken.replace("*", " times ")
    spoken = spoken.replace("/", " divided by ")
    spoken = spoken.replace("+", " plus ")
    spoken = spoken.replace("-", " minus ")
    spoken = spoken.replace("sqrt", "square root of")
    spoken = spoken.replace("pi", " pi ")
    spoken = spoken.replace("oo", "infinity")
    spoken = spoken.replace("log", "logarithm")
    spoken = spoken.replace("sin", "sine")
    spoken = spoken.replace("cos", "cosine")
    spoken = spoken.replace("tan", "tangent")
    spoken = spoken.replace("exp", "e to the power of")
    # Clean multiple spaces
    spoken = re.sub(r'\s+', ' ', spoken).strip()
    return spoken

class MathService:
    @staticmethod
    def solve_problem(intent: str, expression: str, parameters: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Executes math problem solving using SymPy, NumPy, or SciPy based on intent.
        Returns a dictionary containing:
        - result_text: readable text output
        - result_latex: LaTeX formatted string
        - result_spoken: voice-friendly description
        """
        if parameters is None:
            parameters = {}
            
        logger.info(f"MathService executing: intent={intent}, expr={expression}, params={parameters}")
        
        try:
            if intent in ("arithmetic", "scientific"):
                return MathService._solve_arithmetic_scientific(expression)
            elif intent == "solve":
                return MathService._solve_algebra(expression, parameters)
            elif intent == "differentiate":
                return MathService._solve_differentiation(expression, parameters)
            elif intent == "integrate":
                return MathService._solve_integration(expression, parameters)
            elif intent == "limit":
                return MathService._solve_limits(expression, parameters)
            elif intent == "matrix":
                return MathService._solve_matrix(expression, parameters)
            elif intent == "statistics":
                return MathService._solve_statistics(expression, parameters)
            else:
                # Default fallback evaluation
                return MathService._solve_arithmetic_scientific(expression)
        except Exception as e:
            logger.error(f"MathService computation error: {str(e)}")
            raise ValueError(f"Failed to calculate expression. Error details: {str(e)}")

    @staticmethod
    def _solve_arithmetic_scientific(expression: str) -> Dict[str, str]:
        # Clean potential equations (like = ?)
        expr_to_eval = expression.split('=')[0].strip()
        
        # Sympy evaluation
        sym_expr = sp.sympify(expr_to_eval)
        
        # Evaluate to number if simple float
        # Check if expression is numeric or symbolical
        result = sym_expr.evalf() if sym_expr.is_number else sym_expr
        
        # Format decimal places for readability if float
        if isinstance(result, sp.Float):
            result_val = float(result)
            if result_val.is_integer():
                result_text = str(int(result_val))
            else:
                # Round to 5 decimal places for clean voice output
                result_text = f"{result_val:.5f}".rstrip('0').rstrip('.')
        else:
            result_text = str(result)
            
        result_latex = sp.latex(sym_expr) + " = " + sp.latex(result)
        
        # Create speech text
        spoken_expr = clean_mathematical_text(str(sym_expr))
        spoken_result = clean_mathematical_text(result_text)
        result_spoken = f"{spoken_expr} equals {spoken_result}"
        
        return {
            "result_text": f"{expression} = {result_text}" if "=" not in expression else f"{result_text}",
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }

    @staticmethod
    def _solve_algebra(expression: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        # Equations could contain '='. We translate e1 = e2 into e1 - e2 = 0
        eq_parts = expression.split('=')
        if len(eq_parts) == 2:
            eq = sp.Eq(sp.sympify(eq_parts[0].strip()), sp.sympify(eq_parts[1].strip()))
        else:
            # Assume it's equated to 0
            eq = sp.sympify(expression.strip())
            
        # Determine variable to solve for
        var_str = parameters.get("wrt", "x")
        var = sp.Symbol(var_str)
        
        solutions = sp.solve(eq, var)
        
        # Format results
        if not solutions:
            result_text = "No real solutions found"
            result_latex = r"\emptyset"
            result_spoken = "There are no real solutions for " + clean_mathematical_text(var_str)
        else:
            sol_strings = [str(sol) for sol in solutions]
            result_text = f"{var_str} = " + " or ".join(sol_strings)
            result_latex = f"{var_str} = " + ", ".join([sp.latex(sol) for sol in solutions])
            
            spoken_sol = " or ".join([clean_mathematical_text(sol_str) for sol_str in sol_strings])
            result_spoken = f"The solutions for {clean_mathematical_text(var_str)} are {spoken_sol}"
            
        return {
            "result_text": result_text,
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }

    @staticmethod
    def _solve_differentiation(expression: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        expr = sp.sympify(expression)
        var_str = parameters.get("wrt", "x")
        var = sp.Symbol(var_str)
        
        # Derivative order (default 1st derivative)
        order = int(parameters.get("order", 1))
        
        derivative = sp.diff(expr, var, order)
        
        # Construct output
        result_text = str(derivative)
        
        # LaTeX formatting
        diff_notation_latex = f"\\frac{{d^{{{order}}}}}[{{{sp.latex(expr)}}}]{{d{var_str}^{{{order}}}}}" if order > 1 else f"\\frac{{d}}{{d{var_str}}}\\left({sp.latex(expr)}\\right)"
        result_latex = f"{diff_notation_latex} = {sp.latex(derivative)}"
        
        # Spoken text
        spoken_expr = clean_mathematical_text(str(expr))
        spoken_derivative = clean_mathematical_text(str(derivative))
        ord_suffix = "first" if order == 1 else "second" if order == 2 else f"{order}th"
        result_spoken = f"The {ord_suffix} derivative of {spoken_expr} with respect to {var_str} is {spoken_derivative}"
        
        return {
            "result_text": result_text,
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }

    @staticmethod
    def _solve_integration(expression: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        expr = sp.sympify(expression)
        var_str = parameters.get("wrt", "x")
        var = sp.Symbol(var_str)
        
        # Definite vs Indefinite
        lower_limit = parameters.get("lower_limit")
        upper_limit = parameters.get("upper_limit")
        
        if lower_limit is not None and upper_limit is not None:
            # Definite Integral
            low = sp.sympify(lower_limit)
            high = sp.sympify(upper_limit)
            integral = sp.integrate(expr, (var, low, high))
            
            result_text = str(integral)
            result_latex = f"\\int_{{{sp.latex(low)}}}^{{{sp.latex(high)}}} {sp.latex(expr)} \\, d{var_str} = {sp.latex(integral)}"
            
            spoken_expr = clean_mathematical_text(str(expr))
            spoken_low = clean_mathematical_text(str(low))
            spoken_high = clean_mathematical_text(str(high))
            spoken_result = clean_mathematical_text(str(integral))
            
            result_spoken = f"The definite integral of {spoken_expr} with respect to {var_str} from {spoken_low} to {spoken_high} equals {spoken_result}"
        else:
            # Indefinite Integral
            integral = sp.integrate(expr, var)
            
            result_text = f"{integral} + C"
            result_latex = f"\\int {sp.latex(expr)} \\, d{var_str} = {sp.latex(integral)} + C"
            
            spoken_expr = clean_mathematical_text(str(expr))
            spoken_result = clean_mathematical_text(str(integral))
            
            result_spoken = f"The indefinite integral of {spoken_expr} with respect to {var_str} is {spoken_result} plus constant C"
            
        return {
            "result_text": result_text,
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }

    @staticmethod
    def _solve_limits(expression: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        expr = sp.sympify(expression)
        var_str = parameters.get("wrt", "x")
        var = sp.Symbol(var_str)
        
        # Approaches (limit target)
        target_str = str(parameters.get("approaches", "0"))
        target = sp.sympify(target_str)
        
        # Direction (+ or - or default both)
        direction = parameters.get("direction", "") # "+", "-", or "both" / empty
        
        if direction == "+":
            lim = sp.limit(expr, var, target, dir="+")
            dir_text = "from the right"
            dir_latex = "^+"
        elif direction == "-":
            lim = sp.limit(expr, var, target, dir="-")
            dir_text = "from the left"
            dir_latex = "^-"
        else:
            lim = sp.limit(expr, var, target)
            dir_text = ""
            dir_latex = ""
            
        result_text = str(lim)
        result_latex = f"\\lim_{{{var_str} \\to {sp.latex(target)}{dir_latex}}} {sp.latex(expr)} = {sp.latex(lim)}"
        
        spoken_expr = clean_mathematical_text(str(expr))
        spoken_target = clean_mathematical_text(target_str)
        spoken_result = clean_mathematical_text(str(lim))
        
        result_spoken = f"The limit of {spoken_expr} as {var_str} approaches {spoken_target} {dir_text} equals {spoken_result}"
        result_spoken = re.sub(r'\s+', ' ', result_spoken).strip()
        
        return {
            "result_text": result_text,
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }

    @staticmethod
    def _solve_matrix(expression: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Executes matrix operations.
        expression structure is expected to represent matrix elements:
        - "Matrix A: [[1, 2], [3, 4]]"
        - Or raw list string: "[[1, 2], [3, 4]]"
        - For operations involving 2 matrices: "A: [[1,2],[3,4]]; B: [[5,6],[7,8]]"
        """
        operation = parameters.get("operation", "determinant").lower() # determinant, inverse, multiply, transpose, rref, eigenvalues
        
        # Parse matrices from expression using regex
        matrix_matches = re.findall(r'\[\s*(?:\[[^\]]*\]\s*,?\s*)+\]', expression)
        
        if not matrix_matches:
            raise ValueError("No matrix layout detected. Please specify matrices as nested lists, like [[1, 2], [3, 4]].")
            
        # Convert first matrix
        matrix_a_data = eval(matrix_matches[0])
        M_A = sp.Matrix(matrix_a_data)
        
        if operation == "determinant":
            if not M_A.is_square:
                raise ValueError("Matrix must be square to calculate determinant.")
            det = M_A.det()
            result_text = f"det(A) = {det}"
            result_latex = f"\\det(A) = {sp.latex(det)}"
            result_spoken = f"The determinant of the matrix is {clean_mathematical_text(str(det))}"
            
        elif operation == "inverse":
            if not M_A.is_square:
                raise ValueError("Matrix must be square to calculate inverse.")
            if M_A.det() == 0:
                raise ValueError("Matrix is singular (determinant is 0) and cannot be inverted.")
            inv = M_A.inv()
            result_text = f"inv(A) = {str(inv.tolist())}"
            result_latex = f"A^{{-1}} = {sp.latex(inv)}"
            result_spoken = "The inverse matrix is " + clean_mathematical_text(str(inv.tolist()))
            
        elif operation == "transpose":
            trans = M_A.T
            result_text = f"transpose(A) = {str(trans.tolist())}"
            result_latex = f"A^T = {sp.latex(trans)}"
            result_spoken = "The transposed matrix is " + clean_mathematical_text(str(trans.tolist()))
            
        elif operation in ("multiply", "multiplication"):
            if len(matrix_matches) < 2:
                raise ValueError("Matrix multiplication requires two matrices. Please speak or write both.")
            matrix_b_data = eval(matrix_matches[1])
            M_B = sp.Matrix(matrix_b_data)
            
            if M_A.cols != M_B.rows:
                raise ValueError(f"Inner matrix dimensions must match. Matrix A has {M_A.cols} columns but Matrix B has {M_B.rows} rows.")
                
            prod = M_A * M_B
            result_text = f"A * B = {str(prod.tolist())}"
            result_latex = f"A \\cdot B = {sp.latex(prod)}"
            result_spoken = "The matrix product is " + clean_mathematical_text(str(prod.tolist()))
            
        else:
            raise ValueError(f"Matrix operation '{operation}' is not supported.")
            
        return {
            "result_text": result_text,
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }

    @staticmethod
    def _solve_statistics(expression: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Executes statistical calculations.
        Parameters:
        - operation: mean, median, mode, variance, std, correlation
        - data: list of float values
        - data2: second list of values (for correlation)
        """
        operation = parameters.get("operation", "mean").lower()
        
        # Extract number arrays from string (looks for numbers separated by commas, spaces, or in lists)
        arrays = re.findall(r'\[\s*(?:[\d.-]+\s*,?\s*)+\]', expression)
        
        if arrays:
            data = [float(x) for x in re.findall(r'[\d.-]+', arrays[0])]
            data2 = [float(x) for x in re.findall(r'[\d.-]+', arrays[1])] if len(arrays) > 1 else []
        else:
            # Fallback regex search for raw number sequence
            numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', expression)]
            if not numbers:
                raise ValueError("No numeric data list found. Please supply numbers separated by commas or spaces.")
            # If the operation is correlation, we try to split numbers down the middle
            if operation == "correlation":
                mid = len(numbers) // 2
                data = numbers[:mid]
                data2 = numbers[mid:]
            else:
                data = numbers
                data2 = []
                
        if not data:
            raise ValueError("Numeric dataset cannot be empty.")
            
        # Stats computation using pandas and numpy
        df = pd.Series(data)
        
        if operation == "mean":
            val = df.mean()
            result_text = f"Mean: {val:.4f}".rstrip('0').rstrip('.')
            result_latex = f"\\mu = {val:.4f}".rstrip('0').rstrip('.')
            result_spoken = f"The mean of the numbers is {result_text}"
            
        elif operation == "median":
            val = df.median()
            result_text = f"Median: {val}"
            result_latex = f"\\text{{Median}} = {val}"
            result_spoken = f"The median of the numbers is {val}"
            
        elif operation == "mode":
            modes = df.mode().tolist()
            result_text = f"Mode: {modes}"
            result_latex = f"\\text{{Mode}} = {modes}"
            modes_spoken = ", ".join([str(m) for m in modes])
            result_spoken = f"The mode of the numbers is {modes_spoken}"
            
        elif operation in ("variance", "var"):
            # Sample variance by default (ddof=1)
            val = df.var() if len(data) > 1 else 0.0
            result_text = f"Sample Variance: {val:.4f}".rstrip('0').rstrip('.')
            result_latex = f"s^2 = {val:.4f}".rstrip('0').rstrip('.')
            result_spoken = f"The variance of the sample is {result_text}"
            
        elif operation in ("std", "standard deviation", "deviation"):
            val = df.std() if len(data) > 1 else 0.0
            result_text = f"Standard Deviation: {val:.4f}".rstrip('0').rstrip('.')
            result_latex = f"s = {val:.4f}".rstrip('0').rstrip('.')
            result_spoken = f"The standard deviation of the sample is {result_text}"
            
        elif operation in ("correlation", "correlation coefficient", "corr"):
            if not data2:
                raise ValueError("Correlation requires two equal-length datasets.")
            if len(data) != len(data2):
                raise ValueError(f"Datasets must have equal lengths. List A has {len(data)} items, List B has {len(data2)} items.")
            
            coeff, _ = stats.pearsonr(data, data2)
            result_text = f"Pearson Correlation: {coeff:.4f}".rstrip('0').rstrip('.')
            result_latex = f"r = {coeff:.4f}".rstrip('0').rstrip('.')
            result_spoken = f"The Pearson correlation coefficient between datasets is {result_text}"
            
        else:
            raise ValueError(f"Statistical operation '{operation}' is not supported.")
            
        return {
            "result_text": result_text,
            "result_latex": result_latex,
            "result_spoken": result_spoken
        }
