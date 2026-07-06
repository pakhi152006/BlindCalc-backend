import json
import re
import requests
from typing import Dict, Any, Tuple
from app.config import settings, logger
from app.models.schemas import LLMIntentParse

class LLMService:
    @staticmethod
    def parse_query(query: str) -> LLMIntentParse:
        """
        Parses a natural language query into intent, expression, and parameters.
        First tries local Ollama. Falls back to local rule-based parsing if Ollama fails.
        """
        logger.info(f"Parsing query: '{query}'")
        
        # 1. Try Ollama local model
        try:
            return LLMService._query_ollama(query)
        except Exception as e:
            logger.warn(f"Ollama integration unavailable, using rule-based parsing fallback. Error: {str(e)}")
            return LLMService._rule_based_parse(query)

    @staticmethod
    def _query_ollama(query: str) -> LLMIntentParse:
        """
        Sends the query to Ollama API for structured extraction.
        """
        url = f"{settings.OLLAMA_URL}/api/generate"
        
        system_prompt = (
            "You are a mathematical translation parser for BlindCalc AI. "
            "Your job is to read a natural language user query and extract: "
            "1. intent: one of ['arithmetic', 'scientific', 'solve', 'differentiate', 'integrate', 'limit', 'matrix', 'statistics'] "
            "2. expression: a valid SymPy / Python mathematical string (e.g. 'x**2 - 5*x + 6', 'sin(x)/x', '[[1, 2], [3, 4]]', '[5, 10, 15, 20]'). "
            "3. variables: array of variables (usually ['x']). "
            "4. parameters: key-value dictionary for details (e.g., {'wrt': 'x'}, {'lower_limit': 0, 'upper_limit': 1}, {'approaches': '0', 'direction': '+'}, {'operation': 'inverse'}, {'operation': 'mean'}). "
            "5. explanation: a short voice-friendly description of what we are calculating. "
            "6. is_valid: boolean "
            "\n"
            "IMPORTANT: DO NOT perform the actual calculation yourself. The python execution engine handles the calculations. "
            "For factorial operations ALWAYS use factorial(n). NEVER use fac(n), fact(n), or n!. "
            "Output ONLY a raw valid JSON object. No markdown tags, no formatting, no wrapper. Just the JSON."
            "\n"
            "Examples:\n"
            "Input: 'What is 25 plus 47?'\n"
            "Output: {\"intent\": \"arithmetic\", \"expression\": \"25 + 47\", \"variables\": [], \"parameters\": {}, \"explanation\": \"I will calculate 25 plus 47.\", \"is_valid\": true}\n\n"
            "Input: 'Differentiate x cubed plus 2x squared plus 5'\n"
            "Output: {\"intent\": \"differentiate\", \"expression\": \"x**3 + 2*x**2 + 5\", \"variables\": [\"x\"], \"parameters\": {\"wrt\": \"x\", \"order\": 1}, \"explanation\": \"I will differentiate x cubed plus 2x squared plus 5 with respect to x.\", \"is_valid\": true}\n\n"
            "Input: 'Solve x squared minus 5x plus 6 equals zero'\n"
            "Output: {\"intent\": \"solve\", \"expression\": \"x**2 - 5*x + 6\", \"variables\": [\"x\"], \"parameters\": {\"wrt\": \"x\"}, \"explanation\": \"I will solve the equation x squared minus 5x plus 6 equals 0.\", \"is_valid\": true}"
        )
        
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": f"System Guidelines:\n{system_prompt}\n\nUser Input: '{query}'\nJSON Output:",
            "stream": False,
            "options": {
                "temperature": 0.0,  # Highly deterministic parsing
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result_json = response.json().get("response", "").strip()
            # Clean possible markdown block codes (like ```json ... ```)
            result_json = re.sub(r"^```(?:json)?\s*", "", result_json)
            result_json = re.sub(r"\s*```$", "", result_json)
            
            try:
                data = json.loads(result_json)
                return LLMIntentParse(**data)
            except Exception as json_err:
                logger.error(f"Failed to parse JSON response from Ollama: {result_json}, Error: {str(json_err)}")
                raise ValueError("Invalid formatting returned from LLM service.")
        else:
            raise RuntimeError(f"Ollama returned HTTP error code {response.status_code}")

    @staticmethod
    def _rule_based_parse(query: str) -> LLMIntentParse:
        """
        Regex-based parser for offline/no-Ollama fallback.
        Attempts to translate common speech expressions into SymPy formulas.
        """
        q = query.lower().strip()
        
        # Initialize default values
        intent = "arithmetic"
        expression = ""
        parameters = {}
        explanation = ""
        
        # 1. Handle Calculus - Integration
        if "integrate" in q or "integral of" in q:
            intent = "integrate"
            expr_part = q.replace("integrate", "").replace("integral of", "").strip()
            expr_part = expr_part.replace("with respect to x", "").replace("w.r.t x", "").strip()
            
            # Check for definite limits (e.g. from A to B)
            limit_match = re.search(r"from\s+(.+?)\s+to\s+(.+)", expr_part)
            if limit_match:
                parameters["lower_limit"] = limit_match.group(1).strip()
                parameters["upper_limit"] = limit_match.group(2).strip()
                expr_part = expr_part.split("from")[0].strip()
                
            expression = LLMService._clean_to_sympy_math(expr_part)
            parameters["wrt"] = "x"
            explanation = f"I will integrate the expression {expr_part}."
            
        # 2. Handle Calculus - Differentiation
        elif "differentiate" in q or "derivative of" in q:
            intent = "differentiate"
            expr_part = q.replace("differentiate", "").replace("derivative of", "").strip()
            expr_part = expr_part.replace("with respect to x", "").replace("w.r.t x", "").strip()
            expression = LLMService._clean_to_sympy_math(expr_part)
            parameters["wrt"] = "x"
            parameters["order"] = 1
            explanation = f"I will differentiate {expr_part} with respect to x."
            
        # 3. Handle Calculus - Limits
        elif "limit" in q:
            intent = "limit"
            # Limit of f(x) as x approaches target
            # e.g., "limit of sin x by x as x approaches zero"
            expr_part = q.replace("limit of", "").replace("limit", "").strip()
            
            target_match = re.search(r"as\s+\w+\s+approaches\s+(.+)", expr_part)
            if target_match:
                target_str = target_match.group(1).strip()
                target_val = "0"
                if "zero" in target_str or "0" in target_str:
                    target_val = "0"
                elif "one" in target_str or "1" in target_str:
                    target_val = "1"
                elif "infinity" in target_str or "inf" in target_str:
                    target_val = "oo"
                else:
                    target_val = target_str
                    
                parameters["approaches"] = target_val
                expr_part = expr_part.split("as")[0].strip()
            else:
                parameters["approaches"] = "0"
                
            expression = LLMService._clean_to_sympy_math(expr_part)
            parameters["wrt"] = "x"
            explanation = f"I will calculate the limit of {expr_part} as x approaches {parameters['approaches']}."
            
        # 4. Handle Algebra - Solving
        elif "solve" in q:
            intent = "solve"
            expr_part = q.replace("solve", "").strip()
            # Remove trailing 'equals zero' or similar
            expr_part = re.sub(r"equals\s+zero|equals\s+0|=0", "", expr_part).strip()
            expression = LLMService._clean_to_sympy_math(expr_part)
            parameters["wrt"] = "x"
            explanation = f"I will solve the equation {expr_part} equals zero."
            
        # 5. Handle Statistics
        elif any(stat in q for stat in ["mean", "median", "mode", "variance", "standard deviation", "correlation"]):
            intent = "statistics"
            # Extract numbers from string
            numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', q)]
            expression = str(numbers)
            
            if "mean" in q:
                parameters["operation"] = "mean"
                explanation = "I will calculate the mean of the numbers."
            elif "median" in q:
                parameters["operation"] = "median"
                explanation = "I will calculate the median of the numbers."
            elif "mode" in q:
                parameters["operation"] = "mode"
                explanation = "I will find the mode of the numbers."
            elif "variance" in q or "var" in q:
                parameters["operation"] = "variance"
                explanation = "I will calculate the sample variance."
            elif "standard deviation" in q or "std" in q:
                parameters["operation"] = "std"
                explanation = "I will calculate the standard deviation."
            elif "correlation" in q or "corr" in q:
                parameters["operation"] = "correlation"
                explanation = "I will calculate the correlation coefficient."
                # Split lists for correlation
                mid = len(numbers) // 2
                expression = f"A: {str(numbers[:mid])}; B: {str(numbers[mid:])}"
                
        # 6. Handle Matrix Operations
        elif "matrix" in q:
            intent = "matrix"
            if "inverse" in q:
                parameters["operation"] = "inverse"
                explanation = "I will compute the inverse of the matrix."
            elif "transpose" in q:
                parameters["operation"] = "transpose"
                explanation = "I will compute the matrix transpose."
            elif "determinant" in q or "det" in q:
                parameters["operation"] = "determinant"
                explanation = "I will compute the matrix determinant."
            elif "multiply" in q or "multiplication" in q or "times" in q:
                parameters["operation"] = "multiply"
                explanation = "I will multiply the two matrices."
                
            # Search for matrix formats in text: e.g. [[1,2],[3,4]]
            matrix_match = re.search(r'\[\s*(?:\[[^\]]*\]\s*,?\s*)+\]', q)
            if matrix_match:
                expression = matrix_match.group(0)
            else:
                # If they speak numbers, build a dummy matrix or throw error in solver
                expression = q
                
        # 7. Basic Arithmetic and scientific fallbacks
        else:
            intent = "arithmetic"
            expression = LLMService._clean_to_sympy_math(q)
            explanation = f"I will calculate {query}."
            
        return LLMIntentParse(
            intent=intent,
            expression=expression,
            variables=["x"],
            parameters=parameters,
            explanation=explanation,
            is_valid=True
        )

    @staticmethod
    def _clean_to_sympy_math(text: str) -> str:
        """
        Cleans mathematical english text into python math notation.
        """
        t = text.lower().strip()
        
        # Replace mathematical words with operational symbols
        t = t.replace("plus", "+")
        t = t.replace("minus", "-")
        t = t.replace("multiplied by", "*")
        t = t.replace("times", "*")
        t = t.replace("divided by", "/")
        t = t.replace("over", "/")
        t = t.replace("squared", "**2")
        t = t.replace("cubed", "**3")
        t = t.replace("square root of", "sqrt")
        t = t.replace("factorial", "factorial")
        t = t.replace("log base 10 of", "log") # sympy defaults log as ln or we evaluate base 10
        t = t.replace("sine", "sin")
        t = t.replace("cosine", "cos")
        t = t.replace("tangent", "tan")
        t = t.replace("power of", "**")
        t = t.replace("power", "**")
        
        # Remove filler words
        t = t.replace("what is", "")
        t = t.replace("calculate", "")
        t = t.replace("find", "")
        t = t.replace("degrees", " * pi / 180") # Degree conversion to radians
        t = t.replace("degree", " * pi / 180")
        
        # Clean double symbols or spacing
        t = t.replace("  ", " ").strip()
        
        # Implicit multiplications like "2x" -> "2*x"
        t = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1*\2', t)
        
        # Implicit parenthesis for functions e.g. "sin x" -> "sin(x)"
        for func in ["sin", "cos", "tan", "log", "sqrt"]:
            t = re.sub(func + r'\s+([a-zA-Z0-9_]+)', func + r'(\1)', t)
            
        return t
