import json
import re
from typing import Dict, Any

from groq import Groq
from app.config import settings, logger
from app.models.schemas import LLMIntentParse


class LLMService:
    """
    BlindCalc LLM Service (Groq-based)
    Handles:
    - Intent parsing via Groq
    - Rule-based fallback
    """

    client = Groq(api_key=settings.GROQ_API_KEY)

    # ----------------------------
    # PUBLIC ENTRY POINT
    # ----------------------------
    @staticmethod
    def parse_query(query: str) -> LLMIntentParse:
        logger.info(f"Parsing query: {query}")

        try:
            return LLMService._query_groq(query)
        except Exception as e:
            logger.warning(f"Groq failed, falling back to rules. Error: {str(e)}")
            return LLMService._rule_based_parse(query)

    # ----------------------------
    # GROQ LLM CALL
    # ----------------------------
    @staticmethod
    def _query_groq(query: str) -> LLMIntentParse:

        system_prompt = """
You are a mathematical translation parser for BlindCalc AI.

Extract:
1. intent: arithmetic | scientific | solve | differentiate | integrate | limit | matrix | statistics
2. expression: valid SymPy expression
3. variables: list of variables
4. parameters: JSON dict of metadata
5. explanation: short voice-friendly explanation
6. is_valid: boolean

Rules:
- DO NOT calculate anything
- ONLY return raw JSON
- factorial must be factorial(n)
"""

        response = LLMService.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # remove ```json blocks if model adds them
        content = re.sub(r"^```json|```$", "", content).strip()

        data = json.loads(content)

        return LLMIntentParse(**data)

    # ----------------------------
    # RULE BASED FALLBACK
    # (your logic kept, slightly cleaned)
    # ----------------------------
    @staticmethod
    def _rule_based_parse(query: str) -> LLMIntentParse:
        q = query.lower().strip()

        intent = "arithmetic"
        expression = ""
        parameters = {}
        explanation = ""

        if "integrate" in q:
            intent = "integrate"
            expr = q.replace("integrate", "")
            expression = LLMService._clean(expr)
            parameters["wrt"] = "x"
            explanation = f"Integrating {expr}"

        elif "differentiate" in q:
            intent = "differentiate"
            expr = q.replace("differentiate", "")
            expression = LLMService._clean(expr)
            parameters["wrt"] = "x"
            parameters["order"] = 1
            explanation = f"Differentiating {expr}"

        elif "solve" in q:
            intent = "solve"
            expr = q.replace("solve", "")
            expr = re.sub(r"=0|equals zero", "", expr)
            expression = LLMService._clean(expr)
            explanation = f"Solving {expr} = 0"

        elif any(x in q for x in ["mean", "median", "mode", "variance", "std"]):
            intent = "statistics"
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", q)
            expression = str(nums)
            parameters["operation"] = "stats"
            explanation = "Statistical calculation"

        elif "matrix" in q:
            intent = "matrix"
            expression = q
            parameters["operation"] = "matrix_op"
            explanation = "Matrix operation"

        else:
            expression = LLMService._clean(q)
            explanation = f"Calculating {q}"

        return LLMIntentParse(
            intent=intent,
            expression=expression,
            variables=["x"],
            parameters=parameters,
            explanation=explanation,
            is_valid=True
        )

    # ----------------------------
    # CLEANING FUNCTION
    # ----------------------------
    @staticmethod
    def _clean(text: str) -> str:
        t = text.lower()

        replacements = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "over": "/",
            "squared": "**2",
            "cubed": "**3",
            "square root": "sqrt",
            "power of": "**",
            "what is": "",
            "calculate": "",
            "find": "",
        }

        for k, v in replacements.items():
            t = t.replace(k, v)

        t = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", t)

        return t.strip()