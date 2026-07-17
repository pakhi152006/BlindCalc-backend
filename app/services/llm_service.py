import json
import re
from groq import Groq

from app.config import settings, logger
from app.models.schemas import LLMIntentParse


class LLMService:
    """
    BlindCalc LLM Service (Groq-based)

    Responsibilities:
    - Convert natural language math questions into SymPy expressions
    - Extract intent
    - Provide structured JSON output
    - Clean AI-generated expressions before sending to MathService
    """

    client = Groq(api_key=settings.GROQ_API_KEY)

    # ---------------------------------
    # PUBLIC ENTRY POINT
    # ---------------------------------
    @staticmethod
    def parse_query(query: str) -> LLMIntentParse:
        logger.info(f"Parsing query: {query}")

        try:
            return LLMService._query_groq(query)

        except Exception as e:
            logger.warning(
                f"Groq failed, using rule based parser. Error: {str(e)}"
            )
            return LLMService._rule_based_parse(query)

    # ---------------------------------
    # GROQ PARSER
    # ---------------------------------
    @staticmethod
    def _query_groq(query: str) -> LLMIntentParse:

        system_prompt = """
        PASTE YOUR EXISTING PROMPT HERE
        """

        response = LLMService.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

        content = re.sub(
            r"^```json|```$",
            "",
            content
        ).strip()

        data = json.loads(content)

        if "expression" in data:
            data["expression"] = LLMService._clean(
                data["expression"]
            )

        return LLMIntentParse(**data)

    # ---------------------------------
    # RULE BASED FALLBACK
    # ---------------------------------
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

    # ---------------------------------
    # CLEANING FUNCTION
    # ---------------------------------
    @staticmethod
    def _clean(text: str) -> str:

        t = text.lower().strip()

        remove_words = [
            "what is",
            "calculate",
            "find",
            "please",
            "answer",
            "give me",
            "the answer of",
            "the value of",
            "equals",
            "equal to",
            "is",
        ]

        for word in remove_words:
            t = t.replace(word, "")

        numbers = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

        for word, digit in numbers.items():
            t = re.sub(
                rf"\b{word}\b",
                digit,
                t
            )

        replacements = {
            "sum of": "",
            "some of": "",
            "total of": "",

            "multiplied by": "*",
            "divided by": "/",

            "square root of": "sqrt(",
            "square root": "sqrt(",

            "power of": "**",

            "plus": "+",
            "प्लस": "+",

            "minus": "-",
            "माइनस": "-",

            "times": "*",
            "into": "*",

            "divide": "/",
            "over": "/",

            "and": "+",
        }

        for old, new in replacements.items():
            t = t.replace(old, new)

        if "sqrt(" in t and ")" not in t:
            t += ")"

        t = re.sub(
            r"[^0-9a-zA-Z+\-*/().\s]",
            "",
            t
        )

        t = re.sub(
            r"(\d)([a-zA-Z])",
            r"\1*\2",
            t
        )

        t = re.sub(
            r"\s+",
            "",
            t
        )
        logger.info(f"Before cleaning: {text}")
        logger.info(f"After cleaning: {t}")

        return t