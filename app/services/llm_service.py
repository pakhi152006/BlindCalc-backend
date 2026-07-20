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
You are the mathematical language parser for BlindCalc.

Your ONLY job is to convert the user's natural language mathematical question into valid JSON.

DO NOT explain.
DO NOT solve the problem.
DO NOT write markdown.
DO NOT use code blocks.

Return ONLY valid JSON.

The JSON format MUST be:

{
  "intent": "",
  "expression": "",
  "variables": [],
  "parameters": {},
  "explanation": "",
  "is_valid": true
}

Rules:

1. expression MUST be valid SymPy syntax.

Examples:
x squared -> x**2
x cube -> x**3
x to the power 5 -> x**5

sin x -> sin(x)
cos x -> cos(x)
tan x -> tan(x)
cot x -> cot(x)
sec x -> sec(x)
cosec x -> csc(x)

ln(x) -> log(x)
log x -> log(x)

square root of x -> sqrt(x)

pi -> pi
e -> E

2x -> 2*x
3y -> 3*y

2. Detect the intent.

Possible intents:

arithmetic
scientific
solve
differentiate
integrate
limit
matrix
statistics

3. Integration examples

User:
integrate x square

Return:

{
 "intent":"integrate",
 "expression":"x**2",
 "variables":["x"],
 "parameters":{"wrt":"x"},
 "explanation":"Integrate x squared",
 "is_valid":true
}

User:
integration of x square plus sin x

Return:

{
 "intent":"integrate",
 "expression":"x**2 + sin(x)",
 "variables":["x"],
 "parameters":{"wrt":"x"},
 "explanation":"Integrate expression",
 "is_valid":true
}

4. Differentiation

User:
differentiate x cube

Return

{
 "intent":"differentiate",
 "expression":"x**3",
 "variables":["x"],
 "parameters":{"wrt":"x","order":1},
 "explanation":"Differentiate expression",
 "is_valid":true
}

5. Limits

User:
limit of sin x by x as x approaches 0

Return

{
 "intent":"limit",
 "expression":"sin(x)/x",
 "variables":["x"],
 "parameters":{
   "wrt":"x",
   "approaches":"0"
 },
 "explanation":"Evaluate limit",
 "is_valid":true
}

6. Algebra

User:
solve x square minus 4 equals 0

Return

{
 "intent":"solve",
 "expression":"x**2-4=0",
 "variables":["x"],
 "parameters":{"wrt":"x"},
 "explanation":"Solve equation",
 "is_valid":true
}

7. Statistics

Detect mean, median, mode, variance, standard deviation and correlation.

8. Matrix

Detect determinant, inverse, transpose and multiplication.

9. If the user asks for normal arithmetic:

"What is 25 plus 13"

Return

{
 "intent":"arithmetic",
 "expression":"25+13",
 "variables":[],
 "parameters":{},
 "explanation":"Arithmetic calculation",
 "is_valid":true
}

Always return ONLY JSON.
Never explain.
Never solve.
Never use markdown.
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
            expr = data["expression"].strip()
            # Only clean if it still looks like natural language.
            if " " in expr:
                expr = LLMService._clean(expr)
                data["expression"] = expr

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

        if any(k in q for k in ["integrate","integration"]):
            intent = "integrate"
            expr = re.sub(
                r"^(integration of|integrate|integration)\s*",
                "",
                q
                )
            expression = LLMService._clean(expr)
            parameters["wrt"]="x"
            explanation=f"Integrating {expression}"
            
        elif any(k in q for k in ["differentiate","derivative","differentiate of"]):
            intent="differentiate"
            expr = re.sub(
                r"^(differentiate|derivative of|derivative)\s*",
                "",
                q
                )
            expression=LLMService._clean(expr)
            parameters["wrt"]="x"
            parameters["order"]=1
            explanation=f"Differentiating {expression}"
            
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
            # Powers
        t = re.sub(r"x\s*square\b", "x**2", t)
        t = re.sub(r"x\s*cube\b", "x**3", t)
        t = re.sub(r"([a-z])\s*to the power\s*(\d+)", r"\1**\2", t)
        # Constants
        t = t.replace("pi","pi")
        t = re.sub(r"\be\b","E",t)
        # Functions
        funcs = {
            "sin ":"sin(",
            "cos ":"cos(",
            "tan ":"tan(",
            "log ":"log(",
            "ln ":"log(",
            "sqrt ":"sqrt(",
            "square root ":"sqrt(",
            "exp ":"exp("
        }
        for k,v in funcs.items():
            t = t.replace(k,v)
        # close brackets
        for f in ["sin(","cos(","tan(","log(","sqrt(","exp("]:
            if f in t and ")" not in t[t.index(f):]:
                t += ")"

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

        # x square -> x**2
        t = re.sub(r"\bx\s*square\b", "x**2", t)
        t = re.sub(r"\bx\s*cube\b", "x**3", t)
        # x to the power 5 -> x**5
        t = re.sub(r"([a-zA-Z])\s*to\s*the\s*power\s*of\s*(\d+)", r"\1**\2", t)
        # sin x -> sin(x)
        t = re.sub(r"\bsin\s+([a-zA-Z0-9]+)", r"sin(\1)", t)
        t = re.sub(r"\bcos\s+([a-zA-Z0-9]+)", r"cos(\1)", t)
        t = re.sub(r"\btan\s+([a-zA-Z0-9]+)", r"tan(\1)", t)
        t = re.sub(r"\blog\s+([a-zA-Z0-9]+)", r"log(\1)", t)
        t = re.sub(r"\bln\s+([a-zA-Z0-9]+)", r"log(\1)", t)
        t = re.sub(r"\bsqrt\s+([a-zA-Z0-9]+)", r"sqrt(\1)", t)
        # 2x -> 2*x
        t = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", t)
        # xy -> x*y
        
        # remove extra spaces only
        t = re.sub(r"\s+", " ", t).strip()
        logger.info(f"Before cleaning: {text}")
        logger.info(f"After cleaning: {t}")

        return t