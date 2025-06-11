"""
ClassifierScorer implementation for basic Text-to-SQL evaluation.

Takes a natural language query, a corresponding LLM-generated SQL query, and a table schema + (optional) metadata.
Determines if the LLM-generated SQL query is valid and works for the natural language query.
"""

from judgeval.scorers import ClassifierScorer

Text2SQLScorer = ClassifierScorer(
    name="Text to SQL",
    slug="text2sql-1010101010",
    threshold=1.0,
    conversation=[
        {
            "role": "system",
            "content": """You will be given a natural language query, a corresponding LLM-generated SQL query, and a table schema + (optional) metadata.

** TASK INSTRUCTIONS **
Your task is to decide whether the LLM generated SQL query properly filters for what the natural language query is asking, based on the table schema + (optional) metadata. 
Additionally, you should check if the SQL query is valid based on the table schema (checking for syntax errors, false column names, etc.)

** TIPS **
- Look for correct references to the table schema for column names, table names, etc.
- Check that the SQL query can be executed; make sure JOINs, GROUP BYs, ORDER BYs, etc. are valid with respect to the table schema.
- Check that aggregation functions (COUNT, SUM, AVG, etc.) are used appropriately with GROUP BY clauses
- Verify that WHERE conditions use the correct operators and data types for comparisons
- Ensure LIMIT and OFFSET clauses make sense for the query's purpose
- Check that JOINs use the correct keys and maintain referential integrity
- Verify that ORDER BY clauses use valid column names and sort directions
- Check for proper handling of NULL values where relevant
- Ensure subqueries are properly constructed and correlated when needed
- EVEN IF THE QUERY IS VALID, IF IT DOESN'T WORK FOR THE NATURAL LANGUAGE QUERY, YOU SHOULD CHOOSE "N" AS THE ANSWER.

** FORMATTING YOUR ANSWER **
If the SQL query is valid and works for the natural language query, choose option "Y" and otherwise "N". Provide a justification for your decision; if you choose "N", explain what about the LLM-generated SQL query is incorrect, or explain why it doesn't address the natural language query. 
IF YOUR JUSTIFICATION SHOWS THAT THE SQL QUERY IS VALID AND WORKS FOR THE NATURAL LANGUAGE QUERY, YOU SHOULD CHOOSE "Y" AS THE ANSWER. 
IF THE SQL QUERY IS INVALID, YOU SHOULD CHOOSE "N" AS THE ANSWER. 

** YOUR TURN **
Natural language query:
{{input}}

LLM generated SQL query:
{{actual_output}}

Table schema:
{{context}}
        """,
        }
    ],
    options={"Y": 1.0, "N": 0.0},
)
