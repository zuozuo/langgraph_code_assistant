from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class CodeSolution(BaseModel):
    """Schema for code solutions"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

class CodeGenerator:
    def __init__(self, model: str = "gpt-4", provider: str = "openai"):
        self.model = model
        self.provider = provider
        self.llm = self._initialize_llm()
        self.prompt = self._create_prompt_template()
        self.chain = self.prompt | self.llm.with_structured_output(CodeSolution)

    def _initialize_llm(self):
        if self.provider == "openai":
            return ChatOpenAI(temperature=0, model=self.model)
        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=self.model,
                default_headers={"anthropic-beta": "tools-2024-04-04"}
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a coding assistant. Answer the user question based on the provided documentation.
Ensure any code you provide can be executed with all required imports and variables defined.
Structure your answer with:
1. A description of the code solution
2. The imports
3. The functioning code block

Documentation:
{context}

User question:"""),
            ("placeholder", "{messages}")
        ])

    def generate(self, context: str, question: str) -> CodeSolution:
        """Generate code solution for given question and context"""
        return self.chain.invoke({
            "context": context,
            "messages": [("user", question)]
        })
