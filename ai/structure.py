from pydantic import BaseModel, Field

class Structure(BaseModel):
    tldr: str = Field(description="A concise, too long; didn't read summary of the paper")
    motivation: str = Field(description="Describe the motivation behind this paper")
    method: str = Field(description="Summarize the main method(s) proposed or used in this paper")
    result: str = Field(description="Summarize the main results or findings of this paper")
    conclusion: str = Field(description="Summarize the key conclusions of this paper")
