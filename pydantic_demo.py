from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    summary: str = Field(description="Short summary")
    sentiment: Literal["pos", "neg"] = Field(description="Review sentiment")
    pros: Optional[list[str]] = Field(default=None, description="Advantages")
    cons: Optional[list[str]] = Field(default=None, description="Disadvantages")
    name: Optional[str] = Field(default=None, description="Reviewer name")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I recently bought the Samsung Galaxy S24 Ultra.

It is very fast, has an excellent camera, and the battery lasts all day.

However, it is expensive and a bit heavy.

Review by Nitish Singh
""")

print(result)
print(result.name)
print(result.summary)
print(result.sentiment)