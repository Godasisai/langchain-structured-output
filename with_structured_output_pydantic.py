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
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(
        description="A brief summary of the review"
    )
    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either pos or neg"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )
    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast. The 5000mAh battery easily lasts a full day even with heavy use.

The 200MP camera is excellent, but the phone is heavy and expensive.

Pros:
- Powerful processor
- Excellent camera
- Long battery life
- S-Pen support

Review by Nitish Singh
""")

print(result)
print(result.name)
print(result.summary)