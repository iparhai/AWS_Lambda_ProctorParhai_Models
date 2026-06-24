from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from configs import LANGUAGE_CONFIG

ALLOWED_LANGUAGES = list(LANGUAGE_CONFIG.keys())

class CodeRequest(BaseModel):
    language: str = Field(..., example="Python3")
    code: str = Field(..., example='print("Hello, World!")')
    validate_dependencies: bool = Field(False, example=False)
    input: Optional[str] = Field(None, example="User input")
    mode: str = Field("stdin", example="stdin")  # "stdin" or "args"
    args: List[str] = Field(default_factory=list, example=["arg1", "arg2"])

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"Language must be one of {ALLOWED_LANGUAGES}")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in {"stdin", "args"}:
            raise ValueError('Mode must be either "stdin" or "args"')
        return v
    
    @field_validator("code")
    @classmethod
    def validate_code_length(cls, v):
        if len(v) > 10000:  # Limit to prevent excessive execution time
            raise ValueError("Code length exceeds the allowed limit (10,000 characters).")
        return v


class MultiCodeRequest(BaseModel):
    codes: List[CodeRequest] = Field(
        ..., example=[
            {"language": "Python3", "code": 'print("First Code")', "mode": "stdin"},
            {"language": "NodeJS", "code": 'console.log("Second Code")', "mode": "stdin"}
        ]
    )


class BulkCodeRequest(BaseModel):
    language: str = Field(..., example="Python3")
    code: str = Field(..., example='print(input())')
    input: List[str] = Field(..., example=["Input 1", "Input 2", "Input 3"])
    mode: str = Field("stdin", example="stdin")
    args: List[str] = Field(default_factory=list, example=["arg1", "arg2"])

    @field_validator("language")
    @classmethod
    def validate_language_bulk(cls, v):
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"Language must be one of {ALLOWED_LANGUAGES}")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode_bulk(cls, v):
        if v not in {"stdin", "args"}:
            raise ValueError('Mode must be either "stdin" or "args"')
        return v


class CodeResponse(BaseModel):
    output: Optional[str] = Field(None, example="Hello, World!")
    error: Optional[str] = Field(None, example="Error message")
    execution_time: Optional[float] = Field(None, example=0.123)  # Optional for performance tracking


class MultiCodeResponse(BaseModel):
    responses: List[CodeResponse]


class BulkCodeResponse(BaseModel):
    responses: List[CodeResponse]
