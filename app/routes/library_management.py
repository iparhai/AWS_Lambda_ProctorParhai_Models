from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from utils import ADMIN_TOKEN

router = APIRouter(prefix="/update_libs", tags=["Library Management"])

class UpdateLibsRequest(BaseModel):
    language: str
    libraries: list[str]

@router.post("/")
async def update_libraries(request: UpdateLibsRequest, authorization: str = Header(None)):
    if authorization != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

    lang = request.language
    new_libs = request.libraries

    if lang == "Python":
        with open("allowed_python_libs.txt", "a") as f:
            f.write("\n".join(new_libs) + "\n")
    elif lang == "NodeJS":
        with open("allowed_node_libs.txt", "a") as f:
            f.write("\n".join(new_libs) + "\n")
    elif lang == "Java":
        with open("allowed_java_classes.txt", "a") as f:
            f.write("\n".join(new_libs) + "\n")

    return {"message": f"Libraries added to {lang} whitelist."}