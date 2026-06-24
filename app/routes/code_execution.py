import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException
from utils import execute_code
from models import CodeRequest, MultiCodeRequest, BulkCodeRequest, CodeResponse, MultiCodeResponse, BulkCodeResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

router = APIRouter(prefix="/compile", tags=["Code Execution"])

@router.post("/", response_model=CodeResponse, summary="Execute a single code snippet")
async def compile_code(request: CodeRequest):
    start_time = time.time()
    logging.info(f"Received code execution request for {request.language}")

    try:
        response_data = await execute_code(
            request.language, 
            request.code, 
            mode=request.mode, 
            args=request.args, 
            validate_dependencies=request.validate_dependencies, 
            user_input=request.input if request.mode == "stdin" else None
        )

        execution_time = time.time() - start_time
        logging.info(f"Execution successful: {response_data}")

        return CodeResponse(**response_data, execution_time=execution_time)

    except Exception as e:
        logging.exception(f"Unexpected error during code execution: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": "Unexpected error", "details": str(e)})


@router.post("/multi", response_model=MultiCodeResponse, summary="Execute multiple code snippets concurrently")
async def compile_multiple_codes(request: MultiCodeRequest):
    start_time = time.time()
    logging.info(f"Received multiple code execution request with {len(request.codes)} codes")

    async def handle_code_request(code_request: CodeRequest):
        try:
            logging.info(f"Executing {code_request.language} code: {code_request.code[:50]}...")

            response_data = await execute_code(
                code_request.language,
                code_request.code,
                mode=code_request.mode,
                args=code_request.args,
                validate_dependencies=code_request.validate_dependencies,
                user_input=code_request.input if code_request.mode == "stdin" else None
            )

            return CodeResponse(**response_data)

        except Exception as e:
            logging.exception(f"Error executing {code_request.language} code.")
            return CodeResponse(output=None, error=str(e))

    responses = await asyncio.gather(*(handle_code_request(code_request) for code_request in request.codes))

    execution_time = time.time() - start_time
    logging.info(f"Multi-code execution completed in {execution_time:.2f} seconds")

    return MultiCodeResponse(responses=responses)


@router.post("/bulk", response_model=BulkCodeResponse, summary="Execute one code snippet with multiple inputs")
async def compile_bulk_codes(request: BulkCodeRequest):
    start_time = time.time()
    logging.info(f"Received bulk execution request with {len(request.input)} inputs")

    async def handle_code_request(user_input: str):
        try:
            logging.info(f"Executing {request.language} code with input: {user_input[:30]}...")

            response_data = await execute_code(
                request.language, 
                request.code, 
                mode=request.mode, 
                args=request.args, 
                user_input=user_input
            )

            return CodeResponse(**response_data)

        except Exception as e:
            logging.exception(f"Error during execution with input '{user_input}'")
            return CodeResponse(output=None, error=str(e))

    responses = await asyncio.gather(*(handle_code_request(user_input) for user_input in request.input))

    execution_time = time.time() - start_time
    logging.info(f"Bulk code execution completed in {execution_time:.2f} seconds")

    return BulkCodeResponse(responses=responses)