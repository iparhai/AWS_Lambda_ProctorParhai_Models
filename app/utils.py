import asyncio
import subprocess
import re
import os
import tempfile
from fastapi import HTTPException
from configs import LANGUAGE_CONFIG, LANG_PYTHON, LANG_NODEJS, LANG_JAVA, LANG_C, LANG_CPP

ADMIN_TOKEN = "secure_admin_token"

async def run_code(cmd, user_input=None):
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate(input=user_input.encode() if user_input else None)
    return stdout.decode().strip(), stderr.decode().strip()

async def execute_code(language, code, mode="stdin", args=None, validate_dependencies=False, user_input=None):
    """
    Executes code in a specified language securely.

    Parameters:
        - language (str): The programming language.
        - code (str): The source code.
        - mode (str): Execution mode ("stdin" or "args").
        - args (list, optional): Command-line arguments (for "args" mode).
        - validate_dependencies (bool, optional): Whether to validate dependencies.
        - user_input (str, optional): Input for stdin mode.

    Returns:
        dict: {"output": str, "error": str}
    """

    if language not in LANGUAGE_CONFIG:
        raise HTTPException(status_code=400, detail="Unsupported language")

    config = LANGUAGE_CONFIG[language]

    temp_dir = tempfile.TemporaryDirectory()
    try:
        code_file_path = os.path.join(temp_dir.name, f"temp_code{config['ext']}")

        if language == LANG_JAVA and config.get("force_main_class", False):
            code = re.sub(r'public class \w+', "public class Main", code)
            code_file_path = os.path.join(temp_dir.name, "Main.java")

        if language == LANG_NODEJS and config.get("force_exit", False):
            code += "\nprocess.exit(0);\n"

        with open(code_file_path, "w") as temp_file:
            temp_file.write(code)

        executable_file = None
        if "compile" in config:
            executable_file = os.path.join(temp_dir.name, "executable")

            compile_cmd = [cmd.replace("{file}", code_file_path).replace("{executable}", executable_file) for cmd in config["compile"]]
            compile_process = subprocess.run(compile_cmd, capture_output=True, text=True)

            if language == LANG_JAVA:
                code_file_path = os.path.join(temp_dir.name)

            if compile_process.returncode != 0:
                return {"output": None, "error": compile_process.stderr.strip()}

            exec_cmd = [cmd.replace("{file}", code_file_path).replace("{executable}", executable_file) for cmd in config["cmd"]]
        else:
            exec_cmd = [cmd.replace("{file}", code_file_path) for cmd in config["cmd"]]        
        
        
        
        if mode == "args" and args:
            exec_cmd.extend(args)

        stdout, stderr = await run_code(exec_cmd, user_input if mode == "stdin" else None)
        
        return {
            "output": stdout if stdout else None,
            "error": stderr if stderr else None
        }

    except subprocess.TimeoutExpired:
        return {"output": None, "error": "Execution timeout"}
    except Exception as e:
        return {"output": None, "error": str(e)}
    finally:
        temp_dir.cleanup()  # Properly remove temp directory
