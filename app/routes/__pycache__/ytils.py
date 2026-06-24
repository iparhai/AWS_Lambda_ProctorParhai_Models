import subprocess
import re
import multiprocessing
import resource
import os
from fastapi import HTTPException

from configs import LANGUAGE_CONFIG

ADMIN_TOKEN = "secure_admin_token"

def load_allowed_libs(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return set(line.strip() for line in f if line.strip())

    return set()

ALLOWED_PYTHON_LIBS = load_allowed_libs("allowed_python_libs.txt")
ALLOWED_NODE_LIBS = load_allowed_libs("allowed_node_libs.txt")
ALLOWED_JAVA_CLASSES = load_allowed_libs("allowed_java_classes.txt")


def validate_python_dependencies(code):
    imports = re.findall(r'^\s*(?:import|from) (\w+)', code, re.MULTILINE)
    for pkg in imports:
        if pkg not in ALLOWED_PYTHON_LIBS:
            return False, f"Library '{pkg}' is not allowed."
    return True, None

def validate_node_dependencies(code):
    matches = re.findall(r"(?:import\s+\w+\s+from\s+['\"]([\w-]+)['\"]|require\(['\"]([\w-]+)['\"]\))", code)
    for pkg in matches:
        if pkg not in ALLOWED_NODE_LIBS:
            return False, f"Library '{pkg}' is not allowed."
    return True, None

def validate_java_dependencies(code):
    imports = re.findall(r'^\s*import\s+([\w\.]+);', code, re.MULTILINE)
    for pkg in imports:
        if pkg not in ALLOWED_JAVA_CLASSES:
            return False, f"Java package '{pkg}' is not allowed."
    return True, None

def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10)) # 10 second CPU time limit
    resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024)) # 100MB memory limit

def run_code(cmd):
    try:
        # set_limits()
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Execution timed out after {e.timeout} seconds"
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except Exception as e:
        return False, str(e)

def execute_code(language, code, validate_dependencies):
    """
    Executes code in the specified language securely and returns the result.
    """
    if language not in LANGUAGE_CONFIG:
        raise HTTPException(status_code=400, detail="Unsupported language")

    config = LANGUAGE_CONFIG[language]
    code_file = f"/tmp/code{config['ext']}"

    try:
        with open(code_file, "w") as f:
            f.write(code)

        if "compile" in config:
            compile_cmd = [cmd.replace("{file}", code_file) for cmd in config["compile"]]
            subprocess.run(compile_cmd, check=True)

        exec_cmd = [cmd.replace("{file}", code_file) for cmd in config["cmd"]]
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply_async(run_code, (exec_cmd,))
            stdout, stderr = result.get()
            
        return {
            "output": stdout if stdout else None,
            "error": stderr if stderr else None
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or "Compilation/Execution failed")