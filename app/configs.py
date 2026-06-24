LANG_PYTHON = "Python3"
LANG_NODEJS = "NodeJS"
LANG_JAVA = "Java"
LANG_C = "C"
LANG_CPP = "C++"

LANGUAGE_CONFIG = {
    LANG_PYTHON: {
        "ext": ".py",
        "cmd": ["python3", "{file}"],
        "validate": "validate_python_dependencies"
    },
    LANG_NODEJS: {
        "ext": ".js",
        "cmd": ["node", "{file}"],
        "validate": "validate_node_dependencies",
        "force_exit": True  # Ensure the process exits
    },
    LANG_JAVA: {
        "ext": ".java",
        "cmd": ["java", "-cp", "{file}", "Main"],  # Use dynamic file directory
        "compile": ["javac", "{file}"],
        "validate": "validate_java_dependencies",
        "force_main_class": "Main"  # Ensure main class is named correctly
    },
    LANG_C: {
        "ext": ".c",
        "cmd": ["{executable}"],
        "compile": ["gcc", "{file}", "-o", "{executable}"]
    },
    LANG_CPP: {
        "ext": ".cpp",
        "cmd": ["{executable}"],
        "compile": ["g++", "{file}", "-o", "{executable}"]
    }
}
