
# To explain the `__init__.py` files more clearly:

# 1. They're special Python files that mark a directory as a Python package
# 2. They can be completely empty (just create an empty file named `__init__.py`)
# 3. They enable you to use relative imports between your files
# 4. Without them, Python wouldn't recognize your directories as packages that can be imported from

# For example, with this structure, from anywhere in your project you can import:
# ```python
# from src.core.config import GOOGLE_API_KEY
# from src.utils.models_available import check_models  # assuming this function exists
# ```

# The `__init__.py` files are what make these imports possible. When you run the application with `python src/main.py`, Python will know how to find all the modules because of these package markers.



# Note 2
# Scenario 1: Using 'from core.config import xxx' and running 'python src/main.py'
# Python treats the directory where you run the command as the root of the import system
# The command tells Python to run the file at path src/main.py
# Inside main.py, when you use from core.config import xxx, Python looks for a core module in the same directory as main.py

# Scenario 2: Using from src.core.config import xxx and running python -m src.main
# The -m flag tells Python to treat the argument (src.main) as a module path
# Python adds the current directory to the beginning of the import path
# Then it looks for a module named src, then a submodule named main
# Inside main.py, when you use from src.core.config import xxx, it can find src because that's now in the import path

# Scenario 2 works because the -m flag makes Python start from the current directory and navigate through modules by their names
# You can fully qualify your imports from the project root


