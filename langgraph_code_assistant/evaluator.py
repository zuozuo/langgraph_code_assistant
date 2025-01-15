class CodeEvaluator:
    def __init__(self):
        pass

    def check_imports(self, imports: str) -> bool:
        """Check if imports are valid"""
        try:
            exec(imports)
            return True
        except Exception:
            return False

    def check_execution(self, imports: str, code: str) -> bool:
        """Check if code executes successfully"""
        try:
            exec(imports + "\n" + code)
            return True
        except Exception:
            return False

    def evaluate_solution(self, solution) -> dict:
        """Evaluate a complete code solution"""
        return {
            "imports_valid": self.check_imports(solution.imports),
            "code_executes": self.check_execution(solution.imports, solution.code)
        }
