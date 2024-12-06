import streamlit as st
import random
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


class AdvancedMathProblemGenerator:
    def __init__(self):
        random.seed()
        self.problem_types = {
            'Integration': [
                self.generate_rational_power_integral,
                self.generate_sqrt_complex_integral,
                self.generate_power_chain_integral,
                self.generate_logarithmic_integral,
                self.generate_definite_integral,
                self.generate_u_substitution_integral,
                self.generate_trigonometric_integral,
                self.generate_exponential_integral
            ],
            'Derivatives': [
                self.generate_polynomial_derivative,
                self.generate_trigonometric_derivative,
                self.generate_logarithmic_derivative,
                self.generate_chain_rule_derivative
            ],
            'Linear Equations': [
                self.generate_linear_equation,
                self.generate_point_slope_form,
                self.generate_standard_form_line
            ]
        }

        self.difficulty_ranges = {
            'Easy': {
                'coeff_range': (1, 5),
                'power_range': (1, 3),
                'complexity': 1
            },
            'Medium': {
                'coeff_range': (3, 10),
                'power_range': (2, 5),
                'complexity': 2
            },
            'Hard': {
                'coeff_range': (5, 20),
                'power_range': (3, 7),
                'complexity': 3
            },
            'Professional': {
                'coeff_range': (10, 50),
                'power_range': (4, 10),
                'complexity': 4
            }
        }

    def random_coefficient(self, difficulty='Medium', positive_only=True):
        """Generate a random coefficient based on difficulty."""
        ranges = self.difficulty_ranges[difficulty]['coeff_range']
        coefficient = random.randint(ranges[0], ranges[1])
        return abs(coefficient) if positive_only else coefficient * random.choice([-1, 1])

    def random_power(self, difficulty='Medium', max_power=None):
        """Generate a random power based on difficulty."""
        ranges = self.difficulty_ranges[difficulty]['power_range']
        max_power = max_power or ranges[1]
        return random.randint(ranges[0], max_power)

    # Integration Problem Generators
    def generate_rational_power_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        b = self.random_coefficient(difficulty)
        c = self.random_coefficient(difficulty)
        d = self.random_coefficient(difficulty)

        problem = fr"\int \frac{{{a}x^2 + {b}x}}{{{c}x + {d}}} \, dx"

        x = sp.Symbol('x')
        expr = (a * x ** 2 + b * x) / (c * x + d)
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    def generate_sqrt_complex_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        b = self.random_coefficient(difficulty)

        problem = fr"\int x \sqrt{{{a}x^2 + {b}}} \, dx"

        x = sp.Symbol('x')
        expr = x * sp.sqrt(a * x ** 2 + b)
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    def generate_power_chain_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        p = self.random_power(difficulty)

        problem = fr"\int x(x + {a})^{p} \, dx"

        x = sp.Symbol('x')
        expr = x * (x + a) ** p
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    def generate_logarithmic_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty, positive_only=False)
        b = self.random_power(difficulty, max_power=4)

        problem = fr"\int \frac{{(1 + {a}\ln x)^{b}}}{{x}} \, dx"

        x = sp.Symbol('x')
        expr = (1 + a * sp.ln(x)) ** b / x
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    def generate_definite_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        b = self.random_coefficient(difficulty)
        lower = self.random_coefficient(difficulty, max_power=3)
        upper = lower + self.random_coefficient(difficulty, max_power=3)

        problem = fr"\int_{{{lower}}}^{{{upper}}} ({a} + {b}x) \, dx"

        x = sp.Symbol('x')
        expr = a + b * x
        solution = sp.latex(sp.integrate(expr, (x, lower, upper)))

        return problem, solution

    def generate_u_substitution_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        b = self.random_coefficient(difficulty)

        problem = fr"\int \frac{{{a}x + {b}}}{{\sqrt{{x}}}} \, dx"

        x = sp.Symbol('x')
        expr = (a * x + b) / sp.sqrt(x)
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    def generate_trigonometric_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        b = self.random_coefficient(difficulty)

        problem = fr"\int {a}\sin(x) \cos({b}x) \, dx"

        x = sp.Symbol('x')
        expr = a * sp.sin(x) * sp.cos(b * x)
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    def generate_exponential_integral(self, difficulty='Medium'):
        a = self.random_coefficient(difficulty)
        b = self.random_coefficient(difficulty)

        problem = fr"\int {a}x e^{{{b}x}} \, dx"

        x = sp.Symbol('x')
        expr = a * x * sp.exp(b * x)
        solution = sp.latex(sp.integrate(expr, x))

        return problem, solution

    # Add similar generator methods for Derivatives and Linear Equations...

    def generate_problem_set(self, problem_type, difficulty, num_problems):
        """Generate a set of problems for a specific type and difficulty."""
        problem_generators = self.problem_types.get(problem_type, [])
        problems = []

        for _ in range(num_problems):
            generator = random.choice(problem_generators)
            problem, solution = generator(difficulty)
            problems.append((problem, solution))

        return problems


def main():
    st.title("🧮 Advanced Math Problem Generator")

    generator = AdvancedMathProblemGenerator()

    # Sidebar for configuration
    st.sidebar.header("Problem Configuration")
    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        list(generator.problem_types.keys())
    )
    difficulty = st.sidebar.selectbox(
        "Select Difficulty",
        list(generator.difficulty_ranges.keys())
    )
    num_problems = st.sidebar.slider(
        "Number of Problems",
        min_value=1, max_value=20, value=5
    )

    # Generate problems button
    if st.sidebar.button("Generate Problems"):
        problems = generator.generate_problem_set(problem_type, difficulty, num_problems)

        st.header(f"{problem_type} Practice Problems")

        for i, (problem, solution) in enumerate(problems, 1):
            st.markdown(f"**Problem {i}:** ${problem}$")
            with st.expander(f"Solution {i}"):
                st.markdown(f"**Solution:** ${solution}$")


if __name__ == "__main__":
    main()