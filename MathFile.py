import streamlit as st
import random
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


class MathProblemTracker:
    def __init__(self):
        self.user_problems = []
        self.performance_data = {
            'total_problems': 0,
            'solved_problems': 0,
            'difficulty_progression': {},
            'success_rates': {}
        }
        self.user_score = 0
        self.difficulty_level = 'Easy'

    def record_problem_attempt(self, problem, solved, difficulty):
        """Record user's attempt at a problem"""
        self.user_problems.append({
            'problem': problem,
            'solved': solved,
            'difficulty': difficulty,
            'timestamp': datetime.now()
        })

        self.performance_data['total_problems'] += 1
        if solved:
            self.performance_data['solved_problems'] += 1
            self.user_score += self._calculate_score(difficulty)

        self._update_difficulty_progression(difficulty, solved)

    def _calculate_score(self, difficulty):
        """Calculate score based on difficulty"""
        difficulty_scores = {
            'Easy': 10,
            'Medium': 25,
            'Hard': 50,
            'Professional': 100
        }
        return difficulty_scores.get(difficulty, 10)

    def _update_difficulty_progression(self, difficulty, solved):
        """Adjust difficulty based on performance"""
        progression_map = {
            'Easy': ['Easy', 'Medium'],
            'Medium': ['Easy', 'Medium', 'Hard'],
            'Hard': ['Medium', 'Hard', 'Professional'],
            'Professional': ['Hard', 'Professional']
        }

        if difficulty not in self.performance_data['difficulty_progression']:
            self.performance_data['difficulty_progression'][difficulty] = {
                'attempts': 0,
                'successes': 0
            }

        data = self.performance_data['difficulty_progression'][difficulty]
        data['attempts'] += 1
        if solved:
            data['successes'] += 1

        success_rate = data['successes'] / data['attempts'] if data['attempts'] > 0 else 0

        # Dynamically adjust difficulty
        if success_rate > 0.7 and difficulty != 'Professional':
            self.difficulty_level = progression_map[difficulty][1]
        elif success_rate < 0.3 and difficulty != 'Easy':
            self.difficulty_level = progression_map[difficulty][0]


class AdvancedMathProblemGenerator:
    def __init__(self):
        self.problem_tracker = MathProblemTracker()

        # Define problem types dictionary
        self.problem_types = {
            'Algebra': self.generate_algebra_problem,
            'Calculus': self.generate_calculus_problem,
            'Trigonometry': self.generate_trigonometry_problem
        }

        # Define difficulty ranges dictionary
        self.difficulty_ranges = {
            'Easy': (1, 10),
            'Medium': (10, 50),
            'Hard': (50, 100),
            'Professional': (100, 500)
        }

    def generate_problem_set(self, problem_type, difficulty, num_problems):
        """Generate a set of math problems"""
        problem_generator = self.problem_types.get(problem_type)
        if not problem_generator:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        problems = []
        for _ in range(num_problems):
            problem, solution = problem_generator(difficulty)
            problems.append((problem, solution))

        return problems

    def generate_algebra_problem(self, difficulty):
        """Generate an algebra problem based on difficulty"""
        x = sp.Symbol('x')

        # Easy: Simple linear equations
        if difficulty == 'Easy':
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            problem = f"{a}x + {b} = {a * 5 + b}"
            solution = f"x = {5}"

        # Medium: Quadratic equations
        elif difficulty == 'Medium':
            a = random.randint(1, 5)
            b = random.randint(1, 10)
            c = random.randint(1, 10)
            problem = f"{a}x^2 + {b}x + {c} = 0"
            solution = str(sp.solve(a * x ** 2 + b * x + c, x))

        # Hard: More complex algebraic equations
        else:
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            problem = f"{a}x^3 - {b}x = {a * 27 - b * 3}"
            solution = f"x = {3}"

        return problem, solution

    def generate_calculus_problem(self, difficulty):
        """Generate a calculus problem based on difficulty"""
        x = sp.Symbol('x')

        # Easy: Simple derivatives
        if difficulty == 'Easy':
            problem = "Differentiate f(x) = x^2 + 3x"
            solution = "f'(x) = 2x + 3"

        # Medium: Integrals
        elif difficulty == 'Medium':
            problem = "Integrate f(x) = x^3 + 2x"
            solution = "F(x) = (x^4)/4 + x^2 + C"

        # Hard: More complex calculus
        else:
            problem = "Find the derivative of f(x) = sin(x) * e^x"
            solution = "f'(x) = cos(x) * e^x + sin(x) * e^x"

        return problem, solution

    def generate_trigonometry_problem(self, difficulty):
        """Generate a trigonometry problem based on difficulty"""
        # Easy: Simple trig identities
        if difficulty == 'Easy':
            problem = "Simplify sin(x)^2 + cos(x)^2"
            solution = "1"

        # Medium: Trigonometric equations
        elif difficulty == 'Medium':
            problem = "Solve: sin(x) = 0.5"
            solution = "x = π/6 or 5π/6"

        # Hard: Complex trig problems
        else:
            problem = "Find the period of f(x) = tan(2x)"
            solution = "π"

        return problem, solution

    def generate_shareable_link(self, problem_set):
        """Create a shareable link for problem set"""
        # In a real-world scenario, this would be an actual URL generation
        # Here we'll simulate it with base64 encoding
        problem_str = str(problem_set)
        return base64.b64encode(problem_str.encode()).decode()

    def export_to_pdf(self, problem_set, filename='math_problems.pdf'):
        """Export problem set to PDF"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        for i, (problem, solution) in enumerate(problem_set, 1):
            story.append(Paragraph(f"Problem {i}: {problem}", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Solution: {solution}", styles['Normal']))
            story.append(Spacer(1, 12))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def generate_performance_visualization(self):
        """Create performance visualization"""
        performance_data = self.problem_tracker.performance_data

        plt.figure(figsize=(10, 6))
        difficulties = list(performance_data['difficulty_progression'].keys())
        success_rates = [
            (data['successes'] / data['attempts']) * 100
            for data in performance_data['difficulty_progression'].values()
        ]

        plt.bar(difficulties, success_rates)
        plt.title('Problem Success Rates by Difficulty')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode()


def main():
    st.title("🧮 Advanced Math Problem Generator")

    generator = AdvancedMathProblemGenerator()

    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Problem Generator",
        "Performance Tracker",
        "Export & Share",
        "Visualizations"
    ])

    with tab1:
        # Problem generation interface
        problem_type = st.selectbox("Problem Type", list(generator.problem_types.keys()))
        difficulty = st.selectbox("Difficulty", list(generator.difficulty_ranges.keys()))
        num_problems = st.slider("Number of Problems", 1, 20, 5)

        if st.button("Generate Problems"):
            problems = generator.generate_problem_set(problem_type, difficulty, num_problems)

            for i, (problem, solution) in enumerate(problems, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Problem {i}:** ${problem}$")
                with col2:
                    solved = st.checkbox(f"Solved Problem {i}")

                if solved:
                    generator.problem_tracker.record_problem_attempt(problem, True, difficulty)
                    st.success("Great job!")

                with st.expander(f"Solution {i}"):
                    st.markdown(f"**Solution:** ${solution}$")

    with tab2:
        st.header("Performance Tracking")
        tracker = generator.problem_tracker

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Problems", tracker.performance_data['total_problems'])
        with col2:
            st.metric("Solved Problems", tracker.performance_data['solved_problems'])

        st.metric("Current Score", tracker.user_score)
        st.metric("Recommended Difficulty", tracker.difficulty_level)

    with tab3:
        st.header("Export & Share")
        if 'problems' in locals():
            # PDF Export
            pdf_export = generator.export_to_pdf(problems)
            st.download_button(
                label="Download PDF",
                data=pdf_export,
                file_name="math_problems.pdf",
                mime="application/pdf"
            )

            # Shareable Link
            shareable_link = generator.generate_shareable_link(problems)
            st.text_input("Shareable Link", value=shareable_link, disabled=True)

    with tab4:
        st.header("Performance Visualization")
        performance_chart = generator.generate_performance_visualization()
        st.image(base64.b64decode(performance_chart))


if __name__ == "__main__":
    main()
