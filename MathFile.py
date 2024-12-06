import streamlit as st
import random
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import hashlib


class UserProfile:
    def __init__(self, username):
        self.username = username
        self.user_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.total_problems_solved = 0
        self.total_score = 0
        self.problem_history = []
        self.badges = set()
        self.learning_path = {
            'Algebra': {'mastery': 0, 'completed_problems': 0},
            'Calculus': {'mastery': 0, 'completed_problems': 0},
            'Trigonometry': {'mastery': 0, 'completed_problems': 0}
        }

    def update_learning_path(self, problem_type, solved):
        path = self.learning_path[problem_type]
        path['completed_problems'] += 1
        if solved:
            path['mastery'] = min(100, path['mastery'] + 10)
            self.total_problems_solved += 1

        # Award badges based on performance
        if path['mastery'] >= 50:
            self.badges.add(f"{problem_type} Explorer")
        if path['mastery'] >= 80:
            self.badges.add(f"{problem_type} Master")

    def to_dict(self):
        return {
            'username': self.username,
            'user_id': self.user_id,
            'created_at': str(self.created_at),
            'total_problems_solved': self.total_problems_solved,
            'total_score': self.total_score,
            'learning_path': self.learning_path,
            'badges': list(self.badges)
        }


class AdvancedMathProblemGenerator:
    def __init__(self):
        self.seed = random.randint(1, 10000)
        random.seed(self.seed)

        self.problem_types = {
            'Algebra': {
                'generator': self.generate_advanced_algebra_problem,
                'subtypes': ['Linear', 'Quadratic', 'Polynomial', 'Exponential']
            },
            'Calculus': {
                'generator': self.generate_advanced_calculus_problem,
                'subtypes': ['Derivatives', 'Integrals', 'Limits', 'Series']
            },
            'Trigonometry': {
                'generator': self.generate_advanced_trigonometry_problem,
                'subtypes': ['Identities', 'Equations', 'Complex Functions']
            }
        }

    def generate_problem_set(self, problem_type, difficulty, num_problems):
        random.seed(self.seed)
        problem_generator = self.problem_types[problem_type]['generator']

        problems = []
        for _ in range(num_problems):
            # Introduce more randomness
            self.seed += 1
            random.seed(self.seed)

            problem, solution = problem_generator(difficulty)
            problems.append({
                'id': str(uuid.uuid4()),
                'problem': problem,
                'solution': solution,
                'difficulty': difficulty,
                'solved': False
            })

        return problems

    def generate_advanced_algebra_problem(self, difficulty):
        x = sp.Symbol('x')

        if difficulty == 'Easy':
            a, b = random.randint(1, 10), random.randint(1, 10)
            problem = f"Solve: {a}x + {b} = {a * 5 + b}"
            solution = f"x = {5}"

        elif difficulty == 'Medium':
            a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 10)
            problem = f"Solve the quadratic equation: {a}x^2 + {b}x + {c} = 0"
            solution = str(sp.solve(a * x ** 2 + b * x + c, x))

        else:
            a, b = random.randint(1, 10), random.randint(1, 10)
            problem = f"Solve the equation: {a}x^3 - {b}x = {a * 27 - b * 3}"
            solution = f"x = {3}"

        return problem, solution

    def generate_advanced_calculus_problem(self, difficulty):
        x = sp.Symbol('x')

        if difficulty == 'Easy':
            problem = "Find the derivative of f(x) = 3x^2 + 2x"
            solution = "f'(x) = 6x + 2"

        elif difficulty == 'Medium':
            problem = "Integrate f(x) = x^3 + 2x^2 + 5"
            solution = "F(x) = (x^4)/4 + (2x^3)/3 + 5x + C"

        else:
            problem = "Find the derivative of f(x) = sin(x) * e^x"
            solution = "f'(x) = cos(x) * e^x + sin(x) * e^x"

        return problem, solution

    def generate_advanced_trigonometry_problem(self, difficulty):
        if difficulty == 'Easy':
            problem = "Simplify sin(x)^2 + cos(x)^2"
            solution = "1"

        elif difficulty == 'Medium':
            problem = "Solve: sin(x) = 0.5"
            solution = "x = π/6 or 5π/6"

        else:
            problem = "Find the period of f(x) = tan(2x)"
            solution = "π"

        return problem, solution

    def generate_advanced_visualization(self, user_profile):
        # Create a more sophisticated visualization
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)

        # Mastery Level Heat Map
        learning_data = pd.DataFrame.from_dict(user_profile.learning_path, orient='index')
        sns.heatmap(learning_data[['mastery']], annot=True, cmap='YlGnBu', cbar=False)
        plt.title('Learning Path Mastery Levels')

        plt.subplot(2, 2, 2)
        # Problem Solving Distribution
        badges_count = len(user_profile.badges)
        plt.pie([badges_count, 10 - badges_count], labels=['Achieved', 'Pending'], autopct='%1.1f%%')
        plt.title('Badge Progress')

        plt.subplot(2, 2, 3)
        # Problem Type Performance
        problem_types = list(user_profile.learning_path.keys())
        mastery_levels = [data['mastery'] for data in user_profile.learning_path.values()]
        plt.bar(problem_types, mastery_levels)
        plt.title('Problem Type Performance')
        plt.ylim(0, 100)

        plt.subplot(2, 2, 4)
        # Time Series of Problem Solving
        if user_profile.problem_history:
            timestamps = [entry['timestamp'] for entry in user_profile.problem_history]
            solved = [entry['solved'] for entry in user_profile.problem_history]
            plt.plot(timestamps, solved, marker='o')
            plt.title('Problem Solving Trend')

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        plt.close()

        return base64.b64encode(image_png).decode()


def main():
    # Initialize session state for persistent data
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = UserProfile('Anonymous')

    if 'current_problems' not in st.session_state:
        st.session_state.current_problems = []

    st.title("🧮 Advanced Math Problem Generator")

    # Sidebar for user management
    st.sidebar.header("User Profile")
    username = st.sidebar.text_input("Username", value=st.session_state.user_profile.username)
    if st.sidebar.button("Update Profile"):
        st.session_state.user_profile = UserProfile(username)

    generator = AdvancedMathProblemGenerator()

    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Problem Generator",
        "Performance Dashboard",
        "Learning Analytics",
        "Challenge Mode"
    ])

    with tab1:
        st.header("Generate Math Problems")

        # Problem generation interface with more options
        col1, col2, col3 = st.columns(3)
        with col1:
            problem_type = st.selectbox("Problem Type", list(generator.problem_types.keys()))
        with col2:
            difficulty = st.selectbox("Difficulty", ['Easy', 'Medium', 'Hard', 'Professional'])
        with col3:
            num_problems = st.slider("Number of Problems", 1, 20, 5)

        if st.button("Generate Problems"):
            st.session_state.current_problems = generator.generate_problem_set(
                problem_type, difficulty, num_problems
            )

        # Render problems with individual solve tracking
        if st.session_state.current_problems:
            for problem_data in st.session_state.current_problems:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Problem:** ${problem_data['problem']}$")
                with col2:
                    solved = st.checkbox(
                        "Solved",
                        key=problem_data['id'],
                        value=problem_data['solved']
                    )

                # Update problem solve status
                problem_data['solved'] = solved

                if solved:
                    st.session_state.user_profile.update_learning_path(problem_type, True)

                with st.expander(f"Solution"):
                    st.markdown(f"**Solution:** ${problem_data['solution']}$")

    with tab2:
        st.header("Performance Dashboard")
        profile = st.session_state.user_profile

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Problems Solved", profile.total_problems_solved)
        with col2:
            st.metric("Total Score", profile.total_score)
        with col3:
            st.metric("Active Badges", len(profile.badges))

        st.subheader("Learning Path Progress")
        progress_data = pd.DataFrame.from_dict(profile.learning_path, orient='index')
        st.dataframe(progress_data)

    with tab3:
        st.header("Learning Analytics")
        visualization = generator.generate_advanced_visualization(st.session_state.user_profile)
        st.image(base64.b64decode(visualization))

    with tab4:
        st.header("Challenge Mode (Coming Soon)")
        st.info("Future collaborative and competitive features will be added here!")


if __name__ == "__main__":
    main()
