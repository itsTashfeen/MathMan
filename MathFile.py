import streamlit as st
import random
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import uuid
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

class MLPerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.features = []
        self.targets = []

    def train(self, features, targets):
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features, targets)

    def predict_difficulty(self, user_features):
        scaled_features = self.scaler.transform([user_features])
        return self.model.predict(scaled_features)[0]

class UserProfile:
    def __init__(self, username):
        self.username = username
        self.user_id = str(uuid.uuid4())
        self.xp = 0
        self.level = 1
        self.total_problems_solved = 0
        self.learning_path = {
            'Algebra': {'mastery': 0, 'completed_problems': 0, 'difficulty': 'Easy'},
            'Calculus': {'mastery': 0, 'completed_problems': 0, 'difficulty': 'Easy'},
            'Trigonometry': {'mastery': 0, 'completed_problems': 0, 'difficulty': 'Easy'}
        }
        self.achievements = []
        self.diagnostic_results = None

    def update_performance(self, problem_type, solved, difficulty):
        path = self.learning_path[problem_type]
        
        # XP and Level System
        base_xp = {'Easy': 10, 'Medium': 25, 'Hard': 50, 'Professional': 100}
        self.xp += base_xp.get(difficulty, 10)
        self.level = 1 + self.xp // 100

        path['completed_problems'] += 1
        if solved:
            path['mastery'] = min(100, path['mastery'] + 10)
            
            # Dynamic Difficulty Adjustment
            if path['mastery'] > 70 and path['difficulty'] == 'Easy':
                path['difficulty'] = 'Medium'
            elif path['mastery'] > 85:
                path['difficulty'] = 'Hard'

            # Achievement System
            self._check_achievements(problem_type)

    def _check_achievements(self, problem_type):
        mastery = self.learning_path[problem_type]['mastery']
        
        new_achievements = []
        if mastery >= 50:
            new_achievements.append(f"{problem_type} Explorer")
        if mastery >= 80:
            new_achievements.append(f"{problem_type} Master")
        
        self.achievements.extend(set(new_achievements) - set(self.achievements))

class AdvancedMathProblemGenerator:
    def __init__(self):
        self.ml_predictor = MLPerformancePredictor()
        self.problem_types = {
            'Algebra': self.generate_advanced_algebra_problem,
            'Calculus': self.generate_advanced_calculus_problem,
            'Trigonometry': self.generate_advanced_trigonometry_problem
        }

    def generate_problem_set(self, problem_type, difficulty, num_problems):
        generator = self.problem_types[problem_type]
        problems = []
        
        for _ in range(num_problems):
            # Introducing true randomness
            random.seed(datetime.now().timestamp())
            problem, solution = generator(difficulty)
            problems.append({
                'id': str(uuid.uuid4()),
                'problem': problem,
                'solution': solution,
                'difficulty': difficulty,
                'solved': False
            })

        return problems

    def generate_advanced_algebra_problem(self, difficulty):
        random.seed(datetime.now().timestamp())
        x = sp.Symbol('x')
        
        difficulties = {
            'Easy': lambda: (
                f"Solve: {random.randint(1,10)}x + {random.randint(1,10)} = {random.randint(10,50)}",
                f"x = {random.randint(1,10)}"
            ),
            'Medium': lambda: (
                f"Solve the quadratic: {random.randint(1,5)}x^2 + {random.randint(1,10)}x + {random.randint(1,10)} = 0",
                str(sp.solve(sp.randint(1,5) * x**2 + sp.randint(1,10) * x + sp.randint(1,10), x))
            ),
            'Hard': lambda: (
                f"Solve: {random.randint(1,10)}x^3 - {random.randint(1,10)}x = {random.randint(50,200)}",
                f"x = {random.randint(1,10)}"
            )
        }
        
        return difficulties.get(difficulty, difficulties['Easy'])()

    def generate_advanced_calculus_problem(self, difficulty):
        random.seed(datetime.now().timestamp())
        x = sp.Symbol('x')
        
        difficulties = {
            'Easy': lambda: (
                f"Find the derivative of f(x) = {random.randint(1,5)}x^2 + {random.randint(1,10)}x",
                f"f'(x) = {random.randint(1,5)*2}x + {random.randint(1,10)}"
            ),
            'Medium': lambda: (
                f"Integrate f(x) = x^3 + {random.randint(1,10)}x^2 + {random.randint(1,10)}",
                f"F(x) = x^4/4 + {random.randint(1,10)}x^3/3 + {random.randint(1,10)}x + C"
            ),
            'Hard': lambda: (
                f"Find the derivative of f(x) = sin({random.randint(1,5)}x) * e^x",
                f"f'(x) = cos({random.randint(1,5)}x) * e^x + sin({random.randint(1,5)}x) * e^x"
            )
        }
        
        return difficulties.get(difficulty, difficulties['Easy'])()

    def generate_advanced_trigonometry_problem(self, difficulty):
        random.seed(datetime.now().timestamp())
        
        difficulties = {
            'Easy': lambda: (
                f"Simplify sin(x)^2 + cos({random.randint(1,5)}x)^2",
                "1"
            ),
            'Medium': lambda: (
                f"Solve: sin(x) = {random.uniform(0.1, 0.9):.2f}",
                f"x = π/{random.randint(2,6)} or {random.randint(3,7)}π/6"
            ),
            'Hard': lambda: (
                f"Find the period of f(x) = tan({random.randint(1,5)}x)",
                "π"
            )
        }
        
        return difficulties.get(difficulty, difficulties['Easy'])()

def main():
    st.set_page_config(layout="wide")
    st.title("🧮 Advanced Math Problem Generator")

    # Initialize session state
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = UserProfile('Anonymous')
    
    if 'current_problems' not in st.session_state:
        st.session_state.current_problems = []

    generator = AdvancedMathProblemGenerator()

    # Merged Performance and Analytics Tab
    tab1, tab2, tab3 = st.tabs([
        "Problem Generator", 
        "Performance & Analytics", 
        "Challenge Mode"
    ])

    with tab1:
        st.header("Generate Math Problems")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            problem_type = st.selectbox("Problem Type", list(generator.problem_types.keys()))
        with col2:
            difficulty = st.selectbox("Difficulty", ['Easy', 'Medium', 'Hard'])
        with col3:
            num_problems = st.slider("Number of Problems", 1, 20, 5)

        if st.button("Generate Problems"):
            st.session_state.current_problems = generator.generate_problem_set(
                problem_type, difficulty, num_problems
            )

        if st.session_state.current_problems:
            for problem_data in st.session_state.current_problems:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.latex(problem_data['problem'])
                with col2:
                    solved = st.checkbox(
                        "Solved", 
                        key=problem_data['id'], 
                        value=problem_data['solved']
                    )
                
                problem_data['solved'] = solved
                
                if solved:
                    st.session_state.user_profile.update_performance(
                        problem_type, 
                        solved, 
                        problem_data['difficulty']
                    )
                
                with st.expander("Solution"):
                    st.latex(problem_data['solution'])

    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Performance Dashboard")
            profile = st.session_state.user_profile
            
            # Performance Metrics
            cols = st.columns(4)
            metrics = [
                ("Total Problems", profile.total_problems_solved),
                ("XP", profile.xp),
                ("Level", profile.level),
                ("Achievements", len(profile.achievements))
            ]
            
            for col, (name, value) in zip(cols, metrics):
                col.metric(name, value)
            
            # Learning Path Progress
            st.subheader("Learning Path")
            progress_data = pd.DataFrame.from_dict(profile.learning_path, orient='index')
            st.dataframe(progress_data, use_container_width=True)
        
        with col2:
            st.header("Achievements")
            if profile.achievements:
                for achievement in profile.achievements:
                    st.success(f"🏆 {achievement}")
            else:
                st.info("Complete problems to earn achievements!")

    with tab3:
        st.header("Challenge Mode")
        st.info("Competitive features coming soon!")

if __name__ == "__main__":
    main()
