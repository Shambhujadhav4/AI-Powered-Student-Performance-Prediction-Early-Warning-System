# AI-Powered Student Performance Prediction & Early Warning System

## Overview

An intelligent machine learning system designed to predict student academic performance and identify students at risk of failing. The system leverages multiple performance indicators to provide accurate predictions and personalized study recommendations.

**Domain:** Education  
**Repository:** [AI-Powered-Student-Performance-Prediction-Early-Warning-System](https://github.com/Shambhujadhav4/AI-Powered-Student-Performance-Prediction-Early-Warning-System)

## Features

- 📊 **Performance Prediction** - Predict student final grades based on key indicators
- ⚠️ **Early Warning System** - Identify students at risk of failing
- 💡 **Personalized Recommendations** - Provide actionable study suggestions
- 📈 **Data Visualization** - Visualize correlations and trends in student data
- 🎯 **Performance Classification** - Categorize students as "At Risk", "Average", or "High Performer"

## Key Metrics Used

The system makes predictions based on the following student performance indicators:

- **Study Time** (hours per day)
- **Absences** (number of classes missed)
- **Previous Grade** (past academic performance)
- **Assignments Completed** (assignment submission count)
- **Participation Level** (classroom engagement, scale 1-5)

## Project Structure

```
├── app.py                          # Streamlit web application
├── student_prediction.ipynb        # Jupyter notebook with model development
├── student_performance_600.csv     # Dataset with 600 student records
├── student_model.pkl               # Trained ML model
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Technology Stack

- **Python 3.x**
- **Streamlit** - Interactive web application framework
- **Pandas & NumPy** - Data processing and manipulation
- **Scikit-learn** - Machine learning library
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Model development and exploration

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shambhujadhav4/AI-Powered-Student-Performance-Prediction-Early-Warning-System.git
   cd AI-Powered-Student-Performance-Prediction-Early-Warning-System
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Input Student Data:**
   - Study Time (0-10 hours/day)
   - Absences (0-30 classes)
   - Previous Grade (0-100)
   - Assignments Completed (0-10)
   - Participation Level (1-5)

2. **Get Predictions:**
   - Click "Predict Performance" button
   - View predicted final grade
   - See performance category (At Risk/Average/High Performer)

3. **Receive Recommendations:**
   - Get personalized study suggestions
   - Identify areas for improvement
   - Receive positive reinforcement for good habits

### Exploring the Model

Open `student_prediction.ipynb` in Jupyter Notebook to:

- Explore the dataset
- Understand feature correlations
- Review model training process
- Analyze model performance metrics

## Model Details

- **Algorithm:** Random Forest Regressor
- **Training Data:** 600 student records
- **Train-Test Split:** 80-20
- **Random State:** 42 (for reproducibility)

### Performance Thresholds

- **At Risk:** Final Grade < 50
- **Average:** Final Grade 50-74
- **High Performer:** Final Grade ≥ 75

### Recommendations Triggers

- **Low Study Time:** Triggered if study time < 3 hours/day
- **Poor Attendance:** Triggered if absences > 5
- **Low Participation:** Triggered if participation < 3
- **Positive Reinforcement:** When study habits meet targets

## Dataset

The system uses `student_performance_600.csv` containing:
- **Records:** 600 student observations
- **Features:** StudyTime, Absences, PreviousGrade, AssignmentsCompleted, Participation
- **Target:** FinalGrade

## Project Workflow

1. **Data Loading & Exploration**
   - Load CSV dataset
   - Perform exploratory data analysis (EDA)
   - Check for missing values

2. **Data Visualization**
   - Correlation heatmap
   - Study time vs final grade scatter plot
   - Feature distribution analysis

3. **Feature Engineering**
   - Select 5 key predictive features
   - No missing values handling required

4. **Model Training**
   - Split data (80% train, 20% test)
   - Train Random Forest Regressor
   - Save trained model to `student_model.pkl`

5. **Deployment**
   - Streamlit web application for predictions
   - Real-time grade predictions
   - Personalized recommendations

## Performance Metrics

The model evaluation includes:
- Mean Squared Error (MSE)
- Feature importance scores
- Model accuracy on test data

## Requirements

See `requirements.txt` for all dependencies:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Error Handling

The application includes:
- Model file existence validation
- Error handling for model loading
- Input validation for numerical inputs
- User-friendly error messages

## Future Enhancements

- [ ] Add more performance indicators
- [ ] Implement multiple ML algorithms comparison
- [ ] Add student progress tracking
- [ ] Develop mobile application
- [ ] Create admin dashboard for educators
- [ ] Implement real-time alerts for at-risk students
- [ ] Add historical prediction tracking

## Configuration

Key parameters in `app.py`:

```python
MODEL_PATH = "student_model.pkl"              # Path to trained model
GRADE_THRESHOLD_AT_RISK = 50                   # At-risk threshold
GRADE_THRESHOLD_AVERAGE = 75                   # Average-high performer threshold
MIN_STUDY_FOR_GOOD = 3                         # Minimum study hours for good performance
MAX_ABSENCES_FOR_GOOD = 5                      # Maximum absences threshold
MIN_PARTICIPATION = 3                          # Minimum participation level
```

## Troubleshooting

**Issue:** `Model file 'student_model.pkl' not found!`
- **Solution:** Ensure you've run the Jupyter notebook to train the model first

**Issue:** Module import errors
- **Solution:** Verify all dependencies are installed: `pip install -r requirements.txt`

**Issue:** Port 8501 already in use
- **Solution:** Run with different port: `streamlit run app.py --server.port 8502`


---

**Last Updated:** March 17, 2026  
**Status:** Active Development
