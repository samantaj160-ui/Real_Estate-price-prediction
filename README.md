# Real_Estate-price-prediction

This project is a simple Machine Learning based web application that predicts house prices in Bengaluru city.

Users can enter:
- Location
- Total square feet
- Number of bathrooms
- BHK

and the system predicts the estimated property price.

The project also provides:
- Budget based area suggestions
- Property category
- Price per square feet
- Investment analysis

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio

---

## Files Included

| File | Description |
|---|---|
| Real Estate.py | Main project code |
| BHP.csv | Dataset |
| README.md | Project information |

---

## How To Run

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run The Project

```bash
python app.py
```

After running the file, open the local URL shown in terminal.

Example:

```text
http://127.0.0.1:7860
```

---

## Why The Project Does Not Run Directly On GitHub

GitHub is mainly used for storing project files and source code.

Python applications and Gradio interfaces cannot run directly on GitHub pages.

To make the application accessible online, deployment platforms are required such as:
- Hugging Face Spaces
- Render
- Railway
- link for check this("https://huggingface.co/spaces/jit99/Real_Estate-price")

---

## Model Used

Linear Regression

---

## Dataset Information

The dataset contains Bengaluru house details like:
- location
- square feet
- BHK
- bathrooms
- price

which are used to train the prediction model.

---

## Project Objective

The main objective of this project is to understand:
- Data Cleaning
- Feature Engineering
- Machine Learning Model Training
- User Interface Development using Gradio

and build a practical real estate price prediction system.
