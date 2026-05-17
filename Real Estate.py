import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import gradio as gr


df = pd.read_csv("BHP.csv")



# Remove unnecessary columns
df = df.drop(columns=['area_type', 'availability', 'society', 'balcony'])

# Fill missing values
df['location'] = df['location'].fillna('Whitefield')
df['size'] = df['size'].fillna('2 BHK')
df['bath'] = df['bath'].fillna(df['bath'].median())

# Convert size to BHK number
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert total_sqft values properly
def convert_sqft_to_num(x):

    tokens = str(x).split('-')

    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2

    try:
        return float(x)

    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

# Remove invalid rows
df = df.dropna(subset=['total_sqft'])



X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)



column_trans = make_column_transformer(

    (OneHotEncoder(handle_unknown='ignore'), ['location']),

    remainder='passthrough'
)




model = LinearRegression()

pipe = make_pipeline(column_trans, model)



pipe.fit(X_train, y_train)



area_price = df.groupby('location')['price'].mean()

def format_price(price):

    if price >= 100:
        return f"₹ {price/100:.2f} Crore"

    else:
        return f"₹ {price:.2f} Lakhs"


def predict_price(location, sqft, bath, bhk):

    # Check location
    if location not in df['location'].unique():

        return f"""
❌ Location not found in dataset.

Try another Bengaluru location.
"""

    # Input dataframe
    input_data = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    # Predict
    prediction = pipe.predict(input_data)[0]

    # Price per sqft
    price_per_sqft = (prediction * 100000) / sqft

    # Home category
    if sqft < 1000:
        category = "Compact Home"

    elif sqft < 2000:
        category = "Family Home"

    else:
        category = "Luxury Home"

    # Investment score
    if prediction < 50:
        score = "⭐⭐⭐ Good Budget Investment"

    elif prediction < 100:
        score = "⭐⭐⭐⭐ High Potential Area"

    else:
        score = "⭐⭐⭐⭐⭐ Premium Investment"

    # Area average
    avg_area_price = area_price[location]

    # Final Output
    return f"""
🏠  REAL ESTATE REPORT


📍 Location: {location}
💰 Estimated Price:
₹ {format_price(prediction)} 
📏 Total Area:
{sqft} sqft
🛏 BHK:
{bhk}
🛁 Bathrooms:
{bath}

🏡 Property Type:
{category}

💵 Price Per Sqft:
₹ {price_per_sqft:.2f}

📊 Average Area Price:
₹ {format_price(avg_area_price)}

⭐ Investment Score:
{score}

Thank You..

"""



def recommend_area(budget):

    affordable = area_price[area_price <= budget]

    if len(affordable) == 0:
        return "❌ No affordable areas found."

    result = "🏘 Recommended Areas:\n\n"

    for area, price in affordable.sort_values().head(10).items():

        result += f"📍 {area} → ₹ {price:.2f} Lakhs\n"

    return result



locations = sorted(df['location'].unique())



with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
# 🏠 Real Estate Assistant

Predict Bengaluru house prices using Machine Learning.
"""
    )


    with gr.Tab("🏠 Price Prediction"):

        location = gr.Dropdown(
            choices=locations,
            label="Select Location"
        )

        sqft = gr.Number(
            label="Total Square Feet",
            minimum=100,
           
            
        )

        bath = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            label="Bathrooms"
        )

        bhk = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            label="BHK"
        )

        predict_btn = gr.Button("Predict Price")

        output = gr.Textbox(
            label="Prediction Report",
            lines=18
        )
        predict_btn.click(
            fn=predict_price,
            inputs=[location, sqft, bath, bhk],
            outputs=output
        )

   

    with gr.Tab("💰 Budget Recommendation"):

        budget = gr.Number(
            label="Enter Your Budget (Lakhs)"
        )

        recommend_btn = gr.Button("Recommend Areas")

        recommend_output = gr.Textbox(
            label="Recommended Areas",
            lines=12
        )

        recommend_btn.click(
            fn=recommend_area,
            inputs=budget,
            outputs=recommend_output
        )



app.launch(share=True)