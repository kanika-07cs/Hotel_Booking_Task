# 🏨 Hotel Booking Cancellation Prediction

## Project Overview
This project focuses on predicting whether a hotel booking will be **cancelled or not** based on customer and booking details.  
By applying data preprocessing, feature engineering, and advanced machine learning models, this project aims to help hotels:
- Identify potential cancellations early.
- Optimize revenue management.
- Improve customer retention strategies.

**Dataset Link**: https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction

**Streamlit Link**: https://hotelbookingtask.streamlit.app/


## Dataset Description
| Column | Description |
|---------|-------------|
| date of reservation | Date when booking was made |
| number of week nights | Nights booked during weekdays |
| number of weekend nights | Nights booked during weekends |
| number of adults / children | Number of guests |
| type of meal | Meal plan selected |
| room type | Type of room booked |
| market segment type | Market source of the booking |
| car parking space | Whether parking was requested |
| repeated | Whether customer is a returning guest |
| P-C | Indicator related to cancellation probability |
| average price | Average daily rate |
| special requests | Number of requests made |
| booking status | Target variable (Cancelled or Not) |

## Data Preprocessing Steps

1. **Date Handling**  
   Converted date of reservation to datetime format using pd.to_datetime().

2. **Feature Engineering**
   - Created total_nights = number of week nights + number of weekend nights
   - Created total_guest = number of adults + number of children
   - Created a categorical feature booking_window_category from lead time:
     - Short-term (0–150)
     - Mid-term (150–300)
     - Long-term (300–450)

3. **Label Encoding & One-Hot Encoding**
   - Applied LabelEncoder on categorical features like booking status.
   - Applied pd.get_dummies() for type of meal, room type, and market segment type.

4. **Transformation & Skewness Reduction**
   - Used PowerTransformer (Yeo-Johnson) for continuous features:
     - average price
     - total_guest
     - total_nights

5. **Outlier Capping**
   - Capped outliers using the IQR method for numerical features.

6. **Feature Selection**
   - Selected the following features for modeling:
     'total_guest', 'car parking space', 'repeated', 'P-C',
      'average price', 'special requests', 'total_nights', 'booking_window_category',
      ...one-hot encoded columns...

7. **Data Balancing**
   - Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance the classes and prevent bias in prediction.

8. **Data Splitting & Scaling**
   - Split into training (80%) and test (20%) sets.
   - Scaled features using StandardScaler.

## Models Implemented:
- Logistic Regression
- KNN
- SVM (RBF Kernel)
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost 

## Evaluation Metrics
For each model, the following metrics were computed:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
<img width="450" height="455" alt="image" src="https://github.com/user-attachments/assets/4cda5d8b-5989-4ee8-97a5-75a8a2948fc5" />
- Confusion Matrix

## How to Run
1. Clone the repository
- git clone <repository_url>
- cd <repository_folder>
2. Run the Streamlit app
- streamlit run app.py

## Visualisation
- Total Guests per Month by Room Type
<img width="550" height="550" alt="image" src="https://github.com/user-attachments/assets/a48ab149-aa74-4172-a957-dad096a38dbd" />

- Cancellation Rate by Month
<img width="550" height="550" alt="image" src="https://github.com/user-attachments/assets/2c5b36e4-eaff-45ab-adbc-1842b1bef672" />

## Conclusion
- Data preprocessing, transformation, and SMOTE balancing significantly improved model performance.
- XGBoost/RandomForest provided the best predictive accuracy and ROC-AUC, making it ideal for production use.
- The workflow demonstrates the end-to-end machine learning pipeline from raw data → preprocessing → training → tuning → evaluation → model storage.
