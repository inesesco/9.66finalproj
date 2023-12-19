import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the collected survey data
survey_data_path = 'survey_responses.csv'
survey_data = pd.read_csv(survey_data_path)

# Data Preprocessing

# Transforming the data for Bayesian analysis
headline_categories = {
    'Local Election Systems Across America Are Weak and Vulnerable': 'conservative',
    'The New Attack on Hispanic Voting Rights': 'liberal',
    'The Republican leading the probe of Hunter Biden has his own shell company and complicated friends': 'neutral',
    'New critical race theory laws have teachers scared, confused and self-censoring': 'liberal',
    'California Elementary School Students Are Learning To Love The Black Panthers': 'conservative',
    'Florida \'proudly\' teaches African American history, official says, as he defends rejecting AP course': 'neutral'
}

# Assigning categories to the headlines
for headline in headline_categories:
    survey_data[headline + " (Category)"] = headline_categories[headline]
print('done')

# Melting the DataFrame to long format
long_format_data = pd.melt(survey_data, 
                           id_vars=['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', 
                                    'what news sources do you typically read from?'], 
                           value_vars=list(headline_categories.keys()),
                           var_name='Headline', 
                           value_name='Reliability Rating')

# Adding the categories of each headline
long_format_data['Headline Category'] = long_format_data['Headline'].map(headline_categories)

# Mapping headline categories to numerical values
headline_category_numerical = {'conservative': 1, 'neutral': 0, 'liberal': -1}
long_format_data['Headline Category Numerical'] = long_format_data['Headline Category'].map(headline_category_numerical)

# Train a linear regression model for likelihood estimation
X_likelihood = long_format_data[['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', 'Headline Category Numerical']]
y_likelihood = long_format_data['Reliability Rating']
likelihood_model = LinearRegression()
likelihood_model.fit(X_likelihood, y_likelihood)

# Function to compute likelihood under H1
def compute_likelihood_h1(row, model):
    # Create a DataFrame for the input features with correct column names
    X_input = pd.DataFrame([[row['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?'], 
                             row['Headline Category Numerical']]],
                           columns=['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', 
                                    'Headline Category Numerical'])
    
    # Use the model to predict the reliability rating
    predicted_rating = model.predict(X_input)[0]

    # Assuming a normal distribution around the predicted rating to estimate likelihood
    std_dev = 1  # Standard deviation (you can adjust this)
    likelihood = stats.norm(predicted_rating, std_dev).pdf(row['Reliability Rating'])
    
    return likelihood

def compute_likelihood_h2(reliability_rating, data):
    # Calculate the overall probability of the given reliability rating in the dataset
    total_count = len(data)
    rating_count = len(data[data['Reliability Rating'] == reliability_rating])
    return rating_count / total_count if total_count > 0 else 0

# Function to compute the posterior mean
def compute_posterior(prior_mean, model, political_belief, headline_category, prior_variance, likelihood_variance):
    # Predict the base reliability rating using the model
    X_input = pd.DataFrame([[political_belief, headline_category]],
                           columns=['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', 
                                    'Headline Category Numerical'])
    base_likelihood = model.predict(X_input)[0]

    # Adjusting the base likelihood based on the prior
    # Example: Apply a scaling factor based on how far the political belief is from neutral (4 in this case)
    scaling_factor = 1 + abs(political_belief - 4) / 10
    adjusted_likelihood = base_likelihood * scaling_factor

    # Bayesian update
    posterior_mean = (prior_variance * adjusted_likelihood + likelihood_variance * prior_mean) / (prior_variance + likelihood_variance)
    return posterior_mean


# Calculate the standard deviation for political beliefs and reliability ratings
std_dev_political_beliefs = long_format_data['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?'].std()
std_dev_reliability_ratings = long_format_data['Reliability Rating'].std()

print("Standard Deviation of Political Beliefs:", std_dev_political_beliefs)
print("Standard Deviation of Reliability Ratings:", std_dev_reliability_ratings)

# Computing the posteriors using the regression model for likelihood estimation
posteriors = []
for index, row in long_format_data.iterrows():
    political_belief = row['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?']
    headline_category = row['Headline Category Numerical']

    posterior_mean = compute_posterior(political_belief, likelihood_model, political_belief, headline_category, std_dev_political_beliefs**2, std_dev_reliability_ratings**2)
    posteriors.append(posterior_mean)

# Adding the computed posteriors to the long format data
long_format_data['Posterior Mean'] = posteriors

X = long_format_data[['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', 'Headline Category Numerical']]  # Features
y = long_format_data['Posterior Mean']  # Target variable

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)



log_posterior_odds = []

for index, row in long_format_data.iterrows():
    political_belief = row['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?']
    headline_category = row['Headline Category Numerical']
    reliability_rating = row['Reliability Rating']

    # Likelihood under H1 (from regression model)
    likelihood_h1 = compute_likelihood_h1(row, likelihood_model)

    # Likelihood under H2 (overall probability of reliability rating)
    likelihood_h2 = compute_likelihood_h2(reliability_rating, long_format_data)

    # Calculate log odds ratio
    log_odds_ratio = np.log(likelihood_h1 / likelihood_h2) if likelihood_h2 > 0 else float('inf')
    log_posterior_odds.append(log_odds_ratio)

# Adding log posterior odds to the DataFrame
long_format_data['Log Posterior Odds'] = log_posterior_odds

# Predict reliability ratings using the trained model
predicted_ratings = model.predict(X)

# Add the predicted ratings to your DataFrame
long_format_data['Predicted Rating'] = predicted_ratings

# Compare with actual ratings
comparison = long_format_data[['Reliability Rating', 'Predicted Rating']]
print(comparison.head())
comparison.to_csv('comparisons.csv', index=False)

# Visualization
sns.scatterplot(data=long_format_data, x='Reliability Rating', y='Predicted Rating')
plt.xlabel('Actual Reliability Rating')
plt.ylabel('Predicted Reliability Rating')
plt.title('Comparison of Actual vs. Predicted Reliability Ratings')
plt.show()

# Calculate correlation coefficient
correlation = long_format_data[['Reliability Rating', 'Predicted Rating']].corr().iloc[0, 1]
print(f'Correlation Coefficient: {correlation}')

long_format_data.to_csv('long_format_data.csv', index=False)

# Function to calculate accuracy for a given segment
def calculate_accuracy_for_segment(data, model, segment_column, segment_values):
    # Filter the data for the specified segment
    if segment_column == 'how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?':
        segment_data = data[data[segment_column].isin(segment_values)]
    else:
        segment_data = data[data[segment_column] == segment_values]

    # Check if the segment data is empty
    if len(segment_data) == 0:
        return np.nan  # Return NaN if no data available for this segment

    X_segment = segment_data[['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', 'Headline Category Numerical']]
    y_segment = segment_data['Reliability Rating']

    # Predict and calculate accuracy
    predicted_segment = model.predict(X_segment)
    mse_segment = mean_squared_error(y_segment, predicted_segment)
    return mse_segment


# Calculate accuracy for each headline category
accuracy_by_headline = {category: calculate_accuracy_for_segment(long_format_data, model, 'Headline Category', category) for category in ['liberal', 'neutral', 'conservative']}

# Calculate accuracy for each respondent's political belief
# Assuming political beliefs are categorized into 'liberal' (1-3), 'neutral' (4), 'conservative' (5-7)
accuracy_by_respondent = {
    'liberal': calculate_accuracy_for_segment(long_format_data, model, 'how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', range(1, 4)),
    'neutral': calculate_accuracy_for_segment(long_format_data, model, 'how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', [4]),
    'conservative': calculate_accuracy_for_segment(long_format_data, model, 'how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?', range(5, 8))
}

# Output the accuracies
print("Accuracy by Headline Category:", accuracy_by_headline)
print("Accuracy by Respondent's Political Belief:", accuracy_by_respondent)

# Histogram of Log Posterior Odds
plt.figure(figsize=(10, 6))
sns.histplot(log_posterior_odds, kde=True)
plt.title(f'Histogram of Log Posterior Odds. Median = {np.median(log_posterior_odds)}')
plt.xlabel('Log Posterior Odds')
plt.ylabel('Frequency')
plt.savefig('log_posterior_odds_histogram.png')
plt.show()
print('median log posterior:', np.median(log_posterior_odds))

# Box Plot of Log Posterior Odds
plt.figure(figsize=(10, 6))
sns.boxplot(x=long_format_data['Log Posterior Odds'])
plt.title('Box Plot of Log Posterior Odds')
plt.xlabel('Log Posterior Odds')
plt.savefig('log_posterior_odds_boxplot.png')
plt.show()

# Plotting the POlitical Beliefs Histogram
plt.figure(figsize=(10, 6))
sns.histplot(survey_data['how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?'], bins=7, kde=False, color='skyblue')
plt.title(f'Distribution of Respondents\' Political Beliefs. Median = {np.median(survey_data["how would you describe your political beliefs on a scale from 1 to 7 where 1 is most liberal and 7 is most conservative?"])}')
plt.xlabel('Political Beliefs (1=Most Liberal, 7=Most Conservative)')
plt.ylabel('Number of Respondents')
plt.xticks(range(1, 8))
plt.grid(True)
plt.savefig('political_beliefs_histogram.png')
plt.show()