# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# - Version: 1.0
# - last update: 2023-12-08
# - Short description: This notebook models causal inference for demographic variables using donor information.

# ## Install and Import

# Import modules
import warnings
import os
from datetime import date
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ## Notebook Details

# parameters for logging and notebook exports
notebook_name = "03.02_causal_inference-modelling"  # only file name without extension

# ## Configuration

# +
# Plotting
plt.rcParams["figure.figsize"] = (12, 8)
sns.set(rc={"figure.figsize": (12, 8)}, font_scale=0.8)
sns.set(style="darkgrid")

# Pandas
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Warnings
warnings.filterwarnings("ignore")
# -


# Constants
ROOT_PATH = "../"
PATH_DATA = "data/"
PATH_DATA_RAW = "data/raw/"
PATH_DATA_INT = "data/interim/"
PATH_DATA_PRO = "data/processed/"
PATH_LOG = "logs/"
PATH_MOD = "models/"
PATH_REP = "reports/"
PATH_FIG = "reports/figures/"
PATH_HTML = "reports/html/"


# + [markdown] id="6YBbHzbqBzw8"
# ## Load Data

# + id="QUyCYBdWdeWq" outputId="a05ad7b5-365e-4110-d00f-ab95996998b9"
def load_data(path):
    return pd.read_parquet(ROOT_PATH + path)


file = "features_merged.parquet"
df = pd.read_parquet(ROOT_PATH + PATH_DATA_INT + file)
# -


df.shape

# ## Main Part

# ## Unadjusted model (group weights not considered)

# +
columns_to_keep = [
    "age_clean",
    "sex_clean",
    "apo_e4_allele_clean",
    "cerad",
    "education_years",
    "education_years_stages_bin",
    "education_years_quartiles_bin",
    "age_at_first_tbi",
    "age_at_first_tbi_bin",
    "num_tbi_w_loc",
    "control_set",
    "ever_tbi_w_loc_clean",
    "longest_loc_duration_clean",
    #"longest_loc_duration",
    "longest_loc_duration_bin",
    "group_weight",
    "act_demented_clean",
]

# Subset the DataFrame
df_filtered = df[columns_to_keep]
# -

print("Filtered DataFrame:")
df_filtered.info()

# +
# Filter out missing values for the variables of interest
df_unadjusted = df_filtered.dropna(subset=['act_demented_clean', 'ever_tbi_w_loc_clean'])

# Define the outcome and the treatment
outcome_unadjusted = df_unadjusted['act_demented_clean']
treatment_unadjusted = df_unadjusted['ever_tbi_w_loc_clean']

# Add a constant to the treatment for the intercept
treatment_unadjusted_with_const = sm.add_constant(treatment_unadjusted)

# Fit the logistic regression model for the unadjusted model (without group_weight)
unadjusted_model = sm.Logit(outcome_unadjusted, treatment_unadjusted_with_const)
unadjusted_results = unadjusted_model.fit()

# Display the summary of the unadjusted model
print(unadjusted_results.summary())

# +
# Coefficients and standard errors
coefficients = unadjusted_results.params.values
std_errors = unadjusted_results.bse.values

# Variables
variables = unadjusted_results.params.index

# Confidence intervals (assuming 95% confidence)
confidence_intervals = [(c - 1.96 * se, c + 1.96 * se) for c, se in zip(coefficients, std_errors)]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Plot coefficients as bars
ax.barh(variables, coefficients, color='blue', alpha=0.7, label='Coefficient')

# Plot confidence intervals as error bars
for i, (lower, upper) in enumerate(confidence_intervals):
    ax.plot([lower, upper], [i, i], 'ro-', markersize=4, linewidth=1, label='95% CI' if i == 0 else '')

# Add labels and title
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Coefficient')
ax.set_title('Logistic Regression Coefficients with 95% Confidence Intervals\n(Unadjusted Model without Group Weights)')
ax.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()

# -

# **Interpretation:**
# - The output presented is from a logistic regression analysis examining the relationship between the variable `ever_tbi_w_loc_clean` and the likelihood of being `act_demented_clean`. 
# - The model, which includes 107 observations, uses Maximum Likelihood Estimation and converges in 4 iterations. The Pseudo R-squared value of 0.001900 indicates that the model explains very little of the variability in the dependent variable.
# - The intercept of the model is not statistically significant, with a p-value of 0.397, suggesting that when `ever_tbi_w_loc_clean` is 0, the log odds of being `act_demented_clean` are not significantly different from zero. 
# - The coefficient for `ever_tbi_w_loc_clean` is 0.2061, but it is also not statistically significant (p-value: 0.596), implying that changes in this predictor do not have a significant impact on the likelihood of being `act_demented_clean`. 
# - The percentage change in odds for a one-unit increase in `ever_tbi_w_loc_clean` is approximately 22.89%, but due to the lack of statistical significance, this change should not be interpreted as a meaningful effect. 
# - The overall fit of the model is weak, as suggested by the high p-value (0.5960) in the likelihood ratio test, indicating that adding `ever_tbi_w_loc_clean` as a predictor does not significantly improve the model compared to a null model with only an intercept.

# ## Unadjusted model (group weights considered)

# +
# Filter out missing values for the variables of interest
df_weighted = df_filtered.dropna(subset=['act_demented_clean', 'ever_tbi_w_loc_clean', 'group_weight'])

# Define the outcome, the treatment, and the weights
outcome_weighted = df_weighted['act_demented_clean']
treatment_weighted = df_weighted['ever_tbi_w_loc_clean']
weights = df_weighted['group_weight']

# Add a constant to the treatment for the intercept
treatment_weighted_with_const = sm.add_constant(treatment_weighted)

# Fit the logistic regression model using GLM with a binomial family and frequency weights
weighted_model = sm.GLM(outcome_weighted, treatment_weighted_with_const, family=sm.families.Binomial(), var_weights=weights)
weighted_results = weighted_model.fit()

# Display the summary of the weighted unadjusted model
print(weighted_results.summary())


# +
# Coefficients and standard errors
coefficients = weighted_results.params.values
std_errors = weighted_results.bse.values

# Variables
variables = weighted_results.params.index

# Confidence intervals (assuming 95% confidence)
confidence_intervals = [(c - 1.96 * se, c + 1.96 * se) for c, se in zip(coefficients, std_errors)]

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 3))

# Plot coefficients as bars
ax.barh(variables, coefficients, color='blue', alpha=0.7, label='Coefficient')

# Plot confidence intervals as error bars
for i, (lower, upper) in enumerate(confidence_intervals):
    ax.plot([lower, upper], [i, i], 'ro-', markersize=5, linewidth=2, label='95% CI' if i == 0 else '')

# Add labels and title
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Coefficient')
ax.set_title('Logistic Regression Coefficients with 95% Confidence Intervals\n(Unadjusted Model with Group Weights)')
ax.legend(loc='best')

# Show the plot
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# -

# **Interpretation:**
#
# 1. **Dependent Variable and Model Details**:
# - The dependent variable here is `act_demented_clean`.
# - This analysis is based on 107 observations.
# - The model used is a GLM with a binomial family, which is suitable for binary outcomes like dementia presence or absence.
# - The link function is Logit, appropriate for binary logistic regression.
#
# 2. **Model Fit and Significance**:
#    - The Pseudo R-squared (Comparative Fit Index) is 0.3663. This indicates the proportion of variance in the dependent variable that's explained by the model. A value of 0.3663 suggests a moderate fit.
#    - The Log-Likelihood is -2310.1, which is a measure of the model’s goodness of fit.
#
# 3. **Coefficients**:
#    - `const`: The coefficient for the constant (intercept) is -1.1238. This value represents the log odds of the outcome when all predictors are at their reference levels (in this case, when `ever_tbi_w_loc_clean` is 0).
#    - `ever_tbi_w_loc_clean`: The coefficient is 0.5521. This means for every one-unit increase in `ever_tbi_w_loc_clean`, the log odds of being `act_demented_clean` increases by 0.5521, assuming other variables in the model are held constant.
#
# 4. **Standard Error and Z-Value**:
#    - The standard errors for the intercept and `ever_tbi_w_loc_clean` are 0.043 and 0.078, respectively. These measure the variability or uncertainty in the coefficient estimates.
#    - The z-values, which are the coefficients divided by their standard errors, are -26.299 for the intercept and 7.065 for `ever_tbi_w_loc_clean`. High absolute z-values suggest that the coefficients are significantly different from zero.
#
# 5. **P-Values**:
#    - The P-values for both coefficients are less than 0.000, indicating that these effects are statistically significant.
#
# 6. **Confidence Intervals**:
#    - The 95% confidence intervals for the coefficients provide a range of values within which the true population parameter likely falls. For `const`, it’s between -1.208 and -1.040, and for `ever_tbi_w_loc_clean`, it’s between 0.399 and 0.705.
#
# 7. **Percent Change Interpretation**:
#    - To interpret the coefficient of `ever_tbi_w_loc_clean` in terms of percent change, we use the formula: \( \text{% change} = (e^{\text{coef}} - 1) \times 100 \% \).
#    - Applying this to the `ever_tbi_w_loc_clean` coefficient: \( (e^{0.5521} - 1) \times 100 \% \).
#    - The percent change associated with the `ever_tbi_w_loc_clean` coefficient is approximately 73.69%. This means that for every one-unit increase in `ever_tbi_w_loc_clean`, the odds of being `act_demented_clean` increase by about 73.69%, assuming other variables in the model are held constant.
#
# 8. **Summary**:
#    - In summary, this GLM logistic regression model suggests a significant association between `ever_tbi_w_loc_clean` and the likelihood of being `act_demented_clean`, with a moderate fit to the data.
#

# Based on your provided data and format, here is the table for the logistic regression analysis:
#
# **Table: Logistic Regression Analysis of the Association Between History of TBI with Loss of Consciousness and Dementia (considering group weights)**
#
# <br>
#
# | Variable               | Coefficient | Std. Error | z-Value | P-Value | 95% Confidence Interval     |
# |------------------------|-------------|------------|---------|---------|-----------------------------|
# | Constant               | -1.1238     | 0.043      | -26.299 | 0.000   | (-1.208, -1.040)            |
# | Ever TBI w/ LOC (Yes)  | 0.5521      | 0.078      | 7.065   | 0.000   | (0.399, 0.705)              |
#
# Note: The model does not adjust for any other covariates. Pseudo R-squared = 0.3663. The analysis includes 107 observations.

# ## Adjusted model (group weights not considered)

df_filtered.info()

# +
df_adjusted = df_filtered[[
    'act_demented_clean',       # outcome
    'ever_tbi_w_loc_clean',     # treatment
    'age_clean',                # balanced
    'sex_clean',                # balanced
    'apo_e4_allele_clean',      # balanced
    'education_years',          # balanced
    'cerad',                    # balanced
    #'longest_loc_duration_clean',  # only minimal impact
    #'num_tbi_w_loc',               # only minimal impact
]]

# Dropping the records with the missing apo_e4_allele_clean values (-1)
df_adjusted = df_adjusted[~(df_adjusted == -1).any(axis=1)]
df_adjusted.shape

# +
# Define the outcome and the covariates
outcome_adjusted = df_adjusted['act_demented_clean']

covariates_adjusted = [
    'age_clean',
    'sex_clean',
    'apo_e4_allele_clean',
    'education_years',
    'cerad',
    'ever_tbi_w_loc_clean',
    #'longest_loc_duration_clean',  # only minimal impact
    #'num_tbi_w_loc',               # only minimal impact
]

# +
# Add a constant to the covariates for the intercept
X_adjusted_with_const = sm.add_constant(df_adjusted[covariates_adjusted])

# Fit the logistic regression model for the adjusted model (without group_weight)
adjusted_model = sm.Logit(outcome_adjusted, X_adjusted_with_const)
adjusted_results = adjusted_model.fit()

# Display the summary of the adjusted model
print(adjusted_results.summary())


# +
# Coefficients and standard errors
coefficients = adjusted_results.params.values
std_errors = adjusted_results.bse.values

# Variables
variables = adjusted_results.params.index

# Confidence intervals (assuming 95% confidence)
confidence_intervals = [(c - 1.96 * se, c + 1.96 * se) for c, se in zip(coefficients, std_errors)]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Plot coefficients as bars
ax.barh(variables, coefficients, color='blue', alpha=0.7, label='Coefficient')

# Plot confidence intervals as error bars
for i, (lower, upper) in enumerate(confidence_intervals):
    ax.plot([lower, upper], [i, i], 'ro-', markersize=5, linewidth=2, label='95% CI' if i == 0 else '')

# Add labels and title
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Coefficient')
ax.set_title('Logistic Regression Coefficients with 95% Confidence Intervals\n(Adjusted Model without Group Weights)')
ax.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# -

# **Interpretation**:
# 1. **Model Overview**: 
# - Dependent Variable: `act_demented_clean`
# - Number of Observations: 100
# - Model: Logistic Regression (Logit)
# - Number of Iterations: 5
# - Pseudo R-squared: 0.08246, indicating a low to moderate explanatory power of the model.
#
# 2. **Log-Likelihood and Model Fit**:
# - The current function value (Log-Likelihood) is -63.139. This is a measure of the model's fit, with higher values (closer to zero) generally indicating a better fit.
# - LL-Null (Log-Likelihood of the null model) is -68.814, which is the baseline measure without any predictors.
# - The model shows some improvement over the null model, but the relatively small difference suggests that the predictors don't explain a large portion of the variance in the outcome variable.
#
# 3. **Coefficients and Their Interpretation**:
# - `const`: The constant (intercept) is 1.1484. This is the log-odds of the dependent variable being 1 when all other predictors are at 0.
# - `age_clean`: Coefficient is -0.0041, indicating a very small decrease in the log-odds of the dependent variable being 1 for each one-unit increase in age. However, its p-value (0.910) suggests this effect is not statistically significant.
# - `sex_clean`: Coefficient of -0.0707, also not statistically significant (p-value: 0.874).
# - `apo_e4_allele_clean`: Coefficient of 1.0695, which is significant at the 0.05 level. This suggests that having the apo_e4 allele is associated with an increase in the log-odds of the dependent variable being 1.
# - `education_years`: Coefficient of -0.1238. It indicates that more years of education are associated with a decrease in the log-odds of the dependent variable being 1, but this is only marginally significant (p-value: 0.083).
# - `cerad`: Coefficient of 0.2844, not statistically significant (p-value: 0.172).
# - `ever_tbi_w_loc_clean`: Coefficient of 0.3261, not statistically significant (p-value: 0.448).
#
# 4. **Statistical Significance**:
# - The p-values for most predictors are above the conventional threshold of 0.05, indicating that they are not statistically significant predictors of the dependent variable in this model.
# - The only exception is `apo_e4_allele_clean`, which is borderline significant.
#
# 5. **Overall Model Significance**:
# - The LLR (Likelihood Ratio Test) p-value is 0.07817, which is above the standard significance level of 0.05. This suggests that the model as a whole does not significantly improve the prediction over the null model.
#
# 6. **Summary**:
# - While the model includes a range of predictors, most do not significantly contribute to predicting the outcome variable, with the exception of the `apo_e4_allele_clean`. 
# - The overall fit of the model is moderate, and it shows a slight improvement over a model with no predictors. This indicates that further investigation with either additional or different predictors might be needed to better understand the factors influencing the dependent variable.
#

# **Table: Logistic Regression Analysis of Factors Associated with Dementia (group weights not considered)**
#
# <br>
#
# | Variable               | Coefficient | Std. Error | z-Value | P-Value | 95% Confidence Interval     |
# |------------------------|-------------|------------|---------|---------|-----------------------------|
# | Constant               | 1.1484      | 3.507      | 0.327   | 0.743   | (-5.725, 8.022)             |
# | Age                    | -0.0041     | 0.036      | -0.113  | 0.910   | (-0.075, 0.067)             |
# | Sex                    | -0.0707     | 0.447      | -0.158  | 0.874   | (-0.946, 0.805)             |
# | APOE ε4 Allele         | 1.0695      | 0.545      | 1.961   | 0.050   | (0.000, 2.139)              |
# | Education Years        | -0.1238     | 0.072      | -1.732  | 0.083   | (-0.264, 0.016)             |
# | CERAD                  | 0.2844      | 0.208      | 1.365   | 0.172   | (-0.124, 0.693)             |
# | Ever TBI w/ LOC        | 0.3261      | 0.430      | 0.759   | 0.448   | (-0.516, 1.168)             |
#

# ***

# ## Adjusted model (group weights considered)

# +
df_adjusted = df[[
    'act_demented_clean',           # outcome
    'ever_tbi_w_loc_clean',         # treatment
    'age_clean',                    # balanced
    'sex_clean',                    # balanced
    'apo_e4_allele_clean',          # balanced
    'education_years',              # balanced
    'cerad',                        # balanced
    #'age_at_first_tbi',             # only minimal impact
    #'longest_loc_duration_clean',   # only minimal impact
    #'num_tbi_w_loc',                # only minimal impact
    'group_weight',
]]

# Dropping the records with the missing apo_e4_allele_clean values (-1)
df_adjusted = df_adjusted[~(df_adjusted == -1).any(axis=1)]

# Define the outcome and the covariates
outcome_adjusted = df_adjusted['act_demented_clean']

covariates_adjusted = list(df_adjusted.drop(columns=['act_demented_clean', 'group_weight']).columns)

print("Covariates adjusted (selected columns):")
covariates_adjusted

# +
# Add a constant to the covariates for the intercept
X_adjusted_with_const = sm.add_constant(df_adjusted[covariates_adjusted])

# Fit the logistic regression model using GLM with a binomial family and frequency weights
adjusted_model_with_weights = sm.GLM(outcome_adjusted, X_adjusted_with_const, family=sm.families.Binomial(), var_weights=df_adjusted['group_weight'])
adjusted_results_with_weights = adjusted_model_with_weights.fit()

# Display the summary of the adjusted model with weights
print(adjusted_results_with_weights.summary())


# +
# Coefficients and standard errors
coefficients = adjusted_results_with_weights.params.values
std_errors = adjusted_results_with_weights.bse.values

# Variables
variables = adjusted_results_with_weights.params.index

# Confidence intervals (assuming 95% confidence)
confidence_intervals = [(c - 1.96 * se, c + 1.96 * se) for c, se in zip(coefficients, std_errors)]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Plot coefficients as bars
ax.barh(variables, coefficients, color='blue', alpha=0.7, label='Coefficient')

# Plot confidence intervals as error bars
for i, (lower, upper) in enumerate(confidence_intervals):
    ax.plot([lower, upper], [i, i], 'ro-', markersize=5, linewidth=2, label='95% CI' if i == 0 else '')

# Add labels and title
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Coefficient')
ax.set_title('Logistic Regression Coefficients with 95% Confidence Intervals\n(Adjusted Model with Group Weights)')
ax.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# -

# **Interpretation:**
#
# 1. **Model Overview**:
# - Dependent Variable: `act_demented_clean`
# - Number of Observations: 100
# - Model: Generalized Linear Model (GLM) with Binomial family and Logit link
# - Number of Iterations: 5
# - Pseudo R-squared (Comparative Fit Index): 0.9994, indicating an excellent fit of the model to the data.
#
# 2. **Log-Likelihood and Model Fit**:
# - The log-likelihood of the model is -1700.3, a measure of the model's fit to the data.
# - The model's deviance is 3400.6, indicating the unexplained variance in the model.
# - Pearson chi2 of approximately 3490 suggests a good fit of the model.
#
# 3. **Coefficients and Percent Changes**:
# - `const`: Coefficient of 2.3047. For a one-unit increase in the constant, there's an e^2.3047 (approximately 10.02 times) increase in the odds of the dependent variable.
# - `age_clean`: Coefficient of -0.0144. Each additional year decreases the odds of the dependent variable by e^-0.0144 (approximately 1.45%).
# - `sex_clean`: Coefficient of 0.4592. This indicates that being of a certain sex - coded as 1 (male) - increases the odds of the dependent variable by e^0.4592 (approximately 58.30%).
# - `apo_e4_allele_clean`: Coefficient of 1.8388. Presence of the apo_e4 allele increases the odds by e^1.8388 (approximately 6.29 times), increasing the odds by approximately 528.90%.
# - `education_years`: Coefficient of -0.2378. Each additional year of education decreases the odds by e^-0.2378 (approximately 21.21%).
# - `cerad`: Coefficient of 0.3148. An increase in the CERAD score increases the odds by e^0.3148 (approximately 37.02%).
# - `ever_tbi_w_loc_clean`: Coefficient of 0.6945. Having a history of TBI with loss of consciousness increases the odds by e^0.6945, increasing the odds by approximately 100.27%. This means that individuals with such a history are about twice as likely to exhibit the dependent variable compared to those without this history.
#
# 4. **Statistical Significance**:
# - The p-values for each predictor (except for `age_clean`) are less than 0.05, indicating statistical significance.
# - `age_clean` has a p-value of 0.047, suggesting marginal significance.
#
# 5. **Summary**:
# - This model robustly fits the data, as indicated by the high Pseudo R-squared.
# - Significant predictors include sex, apo_e4_allele presence, years of education, CERAD score, and history of TBI with loss of consciousness, each influencing the likelihood of the dependent variable.
# - `age_clean` shows a marginal effect, slightly decreasing the odds of the dependent variable with increasing age.

# **Table: Generalized Linear Model Logistic Regression Analysis of Factors Associated with Dementia (group weights considered)**
#
# <br>
#
# | Variable                  | Coefficient | Std. Error | z-Value | P-Value | 95% Confidence Interval     |
# |---------------------------|-------------|------------|---------|---------|-----------------------------|
# | Constant (Intercept)      | 2.3047      | 0.710      | 3.248   | 0.001   | (0.914, 3.696)              |
# | Age                       | -0.0144     | 0.007      | -1.985  | 0.047   | (-0.029, -0.000)            |
# | Sex                       | 0.4592      | 0.089      | 5.176   | 0.000   | (0.285, 0.633)              |
# | APOE ε4 Allele            | 1.8388      | 0.113      | 16.340  | 0.000   | (1.618, 2.059)              |
# | Education Years           | -0.2378     | 0.015      | -15.729 | 0.000   | (-0.267, -0.208)            |
# | CERAD Score               | 0.3148      | 0.043      | 7.368   | 0.000   | (0.231, 0.399)              |
# | Ever TBI w/ LOC           | 0.6945      | 0.095      | 7.301   | 0.000   | (0.508, 0.881)              |
#

# ### Adjusted model with all variables (group weights considered)

# +
df_adjusted = df[[
    'act_demented_clean',       # outcome
    'ever_tbi_w_loc_clean',     # treatment
    'age_clean',                # balanced
    'sex_clean',                # balanced
    'apo_e4_allele_clean',      # balanced
    'education_years',          # balanced
    'cerad',                    # balanced
    'age_at_first_tbi',             # only minimal impact
    'longest_loc_duration_clean',   # only minimal impact
    'num_tbi_w_loc',                # only minimal impact: should not be used, because very unbalanced: 1 time TBI with 45 donors vs > 1 TBI with 9 donors 
    'group_weight',
]]

# Dropping the records with the missing apo_e4_allele_clean values (-1)
df_adjusted = df_adjusted[~(df_adjusted == -1).any(axis=1)]

# Define the outcome and the covariates
outcome_adjusted = df_adjusted['act_demented_clean']

covariates_adjusted = list(df_adjusted.drop(columns=['act_demented_clean', 'group_weight']).columns)

print("Covariates adjusted (selected columns):")
covariates_adjusted

# +
# Add a constant to the covariates for the intercept
X_adjusted_with_const = sm.add_constant(df_adjusted[covariates_adjusted])

# Fit the logistic regression model using GLM with a binomial family and frequency weights
adjusted_model_with_weights = sm.GLM(outcome_adjusted, X_adjusted_with_const, family=sm.families.Binomial(), var_weights=df_adjusted['group_weight'])
adjusted_results_with_weights = adjusted_model_with_weights.fit()

# Display the summary of the adjusted model with weights
print(adjusted_results_with_weights.summary())

# -

# **Interpretation:**
#
# 1. **Model Overview**:
#    - **Dependent Variable**: The model targets the variable `act_demented_clean`.
#    - **Sample Size**: The analysis involves 100 observations.
#    - **Model Specifications**:
#      - The GLM uses a Binomial family with a Logit link function.
#      - The model has 9 explanatory variables, resulting in 90 degrees of freedom for residuals.
#
# 2. **Model Performance**:
#    - **Log-Likelihood**: The model's log-likelihood is -1697.1.
#    - **Deviance**: The model shows a deviance of 3394.3.
#    - **Pseudo R-squared (Comparative Fit Index)**: The pseudo R-squared value is 0.9994, suggesting a high goodness of fit.
#
# 3. **Coefficients Analysis**:
#    - **Constant Term (Intercept)**:
#      - Coefficient: 2.0993
#      - Standard Error: 0.717
#      - Z-value: 2.926, indicating the intercept is significantly different from zero at the 0.003 level.
#    - **Age (age_clean)**:
#      - Coefficient: -0.0132
#      - Indicates a slight negative association with the dependent variable, but not statistically significant at the traditional 0.05 level (p=0.071).
#    - **Sex (sex_clean)**:
#      - Coefficient: 0.4636
#      - Shows a positive and significant relationship (p<0.001).
#    - **APOE4 Allele (apo_e4_allele_clean)**:
#      - Coefficient: 1.8244
#      - Significantly positively associated with the dependent variable (p<0.001).
#    - **Education Years**:
#      - Coefficient: -0.2302
#      - Indicates a significant negative association (p<0.001).
#    - **CERAD Score**:
#      - Coefficient: 0.3140
#      - Positively related to the dependent variable and significant (p<0.001).
#    - **History of TBI with LOC (ever_tbi_w_loc_clean)**:
#      - Coefficient: 1.0966
#      - Significantly positively associated with the outcome (p=0.002).
#    - **Age at First TBI (age_at_first_tbi)**:
#      - Coefficient: 0.00009341
#      - Not statistically significant (p=0.978).
#    - **Longest LOC Duration (longest_loc_duration_clean)**:
#      - Coefficient: 0.0001
#      - Marginally insignificant (p=0.121).
#    - **Number of TBIs with LOC (num_tbi_w_loc)**:
#      - Coefficient: -0.4384
#      - Shows a negative association, bordering on significance (p=0.062).
#
# 4. **Summary**:
#    - The model suggests significant associations between the outcome (dementia) and variables like sex, APOE4 allele presence, education years, CERAD score, and history of TBI with LOC.
#    - Age, age at first TBI, longest LOC duration, and number of TBIs with LOC show either no significant association or marginal significance.
#
# This model provides valuable insights into factors associated with dementia, highlighting the importance of genetic factors (APOE4 allele), demographic variables (sex, education), and medical history (TBI with LOC).

# ## Adjusted model with all pre-mortality variables (group weights considered and using binned attributes)

# +
df_adjusted = df[[
    'act_demented_clean',        # outcome
    'ever_tbi_w_loc_clean',      # treatment
    #'age_clean',                # balanced: removed because age of death
    'sex_clean',                 # balanced
    'apo_e4_allele_clean',       # balanced
    'cerad',                     # balanced
    'education_years_stages_bin',
    'age_at_first_tbi_bin',      
    'longest_loc_duration_bin',
    #'num_tbi_w_loc',                # should not be used, because very unbalanced: 1 time TBI with 45 donors vs > 1 TBI with 9 donors 
    'group_weight',
]]

# Dropping the records with the missing apo_e4_allele_clean values (-1)
df_adjusted = df_adjusted[~(df_adjusted == -1).any(axis=1)]

# Define the outcome and the covariates
outcome_adjusted = df_adjusted['act_demented_clean']

covariates_adjusted = list(df_adjusted.drop(columns=['act_demented_clean', 'group_weight']).columns)

print("Adjusted covariates (selected columns), with a focus on attributes prior to mortality:")
set(covariates_adjusted)

# +
# Create a label encoder object
le = LabelEncoder()

# Apply label encoding on 'longest_loc_duration_clean' and 'age_at_first_tbi'
df_adjusted['longest_loc_duration_bin'] = le.fit_transform(df_adjusted['longest_loc_duration_bin'])
df_adjusted['age_at_first_tbi_bin'] = le.fit_transform(df_adjusted['age_at_first_tbi_bin'])
df_adjusted['education_years_stages_bin'] = le.fit_transform(df_adjusted['education_years_stages_bin'])

#df_adjusted.head()

# +
# Add a constant to the covariates for the intercept
X_adjusted_with_const = sm.add_constant(df_adjusted[covariates_adjusted])

# Fit the logistic regression model using GLM with a binomial family and frequency weights
adjusted_model_with_weights = sm.GLM(outcome_adjusted, X_adjusted_with_const, family=sm.families.Binomial(), var_weights=df_adjusted['group_weight'])
adjusted_results_with_weights = adjusted_model_with_weights.fit()

# Display the summary of the adjusted model with weights
print(adjusted_results_with_weights.summary())


# +
# Coefficients and standard errors
coefficients = adjusted_results_with_weights.params.values
std_errors = adjusted_results_with_weights.bse.values

# Variables
variables = adjusted_results_with_weights.params.index

# Confidence intervals (assuming 95% confidence)
confidence_intervals = [(c - 1.96 * se, c + 1.96 * se) for c, se in zip(coefficients, std_errors)]

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 4))

# Plot coefficients as bars
ax.barh(variables, coefficients, color='blue', alpha=0.7, label='Coefficient')

# Plot confidence intervals as error bars
for i, (lower, upper) in enumerate(confidence_intervals):
    ax.plot([lower, upper], [i, i], 'ro-', markersize=5, linewidth=2, label='95% CI' if i == 0 else '')

# Add labels and title
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Coefficient')
ax.set_title('Logistic Regression Coefficients with 95% Confidence Intervals\n(Adjusted Model with Group Weights)')
ax.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# -

# **Summary:**
#
# 1. **Model Overview**:
#    - **Number of Observations**: 100
#    - **Degrees of Freedom (Residuals)**: 92
#    - **Degrees of Freedom (Model)**: 7
#    - **Method**: Iteratively Reweighted Least Squares (IRLS)
#    - **Log-Likelihood**: -1800.2, indicating the fit of the model.
#    - **Deviance**: 3600.3, a measure of the model's goodness of fit.
#    - **Pseudo R-squared (Cragg & Uhler’s)**: 0.9953, suggesting a high level of explained variability in the dependent variable by the model.
#
#
# 2. **Coefficients**:
#    - **Constant (Intercept)**: -2.9278. This is the log odds of being `act_demented_clean` when all other predictors are held at zero.
#    - **ever_tbi_w_loc_clean**: Coefficient = 0.6115. This suggests that for each unit increase in `ever_tbi_w_loc_clean`, the log odds of being `act_demented_clean` increase by 0.6115. The percentage change in odds is (exp(0.6115) - 1) * 100% ≈ 84.26%.
#    - **sex_clean**: Coefficient = 0.4171. For each unit increase in `sex_clean`, the log odds increase by 0.4171, or a percentage change of (exp(0.4171) - 1) * 100% ≈ 51.74%.
#    - **apo_e4_allele_clean**: Coefficient = 1.4021. This indicates a 1.4021 increase in the log odds for each unit increase in `apo_e4_allele_clean`, or a percentage change of (exp(1.4021) - 1) * 100% ≈ 305.97%.
#    - **cerad**: Coefficient = 0.4720. Each unit increase in `cerad` is associated with a 0.4720 increase in log odds, or a percentage change of (exp(0.4720) - 1) * 100% ≈ 60.35%.
#    - **education_years_stages_bin**: Coefficient = -0.2907. This indicates a decrease in log odds of 0.2907 for each unit increase, or a percentage decrease of (1 - exp(-0.2907)) * 100% ≈ 25.26%.
#    - **age_at_first_tbi_bin**: Coefficient = 0.4714. This suggests a 0.4714 increase in log odds for each unit increase, or a percentage change of (exp(0.4714) - 1) * 100% ≈ 60.23%.
#    - **longest_loc_duration_bin**: Coefficient = 0.2244. This indicates a 0.2244 increase in log odds per unit increase, or a percentage change of (exp(0.2244) - 1) * 100% ≈ 25.17%.
#
#
# 3. **Statistical Significance**:
#    - All predictors have p-values less than 0.05, indicating that they are statistically significant in predicting the dependent variable `act_demented_clean`.
#
#
# 4. **Interpretation**:
#    - The positive coefficients (e.g., `apo_e4_allele_clean`, `cerad`) suggest that increases in these variables are associated with higher odds of the outcome `act_demented_clean`.
#    - The negative coefficient for `education_years_stages_bin` suggests that higher values of this variable are associated with lower odds of the outcome.
#    - The model appears to be highly predictive, given the high Pseudo R-squared value, but caution is needed as this might also indicate overfitting.

# ## Exports

# +
# Export
# export.to_csv(ROOT_PATH + PATH_DATA_INT + f"{file}.csv")
# export.to_parquet(ROOT_PATH + PATH_DATA_INT + f"{file}.parquet")


# + [markdown] id="ti9wgudgBzw9"
# ## Watermark
# -

# %load_ext watermark

# + id="f7PuaC1JBzw-" outputId="aafeadff-3431-44e5-fa8f-d94281a641d6"
# %watermark

# + id="SU12PVZ2BzxA" outputId="35145233-ba98-423f-ca77-95011e8cd3af"
# %watermark --iversions

# + [markdown] id="BdlIIXzYnOiW"
# -----
#
# -

# ## Snapshot

today = date.today()
output_file = f"{ROOT_PATH}{PATH_HTML}{today}_{notebook_name}.html"
input_file = f"{notebook_name}.ipynb"
print(input_file)
# !jupyter nbconvert --to html {input_file} --output {output_file}

# +
# Construct the output file path
output_file = f"{ROOT_PATH}{PATH_HTML}{today}_{notebook_name}_no_code.html"

# Construct the input file path
input_file = f"{notebook_name}.ipynb"

# Convert the notebook to HTML without the code cells
os.system(f"jupyter nbconvert --to html {input_file} --output {output_file} --no-input")
