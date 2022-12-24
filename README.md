# Credit Risk Analysis for peer to peer lending firm Bandora
The credit risk is analyzed to measure the possibility of loss as result of borrower failing to repay a loan or meeting the loan obligations. The more the credit risk the more it has a negative impact on performance of bank. The credit risk analysis is important because it allows the bank to plan strategies to avoid a negative outcome ahead in future.
- In this project we analyzed the credit risk on individual loans, maximizing the Return on Investment (ROI) by predicting an Eligible Loan Amount and minimizing the financial risk between the lending firm Bandora and the borrowers.

## Understanding the Dataset
Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform  ([**Bondora**](https://www.bondora.com/en/public-reports#dataset-file-format))

- The dataset contains the individual loan details from **2009** to **2019** having both the **defaulted** and **non-defaulted** loans
- The dataset does not contain any target attributes for 
  - defaulted, non-defauted borrowers
  - Eligible Loan Amount (ELA)
  - Return on Investment (ROI)
  - Equated Monthly Installment (EMI) 
- These 4 target attributes will be created using feature engineering

The dataset has 112 attributes intially. The attributes definition of dataset can be found here ([**Data Understanding**](https://colab.research.google.com/drive/1vcujgYWoKM-J_mc23TlaFa_NkWPMTL92?usp=sharing)), it will help you understand the dataset better.

## Preprocessing
As the dataset is not clean, so preprocessing techniques were used to clean the dataset and make it ready for Exploratory Data Analysis (EDA).
- The attributes having greater than 40% missing values were filtered out.
- There are some features which have no role in targets, so they are also removed:
  - 'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth' (because age is already present), 'BiddingStartedOn','UserName','NextPaymentNr','NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther' (As Total income is already present which is total of all these income), 'LoanApplicationStartedDate','ApplicationSignedHour', 'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate', 'LastPaymentOn', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 'StageActiveSince', 'ModelVersion','WorseLateCategory'
- As no time series analysis is involved in our project, so some attributes involving date-time are removed:
   - 'LoanDate', 'FirstPaymentDate','MaturityDate_Original','MaturityDate_Last','LastPaymentOn'
   - note that attribute 'DefaultDate' is not removed, because it contains the default date for those borrowers who defaulted and - for those borrowers who didn't default. This attribute will help us in creating the our 1st target attribute (defaulted, non-defauted borrowers).
-  There are some attributes in numeric form in the dataset but they are actually categorical attributes as per data description such as :
   -  Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea, 
   - These attributes are converted to categorial type by using the respective attribute definition, to make these variables correct for Exploratory Data Analysis (EDA).
   - Like for Gender attribute replacing [0,1,2] by ["Male","Woman","Undefined"] as per attribute definition of Gender.
   - When the data distribution is checked for these attributes, there are some unreplacable values like :
      - 22, 15, 10, 13, 7, 21 in 'LanguageCode' attribute.
      - -1 in 'UseOfLoan'
      -  0, -1 in 'Education'
      - -1 in 'MaritalStatus'
      - -1, 0, 6 in 'EmploymentStatus'
      - -1, 0 in 'OccupationArea'
      - -1 in 'HomeOwnershipType'

- To perform the Credit Risk Analysis, We need to create the 4 Target attributes, 1 for Classification and 3 for Regression
  1) For Classification, we need to create a Default/non-Default binary class Target attribute:
    - Here, status is the variable which help us in creating target variable. The reason for not making status as target variable is that it has three unique values current, Late and repaid. There is no default feature but there is a feature default date which tells us when the borrower has defaulted means on which date the borrower defaulted. So, we will be combining Status and Default date features for creating target variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null. So we will first filter out all the current status records because they are not matured yet they are current loans.
    - The Status attribute consists of three classes : Current, Late, Repaid
    - So filtering out those rows in dataset where status is either Late or Repaid
    - Creating a new variable DefaultTarget, by assigning 1 to those rows where there is a value for defaultDate attribute and 0 to other.
    - It means those borrowers having default date belong to defaulted class in our target attribute
  2) **For Regression, we need to calculate Equated Monthly Installment, Eligible Loan Amount (ELA), Return On investment ROI (Risk to get profit)**
    - The **EMI** is calculated based on the following mathematical formula: **EMI = P × r × (1 + r) ^ n / ((1 + r) ^ n – 1)**
      - Where P = Loan amount. "Amount", r = Rate of interest, which is calculated on a monthly-basis-Interest, n = Loan tenure (in months).
  3) **Eligible Loan Amount**, ELA = Assets (Income) - Liabilities of the borrower
    - **Assets**:
      -  **FreeCash** = ELA
      -   **TotalIncome** - **LiabilitiesTotal** = ELA
      -   Under Concsideration, Eligible Loan Amount means, with respect to a Mortgage Loan that is an Eligible Loan, the lesser of:
        - the Principal Balance of such Eligible Loan, AppliedAmount
        - the Market Value of such Eligible Loan PurchasePrice | BidPrinciple
    -  **Approach Followed :**
        - Calculate AppliedAmount + AppliedAmount*Interest = Total Liabilities Amount
        - Divide by the loan tenure (months)
        - If the result is less than (TotalIncome- LiabilitiesTotal)*30/100
        - Then allow the Applied Amount, If not allow only the result of the previous calculation.
  4) **Preferred ROI**
    - We weren't able to determine the procedure of handling Risk related to loan in order to determine Preferred ROI.
    - In order to complete the task in hand and complete it, we'll calculate ROI instead : 
      - **ROI = Investment Gain / Investment Base**
      - **ROI = Amount lended * interest/100**



## Data Wrangling
Using the techniques of data wrangling, errors were removed, gaps in the dataset were indentified, the data imputation was done to make the data ready for Exploratory Data Analysis (EDA).
- Incorrect data entry is checked for the categorical attributes.
  - City and County
    - Upon exploring the city attribute and checking the distribution
    - The two major cities with counts are Tallinn : 6467 , TALLINN : 5395
    - As we can see these two cities are same, so typing error exists in our dataset.
  - By removing the trailing space (_*) and converting to lower case, the inconsistent data entry is corrected.
- Columns with missing values are indentified:
  - VerificationType 0.058 %, Gender 0.058 %, MonthlyPayment 8.56 %, County 26.5 %, City 6.51 %, Education 0.058 %
  - MaritalStatus 0.058 %, EmploymentStatus 0.25 %, EmploymentDurationCurrentEmployer 1.12 %, OccupationArea 0.11 % 
  - HomeOwnershipType 2.13 %, DebtToIncome 0.058 %, FreeCash 0.05 %, Rating 3.52 %, CreditScoreEsMicroL 33.83 %
  - PreviousRepaymentsBeforeLoan 24.95 %
- Data Imputation
  - The missing values are imputed with different approach for numerical and categorical attributes.
    - After checking the distribution of numerical attributes, it is observed that majority of attributes have highly skewed distribution
    - So to replace the missing values, the median of respective attribute is used instead of mean() because for skewed attributes, we don't use mean() for imputation
    - The missing values in categorical attributes are imputed with the mode() of respective attributes.

## Exploratory Data Analysis
**Introduction:**

- The dataset after data wrangling comprises of 77394 rows and 41 features and 4 targets.
- Dataset comprises of int, float and object data types. 
- In EDA, three type of analysis are performed
  - Univariate Analysis
  - Bivariate Analysis
  - Multivariate Analysis

**Univariate Analysis:**
- In univaraite analysis, each feature is analyzed and explored individually to get hidden insights into it.
- The categorical features are analyzed and explored using Seaborn countplot. The countplot is like an histogram for categorical attributes
- The numerical features are analyzed and explored using Seaborn kdeplot and Displot
- A custom function was coded to calculate the percentage_of_top_n_classes of any categorical column given the n and the column's data
- **The Final Observations are :**
  - The 53.6% customers have 'income and expenses verified' followed by 33.11% customers having 'income unverified'. And the remaining percentages include customers having 'income verified','income unverified, cross-referenced by phone','not set'
  - 66% percent customers are 'male', followed by 27% 'female' and remaining 7% undefined
Majority of loan applicants are male
  - Majority of customers have 'MonthlyPayment' between 0 and 350
  - There is great diversity for County feature.
    - 38% customers have county 'Harju maakond', followed by 10% having 'HARJU' as county.
    - No hard insights can be taken from such diverse feature
  - There is great diversity for City feature similar to County feature.
    - Customers belong to 5438 unique cities.
    - only 15% customers are from 'Tallin' city, Other cities have very low percentages of customers
    - No hard insights can be taken from such diverse feature
  - Majority of customers have secondary education (37%), followed by 'higher education' with 27% customers.A very low no of customers have undefined education
  - 57.15% of customers have undefined marital status, followed by 14.86 % customers who are single, then 12.38% customers who are married and 10.96% customers who are cohabitant.A very small % of customers are divorced (3.98) and widow (0.67).
  - Major distribution include 59.53% of customers with undefined employment status, followed by 35.29% customers who are 'fully employed'. A small 8 % of customers have one of these status ; entrepneur, retireee, self employed and partially employed.
  - Major distribution include 39.02% of customers with 'more than 5 years' duration, followed by 18.46% customers who have duration of 'up to 1 year' and 17.8% having 'up to 5 years' duration. The remaining % of customers have one of these duration ; 'up to 2 years', 'up to 3 years', 'retiree', 'other' ' up to 4 years' and 'trail period'.
  - Major distribution include 33.91% customers who are 'owner' followed by 21.76% having 'tenant, pre-furnished property', 16.51% who are 'living with parents' and 11.32% who have 'mortgage'.The remaining % of customers lie among 'other', 'joint ownership', 'joint tenant' etc
  - Major distribution of customers lie between 0 and 70 DebtToIncome range. A less percentage of customers also have DebtToIncome up to 200.
  - Major distribution of customers have 0 freecash, followed by some percentages having around 250 to 300. A less percentage of customers also have freecash up to 158748 approx.
  - The major distribution of Rating is that 23% customers have rating 'f', 16% have 'hr' rating, '15% have 'e', 15% have 'd' followed by 13% having 'c'.
  - The major distribution of this attrbute is such that 84.5% customers have no creditscoreesmicroL. Only 5.32 % have m1 score, 3% have m5, 2 % have m2 and another 2% have m3 score.
  - 63.14% customers are new Credit Customers. 36.86% of customers are existing Credit Customers.
  - The majority of customers have BidsPortfolioManger in range of 0 and 3000.
  - The major percent of customers have age aroung 23 to 60 years.
  - The major percent of customers have LoanDuration of 60 and around 36 months.
  - The major ExistingLiabilities of customer are in range of 0 and 12.
  - Majority of Customers have 0 RefinanceLiabilities, with some having up to 25.
  - The customers have MonthlyPaymentDay high in number at 1st day,10th day, 15th day and overall the distribution is spread along the month.
  - Majority of customers have no NoOfPreviousLoansBeforeLoan, followed some having up 24 no of previous loans.
  - PreviousEarlyRepaymentsCountBeforeLoans for customer is in higher percentage at 0 value while some having up to 11 value.
  - The Defualted customers are more in number than undefualted customers. The defualted ones are more than 40k and undefaulted are around 35k
  - Major % of customers have 0 BidsApi, while some have up to around 7500.
  - BidsManual attriute has major customers in 0 to 1500 range.
  - No of customers who applied for less amounts are more than such customers who applied for normal aur high amounts.
  - The Amount borrower received has same distribution as AppliedAmount the borrower applied for.
  - The major distribution of Max interest rate accepted is between 0 and 70 as clear from Interest plot.
  - The MonthlyPayment has more density from 0 to 400 range.  
  - The incomeTotal has major of customers lying in range 0 to 3000.
  - The DebtToIncome attribute has more customers in range 0 to around 65.
  - Majority of customer have 0 freecash while some customers have upto around 160000.
  - The PrincipalPaymentsMade are more in range 0 to 4000.
  - InterestAndPenaltyPaymentsMade are more in range 0 to 3000.
  - High % customers have principalbalance in range of 0 to 4000.
  - InterestAndPenaltyBalance has more customers in range 0 to 9000.
  - More customers have AmountOfPreviousLoansBeforeLoan in range 0 to 15000.
  - PreviousRepaymentsBeforeLoan have high customers density in range 0 5000.
  
**Bivariate Analysis:**
- In bivaraite analysis, the features are analyzed and explored with respect to each other to get hidden insights into the relation of different featuers
- The correlation heat map is used to get the insights into the relation among different features
- The categorical features are analyzed and explored using Seaborn countplot. The countplot is like an histogram for categorical attributes
- The numerical features are analyzed and explored using Seaborn kdeplot and Displot
- **The Final Observation are:**
  - The Default/non-default target has a low correlation with BidsPortfolioManager 0.12, LanguageCode 0.14, Country 0.18, Interest 0.17, LoanDuration 0.14
  - Correlation of EmploymentStatus with : BidsPortfolioManager 0.12, BidsApi 0.11, VerificationType 0.3, Gender 0.19
  - Correlation of MaritalStatus with : LanguageCode 0.15, Interest 0.19
  - Correlation of MonthlyPayment with : BidsPortfolioManager 0.32, BidsManual 0.22, NewCreditCustomer 0.15, Country 0.28, AppliedAmount 0.74, Amount 0.6, Interest 0.24
  - Correlation of LoanDuration with : BidsPortfolioManager 0.17, Country 0.21, AppliedAmount 0.31, Amount 0.29, target 0.14
  - Correlation of Interest with : NewCreditCustomer 0.28, LanguageCode 0.5, Country 0.32
  - Correlation of Amount with : BidsPortfolioManager 0.61, BidsManual 0.41, NewCreditCustomer 0.12, Age 0.12, Country 0.23, AppliedAmount 0.89
  - Correlation of AppliedAmount with : BidsPortfolioManager 0.6, BidsManual 0.37, NewCreditCustomer 0.12, Age 0.11, Country 0.23
  - Correlation of Country with : NewCreditCustomer 0.23, LanguageCode 0.2, Age 0.24
  - Correlation of Gender with : NewCreditCustomer 0.12
  - Correlation of LanguageCode with : NewCreditCustomer 0.15
  - Correlation of BidsApi with : BidsPortfolioManager 0.1
  - We can see PrincipalBalance and InterestAndPenaltyBalance attributes have somehow a correlation with target variable of 0.34 and 0.33 respectively.
  - Correlation of Default/non-default Target with : Rating 0.2, Restructured 0.18, CreditScoreEsMicroL 0.16, PrincipalBalance 0.34, InterestAndPenaltyBalance 0.33
  - Correlation of PreviousEarlyRepaymentsCountBeforeLoan with : NoOfPreviousLoansBeforeLoan 0.14, AmountOfPreviousLoansBeforeLoan 0.14, PreviousRepaymentsBeforeLoan 0.14,
  - Correlation of PreviousRepaymentsBeforeLoan with : ExistingLiabitlies 0.15, NoOfPreviousLoansBeforeLoan 0.4, AmountOfPreviousLoansBeforeLoan 0.57, 
  - Correlation of AmountOfPreviousLoansBeforeLoan with : ExistingLiabilities 0.28, MonthlyPaymentDay 0.11, NoOfPreviousLoansBeforeLoan 0.77
  - Correlation of NoOfPreviousLoansBeforeLoan with : ExistingLiabilities 0.33, MonthlyPaymentDay 0.1, 
  - Correlation of InterestAndPenaltyBalance with : ExistingLiabilities 0.19, DebtToIncome 0.21, Rating 0.22, CreditScoreEsMicroL 0.25, PrincipalBalance 0.42,
  - Correlation of PrincipalBalance with : Rating 0.13, InterestAndPenaltyPaymentsMade 0.16 
  - Correlation of InterestAndPenaltyPaymentsMade with : ExistingLiabilities 0.15, RefinanceLiabilities 0.27, DebtToIncome 0.25, Restructured 0.26, PrincipalPaymentsMade 0.46
  - Correlation of PrincipalPaymentsMade with : ExistingLiabilities 0.11, RefinanceLiabilities 0.2, DebtToIncome 0.2,
  - Correlation of CreditScoreEsMicroL with : DebtToIncome 0.14, Rating 0.4
  - Correlation of Restructured with : ExistingLiabilities 0.17, DebtToIncome 0.17
  - Correlation of Freecash with : incomeTotal 0.16
  - Correlation of DebtToIncome with : ExistingLiabilities 0.44, RefinanceLiabilities 0.35, 
  - Correlation of RefinanceLiabilities with : ExistingLiabilities 0.46, 
- **Observations For Categorial Bivariate Analysis**
  - The countplot is like histogram for categorical attributes. So its easy to get insights about such attributes with target attributes
  - New 'NewCreditCustomers' are more likey to default than existing.
  - The customers having 'income and expenses verified', 'income verified' are more likely to default than others.
  - The Estonians are less likely to default and 'finnish','spanish' are more likely to default.
  - The males are more likely to default than females
  - Customers with 'undefined','home improvement''other' use of loan are more likely to default.
  - Customers with vocational education are more likely to default as it has more defualt ratio than other classes.
  - Customers with CredictScoreEsMicroL are more likely to default.
  - Customers with undefined maritalstatus are defaulting more than those with defined maritalstatus.
  - Customers with undefined Employment status are more likely to default than those with defined. SO customers must have defined employmentstatus.
  - Customers with employment duration more than 5 years are defaulting more. so compnany should avoid should customers.
  - Customers with undefined occupationArea are defaulting more.
  - Customrs with 'yes' restructured are more likely to default than others.
  - Tenants, prefurnised property owners are likely to default more as defualt ratio is more this class than owners.
  - The customers with HR rating are defaulting more as this class has more default ratio than F class. we can see from plot of Rating.

**Multivariate Analysis:**
- In multivariate analysis, the relation and trend among multiple features is analyzed at same time to get insights into the dataset.
- it becomes easy when we have performed bivariate analysis and plotted correlation heatmaps.
- it is done among the features having good correlations to expose the trends and insights.
- **FINAL observations are:**
  - Borrowers with low Loan Amount and High interest rates have greater chance of defaulting than with those having low amount and low interest.
  - Borrowers with high loan Amount and less chances have low chances of defaulting.
  - Borrowers with high interest and longer loanDuration have high chances of defaulting than those who have low interest and shorter loanDuration
  - Borrowers having having low Loan amount and a non zero previous loans before loan have low chance of getting defualted.
  - Borrowers having no previous loans before loan have high chances of defaulting .
  - Borrowers having high loan amount and high no of existingliabilities have high chances of defaulting
  - Borrowers having high principal balance and high loan amount have likely less chances of defaulting
  - Borrowers having less interest and low monthlyPayment have likely less chances of defualting
  - Borrowers having low MonthlyPayments and shorter LoanDurations have less chances of defaulting and vice versa.
  - no clear trend among Amount, EmploymentStatus and target Default/non-default.
  - no clear trend among LoanDuration, MaritalStatus and target Default/non-default.
  - no clear trend among Amount, Age and target Default/non-default.
  - no clear trend among BidsPortfolioManager, Amount and target Default/non-default
  - The County and City attributes are very diverse, no clear relationships can be indentified from such diverse features.

**Descriptive Statistics:**
- The Descriptive Analysis of dataset can be found here ([**Descriptive Analysis**](https://colab.research.google.com/drive/1l-Z780JEs44L-PC8n-LU40IUP-oL7-Ph?usp=sharing))

## Feature Engineering

Features in datadset are not ready for model building as we need to treat outliers, encode categorical variables, select features which are actually contributing to prediction of target variable.

**Mutual Information Scores**

1. As Mutual information is univariate scoring method we can not rely much on that to removing features
2. fetures with least scores can be useful when combined with other variables

InterestAndPenaltyBalance                 0.34
PrincipalPaymentsMade                     0.27
PrincipalBalance                          0.25
RecoveryStage                             0.15

Above features are highly contributing towards target variable

**Outliers treatment**

Numerical variables were included with outliers in both direction, we treated all of them with IQR method

*Method*

IQR = Inter quantile range is range between 75th and 25th quantile 
Lower bound = 25th quantile - 1.5*IQR
Upper bound = 75th quantile + 1.5*IQR

Now after defining upper and lower bound
- For each numeric variable, we checked which values are outside of their respective bounds
- Values less than lower bound replaced by "Lower bound" 
- Similarly values higher than upper bound were replaced by "Upper bound"

**Categorical encoding**

'target', 'NewCreditCustomer','VerificationType', 'LanguageCode', 'Gender', 'Country', 'UseOfLoan','Education', 'MaritalStatus', 'EmploymentStatus','EmploymentDurationCurrentEmployer', 'OccupationArea','HomeOwnershipType', 'Restructured', 'CreditScoreEsMicroL'

are the variables which are categorical i.e. "Object" datatype
- we selected these columns for encoding inside pipeline using "make_column_selector" from sklearn.compose module
- One Hot Encoding is used here for categorical variable encoding.
- The reason for using this method is because some variables were ordinal categories and some were nominal categories
    - Encoding these variables with label encoder may result in bad performance and explanability of machine learning model at the end
- From sklearn.preprocessing module we imported and instantiated the One Hot encoder class and used inside column transformer selecting only categorical variables

**Scaling the features**

- We used Ridge regressior as regression algorithm to predict target variables.
- The performance of ridge regressor will be better without any bias to higher values after using scaled features was the main intention of scaling
- We used Standard Scaler from sklearn.preprocessing module

*StandardScaler Methodology*

The standard score of a sample x is calculated as:

z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

- We used "with_mean=False" as we wanted our features not to be centred 

- We used Standard scaling inside pipeline for all the variables (Including one hot encoded target variables)


## Model Building
#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

#### Random Forest Classifier
- The random forest is a classification algorithm consisting of **many decision trees**. It uses bagging and features randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.

#### Linear Regression
- In this method it attempts to model the relationship between two variables by fitting a linear equation to observed data. 
- One variable is considered to be an explanatory variable, and the other is dependent variable.

#### Ridge Regression
- It is the method used for the analysis of multicollinearity in multiple regression data. It is most suitable when a data set contains a higher number of predictor variables than the number of observations.
- Multicollinearity happens when predictor variables exhibit a correlation among themselves. It aims at reducing the standard error by adding some bias in the estimates of the regression.
- The reduction of the standard error in regression estimates significantly increases the reliability of the estimates.

**We tranined 4 models; 2 for classification and 2 for regression**
- When we trained the **logistic regression** model the F1_score we got 82.11%. After tuning parameters using GridSearchCV we got F1_score as 83.36%.
- When we trained **random forest classifier** model using best hyperparameters the F1_score we got 92.86%. After removing unimportant features, we got F1_score as 93.03%.
- When we trained **linear Regression** the R2_score we got is 78.46%.
- When we trained **Ridge Regression** the R2_score we got is 89%.

*Finally, 2 pipelines are deployed:
1. **Random Forest Classifier (Binary Classification)**

2. **Ridge Regression for multi targets (EMI, ELA, ROI)**


## Deployment
Our deployed app can be accessed by following link ([**Credit-Risk-Analysis**](https://arslan1k-ml-deployment-streamlit-app-hy7s2q.streamlit.app/))

### Streamlit
- Streamlit is an open source app framework in Python language. 
- It helps us create web apps for data science and machine learning. 
- It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

