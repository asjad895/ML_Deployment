import pandas as pd
import streamlit as st
import numpy as np
import joblib
import sklearn
import time

classifier_pipeline = joblib.load('RFC_pipeline_Final.joblib')
Regressor_pipeline = joblib.load('RR_pipeline_Final.joblib')

def create_input_Dataframe():
  
  input_dictionary = {
    "LanguageCode" : Language,
    "HomeOwnershipType": HomeOwnershipType,
    "Restructured" : Restructured,
    "IncomeTotal" : IncomeTotal,
    "LiabilitiesTotal" : LiabilitiesTotal,
    "LoanDuration" : LoanDuration,
    "AppliedAmount" : AppliedAmount,
    "Amount": Amount,
    "Interest":Interest,
    "EMI": MonthlyPayment,
    "PreviousRepaymentsBeforeLoan" : PreviousRepaymentsBeforeLoan,
    "MonthlyPaymentDay" :MonthlyPaymentDay,
    
    "PrincipalPaymentsMade" : PrincipalPaymentsMade,
    "InterestAndPenaltyPaymentsMade" : InterestAndPenaltyPaymentsMade,
    "PrincipalBalance" : PrincipalBalance,
    "InterestAndPenaltyBalance" : InterestAndPenaltyBalance,
    "Bids" : BidsPortfolioManager+BidsApi,
    "Rating" : Rating
  }
  DF = pd.DataFrame(input_dictionary,index=[0])
  return DF
  
def create_DF_Regression():
    
  input_dictionary = {
    "Gender" : Gender,
    "Age" :Age,
    "Country" : Country,
    "Education" : Education,
    "MaritalStatus" : MaritalStatus,
    "OccupationArea" : OccupationArea,
    "EmploymentStatus" : EmploymentStatus,
    "EmploymentDurationCurrentEmployer" : EmploymentDurationCurrentEmployer,
    
    "NewCreditCustomer" : NewCreditCustomer,
    "VerificationType" : VerificationType,
    "UseOfLoan" : UseOfLoan,
    
    "LanguageCode" : Language,
    "HomeOwnershipType": HomeOwnershipType,
    "Restructured" : Restructured,
    "IncomeTotal" : IncomeTotal,
    "LiabilitiesTotal" : LiabilitiesTotal,
    "ExistingLiabilities" : ExistingLiabilities,
    "RefinanceLiabilities" : RefinanceLiabilities,
    "DebtToIncome" : DebtToIncome,
    "FreeCash" : FreeCash,
    "PreviousEarlyRepaymentsCountBeforeLoan" : PreviousEarlyRepaymentsCountBeforeLoan,
    "NoOfPreviousLoansBeforeLoan" : NoOfPreviousLoansBeforeLoan,
    "AmountOfPreviousLoansBeforeLoan" : AmountOfPreviousLoansBeforeLoan,
   
    "LoanDuration" : LoanDuration,
    "AppliedAmount" : AppliedAmount,
    "Amount": Amount,
    "Interest":Interest,
    "MonthlyPayment": MonthlyPayment,
    "MonthlyPaymentDay" :MonthlyPaymentDay,
    "PreviousRepaymentsBeforeLoan" : PreviousRepaymentsBeforeLoan,
    
    "PrincipalPaymentsMade" : PrincipalPaymentsMade,
    "InterestAndPenaltyPaymentsMade" : InterestAndPenaltyPaymentsMade,
    "PrincipalBalance" : PrincipalBalance,
    "InterestAndPenaltyBalance" : InterestAndPenaltyBalance,
    "BidsPortfolioManager" : BidsPortfolioManager,
    "BidsApi" : BidsApi,
    "BidsManual" : BidsManual,
    "Rating" : Rating,
    "CreditScoreEsMicroL" : CreditScoreEsMicroL
  }
  
  DF = pd.DataFrame(input_dictionary,index=[0])
  return DF

def Classifier():
  input = create_input_Dataframe()
  prediction = classifier_pipeline.predict(input)
  if prediction==1:
    result = "Defaulter"
  if prediction==0:
    result = "Not Defaulter"
  
  return result

def Regressor():
  input = create_DF_Regression()
  prediction = Regressor_pipeline.predict(input)
  
  return prediction

st.title('Bandora Loan Approval Dashboard')

tab1, tab2 = st.tabs(["**Credit Risk Analysis**", "**Acknowledgment**"])

with tab1:
  st.markdown("The credit risk is analyzed to measure the _**possibility of loss as result of borrower failing to repay a loan or meeting the loan obligations**_. The more the credit risk the more it has a negative impact on performance of bank. The credit risk analysis is important because it allows the bank to plan strategies to avoid a negative outcome ahead in future.")
  st.header("Borrower's Information")
  st.subheader('Personal Background')
  with st.expander("**fill here**"):
    Gender = st.selectbox('Gender',("male","woman","undefined"))
    Age= st.text_input('Age')
    Country = st.selectbox('Country',("ee","fi","es","sk"))
    Education = st.selectbox('Education',("secondary education","higher education","vocational education","basic education",
                                     "primary education","not_present"))
    MaritalStatus = st.selectbox('Marital Status',("single","married","cohabitant","divorced","widow","not_specified"))
    OccupationArea = st.selectbox('Occupation Area',("retail and wholesale","construction","processing","transport and warehousing",
                                                   "healtcare and social help","hospitality and catering","info and telecom",
                                                   "civil service & military","education","finance and insurance","agriculture,forestry and fishing",
                                                   "administrative","energy","art and entertainment","research","real-estate","utlities","mining",
                                                  "not set","other"))

    EmploymentStatus = st.selectbox('Employment Status',("fully employed","entrepneur","retiree","self employed","partially employed","not set"))
    EmploymentDurationCurrentEmployer = st.selectbox('Employment Duration Current Employer',("morethan5years","upto1year","upto5years","upto2years",
                                                                                           "upto3years","retiree","upto4years","other","trialperiod"))

    Language = st.selectbox('Language',("estonian","english", "russian","finnish", "german","spanish","slovakian"))
    HomeOwnershipType = st.selectbox('Home Ownership Type',("homeless","owner","living with parents","tenant, pre-furnished property",
                                                          "tenant, unfurnished property","council house","joint tenant","joint ownership","mortgage",
                                                          "owner with encumbrance","other"))
    Restructured = st.selectbox('Restructured',("yes","no"))
    IncomeTotal = st.text_input('Total Icome')
    LiabilitiesTotal = st.text_input('Total Liabilities')
    ExistingLiabilities = st.text_input('Existing Liabilities')
    RefinanceLiabilities = st.text_input('Refinance Liabilities')
    DebtToIncome = st.text_input('Debt To Income')
    FreeCash = st.text_input('Free Cash')


  st.subheader('Loan Details')
  with st.expander("**fill here**"):
    UseOfLoan = st.selectbox('Use Of Loan',("not set","home improvement", "loan consolidation","vehicle", "business","travel","health",
                                         "education","real estate","purchase of machinery equipment","other  business",
                                         "accounts receivable financing","working capital financing","acquisition of stocks",
                                         "acquisition of real estate","construction finance"))

    NoOfPreviousLoansBeforeLoan = st.text_input('No Of Previous Loans Before Loan')
    AmountOfPreviousLoansBeforeLoan = st.text_input('Amount Of Previous Loans Before Loan')
    NewCreditCustomer = st.selectbox('New Credit Customer',("new","existing"))
    VerificationType = st.selectbox('New Credit Customer',("income and expenses verified","income unverified","income verified",
                                                         "income unverified, cross-referenced by phone","not set"))


    LoanDuration = st.text_input('Loan Duration (in months)') 
    AppliedAmount = st.text_input('Applied Loan Amount')
    Amount = st.text_input('Amount (granted)')
    Interest = st.text_input('Interest')
    MonthlyPayment = st.text_input('Monthly Payment')



  st.subheader('Payment Details')
  with st.expander("**fill here**"):
    PreviousEarlyRepaymentsCountBeforeLoan = st.text_input('Previous Early Repayments Count Before Loan')
    PreviousRepaymentsBeforeLoan = st.text_input('PreviousRepaymentsBeforeLoan')
    MonthlyPaymentDay = st.text_input('MonthlyPaymentDay (digit)')
    PrincipalPaymentsMade = st.text_input('Principal Payments Made') 
    InterestAndPenaltyPaymentsMade = st.text_input('Interest and Penalty Payments Made')

  st.subheader('Balance Details')
  with st.expander("**fill here**"):
    PrincipalBalance = st.text_input('PrincipalBalance')
    InterestAndPenaltyBalance = st.text_input('InterestAndPenaltyBalance')

  st.subheader('Amount of Investment offers made via')
  with st.expander("**fill here**"):
    BidsPortfolioManager = st.text_input('BidsPortfolioManger')
    BidsApi = st.text_input('BidsApi')
    BidsManual = st.text_input('BidsManual')

  st.subheader('Other')
  with st.expander("**fill here**"):
    Rating = st.selectbox('Rating',("a","aa", "b","c", "d","e","f","hr"))
    CreditScoreEsMicroL = st.selectbox('CreditScoreEsMicroL',("m1","m2", "m3","m4", "m5","m6","m7","m8","m9","m10","not set"))

  st.header('Loan Application Status')
  if st.button(label="Check Status"):
    with st.spinner('Analyzing the Provided Information ...'):
      time.sleep(2)
    result = Classifier()
    st.spinner(text="Analyzing the Information")

    if result=="Defaulter":
      st.write("Based on details provided, the user may default so loan is not approved, Thanks!")
      time.sleep(2)
      with st.spinner('Predicting Eligible Loan details ...'):
        Regressor_result = np.array(Regressor())
        time.sleep(2)
        st.header('Eligibile Loan Amount Details')
        st.write("Equated Monthly Installment (EMI) = ",Regressor_result[0,0])
        st.write("Eligible Loan Amount (ELA) = ",Regressor_result[0,1])
        st.write("Return on Investment (ROI) = ", Regressor_result[0,2])

    if result=="Not Defaulter":
      st.write("Your loan is Approved!")

st.header("")
with tab2:
  
  
  st.markdown("The completion of the project [Credit Risk Analysis](https://github.com/Arslan1k/ML_Deployment) is dedicated to ([TechnoColabs](https://technocolabs.com/)). Special thanks to CEO ([Yasin Shah](https://www.linkedin.com/in/yasinshah9598/)) and Mentor ([Mitesh Verma](https://www.linkedin.com/in/mitesh-verma-049b12b3/)) for leading us with precious guidance and experience throught the internship period.")

  st.markdown("**Project Team**")
  st.markdown("- [Arslan Mehmood](https://www.linkedin.com/in/arslan-mehmood-1k/) (Team Leader)")
  st.markdown("- [Kalinga Moharana](https://www.linkedin.com/in/kalinga-moharana-174881205/)")
  st.markdown("- [Aditya Kurhade](https://www.linkedin.com/in/kurhadeaditya//)")
  st.markdown("- [Md Asjadullah](https://www.linkedin.com/in/md-asjad-314501211)")
  st.markdown("- [Robair Garas](https://www.linkedin.com/in/robairgaras) ")
  st.caption("__________________________________________________________________")



