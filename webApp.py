import pandas as pd
import streamlit as st
import sklearn
import base64
import pickle
from pandas import MultiIndex, Int64Index
from urllib.request import urlopen, Request
st.set_page_config(page_title="BANDORA LOAN APPROVAL WEBAPP", page_icon="random", layout="wide", initial_sidebar_state="expanded")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#app background
add_bg_from_local('bckg.jpg')

classifier_pipeline = pickle.load(open('classifier_gboost_pipeline.pkl','rb'))


Regressor_pipeline = pickle.load(open('regression_xgboost_pipeline.pkl','rb'))
@st.experimental_memo
def create_input_Dataframe():
    input_dictionary = {
        "LanguageCode": Language,
        "HomeOwnershipType": HomeOwnershipType,
        "Restructured": Restructured,
        "IncomeTotal": IncomeTotal,
        "LiabilitiesTotal": LiabilitiesTotal,
        "LoanDuration": LoanDuration,
        "AppliedAmount": AppliedAmount,
        "Amount": Amount,
        "Interest": Interest,
        "EMI": EMI,
        "PreviousRepaymentsBeforeLoan": PreviousRepaymentsBeforeLoan,
        "MonthlyPaymentDay": MonthlyPaymentDay,

        "PrincipalPaymentsMade": PrincipalPaymentsMade,
        "InterestAndPenaltyPaymentsMade": InterestAndPenaltyPaymentsMade,
        "PrincipalBalance": PrincipalBalance,
        "InterestAndPenaltyBalance": InterestAndPenaltyBalance,
        "Bids": BidsPortfolioManger + BidsApi,
        "Rating": Rating
    }

    DF = pd.DataFrame(input_dictionary, index=[0])
    return DF


def Classifier():
    input = create_input_Dataframe()
    prediction = classifier_pipeline.predict(input)
    if prediction == 1:
        result = "Defaulter"
        Default=1
    if prediction == 0:
        result = "Not Defaulter"
        Default=0
    return result
@st.experimental_memo
def create_regression_input():
    regr_dict={
        #NUM data
        "BidsPortfolioManager": BidsPortfolioManager,
        "BidsApi": BidsApi,
        "BidsManual": BidsManual,
        "Age": Age,
        "AppliedAmount": AppliedAmount,
        "Amount": Amount,
        "Interest": Interest,
        "LoanDuration": LoanDuration,
        "MonthlyPayment": MonthlyPayment,
        "EmploymentDurationCurrentEmployer": EmploymentDurationCurrentEmployer,
        "IncomeTotal": IncomeTotal,
        "ExistingLiabilities": ExistingLiabilities,
        "LiabilitiesTotal": LiabilitiesTotal,
        "RefinanceLiabilities": RefinanceLiabilities,
        "DebtToIncome": DebtToIncome,
        "FreeCash": Freecash,
        "MonthlyPaymentDay": MonthlyPaymentDay,
        "CreditScoreEeMini'": CreditScoreEeMini,
        "PrincipalPaymentsMade": PrincipalPaymentsMade,
        "InterestAndPenaltyPaymentsMade": InterestAndPenaltyPaymentsMade,
        "PrincipalBalance": PrincipalBalance,
        "InterestAndPenaltyBalance": InterestAndPenaltyBalance,
        "NoOfPreviousLoansBeforeLoan": NoOfPreviousLoansBeforeLoan,
        "AmountOfPreviousLoansBeforeLoan": AmountOfPreviousLoansbeforeloan,
        "PreviousEarlyRepaymentsCountBeforeLoan": PreviousEarlyRepaymentsCountBeforeLoan,
        #cat
        "NewCreditCustomer": NewCreditCustomer,
        "VerificationType": VerificationType,
        "LanguageCode": LanguageCode,
        "Gender": Gender,
        "UseOfLoan": UseOfLoan,
        "Education": Education,
        "MaritalStatus": MaritalStatus,
        "EmploymentStatus": EmploymentStatus,
        "OccupationArea": OccupationArea,
        "HomeOwnershipType": HomeOwnershipType,
        "RecoveryStage": RecoveryStage,
        "Rating": Rating,
        "Restructured": Restructured,
        "CreditScoreEsMicroL":CreditScoreEsMicroL,
        "Default":Default,
        #numerical data

    }
    DF_reg = pd.DataFrame(regr_dict, index=[0])
    return DF_reg
#regression model using
def Regressor():
    input = create_regression_input()
    prediction = Regression_pipeline.predict(input)
    Result=pd.DataFrame({"EMI":None,"ELA":None,"ROI":None})
    Result['EMI']=prediction[0]
    Result['ELA']=prediction[1]
    Result['ROI']=prediction[2]
    return Result
#getting user filled data
@st.experimental_memo
def load_data():
    df=create_regression_input()
    return df


#APP LAYOUT
st.title('Bandora Loan Approval Dashboard')
st.header("Borrower's Information")
op1=st.button(label="Personal Details")
if(op1):
    st.subheader('Personal Background')
    EmploymentDurationCurrentEmployer = st.selectbox('EmploymentDurationCurrentEmployer', (
    "MoreThan5Years", "UpTo1Year", "UpTo5Years", "UpTo3Years", "UpTo4Years", "Other", "TrialPeriod"))
    LanguageCode = st.selectbox('Language',
                                ("estonia","Finish","spanish","other"))
    HomeOwnershipType = st.selectbox('Home Ownership Type', (
    "homeless", "Owner", "other", "Tenant_pre-furnished property", "Living with parents",
    "Mortgage", "Tenant_unfurnished propert", "other", "Joint ownership", "Joint tenant",
    "Council house", "Owner with encumbrance"))
    Gender = st.selectbox('Gender', ("Male", "Woman", "Undefined"))
    Education = st.selectbox('Education', (
    "Basic education", "Primary education", "Vocational education", "Higher education", "other", "Secondary education"))
    MaritalStatus = st.selectbox('MaritalStatus', ("Married", "Cohabitant", "Single", "Divorced", "Widow", "other"))
    IncomeTotal = st.number_input('Total Icome')
    LiabilitiesTotal = st.number_input('Total Liabilities')
    EmploymentStatus = st.selectbox('EmploymentStatus', (
    "Fully employed", "Self-employed_Entrepreneur_Retiree", "Unemployed_Partially employed"))
    OccupationArea = st.selectbox('OccupationArea', (
    "Other", "Mining", "Processing", "Energy", "Utilities", "Construction", "Retail and wholesale",
    "Transport and warehousing",
    "Hospitality and catering", "Info and telecom", "Finance and insurance", "Real-estate", "Research", "Administrative"
    , "Civil service & military", "Education", "Healthcare and social help", "Art and entertainment",
    "Agriculture,forestry and fishing"))
    RefinanceLiabilities = st.number_input('Refinance Liabilities')
    FreeCash = st.number_input('Free Cash')
    ExistingLiabilities = st.number_input('Existing Liabilities')
    DebtToIncome = st.number_input('DebtToIncome')



op2=st.button(label="Loan Details")
if(op2):
    st.subheader('Loan Details')
    NewCreditCustomer = st.selectbox('NewCreditCustomer', ("True", "False"))
    LoanDuration = st.number_input('Loan Duration (in months)')
    AppliedAmount = st.number_input('Applied Loan Amount')
    Amount = st.number_input('Amount (granted)')
    Interest = st.number_input('Interest')
    EMI = st.number_input('Equated Monthly Installment')
    RecoveryStage = st.selectbox('RecoveryStage', ("Collection", "Recovery"))
    UseOfLoan = st.selectbox('UseOfLoan', (
    "other", "Home improvement", "Loan consolidation", "Vehicle", "Travel", "Business", "Education", "Any"))
    VerificationType = st.selectbox('VerificationType', (
    "Not set", "Income unverified", "Income unverified cross-referenced by phone", "Income verified",
    "Income and expenses verified"))
    CreditScoreEsMicroL = st.selectbox('CreditScoreEsMicroL',
                                       ("M", "M3", "M5", "M1", "M9", "M2", "M6", "M4", "M8", "M7", "M10"))
    NoOfPreviousLoansBeforeLoan = st.number_input('No Of Previous Loans Before Loan')
    AmountOfPreviousLoansbeforeloan = st.text_input('Amount Of Previous Loans before loan')
    CreditScoreEeMini = st.number_input('Credit Score Ee Mini')
    PreviousEarlyRepaymentsCountBeforeLoan = st.number_input('PreviousEarlyRepaymentsCountBeforeLoan')
    Restructured = st.selectbox('Restructured', ("False", "True"))

op3=st.button(label="PaymentofPreviousLoan")
if(op3):
    st.subheader('Payment Details')
    PreviousRepaymentsBeforeLoan = st.number_input('PreviousRepaymentsBeforeLoan')
    MonthlyPaymentDay = st.number_input('MonthlyPaymentDay (digit)')
    PrincipalPaymentsMade = st.number_input('Principal Payments Made')
    InterestAndPenaltyPaymentsMade = st.number_input('Interest and Penalty Payments Made')

op4=st.button(label="YourBalance")
if(op4):
    st.subheader('Balance Details')
    PrincipalBalance = st.number_input('PrincipalBalance')
    InterestAndPenaltyBalance = st.number_input('InterestAndPenaltyBalance')
    st.subheader('Amount of Investment offers made via')
    BidsPortfolioManger = st.number_input('Bids through PortfolioManger')
    BidsApi = st.number_input('Bids using Api')
    BidsManual = st.number_input('Bids by Manual')

opt5=st.button(label='OTHER')
if(opt5):
    st.subheader('Other')
    Rating = st.selectbox('Rating', ("A", "AA", "B", "C", "D", "E", "F", "HR"))

st.header('Loan Application Status')
if st.button(label="Check Status"):
    with st.spinner('Analyzing the Provided Information ...'):
        time.sleep(5)
    result = Classifier()
    st.spinner(text="Analyzing the Information")

    if result == "Defaulter":
        st.write("Based on details provided, the user may default so loan is not approved, Thanks!")
        time.sleep(3)
        with st.spinner('Predicting preferred Loan details ...'):
            time.sleep(5)
            result=Regressor()
            st.dataframe(result, use_container_width=st.session_state.use_container_width)
            st.balloons()

    elif result == "Not Defaulter":
        st.write("Congratulations! Your loan is Approved!")
        time.sleep(5)
        with st.spinner('Predicting preferred Loan details ...'):
            time.sleep(5)
            result = Regressor()
            st.dataframe(result, use_container_width=st.session_state.use_container_width)
            st.balloons()





op6=st.button(label="your given detail(want to see)")
if(op6):
    st.checkbox("Use container width", value=False, key="use_container_width")

    df = load_data()

    # Display the dataframe and allow the user to stretch the dataframe
    # across the full width of the container, based on the checkbox value
    st.dataframe(df, use_container_width=st.session_state.use_container_width)
    st.balloons()