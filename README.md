# Credit Risk Analysis for peer to peer lending firm Bandora
The credit risk is analyzed to measure the possibility of loss as result of borrower failing to repay a loan or meeting the loan obligations. The more the credit risk the more it has a negative impact on performance of bank. The credit risk analysis is important because it allows the bank to plan strategies to avoid a negative outcome ahead in future.
- In this project we analyzed the credit risk on individual loans, maximizing the Return on Investment (ROI) by predicting an Eligible Loan Amount and minimizing the financial risk between the lending firm Bandora and the borrowers.
- Machine Learning regression and classification pipelines are deployed to accomplish the objective of credit risk analysis.

## Acknowledgment
The completion of this project is dedicated to ([**TechnoColabs**](https://technocolabs.com/)). Special thanks to CEO ([**Yasin Shah**](https://www.linkedin.com/in/yasinshah9598/)) and Mentor ([**Mitesh Verma**](https://www.linkedin.com/in/mitesh-verma-049b12b3/)) for leading us with precious guidance and experience throught the internship period.

### Project Team
- [Arslan Mehmood](https://www.linkedin.com/in/arslan-mehmood-1k/) (Team Leader)
- [Kalinga Moharana](https://www.linkedin.com/in/kalinga-moharana-174881205/)
- [Aditya Kurhade](https://www.linkedin.com/in/kurhadeaditya//)
- [Md Asjadullah](https://www.linkedin.com/in/md-asjad-314501211)
- [Robair Garas](https://www.linkedin.com/in/robairgaras) 

## Understanding the Dataset
Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform  ([**Bondora**](https://www.bondora.com/en/public-reports#dataset-file-format))

- The dataset contains the individual loan details from **2009** to **2019** having both the **defaulted** and **non-defaulted** loans
- The dataset does not contain any target attributes for 
  - defaulted, non-defaulted borrowers
  - Eligible Loan Amount (ELA)
  - Return on Investment (ROI)
  - Equated Monthly Installment (EMI) 
- These 4 target attributes will be created using feature engineering

The dataset has 112 attributes intially. The following table briefly explains the attribute definitions.

| Feature                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ActiveLateCategory                     | When a loan is in Principal Debt then it will be categorized by Principal Debt days                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ActiveLateLastPaymentCategory          | Shows how many days has passed since last payment and categorised if it is overdue                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ActiveScheduleFirstPaymentReached      | Whether the first payment date has been reached according to the active schedule                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Age                                    | The age of the borrower when signing the loan application                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Amount                                 | Amount the borrower received on the Primary Market. This is the principal balance of your purchase from Secondary Market                                                                                                                                                                                                                                                                                                                                                                                            |
| AmountOfPreviousLoansBeforeLoan        | Value of previous loans                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| AppliedAmount                          | The amount borrower applied for originally                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| AuctionBidNumber                       | Unique bid number which is accompanied by Auction number                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| AuctionId                              | A unique number given to all auctions                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| AuctionName                            | Name of the Auction, in newer loans it is defined by the purpose of the loan                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| AuctionNumber                          | Unique auction number which is accompanied by Bid number                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| BidPrincipal                           | On Primary Market BidPrincipal is the amount you made your bid on. On Secondary Market BidPrincipal is the purchase price                                                                                                                                                                                                                                                                                                                                                                                           |
| BidsApi                                | The amount of investment offers made via Api                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| BidsManual                             | The amount of investment offers made manually                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| BidsPortfolioManager                   | The amount of investment offers made by Portfolio Managers                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| BoughtFromResale_Date                  | The time when the investment was purchased from the Secondary Market                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| City                                   | City of the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ContractEndDate                        | The date when the loan contract ended                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Country                                | Residency of the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| County                                 | County of the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| CreditScoreEeMini                      | 1000 No previous payments problems 900 Payments problems finished 24-36 months ago 800 Payments problems finished 12-24 months ago 700 Payments problems finished 6-12 months ago 600 Payment problems finished < 6 months ago 500 Active payment problems                                                                                                                                                                                                                                                          |
| CreditScoreEsEquifaxRisk               | Generic score for the loan applicants that do not have active past due operations in ASNEF; a measure of the probability of default one year ahead; the score is given on a 6-grade scale: AAA (“Very low”), AA (“Low”), A (“Average”), B (“Average High”), C (“High”), D (“Very High”).                                                                                                                                                                                                                            |
| CreditScoreEsMicroL                    | A score that is specifically designed for risk classifying subprime borrowers (defined by Equifax as borrowers that do not have access to bank loans); a measure of the probability of default one month ahead; the score is given on a 10-grade scale, from the best score to the worst: M1, M2, M3, M4, M5, M6, M7, M8, M9, M10.                                                                                                                                                                                  |
| CreditScoreFiAsiakasTietoRiskGrade     | Credit Scoring model for Finnish Asiakastieto RL1 Very low risk 01-20 RL2 Low risk 21-40 RL3 Average risk 41-60 RL4 Big risk 61-80 RL5 Huge risk 81-100                                                                                                                                                                                                                                                                                                                                                             |
| CurrentDebtDaysPrimary                 | How long the loan has been in Principal Debt                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| CurrentDebtDaysSecondary               | How long the loan has been in Interest Debt                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| DateOfBirth                            | The date of the borrower's birth                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| DebtOccuredOn                          | The date when Principal Debt occurred                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| DebtOccuredOnForSecondary              | The date when Interest Debt occurred                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| DebtToIncome                           | Ratio of borrower's monthly gross income that goes toward paying loans                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| DefaultDate                            | The date when loan went into defaulted state and collection process was started                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| DesiredDiscountRate                    | Investment being sold at a discount or premium                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| EAD1                                   | Exposure at default, outstanding principal at default                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| EAD2                                   | Exposure at default, loan amount less all payments prior to default                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Education                              | 1 Primary education 2 Basic education 3 Vocational education 4 Secondary education 5 Higher education                                                                                                                                                                                                                                                                                                                                                                                                               |
| EL_V0                                  | Expected loss calculated by the specified version of Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| EL_V1                                  | Expected loss calculated by the specified version of Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| EL_V2                                  | Expected loss calculated by the specified version of Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| EmploymentDurationCurrentEmployer      | Employment time with the current employer                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| EmploymentPosition                     | Employment position with the current employer                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| EmploymentStatus                       | 1 Unemployed 2 Partially employed 3 Fully employed 4 Self-employed 5 Entrepreneur 6 Retiree                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ExistingLiabilities                    | Borrower's number of existing liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ExpectedLoss                           | Expected Loss calculated by the current Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ExpectedReturn                         | Expected Return calculated by the current Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| FirstPaymentDate                       | First payment date according to initial loan schedule                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| FreeCash                               | Discretionary income after monthly liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Gender                                 | 0 Male 1 Woman 2 Undefined                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| GracePeriodEnd                         | Date of the end of Grace period                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| GracePeriodStart                       | Date of the beginning of Grace period                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| HomeOwnershipType                      | 0 Homeless 1 Owner 2 Living with parents 3 Tenant, pre-furnished property 4 Tenant, unfurnished property 5 Council house 6 Joint tenant 7 Joint ownership 8 Mortgage 9 Owner with encumbrance 10 Other                                                                                                                                                                                                                                                                                                              |
| IncomeFromChildSupport                 | Borrower's income from alimony payments                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| IncomeFromFamilyAllowance              | Borrower's income from child support                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| IncomeFromLeavePay                     | Borrower's income from paternity leave                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| IncomeFromPension                      | Borrower's income from pension                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| IncomeFromPrincipalEmployer            | Borrower's income from its employer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| IncomeFromSocialWelfare                | Borrower's income from social support                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| IncomeOther                            | Borrower's income from other sources                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| IncomeTotal                            | Borrower's total income                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Interest                               | Maximum interest rate accepted in the loan application                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| InterestAndPenaltyBalance              | Unpaid interest and penalties                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| InterestAndPenaltyDebtServicingCost    | Service cost related to the recovery of the debt based on the interest and penalties of the investment                                                                                                                                                                                                                                                                                                                                                                                                              |
| InterestAndPenaltyPaymentsMade         | Note owner received loan transfers earned interest, penalties total amount                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| InterestAndPenaltyWriteOffs            | Interest that was written off on the investment                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| InterestLateAmount                     | Interest debt amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| InterestRecovery                       | Interest recovered due to collection process from in debt loans                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| LanguageCode                           | 1 Estonian 2 English 3 Russian 4 Finnish 5 German 6 Spanish 9 Slovakian                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| LastPaymentOn                          | The date of the current last payment received from the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| LiabilitiesTotal                       | Total monthly liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ListedOnUTC                            | Date when the loan application appeared on Primary Market                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| LoanDate                               | Date when the loan was issued                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| LoanDuration                           | Current loan duration in months                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| LoanId                                 | A unique ID given to all loan applications                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| LoanNumber                             | A unique number given to all loan applications                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| LoanStatusActiveFrom                   | How long the current status has been active                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| LossGivenDefault                       | Gives the percentage of outstanding exposure at the time of default that an investor is likely to lose if a loan actually defaults. This means the proportion of funds lost for the investor after all expected recovery and accounting for the time value of the money recovered. In general, LGD parameter is intended to be estimated based on the historical recoveries. However, in new markets where limited experience does not allow us more precise loss given default estimates, a LGD of 90% is assumed. |
| MaritalStatus                          | 1 Married 2 Cohabitant 3 Single 4 Divorced 5 Widow                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| MaturityDate_Last                      | Loan maturity date according to the current payment schedule                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| MaturityDate_Original                  | Loan maturity date according to the original loan schedule                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ModelVersion                           | The version of the Rating model used for issuing the Bondora Rating                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| MonthlyPayment                         | Estimated amount the borrower has to pay every month                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| MonthlyPaymentDay                      | The day of the month the loan payments are scheduled for The actual date is adjusted for weekends and bank holidays (e.g. if 10th is Sunday then the payment will be made on the 11th in that month)                                                                                                                                                                                                                                                                                                                |
| NewCreditCustomer                      | Did the customer have prior credit history in Bondora 0 Customer had at least 3 months of credit history in Bondora 1 No prior credit history in Bondora                                                                                                                                                                                                                                                                                                                                                            |
| NextPaymentDate                        | According to schedule the next date for borrower to make their payment                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| NextPaymentNr                          | According to schedule the number of the next payment                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| NextPaymentSum                         | According to schedule the amount of the next payment                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| NoOfPreviousLoansBeforeLoan            | Number of previous loans                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| note_id                                | A unique ID given to the investments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| NoteLoanLateChargesPaid                | The amount of late charges the note has received                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| NoteLoanTransfersInterestAmount        | The amount of interest the note has received                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| NoteLoanTransfersMainAmount            | The amount of principal the note has received                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| NrOfDependants                         | Number of children or other dependants                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| NrOfScheduledPayments                  | According to schedule the count of scheduled payments                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| OccupationArea                         | 1 Other 2 Mining 3 Processing 4 Energy 5 Utilities 6 Construction 7 Retail and wholesale 8 Transport and warehousing 9 Hospitality and catering 10 Info and telecom 11 Finance and insurance 12 Real-estate 13 Research 14 Administrative 15 Civil service & military 16 Education 17 Healthcare and social help 18 Art and entertainment 19 Agriculture, forestry and fishing                                                                                                                                      |
| OnSaleSince                            | Time when the investment was added to Secondary Market                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| PenaltyLateAmount                      | Late charges debt amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| PlannedInterestPostDefault             | The amount of interest that was planned to be received after the default occurred                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| PlannedInterestTillDate                | According to active schedule the amount of interest the investment should have received                                                                                                                                                                                                                                                                                                                                                                                                                             |
| PlannedPrincipalPostDefault            | The amount of principal that was planned to be received after the default occurred                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| PlannedPrincipalTillDate               | According to active schedule the amount of principal the investment should have received                                                                                                                                                                                                                                                                                                                                                                                                                            |
| PreviousEarlyRepaymentsBeforeLoan      | How much was the early repayment amount before the loan                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| PreviousEarlyRepaymentsCountBeforeLoan | How many times the borrower had repaid early                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| PreviousRepaymentsBeforeLoan           | How much the borrower had repaid before the loan                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| PrincipalBalance                       | Principal that still needs to be paid by the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| PrincipalDebtServicingCost             | Service cost related to the recovery of the debt based on the principal of the investment                                                                                                                                                                                                                                                                                                                                                                                                                           |
| PrincipalLateAmount                    | Principal debt amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| PrincipalOverdueBySchedule             | According to the current schedule, principal that is overdue                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| PrincipalPaymentsMade                  | Note owner received loan transfers principal amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| PrincipalRecovery                      | Principal recovered due to collection process from in debt loans                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| PrincipalWriteOffs                     | Principal that was written off on the investment                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ProbabilityOfDefault                   | Probability of Default, refers to a loan’s probability of default within one year horizon.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| PurchasePrice                          | Investment amount or secondary market purchase price                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Rating                                 | Bondora Rating issued by the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Rating_V0                              | Bondora Rating issued by version 0 of the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Rating_V1                              | Bondora Rating issued by version 1 of the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Rating_V2                              | Bondora Rating issued by version 2 of the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| RecoveryStage                          | Current stage according to the recovery model 1 Collection 2 Recovery 3 Write Off                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| RefinanceLiabilities                   | The total amount of liabilities after refinancing                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ReScheduledOn                          | The date when the a new schedule was assigned to the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Restructured                           | The original maturity date of the loan has been increased by more than 60 days                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| SoldInResale_Date                      | The date when the investment was sold on Secondary market                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| SoldInResale_Price                     | The price of the investment that was sold on Secondary market                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| SoldInResale_Principal                 | The principal remaining of the investment that was sold on Secondary market                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| StageActiveSince                       | How long the current recovery stage has been active                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Status                                 | The current status of the loan application                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| UseOfLoan                              | 0 Loan consolidation 1 Real estate 2 Home improvement 3 Business 4 Education 5 Travel 6 Vehicle 7 Other 8 Health 101 Working capital financing 102 Purchase of machinery equipment 103 Renovation of real estate 104 Accounts receivable financing 105 Acquisition of means of transport 106 Construction finance 107 Acquisition of stocks 108 Acquisition of real estate 109 Guaranteeing obligation 110 Other business All codes in format 1XX are for business loans that are not supported since October 2012  |
| UserName                               | The user name generated by the system for the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| VerificationType                       | Method used for loan application data verification 0 Not set 1 Income unverified 2 Income unverified, cross-referenced by phone 3 Income verified 4 Income and expenses verified                                                                                                                                                                                                                                                                                                                                    |
| WorkExperience                         | Borrower's overall work experience in years                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| WorseLateCategory                      | Displays the last longest period of days when the loan was in Principal Debt                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| XIRR                                   | XIRR (extended internal rate of return) is a methodology to calculate the net return using the loan issued date and amount, loan repayment dates and amounts and the principal balance according to the original repayment date. All overdue principal payments are written off immediately. No provisions for future losses are made & only received (not accrued or scheduled) interest payments are taken into account.                                                                                          |


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

<p align="center">
  <img src="/resources/media/1.png">
</p>

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

<p align="center">
  <img src="/resources/media/2.png">
</p>

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

<p align="center">
  <img src="/resources/media/3.png">
</p>

    - The missing values in categorical attributes are imputed with the mode() of respective attributes.

<p align="center">
  <img src="/resources/media/4.png">
</p>

## Exploratory Data Analysis
**Introduction:**

- The dataset after data wrangling comprises of 77394 rows and 41 features and 4 targets.
- Dataset comprises of int, float and object data types. 
- In EDA, three type of analysis are performed
  - Univariate Analysis
  - Bivariate Analysis
  - Multivariate Analysis

### Univariate Analysis:
- In univaraite analysis, each feature is analyzed and explored individually to get hidden insights into it.
- The categorical features are analyzed and explored using Seaborn countplot. The countplot is like an histogram for categorical attributes
- The numerical features are analyzed and explored using Seaborn kdeplot and Displot
- A custom function was coded to calculate the percentage_of_top_n_classes of any categorical column given the n and the column's data
**- Some of the univariate plots are given below:**

<p align="center">
  <img src="/resources/media/eda_1.png">
  <img src="/resources/media/eda_2.png">
  <img src="/resources/media/eda_3.png">
  <img src="/resources/media/eda_4.png">
  <img src="/resources/media/eda_5.png">
</p>

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
  
### Bivariate Analysis:
- In bivaraite analysis, the features are analyzed and explored with respect to each other to get hidden insights into the relation of different featuers
- The correlation heat map is used to get the insights into the relation among different features
- The categorical features are analyzed and explored using Seaborn countplot. The countplot is like an histogram for categorical attributes
- The numerical features are analyzed and explored using Seaborn kdeplot and Displot.

#### The Correlation Heat maps are shown below:

<p align="center">
  <img src="/resources/media/bi_1.png">
  <img src="/resources/media/bi_2.png">
</p>

**- Some of the bivariate plots are given below:**

<p align="center">
  <img src="/resources/media/bi_3.png"><img src="/resources/media/bi_4.png">
  <img src="/resources/media/bi_5.png"><img src="/resources/media/bi_6.png">
</p>

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

### Multivariate Analysis:
- In multivariate analysis, the relation and trend among multiple features is analyzed at same time to get insights into the dataset.
- it becomes easy when we have performed bivariate analysis and plotted correlation heatmaps.
- it is done among the features having good correlations to expose the trends and insights.

**- Some of the multivariate plots are given below:**

<p align="center">
  <img src="/resources/media/m1.png"><img src="/resources/media/m2.png">
  <img src="/resources/media/m3.png"><img src="/resources/media/m4.png">
  <img src="/resources/media/m5.png">
</p>

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

<p align="center">
  <img src="/resources/media/descriptive_statistics.png">
</p>

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

*Finally, 2 pipelines are deployed:*
1. **Random Forest Classifier (Binary Classification)**

2. **Ridge Regression for multi targets (EMI, ELA, ROI)**


## Deployment
Our deployed app can be accessed by following link ([**Credit-Risk-Analysis App**](https://arslan1k-ml-deployment-streamlit-app-hy7s2q.streamlit.app/))

<p align="center">
  <img src="/resources/media/app_1.jpg">
  <img src="/resources/media/app_2.jpg">
  <img src="/resources/media/app_3.jpg">
</p>

### Streamlit
- Streamlit is an open source app framework in Python language. 
- It helps us create web apps for data science and machine learning. 
- It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.
