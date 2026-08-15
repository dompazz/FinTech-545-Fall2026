# Final Exam Instructions

**Be verbose.** Explain clearly your reasoning, methods, and results in your written work.  
No code is necessary, but including it in your answer could result in partial credit.

Written answers are worth the amounts stated.  
**Total points available:** 120 (includes 20 points of extra credit).

---

## General Guidelines

1. **Format:** Answers should be formatted as a PDF. You may convert your Python notebook, if you use one, directly to PDF.  Alternatively, use an 
editor like Word and print to PDF
2. **Restate the question:** Include the question number and restate each question before your answer.
3. **Submission:** When finished, upload your PDF to Canvas along with your code (optional).
4. **Repository:** Do not check code or answers into your repository until after the exam is completed by all.
5. **Data:** Data for problems are available in the repository named by the question number.
6. **Permitted resources:**
   - You may use your notes and the internet **for coding syntax help only.**
   - **You may not use an LLM** – turn off any coding helper you may have installed.
   - **You may not work with other students** – all work must be your own.
7. **Honor Code:** All students will be held to the **Duke Community Standard.**

<p style="margin:0; padding:0;">
<img src="dukecs1.jpeg" alt="Duke Community Standard" style="margin:0; padding:0; display:block;">
<img src="dukecs2.jpeg" alt="Duke Community Standard" style="margin:0; padding:0; display:block;">
</p>
<div style="page-break-after: always;"></div>

# Final Exam Questions

## 1. (50 pts)
**Using `problem1.csv`:**

**The data set contains simulated 1 day returns based on a model for returns.
You want to price options based on this assumption.  The series has been
adjusted so that it maintains the risk neutral assumption on price growth.**

- Current Stock Price - $100
- 1 Day to Maturity
- Annual Risk Free Rate - 4%
- 255 trading days per year
- The Stock pays no dividend

a. What is the price of a put and a call when the Strike ∈ [99,100,101] (15pts)

b. Graph the implied volatility of call option prices using BSM along the 
range of strikes between [95,105] (15pts)

c. Discuss:  Theory would say this should be a flat line.  Why is it not? (20pts)

---

## 2. (50 pts)
**Using `problem2.csv`:**

**The data set contains prices for the SPY. Further there are 2 options to be concerned with**

- Call Option 
  -   Strike=665
  -   Expiration=12/5/2025
  -   Price=7.05
- Put Option 
  -   Strike=655
  -   Expiration=12/5/2025
  -   Price=7.69
- Both are 10 days from expiration (don't worry about the holiday)

**Also assume**

- Div Rate = 1.09%
- rf = 4%
- 255 Trading Days per year
- European Options with continuous compounding on the dividends

a. Find the best fit model between a Normal and a T Distribution for the return of SPY.  Do not remove the mean.  Assume arthimetic returns. (5 pts)

b. Calculate the implied volatility of the Put and the Call (10 pts)

c. You own 1 put and are short 1 call and are long 1 share of the stock.
What is the VaR and ES (in absolute % terms, 5% alpha) of the portfolio holding 
the portfolio until 11/28/2025.  Assume no market holidays,
only weekends (5 trading days). (15 pts)

d. Portfolio Optimizaion (15 pts)

**Find the portfolio of these 3 assets that maximizes the ratio of (mean(simulated return) - rf(for the holding period))/ES(as a %).  Use the same 5 day holding period as above.**

- Assume you can hold fractional shares.  
- You may be short or long up to 2 units of each h ∈ [-2,2] 
- the total value of your portfolio should equal the current portfolio value

**HINTS**
- Current Portfolio Value = 659.67
- rf_holding = rf*5/255
- the ratio of the current portfolio is 
  - (mean(pnl) - rf_holding)/ES(pnl) ≈ 0.0698
- If you have convergence problems, try relaxing your tolerance to converge at 1e-5

e. Plot the portfolio value of your optimal holdings across the range of values from your simulation.  How would you judge the risk of this portfolio on a qualitative basis? (5 pts)

---

## 3. (20 pts)
**Using `problem3_insample.csv` and `problem3_outsample.csv`.**  

**These data are in sample and out of sample monthly returns for a set of stocks.**

a.  Calculate the Max SR portfolio and the Risk Parity Portfolio
using the insample data.  Risk Free Rate = 4%.  Use the insample
mean as the Expected Return.  Constrian your weights to be >=0.  Calculate your covariance using
exponentially weighted covariance (λ=0.97).  Scale both to be annual values. (12 pts)

**HINTS**
- Scale expected return as (1 + er)^12 - 1
- Scale the covariance as ewCovar * 12

b. Assume you hold these portfolios during the out of sample period. You do not rebalance during the period. Calculate the per stock attribution for both ex-post risk and return.  Compare these values to your ex-ante values. (8 pts)

**HINTS**
- Compare both the absolute contribution as well as the percent of total
- Portfolio volatility ex-post will be monthly so do not forget to annualize it.  This is invariant as a % which is why % helps.


---
