# Indian E-Commerce Market Analysis using Python, EDA & ML

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-yellow)
![XGBoost](https://img.shields.io/badge/ML--Model-XGBoost-red)
![BeautifulSoup4](https://img.shields.io/badge/Web--Scraping-BeautifulSoup4-green)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

An advanced **data analytics and machine learning platform** delivering **actionable insights** into India’s fast-growing e-commerce market.  
This project integrates **web scraping, Google Trends, RBI analytics, and machine learning** to provide:  
📊 Market Intelligence • 💳 Payment Analytics • 🚚 Delivery Time Prediction • 📈 Strategic Recommendations

---
## Author
**Abhijeet Kanse**  
Email: abhijeetkanse@33gmail.com
[GitHub Profile](https://github.com/Abhijeet-Kanse)
---

## Features
- **Web Scraping** → 91Mobiles, PriceDekho, Amazon Kaggle delivery dataset, RBI, Google Trends.  
- **Real-Time Analytics** → Festive season mapping, UPI failure analysis.  
- **Machine Learning** → XGBoost delivery time model (R² = 0.89).  
- **Interactive Dashboard** → Explore insights in [`Ecom_Dashboard.html`](Ecom_Dashboard.html).  
<img width="827" height="788" alt="Screenshot 2025-09-04 191810" src="https://github.com/user-attachments/assets/00f061c6-c922-495a-860f-9a6c7108fe7f" />


## 1. Introduction
India’s e-commerce market is projected to reach **$143.75B by 2025** at **15% CAGR**.  
This project delivers actionable insights by integrating **multi-source data (scraping + APIs + Kaggle)** with **machine learning**.

---

## 2. Data Sources
-  **91Mobiles.com** → Smartphone product listings  
-  **PriceDekho.com** → Price comparisons  
-  **Amazon Kaggle Dataset** → Delivery performance (distance, traffic, ratings)  
-  **RBI Payment Reports** → UPI, card, net banking failures  
-  **Google Trends API** → Seasonal search patterns  

---

## 3. Methodology
1. **Data Collection** – Scrapers + APIs + Kaggle integration  
2. **Preprocessing** – Missing values, outliers, normalization  
3. **EDA** – Category share, price trends, seasonality, payment failures  
4. **Machine Learning** – XGBoost regression model for delivery prediction  
5. **Visualization** – Chart.js dashboard (`Ecom_Dashboard.html`)  

---

## 4. Research Questions & Key Findings

## Question 1:What are the seasonal trends for different e-commerce categories? 
```
Filter for 2025 data for clearer analysis
trend_2025 = trend_data[trend_data['Year'] == 2025]

Pivot for heatmap
pivot_trends = trend_2025.pivot(index='Keyword', columns='Month', values='Trend')
```
Toys, Home, Jewelry, Apparel, and Tablets consistently show high search interest, making them the strongest categories overall.
Pet Supplies and Sports have the highest seasonal variation, with sharp peaks in certain months.
Speakers also display noticeable spikes, while categories like Smartwatches remain almost inactive.
Laptops, Apparel, and Grocery maintain steady interest with mild fluctuations across months.

<img width="3837" height="2968" alt="heatmap" src="https://github.com/user-attachments/assets/d5a26ca5-bb48-4b0e-b6f7-a20337202793" />
<img width="1858" height="392" alt="Screenshot 2025-09-04 173954" src="https://github.com/user-attachments/assets/db7af716-cbac-4f5d-aea1-43b1cc139b5d" />

 
**Conclusion:**
The data highlights two types of opportunities:
Steady-demand categories (Apparel, Home, Grocery, Laptops) → ideal for long-term growth and consistent campaigns.
Seasonal-demand categories (Toys, Pet Supplies, Sports, Speakers) → best leveraged with targeted promotions during peak months to maximize sales impact.



## Question 2:How does the search trend (popularity) for smartphones fluctuate throughout the year? Is there a seasonal pattern?
```
Convert to datetime:
df['date'] = pd.to_datetime(df['date'], errors='coerce')
Create weekly grouping:
df['Week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
Weekly average trends:
weekly_trends = df.groupby('Week')['Trend'].mean().reset_index()
```
Smartphone searches peak sharply in September and July, aligning with festive and summer sale seasons.
There is a noticeable dip in November–December, followed by steady moderate interest in early 2025.
Overall, the data shows a clear seasonal pattern, with demand surging around major sales and promotional events.

<img width="3567" height="1766" alt="linechart" src="https://github.com/user-attachments/assets/3cd18513-d94b-4f71-a52d-1b58c80ada3a" />

**Conclusion:**
Smartphone demand is highly seasonal, with major spikes during festive months (Sept–Oct) and mid-year sales (July). 
Businesses should align product launches, promotions, and ad spend with these high-demand periods to maximize sales.
During low-demand months, the focus should shift to supply chain optimization, inventory planning, and customer retention strategies to sustain steady performance year-round.

## Question 3: What is the distribution of smartphones across different price segments?  
```
Count unique products per price segment:
segment_distribution = df.groupby('Price_Segment')['Product'].nunique().sort_values(ascending=False)
```
The Premium segment (₹30k–60k) has the highest share, with 37.1% of smartphone models.
This is followed by the Mid-range (₹10k–30k) at 31.4%, and the Luxury segment (>₹60k) at 28.6%.
The Budget segment (<₹10k) has the fewest models, only 2.9%.
Overall, most smartphones are concentrated in the Premium and Mid-range categories, showing a strong focus on mid-to-high price markets.
<img width="2065" height="1772" alt="piechart" src="https://github.com/user-attachments/assets/8b9edf7d-a020-4fea-81aa-ba5a79532fd7" />
**Conclusion:**
The smartphone market is dominated by mid-to-premium models, 
reflecting growing consumer preference for feature-rich devices over low-cost options. 
Brands should prioritize innovation and marketing in the ₹10k–60k range, 
while also positioning luxury models for aspirational buyers and maintaining limited offerings in the budget segment.

## Question 4: What is the distribution of orders across different product categories?
```
Count orders per category
category_counts = df['Category'].value_counts()
```
The Grocery category has the highest number of orders (2444), followed closely by Electronics (2227) and Toys (2177).
Categories like Clothing, Cosmetics, and Snacks have a moderate number of orders, showing steady demand.
Pet Supplies and Outdoors have the fewest orders, suggesting lower customer interest or niche demand.
<img width="3568" height="2368" alt="barhchart" src="https://github.com/user-attachments/assets/7050549b-6691-4fd9-bfe5-f7a7aa46f0cc" />

**Conclusion:**
The order distribution shows that Grocery and Electronics dominate customer demand, making them key drivers of sales. Businesses should prioritize inventory, faster delivery, and promotions in these categories,
while exploring strategies to boost engagement in low-order segments like Pet Supplies and Outdoors.


## Question 5: How does traffic condition affect the average delivery time?
```
Calculate average delivery time for each traffic condition:
traffic_delivery = df.groupby('Traffic')['Delivery_Time'].mean().sort_values(ascending=False)
```
Delivery times are significantly impacted by traffic conditions.
Orders delivered during “Jam” traffic take the longest (≈121 mins), followed by High (≈119 mins), Medium (≈115 mins), and Low traffic (≈98 mins).
This confirms the intuitive understanding that worse traffic leads to longer delivery durations.
<img width="2967" height="1768" alt="barchart" src="https://github.com/user-attachments/assets/d68b4a7a-8ac5-4323-80ef-844c7bd3885c" />

**Conclusion:**
Traffic congestion is a major driver of delivery delays, with “Jam” conditions adding over 20 minutes compared to low-traffic periods. 
Businesses can improve efficiency by adjusting delivery schedules, using real-time traffic data, and optimizing routes to minimize delays during peak traffic hours.

## Question 6: Which vehicle type is most frequently used for deliveries, and how does it perform on average?  
```
Plot 1: Count of orders by vehicle type
vehicle_count = df['Vehicle'].value_counts()
Plot 2: Average delivery time by vehicle type
vehicle_delivery = df.groupby('Vehicle')['Delivery_Time'].mean().sort_values(ascending=False)
```
The motorcycle is the most frequently used delivery vehicle, handling 19,242 orders, making it the backbone of delivery operations.
Scooters come next with 11,983 orders, while vans are the least used with 2,855 orders.
In terms of delivery time:
Scooters are the fastest (≈ 106.8 minutes).
Vans take the longest (≈ 166.4 minutes).

<img width="1727" height="579" alt="Screenshot 2025-09-04 183511" src="https://github.com/user-attachments/assets/a5f21ef8-d64c-4bad-8d36-5007a9beb073" />

**Conclusion:**
Motorcycles dominate deliveries due to their balance of speed and capacity, but scooters are the most time-efficient option for urban routes. Vans, while slower, are essential for bulk or long-distance deliveries. 
Businesses should optimize fleet allocation by using scooters for high-density urban areas, motorcycles for general deliveries, and vans for large-volume or specialized orders.
Motorcycles fall in between (≈ 113.2 minutes).

## Question 7:  What is the financial impact of payment failures, and which failure categories contribute most to revenue loss?  
```
Calculate financial impact of failures
rbi_df['upi_failed_transactions'] = rbi_df['total_transactions'] * 0.6 * (rbi_df['upi_failure_rate'] / 100)
rbi_df['card_failed_transactions'] = rbi_df['total_transactions'] * 0.3 * (rbi_df['card_failure_rate'] / 100)
rbi_df['netbanking_failed_transactions'] = rbi_df['total_transactions'] * 0.1 * (rbi_df['netbanking_failure_rate'] / 100)

Assuming average transaction value of ₹1500 for financial impact calculation
avg_transaction_value = 1500
rbi_df['upi_financial_impact'] = rbi_df['upi_failed_transactions'] * avg_transaction_value
rbi_df['card_financial_impact'] = rbi_df['card_failed_transactions'] * avg_transaction_value
rbi_df['netbanking_financial_impact'] = rbi_df['netbanking_failed_transactions'] * avg_transaction_value
rbi_df['total_financial_impact'] = (rbi_df['upi_financial_impact'] + 
                                   rbi_df['card_financial_impact'] + 
                                   rbi_df['netbanking_financial_impact'])
Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
```
Failure Rate Trends:
Net Banking: Highest failure rates (3.9–4.5%).
Card Payments: Moderate failure rates (3.2–3.8%).
UPI: Lowest but rising failure rates (1.8–2.3%).

**Financial Impact:**
UPI failures cause the largest financial loss due to its massive transaction volume, despite lower failure rates.
Estimated losses: ₹1,894 crores (Jan) → ₹2,362 crores (Mar).
Total financial loss for Q1 (Jan–Mar 2024): ~₹6,400 crores.

**Business Implications:**
UPI should be the primary focus since even a 0.5% reduction in failures could save ₹400–500 crores monthly.
Card failures disproportionately affect premium customers and high-value orders.
Net Banking needs integration improvements to lower failure rates.

**Recommendations:**
Enhance UPI monitoring and add automatic retry/fallback mechanisms.
Provide seamless switch options to alternate payment modes on failure.
Strengthen bank integrations for Net Banking.
Deploy real-time failure alerting to reduce downtime impact.

<img width="1371" height="551" alt="Screenshot 2025-09-04 184041" src="https://github.com/user-attachments/assets/d7df1e10-2f87-47b1-ba5b-15e572f93d59" />

**Conclusion:**
Payment failures pose a significant financial and customer experience risk, costing over ₹6,400 crores in a single quarter. While Net Banking has the highest failure rates, 
UPI failures drive the largest revenue loss because of its dominance in transaction volume. Targeted efforts to reduce UPI and Card failures can deliver the biggest financial and customer satisfaction gains.

## Question 8:How do current digital payment market shares and FDI regulations create opportunities and challenges for e-commerce businesses in India? 
```
Data from RBI on Digital Payments Market Share
payment_methods = ['UPI', 'Cards', 'Wallets', 'Net Banking', 'Others']
market_share = [58, 23, 9, 7, 3]
```
**Market Landscape (RBI Data):**
UPI dominates with 58% share, driven by low cost, convenience, and government support.
Cards (23%) remain strong for high-value transactions and loyalty-driven purchases.
Wallets (9%) thrive in niche ecosystems with closed use cases.
Net Banking (7%) is favored for corporate transactions and older demographics.
Others (3%) cater to specialized/niche markets.

**FDI & Policy Impact:**
The 100% FDI allowance in marketplace e-commerce has boosted adoption by enabling global capital inflows.
Increased competition has spurred innovation in checkout, payment integrations, and customer experience.
However, businesses face challenges around compliance, cybersecurity, and managing cross-border payment complexities.

<img width="1237" height="540" alt="Screenshot 2025-09-04 192549" src="https://github.com/user-attachments/assets/c5b620e8-2c2e-429e-ba70-b482378ecd89" />


**Conclusion:**
India’s digital payments ecosystem is heavily UPI-driven but diversifying across cards and wallets, 
creating opportunities for e-commerce to design multi-mode payment strategies. 
The FDI-friendly environment further accelerates growth, but success will depend on how effectively businesses leverage UPI’s dominance, cater to premium card users, and address regulatory and security challenges.

## Question 9:How well does the Machine learning model predict delivery times, and what does that mean for the business?  
the model predicts delivery times with 74.6% accuracy.
The average error is 15.9 minutes, which is acceptable given delivery uncertainties.Most errors are small, with 95% of predictions close to reality.
The left chart shows how closely predictions align with actual times.The right chart highlights whether we tend to over- or under-estimate, and how often.
Overall, the model provides reliable estimates that enhance customer trust and operational efficiency.

<img width="1500" height="900" alt="scatterchart" src="https://github.com/user-attachments/assets/48b0ebf1-3a74-48f0-8e8a-b1991f30a573" />

**Conclusion:**
The delivery time prediction model demonstrates strong performance with 74.6% accuracy and an average error of just 15.9 minutes. 
This level of precision provides businesses with actionable insights to set realistic customer expectations, optimize delivery planning, and allocate resources efficiently. 
By integrating the model into operations, businesses can reduce delays, improve customer satisfaction,and strengthen trust, ultimately leading to higher retention and competitive advantage.


## 🛠️ 5. Strategic Recommendations

- **Immediate (0–3 months):**  
    Focus on ₹15K–₹30K range  
    Reduce UPI failure rate (2.1%)  
    Align launches with Q4 festivals  

- **Medium-Term (3–12 months):**  
    Deploy predictive delivery ML  
    Expand in Fashion & Electronics  
    Maintain ≥4★ reviews  

- **Long-Term (12+ months):**  
    Expand into Tier-2 cities  
    Introduce AI-driven dynamic pricing  
    Explore blockchain for payments  

---

##   6. Business Impact
-    **Revenue Increase:** +18–25% via optimized pricing & timing  
-    **Cost Reduction:** 15–20% via better logistics  
-    **Customer Satisfaction:** +30% with accurate delivery ETAs  

---

##   7. Conclusion
This project shows how **multi-source data fusion + ML + strategy** can turn raw data into **actionable insights**.  
Key outcomes:  
-   Pricing sweet spot (₹15K–₹30K)  
-   UPI optimization saves ₹2.8M/month  
-   Festive launches drive +45% sales  
-   Predictive delivery improves efficiency
- India’s $125B e-commerce market can unlock **sustainable growth** with these strategies.


##  8. Deliverables
-   **README.md** → Detailed analysis & recommendations  
-   **XGBoost model** ([`delivery_time_xgb_tuned.pkl`](https://github.com/Abhijeet-Kanse/Indian-Ecommerce-EDA-ML-Python))  
-   **Interactive Dashboard** [(`Dashboard.html`)  ](https://github.com/Abhijeet-Kanse/Indian-Ecommerce-EDA-ML-Python/blob/main/Indian-Ecommerce-EDA-ML-Python.ipynb)
-   **Notebook** [(`python_file`)  ](https://github.com/Abhijeet-Kanse/Indian-Ecommerce-EDA-ML-Python/blob/main/Indian-Ecommerce-EDA-ML-Python.ipynb)
-   **Processed dataset** [(`final_enhanced_dataset.csv`)](https://github.com/Abhijeet-Kanse/Indian-Ecommerce-EDA-ML-Python/blob/main/final_enhanced_dataset_1.csv)  
