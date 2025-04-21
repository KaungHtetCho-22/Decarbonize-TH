# Decarbonize-TH

This project aims to forecast Thailand’s CO₂ emissions and analyze how renewable energy adoption (solar, wind, hydro) can mitigate emissions.


## 🌍 Feature Definitions for CO₂ Emission Prediction

| **Feature Name**                  | **Description** |
|----------------------------------|------------------|
| `year`                           | The calendar year of the observation (e.g., 1990, 2020). Acts as a time indicator to capture historical trends and policy impacts. |
| `population`                     | Total population of Thailand in the given year. Important for scaling emissions and understanding per capita energy demand. |
| `gdp`                            | Gross Domestic Product in constant 2011 international dollars (PPP). Measures the country’s economic activity and development. Economic growth often correlates with increased emissions. |
| `primary_energy_consumption`     | Total primary energy consumed in terawatt-hours (TWh), including fossil fuels, renewables, and nuclear. Closely tied to industrialization and emissions. |
| `oil_co2`                        | CO₂ emissions from oil combustion in million tonnes. Oil is a major energy source, especially in transport and industry. |
| `coal_co2`                       | CO₂ emissions from coal combustion in million tonnes. Coal is a high-emission fossil fuel commonly used in electricity generation. |
| `cement_co2`                     | CO₂ emissions from cement production in million tonnes. Cement emits CO₂ from both fossil fuel use and chemical processes (calcination). |
| `total_ghg`                      | Total greenhouse gas emissions (in million tonnes of CO₂-equivalents), including CO₂, CH₄ (methane), N₂O (nitrous oxide), etc. Captures the full climate impact. |
| `temperature_change_from_ghg`    | Estimated temperature change in °C attributable to all GHGs. Indicates the long-term environmental impact of cumulative emissions. |


## 🧼 Why We Use Forward and Backward Fill (ffill + bfill)

### 📌 Context
Our dataset consists of **yearly, country-level data** for features like:

- `population`, `gdp`, `coal_co2`, `cement_co2`, `total_ghg`, etc.

These features are:
- **Time-series based** (change over time)
- Often **incomplete** in early or scattered years
- Typically **smooth** in progression (not erratic)

---

### 🛠 What is Forward Fill & Backward Fill?

| Method        | Description |
|---------------|-------------|
| `ffill`       | Fills missing value with the **last known non-null** value from previous year |
| `bfill`       | Fills missing value with the **next known non-null** value from future year |

```python
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
```
## 🌏 Justification for Data Split Strategy: Generalization to Thailand

### ✅ Objective
To predict CO₂ emissions for **Thailand** by training on:
- **Training set**: Global countries + selected ASEAN countries (excluding Thailand)
- **Validation set**: Remaining ASEAN countries (excluding Thailand)
- **Test set**: Thailand only

---

### 🧠 Why This Strategy Works

#### 1. **Tests Real-World Generalization**
This setup evaluates whether a model trained on the world (excluding Thailand) can generalize to an **unseen target country**. It simulates deployment where we don’t have training data for the country of interest.

#### 2. **Avoids Data Leakage**
Thailand is completely excluded from training and validation. This ensures a **clean, unbiased evaluation**.

#### 3. **Respects Regional Dynamics**
Using other ASEAN countries for validation leverages **regionally similar patterns** (e.g., climate, energy use) while still keeping Thailand out.

#### 4. **Supports Transfer Learning**
If the model performs well on Thailand, it shows that **shared global/ASEAN dynamics** are sufficient to generalize. If not, it may signal the need for **domain-specific tuning**.

---

### 📊 Dataset Split Summary

| Set        | Includes                          | Purpose                              |
|------------|-----------------------------------|--------------------------------------|
| **Train**  | World + ASEAN (excluding Thailand) | Learn global and regional patterns   |
| **Val**    | ASEAN (excluding Thailand)         | Tune hyperparameters regionally      |
| **Test**   | Thailand only                      | Evaluate model generalization        |

---

### 🧑‍🏫 Questions You Might Be Asked (and How to Answer)

> **Q1: Why exclude Thailand from training?**  
> To assess the model’s ability to generalize to a completely **unseen target**. It simulates a real-world scenario where we deploy the model on a new country.

> **Q2: Isn’t that unfair if Thailand is different?**  
> That’s the point — we’re testing whether the model can **adapt to outliers or domain shifts**. If performance drops, it reveals that Thailand may require **custom tuning**.

> **Q3: Why validate on other ASEAN countries?**  
> ASEAN countries share **geographic and economic similarities** with Thailand. They provide useful validation feedback while preserving Thailand’s isolation.

> **Q4: What if performance is poor on Thailand?**  
> I’d explore domain shift (e.g., different economic or energy structure), and possibly fine-tune with **limited Thailand data** or explore **domain adaptation** techniques.

> **Q5: What does good performance mean?**  
> It confirms that CO₂ emission patterns learned from global and regional data are **general enough** to apply to Thailand. This supports **cross-country generalizability**.

---
