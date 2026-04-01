# 📊 IT Sector Forecasting in Bangladesh

## 🚀 Overview
This project presents a data-driven approach to forecasting ICT export growth in Bangladesh using time-series models. The study compares traditional statistical modeling with modern forecasting techniques to evaluate accuracy, stability, and real-world applicability.

The objective is to provide reliable forecasting insights to support policy planning, investment decisions, and long-term economic strategy.

---

## 🌟 Project Highlights
- Built using **Python + Jupyter Notebook (Anaconda)**  
- Forecasted ICT export growth for **2025–2030**  
- Compared **SARIMA vs Facebook Prophet**  
- Evaluated using **MAPE, RMSE, and R²**  
- Designed as an **applied economic forecasting system**  

---

## 🎯 Problem Statement
Forecasting ICT sector growth in emerging economies like Bangladesh is challenging due to:
- Limited and inconsistent datasets  
- Non-linear economic trends  
- Lack of comparative model evaluation  

This project addresses these challenges by evaluating model performance under real-world data constraints.

---

## 🧠 Models Used
- **SARIMA (Seasonal ARIMA)**  
  - Captures linear trends and seasonality  
  - Suitable for structured time-series data  

- **Facebook Prophet**  
  - Handles non-linear trends and missing data  
  - Automatically detects trend changes  

---

## 📂 Dataset
Data collected from:
- World Bank  
- Bangladesh Bank  
- BASIS  

Includes:
- ICT export revenue  
- GDP  
- Remittance  
- Import/Export indicators  

---

## ⚙️ Project Structure
it-sector-forecasting-bd/
│
├── data/ # Dataset (CSV)
├── code/ # Python scripts (SARIMA, Prophet)
├── paper/ # Thesis document
├── notebooks/ # Jupyter notebooks (coming soon)
└── README.md


---

## 📈 Results Summary
| Model   | MAPE   | RMSE           | R²   |
|--------|--------|----------------|------|
| SARIMA | 17.47% | 110,524,324.65 | -0.01 |
| Prophet| 15.59% | 102,740,903.03 | 0.13  |

### Key Insights:
- Prophet outperforms SARIMA in accuracy and trend capture  
- SARIMA struggles with non-linear economic patterns  
- Prophet provides more stable long-term forecasts  

---

## ▶️ Environment
This project was developed using **Python in Jupyter Notebook via Anaconda**.

---

## ▶️ How to Run

### Option 1: Jupyter Notebook (Recommended)
1. Open **Anaconda Navigator**
2. Launch **Jupyter Notebook**
3. Open the project folder
4. Run the notebook or scripts step by step

### Option 2: Python Scripts
```bash
pip install pandas numpy matplotlib statsmodels prophet scikit-learn
python code/sarima.py
python code/prophet.py
```

---

📊 Output
The scripts generate:
Forecasted ICT export values (2025–2030)
Accuracy metrics (MAPE, RMSE, R²)
Visualization plots with confidence intervals

🔍 Key Contribution
Developed a data-constrained forecasting framework for emerging economies
Compared SARIMA and Prophet in terms of accuracy, stability, and reliability
Provided practical forecasting insights for Bangladesh’s ICT sector

⚠️ Limitations
Limited dataset size
External economic factors not included
No deep learning models (e.g., LSTM) used

🔮 Future Work
Hybrid models (SARIMA + Prophet)
Deep learning approaches (LSTM, Transformers)
Inclusion of external economic indicators

📒 Notebook (Coming Soon)
A Jupyter Notebook version of the full workflow (EDA → SARIMA → Prophet → Evaluation) will be added.

🛠 Repository Status
Core components (dataset, code, and paper) are available.
Notebook workflow and additional visualizations will be added soon.

👤 Author
Mir Shoaib Ahmed
CSE Graduate, AIUB
Focus: Backend Development, Data Systems, Forecasting, Applied Analytics

Md. Tahmid Hasan
CSE Graduate, AIUB

📄 License
MIT License
