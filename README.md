# Global Carbon Monoxide (CO) Emissions Analysis (1970-2035) 📊

This repository contains a comprehensive analysis of global Carbon Monoxide (CO) emissions from 1970 to 2022, including future forecasts up to 2035. The analysis uses the EDGAR (Emissions Database for Global Atmospheric Research) dataset to explore temporal trends, geographic distribution, and country-level patterns, with a special focus on East Africa.

## Table of Contents
- [Key Insights & TL;DR](#key-insights--tldr)
- [Installation & Usage](#installation--usage)
- [Analysis Structure](#analysis-structure)
- [Detailed Analysis Findings](#detailed-analysis-findings)
  - [1. Global Emission Trends (1970-2022)](#1-global-emission-trends-1970-2022)
  - [2. Geographic Distribution](#2-geographic-distribution)
  - [3. East Africa Regional Analysis](#3-east-africa-regional-analysis)
  - [4. Growth Rate Analysis](#4-growth-rate-analysis)
  - [5. Future Forecast (2023-2035)](#5-future-forecast-2023-2035)
- [Project Files](#project-files)

---

## Key Insights & TL;DR

*   **Complex Global Trend**: Global CO emissions do not follow a simple linear path. They rose until ~1990, dipped significantly during the 90s (the "Regulatory Dip"), surged again with globalization until the mid-2000s, and have since entered a plateau with a slight downward trend.
*   **Highly Concentrated Emissions**: A small number of countries are responsible for the vast majority of emissions. The **top 10 emitters**—led by China, the United States, and India—account for over **60%** of all historical CO emissions.
*   **The 90s Dip Explained**: The significant drop in emissions from 1990-2000 is largely attributed to two major global events:
    1.  **Economic Collapse**: The dissolution of the Soviet Union led to the collapse of many of its inefficient, high-polluting heavy industries.
    2.  **Environmental Regulation**: Stricter regulations in Western countries, particularly the **U.S. Clean Air Act Amendments of 1990**, mandated the use of catalytic converters in vehicles, which drastically cut CO output.
*   **The Post-2000 Surge**: The sharp rise in emissions after 2000 was primarily driven by the explosive industrialization and economic growth of China and other developing nations.
*   **East Africa's Position**: While currently low on a global scale, East African nations show diverse trends. **Tanzania and Ethiopia** are the largest historical emitters in the region. Kenya's emissions have been relatively stable but show a slight upward trend in recent years.
*   **Future Outlook**: The Prophet forecasting model, validated with high accuracy (MAPE < 5%), predicts a **slight but steady increase** in global CO emissions towards 2035. However, the confidence interval is wide, indicating that this trend is not guaranteed and depends on continued technological improvements and policy implementation globally.

---

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/carbon-monoxide-analysis.git
    cd "Carbon Monoxide emissions Analysis"
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the analysis:**
    Open `Carbon Monoxide emissions Data Analysis.py` in a compatible environment like VS Code with the Jupyter extension or a Jupyter Notebook and run the cells.
    The script will generate a full `CO_Emissions_Full_Report.html` file in the root directory.

---

## Analysis Structure

The analysis is structured into 15 progressive "Areas," each building upon the previous one to provide deep insights into emission patterns, trends, and future projections.

1.  **Data Loading & Validation**: Health check of the dataset.
2.  **Temporal Trend Analysis**: The big picture of global trends over time.
3.  **Geographic Distribution**: Pinpointing the biggest emitters with charts and an interactive map.
4.  **Comparative Analysis**: A snapshot of where countries stood in the most recent year.
5.  **Growth Rate Analysis**: Identifying the fastest-growing and fastest-declining emitters.
6.  **Time Series for Top Countries**: Tracking the individual journeys of top emitters.
7.  **Correlation Analysis**: Exploring if top emitters have similar emission patterns.
8.  **Clustering Analysis**: Grouping countries into "emission tribes" using K-Means.
9.  **Anomaly Detection**: Spotting unusual spikes or drops in emissions.
10. **Future Predictions (Prophet & ARIMA)**: Forecasting future trends with robust time-series models.
11. **Prophet Model Validation**: Backtesting the forecast model to confirm its accuracy.
12. **Country-Level Forecasting**: Generating specific forecasts for top countries.
13. **East Africa Regional Analysis**: A deep dive into a specific region of interest.
14. **Random Forest Prediction**: An alternative ML model for forecast comparison.
15. **HTML Report Generation**: Compiling all text, tables, and charts into a single, shareable report.
---

## Detailed Analysis Findings

### 1. Global Emission Trends (1970-2022)

The analysis of global emissions over 52 years reveals a multi-stage history:
*   **1970 - 1990 (The Industrial Rise):** A period of steady growth in emissions, fueled by widespread industrial activity in developed nations and increasing vehicle usage without modern emission controls.
*   **1990 - 2000 (The Regulatory Dip):** A notable decline occurred due to the collapse of Soviet-era industry and the successful implementation of environmental policies (like the catalytic converter) in North America and Europe.
*   **2000 - 2010 (The Globalization Surge):** A sharp increase driven by the rapid industrialization of Asia, particularly China, which became a global manufacturing hub.
*   **2010 - 2022 (The Plateau):** Emissions have stabilized and begun a slight downward trend, likely due to stricter emission standards being adopted globally (including in China), technological advancements, and a shift away from heavy industry in some major economies.

### 2. Geographic Distribution

CO emissions are not evenly distributed. The analysis confirms a high concentration among a few key players.

*   **Top Emitters**: China, the United States, India, Brazil, and Indonesia are the top 5 countries with the highest all-time emissions.
*   **Pareto Principle**: The top 20 countries are responsible for over 80% of all emissions, highlighting a significant global disparity.
*   **Interactive Map**: The file `emitters_distribution_map.html` provides an interactive world map visualizing these concentrations, with countries color-coded by emission levels.

### 3. East Africa Regional Analysis

A focused analysis on Kenya and its neighbors provides regional context:

*   **Regional Leader**: The **United Republic of Tanzania** is the largest historical emitter in the selected East African bloc, followed by **Ethiopia**.
*   **Kenya's Profile**: Kenya ranks in the middle of the pack within the region. Its emission trend has been relatively flat for decades but has started to show a consistent, gradual increase since the mid-2000s.
*   **Growth Rates**: The region shows mixed growth. While some nations have stable or declining emissions, others are on an upward trajectory, reflecting different stages of economic development and industrial activity.

### 4. Growth Rate Analysis

The Compound Annual Growth Rate (CAGR) reveals which countries are changing the fastest.

*   **Fastest Growing**: Countries experiencing rapid industrialization or economic shifts show the highest growth rates in emissions.
*   **Fastest Declining**: Many European nations, along with countries that have undergone significant economic transition (e.g., Ukraine), show the fastest declines, indicating successful policy implementation or de-industrialization.

### 5. Future Forecast (2023-2035)

Using the highly accurate Facebook Prophet model, the analysis projects future global trends.

*   **Model Validation**: The model was first backtested on historical data (2012-2022) and achieved a Mean Absolute Percentage Error (MAPE) of less than 5%, confirming its reliability for this dataset.
*   **Projected Trend**: The forecast indicates a continued, slow **increase** in global CO emissions towards 2035.
*   **Uncertainty**: The forecast includes a 95% confidence interval. While the median forecast shows an increase, the lower bound of the confidence interval suggests a decrease is still possible, underscoring the importance of sustained global efforts to control pollution.
*   **Country-Level Forecasts**: Individual forecasts for the top 10 emitting countries have been generated and saved in `country_level_forecasts_2023-2035.csv`.

---

## Project Files

*   `Carbon Monoxide emissions Data Analysis.py`: The main Python script (designed for Jupyter) containing all 15 areas of analysis.
*   `Carbon Monoxide emissions.csv`: The raw input dataset.
*   `country_coordinates.csv`: Helper file providing geographic coordinates for map generation.
*   `requirements.txt`: A list of all Python libraries required to run the analysis.
*   `CO_Emissions_Full_Report.html`: The final, self-contained HTML report with all findings and visualizations.
*   `emitters_distribution_map.html`: An interactive map showing the geographic distribution of total emissions.
*   `country_level_forecasts_2023-2035.csv`: A CSV file containing the emission forecasts for the top 10 countries.
*   `README.md`: This summary file.