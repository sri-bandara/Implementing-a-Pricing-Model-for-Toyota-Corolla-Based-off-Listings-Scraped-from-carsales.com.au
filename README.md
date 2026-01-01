# Toyota Corolla Price Estimator (AU) 🚙

Machine learning model trained on real listings from carsales.com.au that predicts price ranges and classifies Toyota Corolla listings, deployed as both a Gradio app on Hugging Face Spaces and a production-ready FastAPI service using Docker and AWS ECS.

👉🏽 **Live demo:** [PriceMyCorolla-AU on Hugging Face Spaces](https://huggingface.co/spaces/sri-bandara/PriceMyCorolla-AU)  

---

## Overview 📍

- **Input:** Listing features
- **Output:** Predicted price range and classification as below, around or above market price
- **Final Model:** Polynomial Regression (degree 4)
- **Final Performance:** ~1280 MAE
- **Deployment:** (1) Gradio, Hugging Face (2) FastAPI, Docker, AWS ECS

---

## Dataset 📊

The dataset used in this project was collected using a publicly available web scraper on [Apify](https://apify.com/memo23/carsales-cheerio) to extract Toyota Corolla listings from carsales.com.au. Rather than attempting to scrape every available listing, data collection was intentionally constrained to maintain sufficient sample density within each trim and configuration, as sparsely represented trims (e.g. only a few listings) introduce noise and make reliable learning difficult. To further control variance given the dataset size, the data was restricted to automatic transmission vehicles, hatchback and sedan body styles, selected high-volume badges (Ascent, Ascent Sport, Ascent Sport Hybrid, SX, SX Hybrid, ZR, ZR Hybrid), and model years between 2010 and 2025. Due to scraping limitations, approximately 1400 listings were collected in December 2025, providing a representative snapshot of the Australian Toyota Corolla market at that time.

---

## Model Selection 🔍

Model selection focused on balancing accuracy and generalization given the relatively small dataset (~1,400 listings) and the known nonlinear nature of vehicle pricing. A linear regression baseline was first evaluated, but its higher validation error (MAE ≈ 1385) indicated that it could not adequately capture nonlinear effects such as depreciation patterns with mileage or year-based price premiums. Polynomial regression was then tested across degrees 2–6 to explicitly model these nonlinear relationships. Validation performance improved up to degree 4, after which higher-degree models showed increasing validation error despite lower training error, indicating overfitting. Degree 4 achieved the best trade-off, with low validation MAE (~1300) and stable generalization across splits. Tree-based models (Random Forest and XGBoost) achieved much lower training error but showed larger train–validation gaps, particularly for Random Forest, suggesting overfitting. While XGBoost achieved slightly better test MAE, polynomial regression demonstrated more stable behavior and smoother extrapolation at feature extremes, making it a better fit for a user-facing price estimator.

---

## Assumptions and Limitations 🚧

The model assumes that pricing patterns observed in the training data remain representative of the broader Toyota Corolla market, and that key explanatory variables such as year, mileage, trim, and seller type sufficiently capture price variation. As the dataset represents a snapshot of listings scraped in December 2025, the model may be affected by data drift over time due to changes in market conditions. Additionally, the model is constrained to a subset of Corolla configurations and relies on a limited sample size, which may reduce generalization to less common trims. Predictions should therefore be interpreted as indicative estimates rather than precise valuations.

---

## Demo 🔮

<img width="965" height="839" alt="hf" src="https://github.com/user-attachments/assets/43aaa4f7-5deb-4821-880d-708233ead5c5" />
<img width="965" height="644" alt="aws" src="https://github.com/user-attachments/assets/aea437e9-a1be-4c6c-99dc-9ed179ad3ab7" />


