# Toyota Corolla Price Estimator (AU) 🚙

Machine learning model trained on real listings from carsales.com.au that predicts price ranges and classifies Toyota Corolla listings, deployed as both a Gradio app on Hugging Face Spaces and a production-ready FastAPI service using Docker and AWS ECS.

---

## Model Overview 📍

- **Input:** Listing features
- **Output:** Predicted price range and classification as below, around or above market price
- **Final Model:** Polynomial Regression (degree 4)
- **Final Performance:** ~1280 MAE
- **Deployment:** (1) Gradio, Hugging Face (2) FastAPI, Docker, AWS ECS

---

## Dataset 📊

The dataset used in this project was collected using a publicly available web scraper on [Apify](https://console.apify.com/actors/Ljb5XXiCmjlpW16MB/input) to extract Toyota Corolla listings from carsales.com.au. Rather than attempting to scrape every available listing, data collection was intentionally constrained to maintain sufficient sample density within each trim and configuration, as sparsely represented trims (e.g. only a few listings) introduce noise and make reliable learning difficult. To further control variance given the dataset size, the data was restricted to automatic transmission vehicles, hatchback and sedan body styles, selected high-volume badges (Ascent, Ascent Sport, Ascent Sport Hybrid, SX, SX Hybrid, ZR, ZR Hybrid), and model years between 2010 and 2025. Due to scraping limitations, approximately 1,320 listings were collected in December 2025, providing a representative snapshot of the Australian Toyota Corolla market at that time.

---

## Process 🧹

The project followed an end-to-end machine learning workflow, starting with data cleaning and feature engineering on scraped vehicle listings, followed by iterative model selection and evaluation using MAE and RMSE. Multiple regression models were explored, including linear regression, polynomial regression, random forests, and gradient-boosted trees. The final model was packaged into a production-ready inference pipeline and deployed in two ways: as an interactive Gradio application on Hugging Face Spaces, and as a containerized FastAPI service using Docker and AWS ECS.

---

## Assumptions and Limitations 🚧

The model assumes that pricing patterns observed in the training data remain representative of the broader Toyota Corolla market, and that key explanatory variables such as year, mileage, trim, and seller type sufficiently capture price variation. As the dataset represents a snapshot of listings scraped in December 2025, the model may be affected by data drift over time due to changes in market conditions. Additionally, the model is constrained to a subset of Corolla configurations and relies on a limited sample size (~1,320 listings), which may reduce generalization to less common trims. Predictions should therefore be interpreted as indicative estimates rather than precise valuations.
