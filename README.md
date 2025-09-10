# This project analyzes 15,000+ structured video game sales records to predict top-selling games by region. Using Random forest, the model identifies which features (genre, platform, publisher, etc.) best predict regional success. The trained model is deployed on Heroku for interactive predictions.
- This repository contains all the files needed to build and run the Video Game Sales prediction app inside a Docker container, as well as push it to Heroku.
- The core components include Dockerfile, which defines the container image, and app.py, the Flask app that serves predictions.
- The trained Random Forest model is supported by label_encoder.pkl and model_columns.csv to ensure consistent feature encoding and input structure.
- The requirements.txt file lists the dependencies required for the container build. The templates/ folder contains HTML files for the user interface, and VGSales.ipynb provides the notebook with model analysis and training steps.
- An additional file, vgsales_model.py, contains helper functions for loading and using the trained model.
- Finally, multiple screenshots (app pushed to docker ss.PNG, docker cmd ss.PNG, docker local test ss.PNG, docker image to heroku.PNG, docker push to heroku.PNG) document each step of the containerization and deployment process, from local testing to pushing the Docker image to Heroku.
