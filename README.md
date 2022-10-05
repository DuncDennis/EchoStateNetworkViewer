# Reservoir Computing for Time Series Prediction - Streamlit app
A streamlit app to explore Reservoir Computing for time series predictions. 

# App: 
Link to app: https://duncdennis-echostatenetworkviewer-esn-app-main-upload-an-dsrup8.streamlitapp.com/

![image](https://user-images.githubusercontent.com/90915296/194052232-d7eda94e-0185-4abe-a5b8-b3fc80088911.png)


# Features: 

**Create raw data:**
- Either upload your own, or simulate data from a selection of dynamical systems.
- If you choose to simulate data from a dynamical system:
  - Choose from 18 dynamical systems as for example: *Lorenz63*, *Roessler*, 
    *KuramotoSivashinsky*.
  - Adjust the parameters of the dynamical system
  - View the system equations. 
  - View and measure the raw data. 

**Preprocess the raw data:**
  - Perform time delay embedding.
  - Shift and scale the raw data.
  - Add noise.
  - View and measure the preprocessed data.

**Split the preprocessed data into train and predict sections:**
- Choose which parts of the preprocessed data is used for training and for testing 
   (i.e. prediction).
- View the data split.

**Build the reservoir computing setup:**
- Adjust all the Reservoir Computing hyperparameters like *reservoir dimension*, 
  *spectral radius* and more.
- View some RC parameters, like the *Network* or *Input Matrix*

**Train the reservoir:**
- Perform the training.
- View the quality of the training fit. 

**Predict with the reservoir:**
- Predict the prediction-section of the preprocessed data using the trained reservoir. 
- View and measure the predicted vs. the real data. 

**Turn on advanced features:**
- Turn on advanced features by checking a checkbox. 
- Advanced features include:
  - More advanced reservoir computing options that allow for an additional processing 
  layer between the reservoir states and the output fit. 
  - Additional option to "look-under-hood" of the reservoir. View internal reservoir 
  states.

# Note: 
Everything is still beta at the moment.
Things that are not yet implemented: 
- Proper typing for all python files. 
- Documentation and tutorial on how to use. 
- Proper requirements.
- Reformat and further clean up of code. 


# Related repository: 
- [https://github.com/GLSRC/rescomp](https://github.com/GLSRC/rescomp)
