# EchoStateNetworkViewer
A streamlit app to explore Echo State Networks.

# App: 
Link to app: https://duncdennis-echostatenetworkviewer-esn-app-0bm3qk.streamlitapp.com/

![img](https://user-images.githubusercontent.com/90915296/185076401-e3cd836b-b583-4eec-8322-2c71061fe7f8.png)

# Features: 
- Simulate a timeseries: 
  - Choose from 20 dynamical systems.
  - Adjust the system parameters.
  - View the system equation. 
  - Optional preprocessing: Center + shift and add noise.
  - Define how many time-steps are used for training and prediction.  
  - Calculate the largest lyapunov exponent from the system equations. 

- Build the Echo-State-Network:
  - Adjust all ESN parameters like reservoir dimension, input strength, or activation function. 
  - View some properties of the network and the input matrix. 
 
- Train the ESN:
  - View how good the fit was. If there is a big difference between y_train and y_train_fitted, the reservoir prediction will not work well. 

- Predict with ESN:
  - See how good the prediction of the ESN is. 
  - Plot the whole attractor or individual dimensions. 
  - Measure quantities for the true data and the prediction like: value histogram, power spectrum, lyapunov exponent from data. 
  - Calculate some measures based on the difference between the prediction and the real time series like: error or valid time. 
  
- Look under hood of ESN: 
  - Look at the value histogram of internal reservoir states, to see if the input strength is well set. 
  - Plot the timeseries of internal reservoir states for individual reservoir nodes. 
  - Vizualize the W_out matrix. 
  - See how well the standard deviation of the reservoir states during training and prediction fit together. 
  - Plot a 3d plot of three (generalized) reservoir nodes. 
  - Calculate the largest lyapunov exponent of the trained reservoir. 

# Note: 
Everything is very beta at the moment.
Things that are not yet implemented: 
- Proper typing for all python files. 
- Documentation and tutorial on how to use. 
- Proper requirements.
- Reformat and further clean up of code. 


# Related repository: 
- [https://github.com/GLSRC/rescomp](https://github.com/GLSRC/rescomp)
