# emotionDETECTION
Here the project uses the following models 
Deep Learning Model:
- Convolutional Neural Network (CNN): A three-layer model with ReLU activation, maxpooling, and dropout layers to prevent overfitting.
- Output Layer: Softmax activation to predict probabilities for each emotion class.
Machine Learning Models:
1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Decision Tree Classifier
5. Logistic Regression


To do it on your local system

create a folder 
step1) in the folder, save the code in the file emotion_train_.ipynb and save it with the same file name . 
step2)run this code first and you will notice the model in the folder after running the code with name emotiondetector.h5 and json file as emotiondetector.json
your model is trained here till 35 epochs


step3)now save the file emotion_detection.py with the code in the same folder and run it (this is for realtime detection i.e extension of the above mini project)
step4)when you will run emotion_detection.py , to stop the execution of the detection you have to press Q from the keyboard .

note:step 1,2 can be done on jupyter environment but if you wanna do it till step 4 then its better to do it on you local system as realtime need camera access .


pretrained model is uploaded with 54% accuracy.

dont miss step 1,2 before 3,4.... >_<
