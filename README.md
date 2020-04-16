# CSYE7255FinalProject - Spring 2020

## Title
Fake Disaster Tweets Prediction

## Team Information

| Name | Email Address |
| ---- | ------------- |
| Ruonan Ren| ren.ruo@northeastern.edu   |
| Chao Ma   | ma.cha@northeastern.edu   |
| Yan Sun   | sun.yan3@northeastern.edu |

## Description
The purpose of this project is to build a machine learning model to predict which Tweets are about real disasters and which ones are not. There is alse data visualization of tweets anylasis.
This project is inspired by a Kaggle competition. Details can be found at https://www.kaggle.com/c/nlp-getting-started/overview

## How to Run
1. Clone the project
2. Open Intellij
3. Import tweets-training-data as a sbt project
4. Run src/main/scala/edu/neu/coe/csye7200/swing/GUI.scala
5. Input Tweet content
6. Select a machine learning model
7. Input Parameters accordingly
8. Click submit button and get the prediction result
9. Run src/main/scala/edu/neu/coe/csye7200/wordcount/SparkWordCount.scala to get the vegas charts of data visualization
10. Right click on src/test/scala/edu/neu/coe/csye7200/unittests package, seclect Run -> Scalatests to run all the test cases(It may take some time for model accuracy cases)
