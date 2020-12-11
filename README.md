# Hypothesis-testing-Comparing-race-time

Blodomloppet is a race that is arranged in 18 locations around Sweden. The purpose is to draw attention to the vital blood donation and promote a healthy lifestyle. Participants can walk, jog or run 5 or 10 kilometers depending on their conditions. I first participated in this race during my first professional year in 2011 and then I attended again in 2012 in Örebro.

In this project I have web scraped data for the race 2011 and 2012 for all groups and race distances in Örebro and saved them in Excel. Data has been analysed and hypothesis tests have been performed.


Three hypothesis tests are performed with T-test for the means of two independent samples between following groups:  
1.    Men 5km 2011 vs Men 5km 2012
2.    Women 5km 2011 vs Women 5km 2012
3.    Women 5km 2011 vs Men 5km 2011

The null hypothesis is that the average time is equal and the alternative hypothesis is that they are not equal. The hypothesis calculation have been performed with python package scipy.stats and compared with handbook calculation to get a better understanding how the package works. P-values and confidence intervals have been used to evaluate the hypothesis. The results will also be presented graphically to have a visual understanding.

![significance.png](/Pics/significance.png)

### Python package 
* Pandas 
* Numpy 
* Matplotlib 
* Seaborn 
* Scipy
* Math


### Table of contents:
* Exploratory Data Analysis (EDA)
* Data cleaning and manipulation
* Hypothesis 
* Conclusion


<B>NOTE:</B> The formulas in jupyter notebook does not work in Github for some reason. In order to view the whole notebook properly, please paste my repo page in https://nbviewer.jupyter.org/ 

