# 1. Project Title
Customer Segmentation in Bank's Term Deposit Subscription

# 2. Project Description
Marketing campaigns are characterized by focusing on customer needs and their overall satisfaction. Nevertheless, there are different variables that determine whether a marketing campaign will be successful or not. Some important aspects of a marketing campaign are as follows:

Segment of the Population: To which segment of the population is the marketing campaign going to address and why? This aspect of the marketing campaign is extremely important since it will tell which part of the population should most likely receive the message of the marketing campaign.

Distribution channel to reach the customer's place: Implementing the most effective strategy in order to get the most out of this marketing campaign. What segment of the population should we address? Which instrument should we use to get our message out? (Ex: Telephones, Radio, TV, Social Media Etc.)

Promotional Strategy: This is the way the strategy is going to be implemented and how are potential clients going to be addressed. This should be the last part of the marketing campaign analysis since there has to be an in-depth analysis of previous campaigns (If possible) in order to learn from previous mistakes and to determine how to make the marketing campaign much more effective.

You are leading the marketing analytics team for a banking institution. There has been a revenue decline for the bank and they would like to know what actions to take. After investigation, it was found that the root cause is that their clients are not depositing as frequently as before. Term deposits allow banks to hold onto a deposit for a specific amount of time, so banks can lend more and thus make more profits. In addition, banks also hold a better chance to persuade term deposit clients into buying other products such as funds or insurance to further increase their revenues.

You are provided a dataset containing details of marketing campaigns done via phone with various details for customers such as demographics, last campaign details etc. Can you help the bank to predict accurately whether the customer will subscribe to the focus product for the campaign - Term Deposit after the campaign?

# 3. Data Description
Train Set

Contains the ![data](https://github.com/GeoLai/Customer-Segmentation-of-Bank-Termed-Deposit-Subscription/tree/main/dataset) to be used for model building and model testing. It can be segregated into training set and testing set. It has the true labels for whether the customer subscribed for term deposit (1) or not (0)

This data however was presented with issue of imbalanced data where further sampling required. The training of this project is without post-sampling work which can be very biased towards certain class classification. It is adviced to proceed with post-sampling task to balance out the dataset. Please be reminded that this task may has it owns pros and cons. Use it at your discretion.

# 4. How to Install and Run the Project
This project was run in Conda environment using Spyder IDE (Interactive Development Environment). Several essential libraries required to be installed prior to running the code. 

For computer that does not have GPU, you might want to use external workspace such as ![Google Colab](https://colab.research.google.com/?utm_source=scs-index) for running your scripts which no additional modules installation are required.

# 5. How to Use the Project
The full ![code](https://github.com/GeoLai/Customer-Segmentation-of-Bank-Termed-Deposit-Subscription/blob/main/cust_seg_train.py) can be viewed here as a reference. For cleaner code construction, I have written some modules in a separate ![module](https://github.com/GeoLai/Customer-Segmentation-of-Bank-Termed-Deposit-Subscription/blob/main/cust_seg_module.py) file which some of tuning can be done during model training.

Visuals are provided which were generated from data visualization of the data, training curves which displayed in Tensorboard, snippet of training scores, confusion matrix where located in the ![image](https://github.com/GeoLai/Customer-Segmentation-of-Bank-Termed-Deposit-Subscription/tree/main/images) folder.

# 6. Include Credits
Credits to owner of the dataset and the provider to Kaggle.

Data provided by ![Kunal Gupta](https://www.kaggle.com/kunalgupta2616). Purpose for solution ![HackerEarth: HackLife](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)

# 7. Add a License
No license

# 8. Badges
### These codes are powered by
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### If you found this beneficial and help you in way or another. You may want to
![BuyMeAWanTanMee](https://img.shields.io/badge/Buy%20Me%20a%20Wan%20Tan%20Mee-ffdd00?style=for-the-badge&logo=buy-me-a-wantanmee&logoColor=black)

