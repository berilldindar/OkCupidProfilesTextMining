# OkCupidProfilesTextMining
#TextMining 
The overall aim of this study is to show how user data can help both the user and the developer of online dating platforms. User data becomes valuable over time due to its easy accessibility and storability. We can do further analysis by finding relationships and outliers between the data. The benefit of finding relationships in data can help users find relationships that match other users. Detection of outliers can help the developer filter out spam or low-quality accounts in their recommendation system. Finding inconsistencies or missing values ​​in the data can help both the user and the developer figure out what to report and what not. We will see in this study the importance of something as simple as skipping data for privacy concerns like income or education. Instead of requesting a classic level of education like “High School Graduation” or “College Leaving”, users can choose an answer like “Work in Space Camp”. This benefits the overall system by allowing users to have a full profile and developer with less missing value to work with.
After analyzing the data, we use a short machine learning algorithm to build a model that predicts the gender of the user. While predicting gender is useful for some users, it is not for others. Other combinations of variables can be used to create other models, and models can be easily replicated. Other models may include: Education, body type and age to estimate income level. In addition, drinking and drug habits can be looked at to predict the type of job. The models we build for prediction can first help the developer match users with recommendations. It can also help fill in missing user profiles which helps users to get a full profile.


OkCupid Dataset
The data consists of publicly available profiles of 59,946 OkCupid users who live within 25 miles of San Francisco, had active profiles on June 26, 2012, were online in the previous year, and had at least one picture on their profile.
Using a Python script, data was scraped from users' public processes on June 30, 2012; Non-public information such as messaging could not be accessed.
Variables include typical user information (such as gender, sexual orientation, age, and ethnicity) and lifestyle variables (such as eating and drinking habits, smoking habits). In addition, the answers to 10 questions directed to all Okcupid users are also included in the data set.
However, we will examine each to understand word responses, numerical responses, and written responses.

Overview of OkCupid Profiles
When we look at the variables listed from the data frame, we can count 31 variables. As an example, let's say you want people to have 5 matching criteria in order to be matched. These criteria may include: Approximately 30 years old, drug-free, college-educated, male, and living in New York. Since we have 31 variables instead of 5, it is too difficult to filter who the user matches with. It also ensures that if a user's profile has missing sections, it will still match. Useful for many variables is the ability to combine certain variables to predict another. Another example is using education and work to estimate income. With these one can make a rough estimate. However, it is possible to use machine learning to teach an algorithm how to make an accurate prediction based on user data.
We can see our variables in the image;
![image](https://user-images.githubusercontent.com/53093904/169523467-fc7308cc-567c-4bab-92be-2ce7b57c99a2.png)

Age Variable
The average user age is 32. This practice makes sense after college or when people will be using dating sites to find common ground. The minimum age is 18. This should be the lowest level as users are under the age of 18 and it is illegal to use a dating site in the US. Usually the age range is around 26-37 years old. The maximum age is 110, which seems like an outlier, but with over 50,000 data points it doesn't affect the averages too much in the mid-range. Another observation that can be made is that users don't have much room to lie about their age if they decide to do so because they also have a profile picture. If they choose to lie, it shouldn't be too far from the standard deviation, they can tell a maximum of 9 years more or less.

![image](https://user-images.githubusercontent.com/53093904/169523633-59a9ec70-d171-4410-addb-7fd9e0600f2b.png)

Body Type Variable
There are 11 types for body types. The top count is the "average" body type with 14,652 users, which makes sense since it's average. The lowest count is "unspecified" with 198 users, which may have meaning with machine learning. Some interesting body types here are "jacked", "curvy", "fit" and "overweight". The first two can predict gender and the last two can predict diet. With a combination of other variables, it can very well predict variables not listed in this study.
![image](https://user-images.githubusercontent.com/53093904/169523765-e855c04b-a13a-4a93-8022-937d1652af06.png)

Diet Variable
Eating habits are also very similar to body type. Most popular habit: “mostly anything” with 16,585 users and least habit: “halal” or “kosher” with 11 users each. It can be used to predict variables such as dietary habits, religion. Although many users overlook these habits when exiting, we think it will be very useful in estimating a person's other factors.
![image](https://user-images.githubusercontent.com/53093904/169523920-72f51c06-9f92-4512-97af-1baa9241a0c2.png)

İncome Variable
     A value of -1 indicates users who do not enter data here, and 48,442 does not prefer to list their income. While it's really nice to have a revenue estimator, training for machine learning becomes difficult when more than half of the data is missing for users. It is also important to pay attention to honesty in this section.
     ![image](https://user-images.githubusercontent.com/53093904/169524044-83b17b19-3f8b-4592-aa08-276ba93755b6.png)

Education Variable
Education is a great variable for “smart” matches. It helps users with similar goals in similar situations. The interesting category in the education variable is 'space camp'. However, it is debatable whether all answers are honest. Education may be a good option for matches by age and income. The largest group is university graduates with 23,959 users and at least 11 users are medical school graduates.
![image](https://user-images.githubusercontent.com/53093904/169524118-5c277c20-6b6a-4202-ac0b-4e23b47a7a05.png)

Essay Variable
Unlike many answers readily available for choices, this way users are given the opportunity to express themselves in words. In this way it is possible to explain other variables such as income and religion. Or they can use this section to prepare for meetings. Either way, this is an excellent way for users to get to know each other.

On the right, 10 questions asked to users are displayed. A separate column was created for each question in the data set.

The titles are:
● 0: Self Summary
● 1: What am I doing with my life…
● 2: I'm really good…
● 3: The first thing people usually notice about me…
● 4: Favorite books, movies, show music and food…
● 5: Six things I could never do without…
● 6: I spend a lot of time thinking…
● 7: On a typical Friday night I…
● 8: The most private thing I am willing to accept…
● 9: If you want to message me...
First User Profiles
![image](https://user-images.githubusercontent.com/53093904/169524432-884babe9-2ce6-421d-9105-faa915cae056.png)

Data Cleaning
We clean the data before processing and analyzing it. We also focus on overreactions that deviate from the mean response. We clean up numbers and essays the same way, so we can find averages and make them machine learning ready.
We know that men are generally taller than women. The average male is 70 inches tall, the average female is 65 inches tall.
![image](https://user-images.githubusercontent.com/53093904/169524760-2970c713-8cdb-4aee-9942-0c432762778f.png)

She has a female standing 4 inches tall when we check the outliers to assess the integrity of the data. There is also a 1 inch tall male.
![image](https://user-images.githubusercontent.com/53093904/169524822-93ff0aea-4d1d-4027-a76f-9374fe221f32.png)

Here is a female standing 95 inches tall. He's also a 95-inch-tall male. Just for reference, the average NBA basketball player is 79 inches tall.
![image](https://user-images.githubusercontent.com/53093904/169524879-bdb8b077-7213-44c6-973d-bc640904d4e2.png)

Data Analysis

If we visualize the data for the variable for length, we can more easily see that men are, on average, taller than women. We got rid of outliers during the data cleaning phase. That's why we're seeing results between 55-80 inches.
![image](https://user-images.githubusercontent.com/53093904/169525002-acd198bd-e86c-47aa-8f8d-e471b58517c4.png)

Visualization of the words used by school leavers and doctoral graduates with wordcloud;
![image](https://user-images.githubusercontent.com/53093904/169525232-0bca423a-246e-4e8f-901e-257a0c38e147.png)
![image](https://user-images.githubusercontent.com/53093904/169525256-bd239fee-0f96-475b-8064-40e383746c3a.png)

Comparison of words used by school dropouts and doctoral graduates
![image](https://user-images.githubusercontent.com/53093904/169525652-bbd5e373-1414-458a-bdd9-c380f87da932.png)

Repeating words by gender;
![image](https://user-images.githubusercontent.com/53093904/169526065-e9672d22-d1fb-4c64-a43b-8821e8de5fc8.png)

We may use this data to predict integrity in user profiles. If a college graduate's remarks deviated excessively from the average, we can quickly check their other answers to see if they were honest. Having the right profiles should lead to happier users. If a user gets a match based on incorrect information, it could further jeopardize their interest in using the website after a bad match based on incorrect information.


Modeling of Data
The final part of our analysis focused on modeling data with the help of Machine Learning. The overall goal here is to be able to predict data that will help OkCupid users match more easily. Our data is labeled and we are trying to predict categories, so we use the Naive Bayes method. Naive Bayes methods are a set of supervised learning algorithms based on the application of Bayes' theorem with a "naive" assumption of independence between each pair of features.

Researching Data
Here we check different categories in guessing OkCupid users. We have several variables at our disposal. One of the things we can do is combine categories for clustering. However, we have too much data for that. With over half a million essays, the sample size may be sufficient to predict another category. One very simple thing we can do is predict gender. However, we can predict education, income or any other category with the same algorithms that will be used here.
    The reason we use gender to predict first is because it has the least amount of missing data. It is also very difficult to lie about your gender compared to all other variables. We can also use a combination of variables such as body type and diet if we don't have enough data to predict another category.
![image](https://user-images.githubusercontent.com/53093904/169529236-036b9295-60bd-43ad-8f6c-3e6a93b49c48.png)

We reduced all essays to a single column on the basis of gender.
![image](https://user-images.githubusercontent.com/53093904/169529302-522a6533-7d2d-4336-ae7f-dbc971738cfa.png)

Vector Space Model for Naive Bayes

First, we transform the essays into a vector space model to act as a search engine for machine learning.
![image](https://user-images.githubusercontent.com/53093904/169529362-342d2e27-6611-4f69-9869-db207141bb67.png)

Next, we use MultinomialNB, the Naive Bayes algorithm for multinomial distributed data. MultinominalNB is also one of two classical naive Bayesian variants used in text classification.
![image](https://user-images.githubusercontent.com/53093904/169529436-71b854a1-9dd5-48ea-96d3-7e509c000a08.png)

We find our training and test accuracy values according to the scores.
![image](https://user-images.githubusercontent.com/53093904/169529478-05d1a662-709f-4af5-8e0a-2fef8ec3b192.png)
![image](https://user-images.githubusercontent.com/53093904/169529561-299fe165-f80b-4e1e-bff2-eb40fb0f49da.png)
![image](https://user-images.githubusercontent.com/53093904/169529579-11ce5637-9a8a-4e60-9bbb-5b6f86835ca3.png)
![image](https://user-images.githubusercontent.com/53093904/169529597-31a904ac-08bb-4411-bfd4-d5380d2c91d4.png)


Gender Prediction in Words
![image](https://user-images.githubusercontent.com/53093904/169529661-30686f79-6fa8-4ed7-aba7-34e8ef03b9d8.png)
![image](https://user-images.githubusercontent.com/53093904/169529688-21a8d5ad-57c8-496a-8e71-d486d337b302.png)

Phrase-based gender prediction
The first example is very simple. As expected, we get a good result “with a 96.81 percent probability of a man trying.”
![image](https://user-images.githubusercontent.com/53093904/169529803-adc1ffd5-d37f-49b5-93e9-78750dda6f7b.png)

The second example is similar to the first, “we get a good 87.92% result from a woman's trial”.
![image](https://user-images.githubusercontent.com/53093904/169529821-25a45bb9-5abe-4364-a09f-bb960e75b03b.png)

Let's try to make clever manipulations on a simple sentence that will reverse its meaning. Let's start with the following example;
![image](https://user-images.githubusercontent.com/53093904/169529959-3074c9fc-f5c6-4871-9c2d-9e99f5005b8e.png)

After seeing the positive result above, we're trying to see if we can get the opposite by changing the word "like" to "hate".
![image](https://user-images.githubusercontent.com/53093904/169530003-3e46dfb7-7986-47a8-bbc6-fa534ec5c53d.png)
Our model is not that good, still classifies the user as a male, but with less confidence.

Possible Improvements
We can take a deeper look at the text analysis we use by seeing how our model correctly distinguishes between the words "like" and "hate". Another thing we can look at is n-grams.
In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sequence of text or speech. Depending on the application, the items can be phonemes, syllables, letters, words or base pairs. N-grams are typically collected from a body of text or speech. When the elements are words, n-grams may also be called shingles.
The n grams of size 1 are called "unigrams"; size 2 is a "bigram" (or less commonly "digram"); size 3 is a "trigram". Larger sizes are sometimes called the n-value in modern language, e.g. "four grams", "five grams" etc. It could be a simple example from our previously used example. For a unigram: "I" "hate" "computers". This will be more influenced by the term: computers, which is usually associated with men. For two grams: "I hate" "I hate computers". This would be more influenced by the term "hate of computers", which is not usually associated with men.
![image](https://user-images.githubusercontent.com/53093904/169530276-daa8610f-c9fa-45ef-b874-393ffb8e5ff5.png)
Bigrams results are better than the 73/71 model, but it's an overfitted model. We should skip this.

Although the implementation of n-grams may yield better results, we should not dismiss other classifiers such as the random forest. Perhaps other classifiers that didn't perform as well would have been more helpful if we added more variables to our articles, such as orientation, diet, and height.

Results and Recommendations for OkCupid
There are some results in this study that may help OkCupid and its users. We focus on getting more data, filtering out inconsistent users and recommendations for users. It does a great job of getting more data, having lots of variables so users may have little or no response to some of them. However, data is more important to OkCupid than its users. It can be used to improve your platform to beat your opponents. For example, OkCupid may send its users an anonymous survey asking which 5 variables they most search for in a partner. Or they can even do a blind study to see what users are filtering out the most. Assuming users care most about income and education, OkCupid should do whatever it takes to ensure answers are received. Refusing to share revenue or even "space camp" for education may not be useful for future work to help users get what they want.



Filtering Inconsistent Users
Nothing is more frustrating than finding a match online and finding out you've been hunted. “Catfished” is defined by the urban dictionary as “being cheated on with false information online.” This can be tracked by inconsistent responses in a user's profile. Some of the ways Okcupid can prevent this is by comparing similar response categories to the average. For example, if a 19-year-old college student became a millionaire, other It should be compared to what 19-year-old college students earn. Instead of banning their accounts, OkCupid may give them fewer matches or matches with other inconsistent users. The reason Okcupid doesn't ban inconsistent users outright is because sometimes the user is really honest. Also, banning users will generate less revenue for OkCupid overall .

Recommendation for Users
Let's say a user is difficult to respond to or is unaware that a good full profile will result in higher quality matches. A full profile can mean honesty and transparency for users. This is much better than a poor profile that can lead to a feeling of blind history. To assist users, OkCupid can use Machine Learning predictions to help users auto-complete their profiles or even encourage users through positive reinforcement. Badges can be earned for full profiles.



If you want to contribute to our project, you can contact us.




