# Sentiment Analysis of IMDb Movie Reviews
## Authors
* Claire Brekken
* Rachel Brynsvold
* Asher Khan
## Description 
In this project, we construct a model that predicts the probability that a given movie review has positive sentiment. The dataset consists of 50,000 IMDB movie reviews, each with a unique id, numerical score from the reviewer, the text of the review and a flag indicating if the review was negative (0) or positive(1). For the numerical score, 1-4 correspond with a negative review, 7-10 correspond with a positive review. The dataset does not contain any reviews with a score of 5 or 6. With this data set, we constructed a vocabulary of 775 n-grams to use as features to train a logistic regression on each of the 5 train/test splits. The input to the model is 775 features representing our 775 n-gram vocabulary, each with a value 0 or 1 for indicating if that n-gram exists in the given review. The output of the model is a probability that the review has a positive sentiment.
## Technical Details
### Preprocessing
The first step in developing our logistic regression model was determining which n-grams within the reviews were most indicative of a positive or negative sentiment. To do this, we loaded the entire dataset and tokenized each review before creating a vocabulary of n-grams with up to 4 words. We ignored stop words when creating this initial vocabulary and then pruned the vocabulary to only include n-grams that had at least 10 occurrences and made up more than 0.1% of the vocabulary but less than 50% to make sure we aren’t including words that aren’t common or words that occur too often to be indicative of positive or negative sentiment. We determined our final vocabulary by training a logistic regression model with Elastic Net and using the minimum number of non-zero beta values returned from this model that gave us the desired AUC on all 5 train/test splits.  To eliminate a source of data leakage, we also removed n-grams that effectively placed the score in the review (e.g. 7_out_of_10). The described preprocessing steps resulted in a final vocabulary of 775 n-grams.
### Models and Tuning
Once we had our vocabulary determined, we trained a Logistic Regression with Elastic Net cross validation to generate predictions on each of the 5 splits. We tested out a range of alpha values to use in Elastic Net and found that alpha = 0.1 gave us the desired AUC for all 5 splits with this vocabulary.
Model Validation
|Split|AUC|
|-----|---|
|1|0.9606|
|2|0.9612|
|3|0.9603|
|4|0.9613|
|5|0.9600|

Total Runtime: 2.416624 minutes

Run on a MacBook Pro, 2.8GHz, 16GB Memory

While our model meets the desired AUC on all splits, there are improvements we could make in the future to enhance this model. Most improvements are related to additional cleaning or adjusting how our vocabulary is constructed. Using punctuation such as exclamation points or question marks could be good indicators or positive or negative reviews. We could also try using longer or shorter n-grams to capture more or less nuance in the reviews, or further clean up the n-grams when generating the vocabulary such as replacing contractions with root words or removing n-grams that are completely contained in other n-grams.

### Interpretability
We chose to use a logistic regression because of the explainability of the model. If we were to try and explain this model to someone who is evaluating the correctness of our review classifier, we would say that our model has learned that there are certain words that are used more often in positive or negative reviews and has determined weights (also known as beta values) to apply to each of these words to determine the probability that a review is positive. With a vocabulary of the most significant n-grams, we use the weight of each vocabulary word that is present in the review to generate our prediction. So, larger/more positive weights are assigned to n-grams that are more indicative of a positive review and the smaller/more negative weights are assigned to n-grams associated with negative reviews. 

To assess the interpretability of our vocabulary, we looked at which features (n-grams) had the most significant contribution to our predictions in the form of the beta value associated with each feature in an Elastic Net Logistic Regression model trained on the entire dataset. When looking at the features that had the most positive beta-values, n-grams such as “just_great”, “definitely_worth”, and “refreshing” make sense in terms of a positive review.

![alt text](https://github.com/asher657/imdb-sentiment-analysis/blob/main/MostPositive.png?raw=true)

 Similarly, we were able to gather features with the most negative beta values and n-grams such as “lost_interest”, “only_redeeming”, and “stinker” make sense to indicate a review with negative sentiment. 
 
![alt text](https://github.com/asher657/imdb-sentiment-analysis/blob/main/MostNegative.png?raw=true)

One example where our model predicted a negative sentiment on a review that actually had a rating of 10 was review id 3794 with the following review:

> "I would like to vent my displeasure at NBC Canceling Las Vegas. The show had been Top Notch for the past 5years. Tom Sellecks addition was great. He really brought a nice fresh addition to the show. What does NBC have now? Lame reality and night time game shows. I mean come on Keep the Old and Tired Law and Order? Not even putting Jack McCoy as DA can keep the show interesting. Gee let's keep quality program like Deal or No Deal or ED? ER should be put out to pasture to. NBC is worse now than it was in Pre Seinfeld Cheers days. With cable and internet, NBC cannot afford to fall flat on its face.PLEASE BRING BACK VEGAS! i remember when Homicide Life on the Street ended the way it did. At least they had a two hour series final. Hey CBS are you listening? Please pick up Vegas it is a great show."

As a human reading this review, it is clear why our model predicted a negative sentiment: the reviewer uses the review to criticize NBC for cancelling the show, rather than giving the reasons they love the show enough to give it the 10 rating. We see many words from this review like “lame”, “worse”, “tired”, and “least” that are in our vocabulary and that our model has found to be predictive of negative reviews. The review also contains vocabulary words “great”, “fresh”, and “nice”, which our model has found to be predictive of positive reviews, but none of these words have a large positive trained beta value.  Interestingly, it seems our model has actually done a good job of capturing the sentiment of this review’s text - it is quite negative.  The explanation for this false negative is that the content of the review does not match the score the reviewer gave the show.

While the example above is easily explained, it is also worth including some discussion of the tradeoffs between interpretability and performance with respect to vocabulary size.  A small vocabulary allows better interpretability of the vocabulary itself (understanding why certain words are included and if they indicate a positive or negative connotation). However the model’s performance should generally improve from having a larger vocabulary, since predicting using more words will increase the likelihood of capturing the nuances in some of the reviews that end up as false positives or false negatives due to more ambiguous language.

