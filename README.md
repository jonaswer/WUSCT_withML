# Psychological background

The Washington University Sentence Completion Test (WUSCT) created by Jane Loevinger in 1979 measures the ego development of a human along Loevinger's stages of ego development. More Information can be gathered in recent research papers or articles on cognitive development, which are still refering to the research of Loevinger and her initial model. (https://www.researchgate.net/publication/356357233_Ego_Development_A_Full-Spectrum_Theory_Of_Vertical_Growth_And_Meaning_Making, https://www.researchgate.net/publication/350683488_Measuring_Ego_Development, i.a.)

This repository is an approach to solving the Washington sentence completion test (WUSCT)
with Natural Language Processing (NLP) methods with public available data of former WUSCT tests, available at [click here](https://osf.io/jw7dy/). 

# Computational methods

The WUSCT contains relatively small sentences, that need to be classified into 
eight categories. The following chapter describes the different characteristics of 
the selected architecture and preprocessing routines based on the underlying goal 
to distinguish the answers to the test. 

Firstly, before tokenizing the sentences into discrete elements for the machine
learning model, we preprocess the data. Unimportant words, such as the, a, or it will 
be deleted, and the conjugations, as well as declinations of words, will be replaced
with their infinitive versions. This is known as Stopwords in NLP [1].
Furthermore, special signs are filtered and the sentence is transferred to lowercase
letters completely.

The tokenization is done with a maximum number of words of 11000 with about 10000 
unique words in the data and a maximum length of words of 50 for each test answer.

The dataset used by Lanning, K. [2] is highly unbalanced. The early and late development 
stages are underrepresented, while the conventional answers are highly overrepresented. 
Two. As a consequence, we add some augmented data to the underrepresented classes and also 
use class weights to adjust the impact of overrepresented data lower.

We transfer test scores to binary data, which is known as one hot encoding and is typical 
in multi-class classifications. [3]

A recurrent neural network with long-term short memory (LTSM) is used with validation data
for hyperparameter optimization as a typical approach for NLP. We use 80% of the data 
as the training dataset and 20% of this training dataset for validation

The confusion matrix shows the result for the independent test dataset. The center 
of every actual value is also the predicted value. 

All metrics can be sent by request on my GitHub account.

# Example

In this example 18 sentence beginnings from Lenning's sample were used. The following tables shows some sentences and their initial 
score as well as the average score

sentence nr. | beginning | completion | score
-------- | -------- | -------- | --------
1   | When a child will not join in group activities | take time with him and eventually he will come around when he sees how much fun it is | 7
2   | Raising a family   | takes a great deal of patience and a sense of humor | 6
3   | Being with other people | that I do not know annoys me | 4
4   | The thing I like about myself is   | that fact that i am going back to school and doing it for myself | 6
5   | My mother and I | tend to fight | 4
6   | What gets me into trouble is   | taking on more than I can handle | 6
..  | ..  | .. | ..
18   | Rules are  | to be broken | 4

The full question and answer set is in lenning_template_csv. The average of these random 18 answers is 5.05, 
which equals the stage "self-aware" be Loevinger. Using these 18 Answers for the developed neural network gives a similar average with the following plot:

![graph](barchart_example.png)

# Usage 

The code is deployed webbased and can be used at .. free of charge

