### Probability 

Probability is used in contexts where you wish to know the possibility/odds of an event.

***examples*** 

- Probability of obtaining an even number in a die roll?
- Probability of drawing an ace of diamonds from a deck?

When translated to ML, probability can be thought of as:
- What is the probability that a transaction is fraud?
- What is the probability that an image depicts a cat?

When calculating probability, the model’s parameters are known. Also, we assume that they are trustworthy.
For instance, to determine the probability of a head in a coin toss, we mostly assume and trust that it is a fair coin.

### Likelihood

Likelihood, on the other hand, is about explaining events that have already occurred.Unlike probability (where parameters are known and assumed to be trustworthy).likelihood helps us determine if we can trust the parameters in a model based on the observed data.

In maximum likelihood estimation, you have some observed data and you are trying to determine the specific set of parameters (θ) that maximize the likelihood of observing the data.

Using the term “likelihood” is like:
I have a possible explanation for my data. (In the above illustration, “explanation” can be thought of as the parameters you are trying to determine)

How well does my explanation explain what I’ve already observed? This is precisely quantified with likelihood.
For instance:
- Observation: The outcomes of 10 coin tosses are “HHHHHHHTHH”.
- Explanation: I think it is a fair coin (p=0.5).
- What is the likelihood that my above explanation is true based on the observed data?