### An explanation to Maximum Likelihood

When using statistical model, it is essentially to find the specific set of paramters that maximizes the likelihood of observing given data. More formally for the given data (X,y) we are looking to find set of parameters (theta)

max L(y | X; theta)

In simple words, maximize the likelihood of observing y given X when the prediction is paramterized by some parameters theta

MLE is a method for estimating the parameters of a statistical model by maximizing the likelihood of the observed data.

P (event | explanation) or P(data | Parameters)

#### Example -
Essentially, we maximized the conditional probability of the event (perfect score) given an explanation.

- The event is what we observed — “Everyone getting a perfect score.”
- An “explanation” represents a possible cause we came up with.

We formulated our problem this way because there’s a probability of:

- “Perfect score” given “Everyone studied thoroughly.”
- “Perfect score” given “The paper was leaked.”
- “Perfect score” given “The invigilator allowed them to cheat.”

#### Conclusion

After estimating the probability, we picked the explanation with the highest conditional probability P(event|explanation).Simply put, we tried to find the scenario that most likely led to the observed event.
