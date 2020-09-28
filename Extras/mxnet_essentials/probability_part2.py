############ Axioms ############

"""
P(A, B) is the joint probality, where given 'a' and 'b': A=a, B=b simulataneously.
For all values of a and b, P(A, B) <= P(A=a). (ofc)
=> 0 <= P(A, B) / P(A) => This is called "Conditional Probability"
The probability that B happens, provided that A has happened.
From this we Derive The bayes Theorem.
P(B|A) = P(A, B) / P(A) # CONDITIONAL PROBABILY
SO it can be inferred that:
=> P(A, B) = P(B|A) * P(A)
=> P(A, B) = P(A|B) * P(B)
=> P(A|B) = P(B|A) * P(A) / P(B)
The final versions:
=> P(B|A) = P(A|B) * P(B) / P(A)
=> P(A|B) = P(B|A) * P(A) / P(B)
"""
