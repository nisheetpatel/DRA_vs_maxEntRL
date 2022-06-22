# [DRA](https://papers.nips.cc/paper/2020/hash/c4fac8fb3c9e17a2f4553a001f631975-Abstract.html) vs [Max Ent RL](https://arxiv.org/pdf/1801.01290.pdf)

In this project, we would like to analyze whether memory resource allocation & entropy regularization in RL as two sides of the same coin.

### Key questions

1. Are they behaviorally identical?
   - so far, simulations indicate that they are largely identical
2. Can we cast them in the same framework to analyze the differences analytically?
   - yes, using [RL as inference](https://arxiv.org/abs/1805.00909)
3. What are the implications for neuroscientists?

### Frequency and stakes for MaxEntRL

#### Source of stochasticity in MaxEntRL

1. Stochastic policy
   - The policy is a softargmax with temperature $\alpha$ and $q_\text{soft}(s,a)$ as arguments
   - The higher the $\alpha$, the more stochastic the policy gets
2. "Soft" q-values include entropy
   - The soft q-values don't only rely on reward, but they also incorporate the entropy of the policy in that state

##### Implications for stakes

All else being equal, if the stakes at two states are different, then the one with the higher stakes will have a higher choice probability for the better option since all that matters for the policy entropy is the stakes at the state.

##### Implications for frequency

All else being equal, a state that is visited more frequently has more of a bearing on the overall reward. If both states had equal policy entropy, marginally reducing the policy entropy in the more frequent state would result in a higher gain in reward than the equivalent change in the less frequent state. Hence, at convergence, the state that is visited more frequently would have a higher choice probability for the better option.
