# [DRA](https://papers.nips.cc/paper/2020/hash/c4fac8fb3c9e17a2f4553a001f631975-Abstract.html) vs [Max Ent RL](https://arxiv.org/pdf/1801.01290.pdf)

In this project, we would like to analyze whether memory resource allocation & entropy regularization in RL can be thought of as two sides of the same coin. Refer to [the documents](docs) for the pre-draft collection-of-thoughts-and-results. The code is currently functional but not packaged well into modules; I will update it before publishing.


### Key questions

1. Are they behaviorally identical?
   - so far, [simulations](docs/Simulations%20DRA%20vs%20max-entropy%20RL.pdf) indicate that they are largely identical
   - look at this intuition behind the [similarity](docs/MaxEntRL%20frequency%20vs%20stakes.pdf) and [differences](docs/differences%20DRA%20maxEnt.pdf)
2. Can we cast them in the same framework to analyze the differences analytically?
   - yes, by shining the [inference lamp](docs/Inference%20lamp.pdf) on DRA
      - [full derivation by Luigi](docs/DRA_as_inference_LuigiAcerbi.pdf)
   - [meeting notes](docs/Luigi%20Alex%20DRA%20as%20inference.pdf), [some linked ideas](docs/Regularization%20in%20RL.pdf), [other](docs/DRA%20analytical%20gradient%20for%202%20options.pdf)
3. What are the implications for neuroscientists?
   - A little blurb about [process vs normative models](docs/Process%20vs%20normative%20modeling.pdf), where DRA can be thought of as the normative approach and max-entropy RL as a heuristic/process model that turns out to share a common framework
   - A few thoughts in the last section of [the main document](docs/000%20DRA%20vs%20Max%20Ent%20RL.pdf)
