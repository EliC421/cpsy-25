# presentation outline

***Background*** 
- Need for interpretable and personalized risk detection
- existing models fail to integrate all available data in an interpretable fashion

***Rhetoric***
- Computational Psychiatry symposium
- won last year need to take the cake again
- eclectic mix of research from strictly biological to strictly psychological and everything in between
- won last year on strictly biological with cognition as an anchor

***Data used***
- Shared baseline data from 3 studies
- Demographics (ASE)
- Health History ()
- WMH ratio (WMH/ICV)
- Cognition (DSST SCORE)
- Activity Habits & Sleep (GGIR - physical activity by day combination, sleep efficiency, inactivity)

***Research Question***
- how best to make one number that best represents someones overall health in terms of brain aging?
- must include all factors associated with brain aging: demos, hh, exercise habits, brain measures, cognition

***Current Modeling Problems***
- **How to handle skews under the laplacian assumption**
- **What dv/iv mix is best for power, rhetoric, and soundness?**

***Model***
- Reduced Rank Regression adds rank constarin (min(mxn)) for mxn matrix C (coefficients)
- Allows to separate C into AB^T
- Bayesian proportion sampled from laplace distribution (A), B maps from latent to response and is lognorm
- Monotonic effect for arbitrary spacing of ordinal predictors (effect)
- Dirichlet ensures all simplex components of monotonic transform add to 1, is also uniform (equally likely)

***Potential Models***
- preds:resps
- demo, hh, act : wmh, cog -> others to brain
- demo, wmh, cog : wmh, act -> non biological to biological

***Predictors***

**Demographics**
- show fig, no issues

**DSST**
- show fig, no issues

**GGIR**
- wait for eli

**WMH** FOCUS!
- Distribution heavily skewed right (yay! but also not yay)
- log transform keeps original shape, power transform lambda = -0.92 (wow...) and has normal shape

**Health History** FOCUS ON LIQUOR!
- most look good, liquor needs help
- show transforms




>>>>> CHATS VERSION

Presentation Outline

1. Introduction

Background
	•	The need for interpretable and personalized risk detection.
	•	Existing models fail to integrate all available data in an interpretable fashion.

Rhetoric
	•	Presenting at the Computational Psychiatry Symposium.
	•	Previous winner—aiming to win again.
	•	Research spans from strictly biological to strictly psychological and everything in between.
	•	Last year’s success was in strictly biological models, with cognition as an anchor.

⸻

2. Data Overview

Data Sources
	•	Shared baseline data from three studies.
	•	Key Variables:
	•	Demographics (ASE)
	•	Health History
	•	WMH Ratio (WMH/ICV)
	•	Cognition (DSST Score)
	•	Activity Habits & Sleep (GGIR—physical activity by day, sleep efficiency, inactivity)

⸻

3. Research Question
	•	Goal: How can we create a single number representing overall brain aging health?
	•	Essential Factors:
	•	Demographics
	•	Health History
	•	Exercise Habits
	•	Brain Measures
	•	Cognition

⸻

4. Current Modeling Challenges
	•	Skewness Under the Laplacian Assumption
	•	How to effectively transform skewed distributions?
	•	Optimal IV/DV Mix
	•	Balancing power, rhetorical strength, and scientific soundness.

⸻

5. Modeling Approach

Core Model: Reduced Rank Regression (RRR)
	•	Constraint on Rank: Minimizes dimensions while preserving key relationships.
	•	Matrix Factorization: C = AB^T decomposition.
	•	Bayesian Framework:
	•	A: Proportion sampled from a Laplace distribution.
	•	B: Maps from latent space to response, modeled as log-normal.
	•	Handling Ordinal Variables:
	•	Monotonic Transformations allow meaningful spacing.
	•	Dirichlet Prior ensures simplex components sum to 1, maintaining interpretability.

⸻

6. Potential Modeling Approaches
	•	Predictors → Response Mapping
	•	Option 1: Demographics, Health History, Activity → WMH, Cognition → Other brain measures.
	•	Option 2: Demographics, WMH, Cognition → WMH, Activity → Mapping non-biological to biological measures.

⸻

7. Key Predictors & Findings

Demographics
	•	Visualization Ready: No major issues.

Cognition (DSST)
	•	Visualization Ready: No major issues.

Activity & Sleep (GGIR)
	•	Pending Update: Waiting on Eli’s input.

WMH – Key Focus Area
	•	Right-Skewed Distribution (Important but problematic).
	•	Transformations:
	•	Log transformation maintains shape.
	•	Power transformation (\lambda = -0.92) normalizes distribution.

Health History – Focus on Liquor Consumption
	•	Most variables look good, but liquor data needs attention.
	•	Planned Fix: Apply transformations and assess impact.

⸻

8. Conclusion & Next Steps
	•	Finalizing Transformations
	•	Address skewness in WMH and Liquor data.
	•	Optimizing Model
	•	Evaluate different IV/DV combinations for best interpretability.
	•	Refining the Narrative
	•	Ensure clarity in presenting results at Computational Psychiatry Symposium.

⸻

This version enhances clarity, structure, and professionalism while keeping the content engaging and competitive for the symposium. 🚀
