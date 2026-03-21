2025 IEEE International Conference on Data Mining (ICDM)

# DiffuGC: Diffusion Model Can Help Discover Granger Causality from Interventional Time Series


Bo Liu [1] _[,]_ [3], Hongyan Li [1] _[,]_ [3] _[,][∗]_, Shenda Hong [2] _[,][∗]_

1 _School of Intelligence and Science Technology, Peking University_

2 _Health Science Center of Peking University, Peking University_
3 _State Key Laboratory of General Artificial Intelligence, Peking University_

_∗_ Corresponding authors: leehy@pku.edu.cn, hongshenda@pku.eud.cn



_**Abstract**_ **—Discovering Granger causality from time series data**
**is fundamental to understanding dynamic systems, yet most**
**existing methods struggle with unknown intervention targets**
**or causal structures in real-world scenarios. In this paper, we**
**propose DiffuGC, a novel diffusion-based framework that unifies**
**observational and interventional causal discovery through a gen-**
**erative denoising process. By introducing diffusive interventions,**
**which apply progressive interventions without any prior knowl-**
**edge, DiffuGC amplifies causal signals while preserving structural**
**information. Furthermore, we introduce a denoising NoiFormer**
**with adaptive attention to both short- and long-term causal**
**dependencies, which disentangles trend and seasonal components**
**to enable accurate reconstruction of causal structures from**
**interventional data. To the best of our knowledge, we are the first**
**to integrate diffusion models with interventional Granger causal**
**discovery. Extensive experiments on synthetic, quasi-real, and**
**real-world benchmarks demonstrate that DiffuGC consistently**
**outperforms state-of-the-art baselines in both observational and**
**interventional data. Moreover, we introduce an intriguing notion,**
**Causality Acceleration, characterized by the early emergence of**
**informative causal patterns within the diffusion path, which may**
**open up promising directions for future research on efficient and**
**adaptive causal discovery.**

_**Index Terms**_ **—Granger causal discovery, Diffusion models, Ob-**
**servational and interventional time series, Diffusive intervention**


I. I NTRODUCTION


Causal interpretation of the observed time-series data can
help answer fundamental causal questions and advance scientific discoveries in various disciplines such as medical and
financial fields [1], [2]. Researchers in the past decades have
been dedicated to discovering causal graphs from observed
time-series and made large progress, for example, Granger
causality [3], transfer entropy (TE) [5], constraint-based
methods (e.g., PCMCI [6]), noise-based methods (e.g., VarLiNGAM [4]). Unfortunately, conditional TE and constraintbased algorithms struggle to scale due to their iterative
procedures and reliance on density estimation, highlighting
the need for more efficient techniques in discovering causal
relationships. Recently, deep learning methods have shown
strong potential in Granger causal discovery [2], [7], [8],
demonstrating effectiveness in modeling pairwise interactions
in domains like gene regulatory networks [2], [38].

Despite recent progress, most existing methods infer causal
graphs from purely observational time series. However, learn

2374-8486/25/$31.00 ©2025 IEEE
DOI 10.1109/ICDM65498.2025.00056



Fig. 1: Integrating diffusion models into causal discovery addresses key challenges: (a) observational data cannot uniquely
identify causal structures; (b) traditional interventions require
known targets or causal graphs; (c) diffusion enables progressive, prior-free intervention, facilitating causal inference.


ing causal structures solely from observational data remains
fundamentally limited. Under the faithfulness assumption, the
true causal structure is only identifiable up to a Markov Equivalence Class (MEC) [12], where multiple distinct graphs yield
identical observational distributions [10], [11]. As illustrated in
Fig. 1, even structurally different causal graphs can produce
indistinguishable time series when noise terms _ϵ_ _t_ are independent and Gaussian. Encouragingly, this identifiability issue
can be alleviated by incorporating interventional data, which
introduces distribution shifts across different conditions. Such

shifts present both challenges and opportunities for causal
discovery: under interventions, the causal structure becomes
identifiable within a stricter Interventional MEC (I-MEC) [13],

[14]. With sufficiently diverse interventions, even full recovery
of the true causal graph becomes theoretically possible [15].

While some prior studies have explored causal discovery
from interventional time series, they often rely on strong
assumptions—such as known intervention targets [13], [18] or



487



Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


access to ground-truth causal structures to guide the intervention process [17]. However, such assumptions rarely hold in
real-world settings. Interventions are often imperfect and their
targets unknown, as in cases involving environmental shifts
or system malfunctions [17]. Furthermore, real-world time series frequently exhibit heterogeneity: observational segments,
perfect interventions, and imperfect interventions may coexist
within the same dataset. This diversity substantially increases
the complexity of causal discovery under interventional settings. Consequently, successfully deciphering causal structures
within time series that incorporate both observational and
interventional data, especially when interventions are unknown
or imperfect, continues to be a crucial and predominantly
unresolved challenge.


To address these challenges, we propose DiffuGC, a novel
framework for Granger causal discovery that integrates forward diffusive intervene process with reverse causal discovery
mechanism. To the best of our knowledge, DiffuGC is the
first to incorporate Denoising Diffusion Probabilistic Models
(DDPMs) [16] into the context of interventional Granger
causality. By progressively injecting noise in the forward
diffusion process, DiffuGC constructs a natural continuum
from observational time series to imperfect and ultimately
perfect interventions—without requiring any prior knowledge
of intervention targets. To reconstruct causal structures from
such heterogeneous data, we introduce a denoising NoiFormer
with adaptive attention to both short- and long-term causal
dependencies, which disentangles trend and seasonal components to enable accurate reconstruction of causal structures

from interventional data. Throughout the full diffusion trajectory, DiffuGC learns to reconstruct clean causal patterns
from time series subjected to varying degrees of intervention,
ranging from none to complete. This enables DiffuGC to
handle real-world scenarios involving complex, unknown, and
imperfect interventions in a unified and fully data-driven
manner. Code of DiffuGC is avaliable and anonymous at
https://anonymous.4open.science/r/DiffusionGC. In summary,
the main contributions of this paper are as follows:


_•_ We introduce DiffuGC, a novel framework that combines


forward diffusive intervention with reverse causal discov
ery. DiffuGC is the first to incorporate diffusion models
into interventional Granger causal discovery. By leveraging DDPMs, it generates intervention-aware time series
without requiring any prior knowledge, even in complex
scenarios involving both observational and interventional
data, and enables step-wise learning of causal features
throughout the diffusion trajectory.

_•_ We develop NoiFormer, a Transformer-based denoiser

that adaptively models both short- and long-term causal
dependencies. It disentangles trend and seasonal components from noisy sequences, enabling causal graph
reconstruction under varying degrees of intervention.

_•_ Extensive experiments on synthetic, quasi-real, and real
world datasets show that DiffuGC consistently outperforms state-of-the-art baselines under both observational



and interventional conditions.


_•_ We introduce the notion of Causality Acceleration, sug
gesting new directions for efficient and adaptive causal
discovery.


II. R ELATED W ORK


_A. Causal Discovery from Intervention Data_


In parametric analyses, datasets from various distributions,
or domains, are considered interventional data. MC/IB [19]
explored causal graph learning in linear systems using observational data across domains with varying causal coefficients.
DCDI [13] proposed a differentiable causal graph learning
method for static data using score functions to manage perfect, imperfect, and unknown interventions, identifying the
I-MEC. Recent advances integrate large language models
with score-based temporal causal discovery [20] to create
intervention-aware scoring via semantic pattern recognition.
Many intervention-based causal discovery methods depend
on scores, limiting scalability and expressiveness. However,
Granger causality, compatible with neural networks, promises
scalable and dynamic causal inference.


_B. Neural Granger Causal Discovery_


Granger causality [3] identifies causality in time-series: if
past data of variable _A_ improves predictions of variable _B_,
then _A_ Granger causes _B_ . Deep learning integration with
Granger causality is promising for enhancing causal modeling.
NGC [2] introduced a sparse-input MLP and LSTM for
nonlinear Granger causality. eSRUS [9] proposed a recurrent
unit with groupwise regularized inputs. CR-VAE [8] incorporates Granger causality using a causal variational autoencoder.
CUTS [7] created a method for a causal adjacency matrix
in high-dimensional data with sparse regularization. These
methods infer Granger causal graphs but don’t fully leverage
interventional data or solve identifiability issues. Integrating
Granger causality with interventional analysis is mostly limited to IGC [17], which relies on impractical assumptions
and prior causal knowledge, highlighting gaps in identifying
Granger causality in observational data with unknown interventions and hidden mechanisms.


_C. Denoising Diffusion Probabilistic Models_


Denoising diffusion probabilistic models (DDPMs) are innovative generative frameworks [16]. These models gradually
convert noise to structured data through a denoising process
guided by learned reverse transitions. Diffusion-based methods
excel in time series forecasting, capturing uncertainty and
complex temporal dependencies [21], [22]. Recent research
explores integrating DDPMs with causal reasoning by creating
causal variants [24], [25]. However, these studies mainly
simulate rather than discover causal mechanisms. Research on

using diffusion models for causal discovery is limited [26],

[27], and interventional Granger causal discovery with DDPMs
remains unexplored.



488


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


III. P RELIMINARIES


_A. Problem Formualtion_


**Granger Causality Discovering:** Consider a complex dynamical system modeled by a multivariate time series **X** =
_{X_ 1 _, . . ., X_ _d_ _} ∈_ R _[D][×][T]_ with _D_ variables. The system’s dynamics are recorded over _T_ observation points, _t ∈_ (1 _, . . ., T_ ).
The causal relationships between variables are described by
this structural model:


_X_ _i,t_ +1 = _g_ _i_ ( _X_ 1 _,<t_ _, ..., X_ _D,<t_ ) + _ϵ_ _l,n_ (1)


where _X_ _j,<t_ = ( _X_ _j,_ 1 _, . . ., X_ _j,t−_ 1 ) denotes the past of time
series _i_ and _g_ _i_ ( _·_ ) is a function mapping the past of all the _D_
time series to series _i_ .


Time series _j_ is Granger non-causal for time series _i_ if for
all _X_ 1 _,<t_ _, ..., X_ _D,<t_ and all _X_ _j,<t_ _[′]_ _[̸]_ [=] _[ X]_ _[j,<t]_ [:]


_g_ _i_ ( _X_ 1 _,<t_ _, . . ., X_ _j,<t_ _, . . ., X_ _D,<t_ )

(2)

= _g_ _i_ ( _X_ 1 _,<t_ _, . . ., X_ _j,<t_ _[′]_ _[, . . ., X]_ _[D,<t]_ [)]


which implying _g_ _i_ ( _·_ ) does not depend on _X_ _j,<t_ . Often the
Equation 2 is reflected on the parameters of neural networks.
**Causal Graph Construction:** Let _G_ = ( _V, E_ ) be the Granger
causal graph, with _V_ as nodes and _E_ as edges. Set _V_ includes
_D_ dependent time series ( _X_ 1 _, . . ., X_ _d_ ). An edge from node _X_ _i_
to _X_ _j_ exists if: 1) _i ̸_ = _j_ values of _X_ _i_ give unique, significant
predictive information about _X_ _j_ ; and 2) for _i_ = _j_, _X_ _i_ selfcauses _X_ _i_ .


_B. Intervention_


An intervention on a variable involves replacing its conditional density with a new one. We also define the **interven-**
**tional targets**, a set of variables subjected to simultaneous
intervention, and the **interventional family** _I_ := ( _I_ 1 _, . . ., I_ _O_ ),
where _O_ counts the interventions. The observation context,
with no intervention, is denoted by _I_ 1 := _∅_ . The likelihood
for the _k_ th intervention is expressed as:




_[̸]_



Fig. 2: Illustration of the proposed DiffuGC architecture.


_A. Forward Intervention Dynamics: Diffusive Interventions on_

_[̸]_ _Time Series_


Within the framework of the diffusion model, the forward
intervention process _q_ is a Markov process in which Gaussian
noise is gradually added to the time series at each step
until the original data _x_ [0] _∼_ _q_ ( _x_ [0] ) transforms into complete
Gaussian noise _x_ _[N]_, following an increasing variance schedule
_β_ 1 _, . . ., β_ _N_ with _β_ _n_ _∈_ (0 _,_ 1) a variance at diffusion step _n_ .
The approximate posterior _q_ ( _x_ _[<N]_ _|x_ [0] ) is then computed as
follows:




_[̸]_


_q_ ( _x_ _[<N]_ _|x_ [0] ) =




_[̸]_


_N_
�


_n_ =1




_[̸]_


_q_ ( _x_ _[n]_ _|x_ _[n][−]_ [1] ) (4)




_[̸]_


_q_ ( _x_ _[n]_ _|x_ _[n][−]_ [1] ) = _N_ ( _x_ _[n]_ ;




_[̸]_


�1 _−_ _β_ _n_ _x_ _[n][−]_ [1] _, β_ _n_ **I** ) (5)




_[̸]_


where _N_ denotes the total number of steps in the forward
process. The forward process variance _β_ _[n]_ in (2) can be
implemented with the reparameterization trick:




_[̸]_


~~�~~ _β_ _n_ _ϵ,_ (6)




_[̸]_


_x_ _[n]_ =


where _ϵ ∼N_ (0 _, I_ ).




_[̸]_


�1 _−_ _β_ _n_ _x_ _[n][−]_ [1] +




_[̸]_


�

_j∈I_ _o_




_[̸]_


A notable property of the forward process is that using
notation _α_ _n_ = 1 _−_ _β_ _n_ and _[√]_ ~~_α_~~ _n_ = [�] _[n]_ _s_ =1 _[α]_ _[s]_ [, we can sample]

_x_ _[t]_ at any arbitrary time step t in a closed form:


_q_ ( _x_ _[n]_ _|x_ [0] ) = _N_ ( _x_ _[n]_ ; _[√]_ _α_ ¯ _n_ _x_ [0] _,_ (1 _−_ _α_ ¯ _n_ ) **I** ) (7)




_[̸]_


_p_ [(1)] _j_ [(] _[X]_ _[j]_ _[|]_ [PA][(] _[X]_ _[j]_ [))]




_[̸]_


_p_ [(] _j_ _[o]_ [)] [(] _[X]_ _[j]_ _[|]_ [PA][(] _[X]_ _[j]_ [))][ (3)]




_[̸]_


_p_ [(] _[o]_ [)] ( **X** ) :=




_[̸]_


�

_j̸∈I_ _o_




_[̸]_


1 _−_ _α_ ¯ _n_ _ϵ_ (8)




_[̸]_


Interventions are of two types: 1) imperfect (or soft) interventions [23] and 2) perfect (or hard) interventions [13],
which disconnect a node from its parent nodes _j_ _∈_ _I_ _o_
_p_ [(] _j_ _[o]_ [)] [(] _[X]_ _[j]_ _[|]_ [PA][(] _[X]_ _[j]_ [)) =] _[ p]_ [(] _j_ _[o]_ [)] [(] _[X]_ _[j]_ [)][. Typically, intervention methods]

assume known targets, limiting Granger causal discovery when
targets are unknown or cannot be predetermined.


IV. D IFFU GC


In this section, we introduce DiffuGC mainly consists of
DDPMs, in which Granger causal discovery is embedded, as
shown in Fig.2.




_[̸]_


Intermediate noised data points _x_ _[n]_ represent partially intervened samples, which transform into fully intervened time
series after the final step _N_, _x_ _[n]_ . This transition helps the
model capture various intervention intensities, enriching causal
discovery and boosting its ability to identify interventionaware causal relationships.


_B. Reverse Causal Discovery: Denoising on Heterogeneous_
_Time Series_


The DDPMs’ reverse process is a Markov process that
restores data by incrementally removing noise from corrupted




_[̸]_


_x_ _[n]_ = _[√]_ _α_ ¯ _n_ _x_ [0] +




_[̸]_


_√_




_[̸]_


489


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


time series _x_ [0] _x_ _[N]_ . A neural network learns to denoise samples
through this reverse transition _θ_ :


_p_ _θ_ ( _x_ _[n][−]_ [1] _|x_ _[n]_ ) = _N_ ( _x_ _[n][−]_ [1] ; _µ_ _θ_ ( _x_ _[n]_ _, n_ ) _,_ _β_ [˜] _n_ **I** ) (9)


where _µ_ _θ_ ( _x_ _[n]_ _, n_ ) is the mean of reverse distribution obtained
from neural networkInspired by [21], we directly predict the original time _θ_, _β_ [˜] _n_ = [1] _[−]_ 1 _−_ _[α]_ [¯] _α_ _[n]_ ¯ _[−]_ _n_ [1] _[β]_ _[n]_ [.]

series _x_ [0] from each intermediate state _x_ _[n]_ in the forward

diffusion process, enabling the extraction of Granger causal
relationships from interventional data. Specifically, for each
time series _i_, we construct a dedicated fitting function _f_ _θ_ _i_ to
model its causal dependencies under the dynamics of diffusion:


_x_ ˆ [0] _i_ [=] _[ f]_ _[θ]_ _i_ [(] _[x]_ _[n]_ _[, n]_ [)] (10)


_[̸]_



where _x_ _[n,t]_ _i_ is the value of the interventional time series _i_ at

time _t_ and diffusion step _n_ . ( _t_ _s_ _, . . ., t −_ 1) is the prediction
look-back window, with its earlier segment ( _t_ _s_ _, . . ., t_ _e_ ) as
the encoder input and the latter segment ( _t_ _e_ _, . . ., t −_ 1) as
the decoder input. TFA (Temporal Fourier Attention layer)
integrates encoder and decoder outputs using self and crossattention to model trends, seasonality, and residuals in time
series. _w_ ( _[l,n]_ _·_ ) [,] _[ i][ ∈]_ [(1] _[,][ · · ·][, L]_ [)][ indicate the relevant decoder block]

index at diffusion step _n_ .
**Trend Synthesis** . The trend component describes the smooth
underlying mean of the data, which aims to model slowvarying behavior. To produce reasonable trend components,
we use the polynomial regressor to model the trend _V_ _tr_ _[n]_ [as]

follows:


_[̸]_



As the forward interventional process unfolds, the function
_f_ _θ_ _i_ progressively learns from each intermediate diffused state
to approximate the underlying dynamics of time series variable
_i_ . Building upon Eq. 10, Granger causality is ultimately
determined based on the following criterion for all _x_ _[′][n]_ _[̸]_ [=] _[ x]_ _[n]_ [:]



_tr_ [)] _[,]_ _**C**_ = [1 _, c, . . ., c_ _[s]_ ]


_[̸]_



_V_ _tr_ _[n]_ [=]


_[̸]_



_V_ _[n]_


_[̸]_



_D_
�


_[̸]_



( _**C**_ _·_ Linear( _w_ _tr_ _[l,n]_


_[̸]_



_tr_ _[l,n]_ [) +] _[ X]_ _[ l,n]_ _tr_


_[̸]_



_j_ _[̸]_ [=] _[ x]_ _[n]_ _j_




_[̸]_ _j_ [:]



_i_ =1

(14)

_[̸]_ where _X_ _tr_ _[l,n]_ is the mean value of the output of the ith


_·_
decoder block, and ’ ’ denotes tensor multiplication. Here
slow-varying poly space _C_ is the matrix of powers of vector
_c_ = [0 _,_ 1 _,_ 2 _, · · ·, τ −_ 2 _, τ −_ 1] _T/τ_, and _s_ is a small degree to
model low frequency behavior.
**Seasonality & Error Synthesis** . We aim to extract nontrend components from the model input, including seasonal
and error elements, by representing seasonality with Fourier
synthetic layers and bases. We also observe that at larger
diffusion steps, the intervened time series tends to preserve
long-term trends and variations, which correspond to lowfrequency components in the Fourier domain, reflecting longterm causal effects. In contrast, at smaller diffusion steps,
the sequences contain less noise and retain more short-term
fluctuations, associated with high-frequency components and
short-term causal effects. Motivated by this insight, we propose assigning noise-aware frequency weights to high- and
low-frequency channels post-Fourier transform, enabling the
model to adaptively focus on the most informative frequency
components under different noise levels.




_[̸]_


_[n]_ _d_ [)] _[, n]_ [)]




_[̸]_


_f_ _θ_ _i_ (( _x_ _[n]_ 1




_[̸]_


_[n]_ _j_ _[, . . ., x]_ _[n]_ _d_




_[̸]_


_j_ _[, . . ., x]_ _[n]_ _d_




_[̸]_


_[n]_ 1 _[, . . ., x]_ _[n]_




_[̸]_


_[n]_ _d_ [)] _[, n]_ [) =] _[ f]_ _[θ]_ _i_ [((] _[x]_ 1 _[n]_




_[̸]_


_[n]_ 1 _[, . . ., x]_ _[′][n]_




_[̸]_


(11)


_C. Shared Denoing NoiFormer_

_[̸]_



Fig. 3: The network architecture of NoiFormer.


We introduce NoiFormer, a Transformer-based encoderdecoder model, to extract dynamics from noise-corrupted
sequences (Fig. 3). NoiFormer decomposes inputs into trends,
seasonality, and error to better model noisy temporal patterns.
Trends depict smooth changes, seasonality captures periodic
patterns, and errors are the residuals. For Granger causality, the
time series passes through a GC layer for embedding, where
the weight matrix _W_ [0] indicates the causal impact of _D_ on
target _i_ . NoiFormer’s attention layers then model features and
extract causal information.


**h** _[t]_ 1 [=] _[ W]_ [ 0] _[x]_ _[n,]_ [(] _[t]_ _[s]_ _[,...,t]_ _[e]_ [)] [ +] _[ b]_ [0] [ +] _[ PE]_ _[pos]_ (12)




_[̸]_


_A_ [(] _l,n_ _[k]_ [)] [=]




_[̸]_


�� _F_ ( _w_ _seasl,n_ [)] _[k]_




_[̸]_


�� (15)




_[̸]_


Φ [(] _l,n_ _[k]_ [)] [=] _[ ϕ]_




_[̸]_


�




_[̸]_


�




_[̸]_


_F_ ( _w_ _seas_ _[l,n]_ [)] _[k]_




_[̸]_


(16)




_[̸]_


_κ_ _l,n_




_[̸]_


(1) _, · · ·, κ_ [(] _l,n_ _[K]_ [)] [=] _k∈{_ arg TopK 1 _,···,⌊τ/_ 2 _⌋_ +1 _}_




_[̸]_


_{A_ [(] _l,n_ _[k]_ [)] _[}]_ (17)




_[̸]_


_{A_ [(] _[k]_ [)]




_[̸]_


(1) _, · · ·, κ_ _LFl,n_




_[̸]_


(1) _, · · ·, κ_ _LF_




_[̸]_


_κ_ _[LF]_

_l,n_




_[̸]_


_κ_ _[LF]_




_[̸]_


( _K_ _LF_ ) = arg(Φ [(] _l,n_ _[k]_ [)] _[<]_ [ max][(Φ] _[l,n]_ [)] _[/]_ [2)]

_k∈κ_ _l,n_ [(1)] _,···,κ_ [(] _l,n_ _[K]_ [)]

( _K_ _HF_ ) = arg(Φ [(] _l,n_ _[k]_ [)] _[>]_ [ max][(Φ] _[l,n]_ [)] _[/]_ [2)]

_k∈κ_ _l,n_ [(1)] _,···,κ_ [(] _l,n_ _[K]_ [)]




_[̸]_


_κ_ _[HF]_

_l,n_




_[̸]_


(1) _, · · ·, κ_ _HFl,n_




_[̸]_


( _l,n,LFk_ ) _[τc]_ [ + Φ]




_[̸]_


(18)


(19)


(20)




_[̸]_


_S_ _l,n_ _[LF]_ [=]




_[̸]_


_K_ _LF_
�


_k_ =1




_[̸]_


_κ_ [(] _[k]_ [)]




_[̸]_


_l,n,LF_
_l,n_ [cos(2 _πf_ _κ_ ( _k_ )




_[̸]_


_κ_ [(] _[k]_ [)]

_l,n_ _l,n,LF_ )+




_[̸]_


_A_




_[̸]_


_w_ _[l,n]_




_[̸]_


_[l,n]_

( _tr_ ) _[, w]_ ( _[l,n]_




_[̸]_


( _[l,n]_ _seas_ ) [=][ TFA][(] **[h]** _[t]_




_[̸]_


_[t]_ 1 _[, x]_ _[n,]_ [(] _[t]_ _[e]_ _[,...,t][−]_ [1)] [)] (13)




_[̸]_


cos(2 _πf_ [¯] _κ_ ( _l,n,LFk_ ) _[τc]_ [ + ¯Φ]




_[̸]_


_κ_ [(] _[k]_ [)]

_l,n_ _l,n,LF_ )]




_[̸]_


490


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


( _l,n,HFk_ ) _[τc]_ [ + Φ]



_S_ _l,n_ _[HF]_ [=]



_K_ _HF_
�


_k_ =1



_κ_ [(] _[k]_ [)]



_l,n,HF_
_l,n_ [cos(2 _πf_ _κ_ ( _k_ )



_κ_ [(] _[k]_ [)]

_l,n_ _l,n,HF_ )+



_κ_ [(] _[k]_ [)]



_A_



(21)



cos(2 _πf_ [¯] _κ_ ( _l,n,HFk_ ) _[τc]_ [ + ¯Φ]



cos(2 _πf_ [¯] _κ_ ( _k_ )



_κ_ [(] _[k]_ [)]

_l,n_ _l,n,HF_ )]



_κ_ [(] _[k]_ [)]



where the function TopK is utilized to extract the top _K_
amplitudes, where _K_ is considered a hyperparameter. The
terms _A_ [(] _l,n_ _[k]_ [)] [and][ Φ] [(] _l,n_ _[k]_ [)] [denote the amplitude and phase of the]

_k_ -th frequency component after applying the discrete Fourier
transform, _F_ . The symbol _f_ _k_ signifies the Fourier frequency
associated with index _k_, and ( [¯] _·_ ) indicates the conjugate of ( _·_ ).

Avoiding the introduction of additional hyperparameters,
we use _[√]_ ~~_α_~~ ~~¯~~ _n_ as the adaptive noise-aware weighting factor,
enabling the model to dynamically adjust its sensitivity to
the noise level at each diffusion step. Ultimately, the original
signal can be reconstructed using the following equations:



_L_
�


_l_ =1



_S_ _l,n_ _[HF]_ [+] _[R]_



_S_ _[HF]_



Fig. 4: Summary of DiffuGC’s performance across six benchmark datasets


where ( _x_ ) + = max(0 _, x_ ). Training uses two optimization
methods: proximal gradient on input layer weights _W_ [(0)], and
SGD on other parameters.


**Algorithm 1** Proximal gradient descent with line search
algorithm.


**Require:** _λ >_ 0

_m_ = 0, initialize Θ _, W_ [0] (0)
**while** not converged **do**



_x_ ˆ [0] ( _x_ _[n]_ _, n, θ_ ) = _V_ _tr_ _[n]_ [+(1] _[−√][α]_ [¯] _[n]_ [)]



_L_
�


_l_ =1



_S_ _l,n_ _[LF]_ [+(] _[√][α]_ [¯] _[n]_ [)]



_S_ _[LF]_



(22)
where _R_ represents the output from the final decoder block
and can be considered as the combined result of residual

periodicity along with other noise.


_D. Granger Casual Disvocery_


While modeling target variable _i_, NoiFormer learns from
each diffusion step, and the GC layer updates causal weights
from time series _D_ to variable _i_ . We apply a regularization
term on _W_ [0] to the training loss to promote sparsity in the
causal matrix _W_ [0], improving interpretability.



**end for**

**end while**
**return** ( **W** [0] ( _i_ ))



compute _∇L_ _pred_ by BPTT and pdate Θ except _W_ [0] using
SGD.

_i_ = _i_ + 1
determine _γ_ by line search.
**for** _j_ = 1 to _m_ **do**



**W** [0] ( _i_ )�



�



_W_ : [0] _j_ [(] _[i]_ [+1) = soft]



�



_W_ [0]



: [0] _j_ [(] _[i]_ [)] _[ −]_ _[γ][∇]_ _W_ [0]



: [0] _j_ _[L]_ _[pred]_



_, γλ_



�



(ˆ _x_ [0] _i_ _[,t]_ _−_ _f_ _θ_ _i_ ( _x_ _[n,]_ [(] _[t]_ _[e]_ _[,][···][,t][−]_ [1)] _, n_ ) [2] )

� � ~~�~~ �
_L_ _pred_



_∥_ _W_ : [0] _j_ _[∥]_ [2]



_D_


+ _λ_

�

_j_ =1



~~�~~ � ~~�~~ �
_L_ _penalty_



_L_ =



_T_
�


_t_ =2



(ˆ _x_ [0] _i_ _[,t]_



(23)
_1) Optimizing the Penalized Objective:_ We use proximal
gradient descent [33] to optimize the nonconvex objectives of
Eq. 23. This approach is crucial in inducing zeros in input
matrix columns, key for interpreting Granger non-causality.
A line search might be added to the algorithm to ensure
local minimum convergence [34]. The algorithm updates the
network weights _**θ**_ _i_ iteratively starting with _W_ [(0)] by



_W_ [0] ( _i_ + 1) = prox _γ_



_W_ [0] ( _i_ ) _−_ _γ∇L_ _pred_ ( _W_ [0] ( _i_ ))�



_W_ [0] ( _i_ ) _−_ _γ∇L_ _pred_ ( _W_ [0] ( _i_ ))



�



(24)



_2) Model Complexity:_ DiffuGC’s complexity arises from
_d_ -fold forward intervention and backward causal discovery.
Forward diffusion complexity is _O_ ( _Nd_ ) over _N_ steps. In
the backward phase, a NoiFormer module with _L_ layers
processes a time series of length _T_ and hidden dimension
_H_, with complexity _O_ ( _LT_ [2] _H_ ). All _d_ instances execute independently in parallel, achieving a parallel-time complexity of
_O_ ( _Nd_ + _LT_ [2] _H_ ), enabling scalability for large causal graphs.


V. E XPERIMENTS


We evaluate DiffuGC’s performance using two methods: (1)
using only observational time series data, and (2) combining
observational and interventional time series data. Figure 4
summarizes its performance on six benchmark datasets. We
also apply DiffuGC to a real-world causal discovery task,
aligning with established medical research findings and showcasing its practical effectiveness.


_A. Datasets_


We tested our approach on 2 synthetic (VAR, Lorenz-96)
and 4 public quasi-real datasets (fMRI, Air Quality Index,



where prox _γ_ is the proximal operator with step size _γ_ ; _L_ _pred_
is the convex part of _L_ .

The weights’ group lasso penalty proximal step involves a
group soft-thresholding operation [33]:



� +



�



prox _γλ_ ( _W_ : [0] _j_



: [0] _j_ [) =][ soft][(] _[W]_ [ 0] : _j_




[ 0] : _j_ _[, γλ]_ [) =]



�



_λγ_
1 _−_ _||W_ : [0]



: [0] _j_ _[||]_ [2]



_W_ : [0] _j_ [(25)]



_W_ [0]



491


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


Traffic, Medical). The synthetic datasets are observational,
while the quasi-real datasets are interventional due to modifications embedding known causal structures.


_•_ VAR (linear): As a linear dataset, VAR datasets are

simulated following:



_x_ _[t]_ =



_τ_ _max_
�


_τ_ =1



**A** _τ_ _x_ _[t][−][τ]_ + _e_ _t_ _,_ (26)



_C. Implementation Details_


We used the official implementations of baseline methods
with parameter adjustments for our setup. Hyperparameter
search was conducted over batch size _b ∈_ 16 _,_ 32 _,_ 64 _,_ 128,
learning rate _lr ∈_ (5 _×_ 10 _[−]_ [4] _,_ 1 _×_ 10 _[−]_ [4] ), weight coefficient _λ ∈_
(1 _×_ 10 _[−]_ [1] _,_ 1 _×_ 10 _[−]_ [2] ), and proximal step size _γ ∈_ 1 _,_ 5 _,_ 10 _,_ 20.
Diffusion models were tested with a noise level of _N_ = 1000.


_D. Experiment Results_


In this section, we evaluate the effectiveness of DiffuGC in
discovering Granger causalit. We highlight the best and the
second best in bold and with underlining, respectively.
**VAR** . We simulated _D_ = 10 _,_ 20 _,_ 40 time series over _T ∈_
(500 _,_ 1000) observations with a maximum time lag of _τ ∈_
5 _,_ 20. Table I shows DiffuGC’s performance on VAR datasets
across different dimensionalities, lengths, and lags. DiffuGC
consistently achieves the highest AUROC, AUPRC, and F1
scores, with the lowest SHD. Even in challenging conditions
(D=30, Lag=20), DiffuGC outperforms GC and TCDF, which
degrade significantly. Compared to neural methods like NGC,
CR-VAE, and CUTS, DiffuGC shows stronger generalization
and better causal structure recovery. It also demonstrates more
robustness than IGC, with less variability and consistently
lower SHD, confirming its reliability in identifying causal
relations in complex scenarios.
**Lorenz-96** . For this dataset, we simulated _T ∈_ (500 _,_ 1000)
observations with _D ∈_ (20 _,_ 40), _F ∈_ 10 _,_ 20. _F_ is a forcing constant, and higher _F_ values increase chaotic behavior
and nonlinearity. Table II shows DiffuGC’s performance on
the Lorenz-96 system across different dimensions and chaos
levels. DiffuGC consistently outperforms existing methods. It
excels in both low-dimensional, mildly nonlinear and highdimensional, highly chaotic scenarios, showing superior accuracy and stability in structural recovery with low SHD
values. Unlike others, DiffuGC maintains robust performance
under strong chaos, demonstrating strong adaptability and
generalization in complex nonlinear systems.
**fMRI** . The simulated fMRI BOLD signals, generated using
DCM with a nonlinear balloon model, include time series
for various brain ROIs across 28 sub-datasets, each with 50
unique-subject features. This study evaluates all subjects and
simulations, unlike previous limited studies. Table III shows
that DiffuGC excelled in 22 out of 28 simulations, handling
complex scenarios like global mean interference and time lags.
While other baselines excel sporadically, DiffuGC consistently
provides robust, high-quality results, demonstrating its superior generalization and resilience. This confirms DiffuGC’s
practical utility in brain connectivity inference.
**CausalTime** For the new benchmark datasets, we followed the
experimental settings described by [28]. Table IV details the
performance of various methods across three quasi-realistic
domains from the CausalTime benchmark. DiffuGC emerges
as the top performer across all domains and evaluation criteria,
consistently surpassing the baselines. Notably, its superiority
is especially pronounced in complex domains characterized
by distribution shifts and diverse interventions. These findings



where the matrix **A** _τ_ is the sparse autoregressive coefficients for time lag _τ_ . Time-series _i_ Granger cause timeseries _j_ if _∃τ ∈{_ 1 _, ..., τ_ _max_ _}, a_ _τ,ij_ _>_ 0.

_•_ Lorenz-96 (non-linear): As a nonlinear model to simu
late climate dynamics, the p-dimensional Lorenz-96 [30]
model is defined as:

_dxdt_ _[n]_ _i_ [= (] **[x]** _[i]_ [+1] _[ −]_ **[x]** _[i][−]_ [2] [)] **[x]** _[i][−]_ [1] _[ −]_ **[x]** _[i]_ [ +] _[ F]_ (27)


where _x_ _−_ 1 = _x_ _p−_ 1, _x_ [0] = _x_ _p_ and _x_ _p_ +1 = _x_ 1 . _F_ is the
forcing constant which is set to be 10.


_•_ fMRI: fMRI [29] serves as a benchmark for causal dis
covery, comprising realistic simulations of blood-oxygenlevel-dependent (BOLD) time series generated via the
dynamic causal modelling framework for functional magnetic resonance imaging (fMRI) forward model.

_•_ Air Quality Index, Traffic and Medical: The CausalTime

[28] pipeline, serving as a novel benchmark dataset,
produces time series that closely mimic real-world data,
incorporating ground truth causal graphs for quantitative
performance assessment. The CausalTime benchmark encompasses three domains of quasi-real time series: Air
Quality Index, Traffic, and Medical.


_B. Baselines and Metrics_


_1) Baselines:_ **GC** [3] (Vector AutoRegressive) is a linear
model for Granger causality tests. **PCMCI** [6] uses conditional independence tests with optimized conditioning sets for
inferring causal structure. **NGC** [2] implements a componentwise LSTM with sparse input weights to infer non-linear
Granger causality. **DyNoTears** [31] is a score-based method
with continuous optimization for learning causal structure.
**CR-VAE** [8] are specialized RNNs for identifying the structure
of non-linear Granger causal networks. **CUTS** [7] is a neural
Granger causal discovery algorithm for imputed and highdimensional data. **IGC** [17] combines intervention modeling
and Granger causality to robustly infer multivariate TS causal
structures, accommodating distribution shifts or interventions.

_2) Evaluation Metrics:_ We use four standard evaluation
metrics: **AUROC** Area Under the Receiver Operating Characteristic curve shows the area under the true positive rate
vs. false positive rate curve across thresholds; **AUPRC** Area
Under the Precision-Recall Curve analyzes precision vs. recall
at various thresholds; **F1 Score** measures the harmonic mean
of precision and recall; **SHD** Structural Hamming Distance
counts incorrectly predicted edge states.



492


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


TABLE I: Comparisons of AUROC and AUPRC for Granger causality among different approaches on VAR dataset.






|VAR|Metrics|GC|PCMCI|NGC|DyNoTears|CR-VAE|CUTS|IGC|DiffuGC|
|---|---|---|---|---|---|---|---|---|---|
|D=10, T=1000, Lag=5|AUROC (_↑_)<br>AUPRC(_↑_)<br>F1(_↑_)<br>SHD(_↓_)|0.687(±0.071)<br>0.676(±0.015)<br>0.650(±0.010)<br>8(±2)|0.888(±0.035)<br>0.863(±0.015)<br>0.789(±0.025)<br>4(±2)|0.925(±0.011)<br>0.921(±0.010)<br>0.917(±0.018)<br>2(±1)|0.806(±0.020)<br>0.788(±0.006)<br>0.750(±0.015)<br>1(±0)|0.955(±0.001)<br>0.989(±0.015)<br>0.930(±0.009)<br>1(±0)|0.925(±0.007)<br>0.927(±0.010)<br>0.920(±0.005)<br>2(±0)|**1.000(±0.000)**<br>**1.000(±0.000)**<br>**1.000(±0.000)**<br>**0(±0)**|**1.000(±0.000)**<br>**1.000(±0.000)**<br>**1.000(±0.000)**<br>**0(±0)**|
|D=20, T=1000, Lag=5|AUROC (_↑_)<br>AUPRC (_↑_)<br>F1(_↑_)<br>SHD(_↓_)|0.598(±0.020)<br>0.605(±0.014)<br>0.599(±0.031)<br>38(±4)|0.666(±0.020)<br>0.743(±0.017)<br>0.701(±0.028)<br>33(±2)|0.833(±0.025)<br>0.840(±0.015)<br>0.852(±0.017)<br>15(±3)|0.759(±0.020)<br>0.745(±0.015)<br>0.733(±0.015)<br>14(±3)|0.925(±0.015)<br>0.965(±0.018)<br>0.930(±0.024)<br>8(±2)|0.947(±0.010)<br>0.980(±0.032)<br>0.963(±0.015)<br>6(±2)|0.970(±0.019)<br>0.970(±0.020)<br>0.969(±0.017)<br>11(±1)|**0.982**_±_**0.033**<br>**0.989(±0.018)**<br>**0.980(±0.017)**<br>**5(±1)**|
|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|D=20, T=500, Lag=20<br>AUROC (_↑_)<br>0.569(±0.020)<br>0.598(±0.030)<br>0.820(±0.020)<br>0.698(±0.010)<br>0.830(±0.013)<br>0.842(±0.025)<br>0.935(±0.015)<br>**0.971(±0.012)**<br>AUPRC (_↑_)<br>0.588(±0.025)<br>0.587(±0.030)<br>0.810(±0.021)<br>0.675(±0.025)<br>0.835(±0.015)<br>0.837(±0.027)<br>0.925(±0.016)<br>**0.973(±0.006)**<br>F1(_↑_)<br>0.708(±0.018)<br>0.609(±0.045)<br>0.823(±0.005)<br>0.678(±0.002)<br>0.810(±0.025)<br>0.823(±0.015)<br>0.946(±0.012)<br>**0.955(±0.002)**<br>SHD(_↓_)<br>179(±4)<br>165(±12)<br>65(±15)<br>95(±6)<br>99(±5)<br>22(±3)<br>59(±7)<br>**18(±2)**|
|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|D=30, T=1000, Lag=20<br>AUROC (_↑_)<br>0.599(±0.029)<br>0.566(±0.025)<br>0.789(±0.015)<br>0.649(±0.015)<br>0.785(±0.026)<br>0.838(±0.020)<br>0.919(±0.003)<br>**0.945(±0.005)**<br>AUPRC (_↑_)<br>0.578(±0.020)<br>0.589(±0.024)<br>0.813(±0.021)<br>0.643(±0.017)<br>0.845(±0.015)<br>0.837(±0.017)<br>0.902(±0.018)<br>**0.956(±0.010)**<br>F1(_↑_)<br>0.710(±0.010)<br>0.578(±0.032)<br>0.790(±0.010)<br>0.638(±0.022)<br>0.805(±0.015)<br>0.810(±0.029)<br>0.938(±0.009)<br>**0.950(±0.002)**<br>SHD(_↓_)<br>164(±3)<br>158(±10)<br>58(±9)<br>85(±2)<br>83(±10)<br>79(±6)<br>33(±6)<br>**10(±2)**|



TABLE II: Comparisons of AUROC and AUPRC for Granger causality among different approaches on Lorenz-96 dataset.






|Lorenz-96|Metrics|GC|PCMCI|NGC|DyNoTears|CR-VAE|CUTS|IGC|DiffuGC|
|---|---|---|---|---|---|---|---|---|---|
|D=10, T=1000, F=10|AUROC (_↑_)<br>AUPRC(_↑_)<br>F1(_↑_)<br>SHD(_↓_)|0.658(±0.040)<br>0.588(±0.033)<br>0.601(±0.019)<br>17(±2)|0.598(±0.045)<br>0.601(±0.005)<br>0.565(±0.018)<br>15(±1)|0.938(±0.001)<br>0.945(±0.008)<br>0.930(±0.001)<br>6(±0)|0.765(±0.020)<br>0.785(±0.015)<br>0.734(±0.016)<br>10(±1)|0.928(±0.040)<br>0.922(±0.005)<br>0.922(±0.027)<br>7(±1)|0.928(±0.030)<br>0.916(±0.017)<br>0.905(±0.006)<br>2(±0)|**1.000(±0.000)**<br>**1.000(±0.000)**<br>**1.000(±0.000)**<br>7(±0)|**1.000(±0.000)**<br>**1.000(±0.000)**<br>**1.000(±0.000)**<br>**3(±0)**|
|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|D=20, T=1000, F=10<br>AUROC (_↑_)<br>0.573(±0.026)<br>0.608(±0.022)<br>0.862(±0.018)<br>0.713(±0.020)<br>0.923(±0.013)<br>0.850(±0.020)<br>0.934(±0.018)<br>**0.979**_±_**0.033**<br>AUPRC (_↑_)<br>0.560(±0.014)<br>0.634(±0.015)<br>0.875(±0.009)<br>0.715(±0.028)<br>0.893(±0.020)<br>0.867(±0.021)<br>0.946(±0.015)<br>**0.984(±0.018)**<br>F1(_↑_)<br>0.686(±0.024)<br>0.635(±0.010)<br>0.873(±0.021)<br>0.728(±0.022)<br>0.903(±0.006)<br>0.822(±0.019)<br>0.931(±0.011)<br>**0.980(±0.017)**<br>SHD(_↓_)<br>48(±2)<br>42(±2)<br>12(±2)<br>29(±3)<br>9(±1)<br>14(±3)<br>10(±2)<br>**8(±1)**|
|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|D=20, T=500, F=20<br>AUROC (_↑_)<br>0.540(±0.018)<br>0.575(±0.015)<br>0.775(±0.016)<br>0.656(±0.023)<br>0.853(±0.020)<br>0.813(±0.038)<br>0.903(±0.020)<br>**0.943(±0.008)**<br>AUPRC (_↑_)<br>0.568(±0.010)<br>0.586(±0.010)<br>0.780(±0.014)<br>0.665(±0.012)<br>0.867(±0.018)<br>0.862(±0.017)<br>0.925(±0.022))<br>**0.950(±0.015)**<br>F1(_↑_)<br>0.690(±0.018)<br>0.571(±0.012)<br>0.770(±0.010)<br>0.725(±0.013)<br>0.565(±0.017)<br>0.810(±0.026)<br>0.915(±0.004)<br>**0.944(±0.006)**<br>SHD(_↓_)<br>197(±3)<br>182(±10)<br>82(±7)<br>141(±5)<br>103(±8)<br>70(±19)<br>78(±8)<br>**23(±3)**|
|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|D=40, T=1000, F=20<br>AUROC (_↑_)<br>0.560(±0.019)<br>0.557(±0.013)<br>0.719(±0.015)<br>0.716(±0.018)<br>0.743(±0.021)<br>0.825(±0.006)<br>0.907(±0.008)<br>**0.932(±0.005)**<br>AUPRC (_↑_)<br>0.556(±0.010)<br>0.543(±0.029)<br>0.766(±0.011)<br>0.687(±0.017)<br>0.809(±0.017)<br>0.829(±0.005)<br>0.909(±0.017)<br>**0.940(±0.016)**<br>F1(_↑_)<br>0.571(±0.015)<br>0.568(±0.042)<br>0.767(±0.029)<br>0.755(±0.010)<br>0.768(±0.024)<br>0.774(±0.017)<br>0.913(±0.002)<br>**0.938(±0.009)**<br>SHD(_↓_)<br>169(±8)<br>170(±12)<br>152(±10)<br>90(±8)<br>91(±9)<br>77(±10)<br>32(±10)<br>**15(±5)**|



demonstrate that DiffuGC effectively addresses real-world
time series challenges, establishing it as a highly robust tool
for interventional Granger causal discovery.


_E. Results on Intervened Synthethic Time Series_


We perform controlled interventions on synthetic VAR and
Lorenz-96 datasets to study their impact on causal discovery
from interventional time series. We focus on the duration

of interventions and the number of variables affected. To

highlight intervention effects, we replace selected variables
with Gaussian noise, which isolates and amplifies their impact
on causal structure accuracy.
**Percentage of intervention time** . We intervene on 5%, 10%,
20%, and 50% of the time series duration to evaluate how
intervention length affects Granger causal discovery. Figure 5
shows that as intervention increases from 5% to 50%, most
baseline methods suffer significant performance decline in AUROC and AUPRC due to noise sensitivity. In contrast, DiffuGC
consistently achieves high performance and robustly handles
various intervention durations in both linear and nonlinear set
tings, demonstrating its resilience to prolonged interventions
and capability in heterogeneous TS environments.
**Percentage of intervention variables** . We randomly intervene
on 10%, 20%, 50%, 100% of the time series to analyze



how the proportion of intervention variables affects Granger
causal discovery under interventional settings. Figure 6 shows
the performance of various causal discovery methods under
increasing percentages of intervention variables. As the proportion of intervened variables grows from 10% to 100%,
most baseline methods suffer noticeable declines in F1 Score

and increases in SHD, indicating a reduction in precision and
structural accuracy. In contrast, DiffuGC maintains both high
F1 and consistently low SHD across all settings, demonstrating
its robustness in the presence of large-scale interventions. This
resilience highlights DiffuGC’s ability to integrate heterogeneous interventional signals without being misled by extensive
structural perturbations, which often degrade the performance
of other approaches.


_F. Results on Real-World fMRI Dataset_


We validate our model by comparing it to alternative methods using a real resting-state fMRI [1] dataset from the enhanced
NKI Rockland sample [37] (Fig. 7). The dataset offers voxellevel BOLD signals, grouped into regions of interest (ROIs)
via anatomical parcellation. We examine seven ROIs: posterior
and anterior cingulate cortexes, middle temporal and angular
gyri in both hemispheres (PCC, LACC, LMTG, LAG, RACC,


1 https://nilearn.github.io/stable/index.html



493


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


TABLE III: AUROC of the fMRI bold signals dataset


AUROC
Subject

GC NGC PCMCI DyNoTears CR-VAE CUTS IGC **DiffuGC**


Sim1 0.579 _±_ 0.045 0.715 _±_ 0.032 0.749 _±_ 0.053 0.18 _±_ 0.046 0.801 _±_ 0.048 0.813 _±_ 0.046 **0.818** _±_ 0.032 0.803 _±_ 0.064

Sim2 0.630 _±_ 0.038 0.739 _±_ 0.042 0.709 _±_ 0.032 0.681 _±_ 0.028 0.831 _±_ 0.026 0.848 _±_ 0.034 0.826 _±_ 0.038 **0.846** _±_ 0.036

Sim3 0.516 _±_ 0.027 0.724 _±_ 0.056 0.728 _±_ 0.048 0.698 _±_ 0.034 0.848 _±_ 0.021 0.850 _±_ 0.026 0.825 _±_ 0.037 **0.868** _±_ 0.027

Sim4 0.602 _±_ 0.047 0.708 _±_ 0.025 0.727 _±_ 0.047 0.676 _±_ 0.042 0.849 _±_ 0.015 0.864 _±_ 0.028 0.868 _±_ 0.020 **0.901** _±_ 0.016

Sim5 0.585 _±_ 0.035 0.747 _±_ 0.035 0.729 _±_ 0.032 0.785 _±_ 0.026 0.828 _±_ 0.030 0.845 _±_ 0.042 0.839 _±_ 0.045 **0.855** _±_ 0.037

Sim6 0.611 _±_ 0.092 0.766 _±_ 0.027 0.809 _±_ 0.033 0.829 _±_ 0.037 0.877 _±_ 0.036 0.898 _±_ 0.034 0.888 _±_ 0.028 **0.907** _±_ 0.025

Sim7 0.617 _±_ 0.043 0.775 _±_ 0.045 0.740 _±_ 0.028 0.795 _±_ 0.038 0.837 _±_ 0.036 0.848 _±_ 0.047 0.831 _±_ 0.037 **0.895** _±_ 0.038

Sim8 0.518 _±_ 0.090 0.602 _±_ 0.087 0.628 _±_ 0.066 0.597 _±_ 0.113 0.613 _±_ 0.083 0.677 _±_ 0.065 0.702 _±_ 0.055 **0.753** _±_ 0.067

Sim9 0.601 _±_ 0.085 0.701 _±_ 0.043 0.706 _±_ 0.066 0.708 _±_ 0.080 0.732 _±_ 0.077 0.805 _±_ 0.062 0.802 _±_ 0.068 **0.811** _±_ 0.077

Sim10 0.640 _±_ 0.080 0.665 _±_ 0.109 0.680 _±_ 0.072 0.723 _±_ 0.120 0.672 _±_ 0.070 **0.781** _±_ 0.065 0.770 _±_ 0.085 0.772 _±_ 0.070

Sim11 0.608 _±_ 0.037 0.725 _±_ 0.038 0.738 _±_ 0.027 0.764 _±_ 0.037 0.802 _±_ 0.028 0.808 _±_ 0.023 0.815 _±_ 0.026 **0.821** _±_ 0.024

Sim12 0.545 _±_ 0.033 0.737 _±_ 0.037 0.726 _±_ 0.040 0.777 _±_ 0.031 0.788 _±_ 0.046 0.805 _±_ 0.040 0.822 _±_ 0.042 **0.850** _±_ 0.030

Sim13 0.580 _±_ 0.042 0.648 _±_ 0.081 0.670 _±_ 0.087 0.676 _±_ 0.081 0.680 _±_ 0.090 0.704 _±_ 0.066 0.728 _±_ 0.070 **0.751** _±_ 0.074

Sim14 0.539 _±_ 0.067 0.681 _±_ 0.070 0.664 _±_ 0.083 0.708 _±_ 0.070 0.735 _±_ 0.062 0.750 _±_ 0.071 0.755 _±_ 0.060 **0.792** _±_ 0.072

Sim15 0.633 _±_ 0.099 0.600 _±_ 0.088 0.599 _±_ 0.075 0.655 _±_ 0.065 0.688 _±_ 0.085 0.726 _±_ 0.080 **0.764** _±_ 0.091 0.740 _±_ 0.075

Sim16 0.603 _±_ 0.122 0.644 _±_ 0.086 0.625 _±_ 0.080 0.633 _±_ 0.086 0.622 _±_ 0.118 0.711 _±_ 0.080 0.700 _±_ 0.118 **0.744** _±_ 0.088

Sim17 0.578 _±_ 0.053 0.703 _±_ 0.042 0.687 _±_ 0.043 0.768 _±_ 0.038 0.781 _±_ 0.040 0.839 _±_ 0.030 0.855 _±_ 0.044 **0.878** _±_ 0.032

Sim18 0.555 _±_ 0.076 0.674 _±_ 0.052 0.680 _±_ 0.062 0.736 _±_ 0.050 0.743 _±_ 0.057 0.818 _±_ 0.051 **0.821** _±_ 0.052 0.807 _±_ 0.062

Sim19 0.566 _±_ 0.041 0.788 _±_ 0.050 0.722 _±_ 0.061 0.819 _±_ 0.042 0.833 _±_ 0.040 0.867 _±_ 0.031 0.860 _±_ 0.032 **0.892** _±_ 0.032

Sim20 0.583 _±_ 0.091 0.814 _±_ 0.032 0.750 _±_ 0.057 0.840 _±_ 0.043 0.865 _±_ 0.022 0.902 _±_ 0.032 0.880 _±_ 0.015 **0.909** _±_ 0.037

Sim21 0.563 _±_ 0.082 0.689 _±_ 0.057 0.700 _±_ 0.043 0.688 _±_ 0.049 0.632 _±_ 0.067 0.770 _±_ 0.060 0.758 _±_ 0.055 **0.801** _±_ 0.060

Sim22 0.578 _±_ 0.050 0.703 _±_ 0.057 0.712 _±_ 0.050 0.658 _±_ 0.063 0.654 _±_ 0.058 0.788 _±_ 0.050 0.789 _±_ 0.050 **0.818** _±_ 0.060

Sim23 0.588 _±_ 0.092 0.609 _±_ 0.081 0.614 _±_ 0.082 0.572 _±_ 0.083 0.613 _±_ 0.094 0.635 _±_ 0.086 **0.696** _±_ 0.076 0.657 _±_ 0.077

Sim24 0.530 _±_ 0.120 0.545 _±_ 0.061 0.551 _±_ 0.081 0.521 _±_ 0.078 0.585 _±_ 0.12 **0.605** _±_ 0.072 0.565 _±_ 0.073 0.580 _±_ 0.092

Sim25 0.537 _±_ 0.050 0.624 _±_ 0.066 0.638 _±_ 0.058 0.600 _±_ 0.060 0.637 _±_ 0.040 0.700 _±_ 0.062 0.713 _±_ 0.053 **0.757** _±_ 0.058

Sim26 0.579 _±_ 0.089 0.601 _±_ 0.055 0.620 _±_ 0.077 0.585 _±_ 0.053 0.619 _±_ 0.049 0.677 _±_ 0.089 0.695 _±_ 0.072 **0.708** _±_ 0.093

Sim27 0.626 _±_ 0.062 0.638 _±_ 0.090 0.682 _±_ 0.062 0.607 _±_ 0.071 0.639 _±_ 0.080 0.699 _±_ 0.069 0.713 _±_ 0.061 **0.745** _±_ 0.082

Sim28 0.644 _±_ 0.052 0.700 _±_ 0.050 0.728 _±_ 0.045 0.631 _±_ 0.040 0.732 _±_ 0.072 0.755 _±_ 0.083 0.760 _±_ 0.060 **0.810** _±_ 0.072


TABLE IV: Comparative performance on CausalTime benchmark datasets.


|Methods|Metrics|AQI|Traffci|Medical|
|---|---|---|---|---|
|GC|AUROC<br>AUPRC|0.447_±_0.014<br>0.645_±_0.027|0.451_±_0.026<br>0.278_±_0.022|0.531_±_0.041<br>0.421_±_0.028|
|CUTS|AUROC<br>AUPRC|0.624_±_0.0434<br>0.553_±_0.036|0.618_±_0.025<br>0.352_±_0.022|0.368_±_0.021<br>0.353_±_0.014|
|NGC|AUROC<br>AUPRC|0.717_±_0.019<br>0.717_±_0.020|0.603_±_0.025<br>0.358_±_0.049|0.574_±_0.014<br>0.463_±_0.012|
|CR-VAE|AUROC<br>AUPRC|0.504_±_0.025<br>0.338_±_0.030|0.524_±_0.041<br>0.421_±_0.025|0.524_±_0.013<br>0.490_±_0.021|
|PCMCI|AUROC<br>AUPRC|0.426_±_0.044<br>0.669_±_0.015|0.536_±_0.016<br>0.392_±_0.022|0.665_±_0.021<br>0.548_±_0.024|
|IGC<br>AUROC<br>0.784_±_0.021<br>0.619_±_0.045<br>0.739_±_0.012<br>AUPRC<br>0.754_±_0.012<br>0.564_±_0.017<br>0.542_±_0.050|IGC<br>AUROC<br>0.784_±_0.021<br>0.619_±_0.045<br>0.739_±_0.012<br>AUPRC<br>0.754_±_0.012<br>0.564_±_0.017<br>0.542_±_0.050|IGC<br>AUROC<br>0.784_±_0.021<br>0.619_±_0.045<br>0.739_±_0.012<br>AUPRC<br>0.754_±_0.012<br>0.564_±_0.017<br>0.542_±_0.050|IGC<br>AUROC<br>0.784_±_0.021<br>0.619_±_0.045<br>0.739_±_0.012<br>AUPRC<br>0.754_±_0.012<br>0.564_±_0.017<br>0.542_±_0.050|IGC<br>AUROC<br>0.784_±_0.021<br>0.619_±_0.045<br>0.739_±_0.012<br>AUPRC<br>0.754_±_0.012<br>0.564_±_0.017<br>0.542_±_0.050|
|**DiffuGC**<br>AUROC<br>**0.805**_±_**0.014**<br>**0.708**_±_**0.010**<br>**0.772**_±_**0.011**<br>AUPRC<br>**0.786**_±_**0.024**<br>**0.607**_±_**0.023**<br>**0.656**_±_**0.016**|**DiffuGC**<br>AUROC<br>**0.805**_±_**0.014**<br>**0.708**_±_**0.010**<br>**0.772**_±_**0.011**<br>AUPRC<br>**0.786**_±_**0.024**<br>**0.607**_±_**0.023**<br>**0.656**_±_**0.016**|**DiffuGC**<br>AUROC<br>**0.805**_±_**0.014**<br>**0.708**_±_**0.010**<br>**0.772**_±_**0.011**<br>AUPRC<br>**0.786**_±_**0.024**<br>**0.607**_±_**0.023**<br>**0.656**_±_**0.016**|**DiffuGC**<br>AUROC<br>**0.805**_±_**0.014**<br>**0.708**_±_**0.010**<br>**0.772**_±_**0.011**<br>AUPRC<br>**0.786**_±_**0.024**<br>**0.607**_±_**0.023**<br>**0.656**_±_**0.016**|**DiffuGC**<br>AUROC<br>**0.805**_±_**0.014**<br>**0.708**_±_**0.010**<br>**0.772**_±_**0.011**<br>AUPRC<br>**0.786**_±_**0.024**<br>**0.607**_±_**0.023**<br>**0.656**_±_**0.016**|



RMTG, RAG). Some regions correlate during rest. For each
ROI, voxel signals are averaged to form a representative time
series for causal discovery.

According to Fig. 7, DiffuGC identifies the PCC as the
main cortical hub, causally influencing the AG, MTG, and
LACC, which is consistent with existing neuroscientific studies
in [35]. Moreover, the detected causal path MTG _→_ AG aligns
with previously reported interactions in [36]. Compared to the
state-of-the-art baseline IGC for interventional Granger causal



Fig. 5: Comparing causal discovery methods with varied
intervention points (5%, 10%, 20%, 50%) on synthetic linear
(a,b) and nonlinear (c,d) datasets.


discovery, it fails to uncover the core causal role of PCC.


_G. Ablation study_


We compare two intrinsic noise schedules for diffusive
interventions: linear and cosine _β_ _n_ . As shown in Fig. 8(a),
the linear schedule injects stronger noise early, while the



494


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


Fig. 6: Performance of causal discovery methods with varying
percentages of intervention variables on synthetic datasets
using (a) F1 Score and (b) SHD.


Fig. 7: Causal graphs of 6 selected ROIs estimated by our
method (a) and baseline IGC (b) from real-world fMRI dataset.


cosine schedule introduces weaker initial perturbations that
increase gradually. This difference shapes the intervention
trajectories: Figs. 8(b)–(c) show that linear scheduling perturbs
early dynamics more aggressively, whereas cosine scheduling
preserves short-term patterns and applies smoother transitions
across diffusion steps. Performance comparisons on three
datasets are reported in Figs. 8(d)–(e). On synthetic datasets
(VAR and Lorenz-96), both schedules perform comparably,
confirming that DiffuGC remains robust regardless of the
chosen strategy. On the more complex fMRI dataset, however,
the cosine schedule achieves slightly better results, suggesting that its gradual noise injection better preserves temporal
dependencies under high noise.


_H. Case Study–Causality Acceleration_


Our experiments reveal a fascinating phenomenon. Figure 9
shows how causal strength changes through diffusion steps for
both truly causal and non-causal variable pairs. Significantly,
we find that the distinction between causal and non-causal
relationships appears early in the diffusion process—true
causal edges maintain high strength initially and diminish
slowly, while non-causal connections stay weak throughout.
This observation leads us to introduce a novel concept: Causality Acceleration, denoting the speed at which causal signals
become apparent through successive noisy interventions. Essentially, insightful causal patterns aren’t spread uniformly
across the entire diffusion path; instead, they become evident
more quickly during the initial to middle diffusion stages.
This realization suggests new optimization opportunities: by



Fig. 8: Comparison of linear and cosine noise schedules in
diffusive interventions.


Fig. 9: Causality Acceleration—evolution of causal strength
across diffusion steps.


directing computational effort or implementing early stopping
at the most significant steps, we may reduce training costs
while maintaining precision.


VI. C ONCLUSION


In this work, we introduced DiffuGC, a diffusion-based
framework for Granger causal discovery that integrates observational and interventional time series. By coupling progressive diffusive interventions with the Transformer-based
denoiser NoiFormer, DiffuGC reliably reconstructs causal
structures under noise, heterogeneity, and distribution shifts.
Extensive experiments on synthetic, quasi-real, and real-world
datasets demonstrate consistent improvements over state-ofthe-art baselines in accuracy, robustness, and generalization. Moreover, we identified a notable phenomenon, termed
Causality Acceleration, where critical causal signals emerge
early in the diffusion process, providing new insights into
the dynamics of diffusion-based inference. These findings
highlight both the empirical effectiveness and the theoretical
potential of DiffuGC. Overall, our study establishes DiffuGC
as a principled and adaptable framework for reliable causal
discovery in complex time series, and points to promising
future directions in efficient, data-driven causal inference.



495


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


VII. A CKNOWLEDGEMENTS


This work is supported by the National Natural Science
Foundation of China (No.62172018, No.62102008) and CCFZhipu Large Model Innovation Fund (CCF-Zhipu202414).


R EFERENCES


[1] C. K. Assaad, E. Devijver, and E. Gaussier, “Survey and evaluation

of causal discovery methods for time series,” _Journal of Artificial_
_Intelligence Research_, vol. 73, pp. 767–819, 2022.

[2] A. Tank, I. Covert, N. Foti, A. Shojaie, and E. Fox, “Neural granger

causality,” _IEEE Transactions on Pattern Analysis and Machine Intelli-_
_gence,IEEE Transactions on Pattern Analysis and Machine Intelligence_ .

[3] C. W. J. Granger, “Investigating causal relations by econometric models

and cross-spectral methods,” _Econometrica_, vol. 37, no. 3, pp. 424–438,
1969.

[4] S. Shimizu, P. O. Hoyer, A. Hyv¨arinen, A. Kerminen, and M. Jordan,

“A linear non-gaussian acyclic model for causal discovery.” _Journal of_
_Machine Learning Research_, vol. 7, no. 10, 2006.

[5] T. Schreiber, “Measuring information transfer,” _Physical Review Letters_,

vol. 85, no. 2, pp. 461–464, 2000.

[6] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic,

“Detecting and quantifying causal associations in large nonlinear time
series datasets,” _Science Advances_, vol. 5, no. 11, p. eaau4996, 2019.

[7] Yuxiao Cheng, Runzhao Yang, Tingxiong Xiao, Zongren Li, Jinli Suo,

Kunlun He, and Qionghai Dai, “Cuts: Neural causal discovery from
irregular time-series data,” in _Proceedings of the Eleventh International_
_Conference on Learning Representations (ICLR)_, Kigali, Rwanda, May
1–5, 2023.

[8] Hongming Li, Shujian Yu, and Jos´e C. Pr´ıncipe, “Causal recurrent vari
ational autoencoder for medical time series generation,” in _Proceedings_
_of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI)_,
Washington, DC, USA, Feb. 7–14, 2023, pp. 8562–8570.

[9] Saurabh Khanna and Vincent Y. F. Tan, “Economy statistical recurrent

units for inferring nonlinear Granger causality,” in _Proceedings of the 8th_
_International Conference on Learning Representations (ICLR)_, Addis
Ababa, Ethiopia, Apr. 26–30, 2020.

[10] S. Ren and P. Li, “Flow-based perturbation for cause-effect inference,” in

_Proceedings of the 31st ACM International Conference on Information_
_& Knowledge Management_, 2022, pp. 1706–1715.

[11] S. Ren, H. Yin, M. Sun, and P. Li, “Causal discovery with flow-based

conditional density estimation,” in _2021 IEEE International Conference_
_on Data Mining (ICDM)_ . IEEE, 2021, pp. 1300–1305.

[12] T. S. Verma and J. Pearl, “Equivalence and synthesis of causal models,”

in _Probabilistic and Causal Inference: The works of Judea Pearl_, 2022,
pp. 221–236.

[13] P. Brouillard, S. Lachapelle, A. Lacoste, S. Lacoste-Julien, and

A. Drouin, “Differentiable causal discovery from interventional data,”
_Advances in Neural Information Processing Systems_, vol. 33, pp.
21 865–21 877, 2020.

[14] K. Yang, A. Katcoff, and C. Uhler, “Characterizing and learning

equivalence classes of causal dags under interventions,” in _International_
_Conference on Machine Learning_ . PMLR, 2018, pp. 5541–5550.

[15] Frederick Eberhardt, “Almost optimal intervention sets for causal discov
ery,” in _Proceedings of the 24th Conference on Uncertainty in Artificial_
_Intelligence (UAI)_, Helsinki, Finland, July 9–12, 2008, pp. 161–168.

[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion proba
bilistic models,” in _Advances in Neural Information Processing Systems_,
vol. 33, NeurIPS 2020, virtual conference, Dec. 6–12, 2020.

[17] Ziyi Zhang, Shaogang Ren, Xiaoning Qian, and Nick Duffield, “Learn
ing flexible time-windowed Granger causality integrating heterogeneous
interventional time series data,” in _Proceedings of the 30th ACM_
_SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)_,
Barcelona, Spain, Aug. 25–29, 2024, pp. 4408–4418.

[18] Tian Gao, Debarun Bhattacharjya, Elliot Nelson, Miao Liu, and Yue

Yu, “Idyno: Learning nonparametric DAGs from interventional dynamic
data,” in _Proceedings of the International Conference on Machine_
_Learning (ICML)_, Baltimore, Maryland, USA, July 17–23, 2022, vol.
162, pp. 6988–7001.

[19] A. Ghassami, N. Kiyavash, B. Huang, and K. Zhang, “Multi-domain

causal structure learning in linear systems,” in _Advances in Neural_
_Information Processing Systems_, vol. 31, 2018.




[20] Peiwen Li, Xin Wang, Zeyang Zhang, Yuan Meng, Fang Shen, Yue

Li, Jialong Wang, Yang Li, and Wenwu Zhu, “Realtcd: Temporal
causal discovery from interventional data with large language model,” in
_Proceedings of the 33rd ACM International Conference on Information_
_and Knowledge Management (CIKM)_, 2024, pp. 4669–4677.

[21] Xinyu Yuan and Yan Qiao, “Diffusion-ts: Interpretable diffusion for

general time series generation,” _arXiv preprint arXiv:2403.01742_, 2024.

[22] Lifeng Shen and James Kwok, “Non-autoregressive conditional diffusion

models for time series prediction,” in _Proceedings of the International_
_Conference on Machine Learning (ICML)_, 2023, pp. 31016–31029.

[23] J. Peters, D. Janzing, and B. Sch¨olkopf, _Elements of causal inference:_

_foundations and learning algorithms_ . The MIT Press, 2017.

[24] Anirudh Komanduri, Changlong Zhao, Fanbo Chen, and Xiaolin Wu,

“Causal diffusion autoencoders: Toward counterfactual generation via
diffusion probabilistic models,” in _Proceedings of the 27th European_
_Conference on Artificial Intelligence (ECAI)_, 2024, pp. 2516–2523.

[25] Chao Deng, Danyang Zhu, Kaixin Li, Shichao Guang, and Heng Fan,

“Causal diffusion transformers for generative modeling,” _arXiv preprint_
_arXiv:2412.12095_, 2024.

[26] Pedro Sanchez and Sotirios A. Tsaftaris, “Diffusion causal models for

counterfactual estimation,” _arXiv preprint arXiv:2202.10166_, 2022.

[27] Pedro Sanchez, Xiao Liu, Alison Q. O’Neil, and Sotirios A. Tsaftaris,

“Diffusion models for causal discovery via topological ordering,” _arXiv_
_preprint arXiv:2210.06201_, 2022.

[28] Yuxiao Cheng, Ziqian Wang, Tingxiong Xiao, Qin Zhong, Jinli Suo,

and Kunlun He, “Causaltime: Realistically generated time-series for
benchmarking of causal discovery,” _arXiv preprint arXiv:2310.01753_,
2023.

[29] Stephen M. Smith, Karla L. Miller, Gholamreza Salimi-Khorshidi,

Matthew Webster, Christian F. Beckmann, Thomas E. Nichols, Joseph
D. Ramsey, and Mark W. Woolrich, “Network modelling methods for fMRI,” _NeuroImage_, vol. 54, pp. 875–891, 2011. doi:
10.1016/j.neuroimage.2010.08.063.

[30] Edward N. Lorenz, “Predictability: A problem partly solved,” in _Pro-_

_ceedings of the Seminar on Predictability_, vol. 1, 1996.

[31] Roxana Pamfil, Nisara Sriwattanaworachai, Shaan Desai, Philip Pilger
storfer, Konstantinos Georgatzis, Paul Beaumont, and Bryon Aragam,
“Dynotears: Structure learning from time-series data,” in _Proceedings_
_of the International Conference on Artificial Intelligence and Statistics_
_(AISTATS)_, 2020, pp. 1595–1605.

[32] Meike Nauta, Doina Bucur, and Christin Seifert, “Tcdf: Causal dis
covery with attention-based convolutional neural networks,” _Machine_
_Learning and Knowledge Extraction_, vol. 1, pp. 312–340, 2019. doi:
10.3390/make1010019.

[33] N. Parikh, S. Boyd _et al._, “Proximal algorithms,” _Foundations and_

_Trends in Optimization_, vol. 1, no. 3, pp. 127–239, 2014.

[34] Pinghua Gong, Changshui Zhang, Zhaosong Lu, Jianhua Huang, and

Jieping Ye, “A general iterative shrinkage and thresholding algorithm
for non-convex regularized optimization problems,” in _Proceedings of_
_the International Conference on Machine Learning (ICML)_, 2013, pp.
37–45.

[35] Hao Yan, Yu Zhang, Huasheng Chen, Yuhui Wang, and Yijun Liu, “Al
tered effective connectivity of the default mode network in resting-state
amnestic type mild cognitive impairment,” _Journal of the International_
_Neuropsychological Society_, vol. 19, no. 4, pp. 400–409, 2013. doi:
10.1017/S1355617712001580.

[36] Wenbin Guo, Feng Liu, Changqing Xiao, Miaoyu Yu, Zhikun Zhang,

Jianrong Liu, Jian Zhang, and Jingping Zhao, “Increased causal connectivity related to anatomical alterations as potential endophenotypes
for schizophrenia,” _Medicine_, vol. 94, no. 42, p. e1493, 2015. doi:
10.1097/MD.0000000000001493.

[37] Kate Brody Nooner, Stanley J. Colcombe, Russell H. Tobe, Maarten

Mennes, Melissa M. Benedict, Alexis L. Moreno, et al, “The NKIRockland sample: A model for accelerating the pace of discovery
science in psychiatry,” _Frontiers in Neuroscience_, vol. 6, 2012. doi:
10.3389/fnins.2012.00152.

[38] Alexander P. Wu, Rohit Singh, and Bonnie Berger, “Granger causal

inference on DAGs identifies genomic loci regulating transcription,”
in _Proceedings of the Tenth International Conference on Learning_
_Representations (ICLR)_, Virtual Event, Apr. 25–29, 2022.



496


Authorized licensed use limited to: Tsinghua University. Downloaded on March 14,2026 at 07:48:44 UTC from IEEE Xplore. Restrictions apply.


