# Bibliographic notes on earthquake discrimination

Start off with one big file, and split into meaningful sections at some later
date.

## Outlier detection

### Gupta, M. et al., 2014. Outlier Detection for Temporal Data: A Survey.
***Ieee Transactions on Knowledge and Data Engineering, 26(9), pp.2250–2267.***

A common characteristic of all temporal **outlier analysis** is that temporal
continuity plays a key role in all these formulations, and unusual changes,
sequences, or temporal patterns in the data are used in order to model
outliers. In this sense, time forms the contextual variable with respect to
which all analysis is performed. [...] 
In time-series data (e.g., sensor readings) the importance of temporal
continuity is paramount, and all analysis is performed with careful use of
reasonably small windows of time (the contextual variable). [...]
**Specific Challenges for Outlier Detection for Temporal Data**: While
temporal outlier detection aims to find rare and interesting instances, as in
the case of traditional out- lier detection, new challenges arise due to the
nature of temporal data :

1. A wide variety of anomaly models are possible depending upon the specific
data type and scenario. This leads to diverse formulations, which need to
be designed for the specific problem. For arbitrary applications, it may
often not be possible to use off- the-shelf models, because of the wide
variations in problem formulations. This is one of the motivating reasons
for this survey to provide an overview of the most common combinations of
facets explored in temporal outlier analysis.
2. Since new data arrives at every time instant, the scale of the data is very
large. This often leads to processing and resource-constraint challenges. In
the streaming scenario, only a single scan is allowed. Traditional outlier
detection is much easier, since it is typically an offline task.
3. Outlier detection for temporal data in distributed scenarios poses
significant challenges of minimizing communication overhead and computational
load in resource-constrained environments.

**Direct Detection of Outlier Time Series**
Given: A database of time series,Find: All anomalous time series.
It is assumed that most of the time series in the database are normal while a
few are anomalous. Similar to tradi- tional outlier detection, the usual
recipe for solving such problems is to first learn a model based on all the
time series sequences in the database, and then compute an out- lier score
for each sequence with respect to the model. The model could be supervised or
unsupervised depending on the availability of training data.

**Window-Based Detection of Outlier Time Series**
Given: A database of time series,
Find: All anomalous time windows, and hence anomalous time series.
Compared to the techniques in the previous subsection, the test sequence is
broken into multiple overlapping sub- sequences (windows). The anomaly score
is computed for each window, and then the anomaly score (AS) for the entire
test sequence is computed in terms of that of the individual windows.
Window-based techniques can perform better localization of anomalies,
compared to the tech- niques that output the entire time series as outliers
directly. These techniques need the window length as a parameter. Windows
are called fingerprints, pattern fragments, detectors, sliding windows,
motifs, and n-grams in various contexts. In this methodology, the techniques
usually maintain a normal pattern database, but some approaches also
maintain a negative pattern or a mixed pattern database.

### Liu, B. et al., 2014. An Efficient Approach for Outlier Detection with Imperfect Data Labels.
***Ieee Transactions on Knowledge and Data Engineering, 26(7), pp.1602–1616.***

This paper presents a novel outlier detection approach to address data with
imperfect labels and incorporate limited abnormal examples into learning.
[...]
Though much progress has been done in support vector data description for
outlier detection, **most of the existing works on outlier detection always
assume that input training data are perfectly labeled for building the
outlier detection model or classifier**. However, we may collect the data with
imperfect labels due to noise or data of uncertainty.[...]
Therefore, it is necessary to develop outlier detection algorithms to handle
imperfectly labeled data. [...]
Our proposed approaches first capture local data information by
generating likelihood values for input examples, and then incorporate such
information into support vector data description framework to build a more
accurate outlier detection classifier.[...]

**Difference from Imbalanced Data Classification**
The outlier detection problem that we consider in this paper is also related to
the problem of imbalanced data classifi- cation [32], in which outliers
corresponding to the negative class are extremely small in proportion as
compared to the normal data corresponding to the positive class.
We briefly review the research on imbalanced data [32]–[34] as follows. In
general, **previous work on imbalanced data classification** falls into two main
categories. The first category attempts to **modify the class distribution of
training data before applying any learning algorithms** [35]. This is usually
done by over-sampling, which replicates the data in the minority class, or
under-sampling, which throws away part of the data in the majority class. The
second category focuses on **making a particular classifier learner cost
sensitive**, by setting the false positive and false negative costs very
differently and incorporating the cost factors into the learning process [32].
Representative methods include cost-sensitive decision trees [36] and cost-
sensitive SVMs [37]–[40]. **In cost-sensitive SVMs, the cost factors of two
classes are set differently so that the cost factors can affect the decision
boundary.** **When imbalanced data are present, researchers have argued for the
use of ranking-based metrics, such as the ROC curve and the area under ROC
curve (AUC) [41] instead of using accuracy.**
The difference between imbalanced data classification and our outlier detection
problem is that: **in imbalanced data classification, the examples from one or
more minority classes are often self-similar, potentially forming compact
clusters**, while in outlier detection, the outliers are typi- cally scattered
around normal data so that the distribution of the negative class cannot be
well represented by the very few negative training examples. To solve our
problem, we can exploit cost-sensitive learning algorithms, but the false
positive and false negative costs are usually unknown to us in real life
applications. Therefore, we exploit a novel one- class classification method
for outlier detection, which aims at building decision boundary around the
normal data, and utilizes the few negative examples to refine the boundary to
build an outlier detection classifier.
