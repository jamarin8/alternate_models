from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
import math

'''
anomaly probability mapping
'''
def isoforest_xgboost_compare(df, perc_cont, full_labels, fm_scores, adjustment='dominance'):
    iforest = IsolationForest(n_estimators=max(240, min(500, 100 * np.log10(df.shape[0]) // 2)),
                              max_samples='auto',
                              contamination=perc_cont,
                              max_features=len(features) - 1, bootstrap=False, n_jobs=-1, random_state=1)
    '''
    the indexing below takes the full_labels dataframe with its time indexing and 
    joins it to the time segmented selected by the user or the 30-60-90
    timedict_normalized['60'].join(fraud_column).join(fm_5)
    '''
    df = df.join(full_labels).join(fm_scores)
    labels = df.fraud_specified
    fm_5_scores = df.fm_5_score
    '''
    non-critical features dropped for purposes of isolation forest calculation which only takes numeric features
    '''
    df_ = df.drop(['fraud_specified', 'fm_5_score'], axis=1)

    iso_pred = iforest.fit_predict(df_)
    iso_pred = np.where(iso_pred == -1, 1, 0)
    '''
    iforest anomaly scores converted to probabilities according to
    https://stackoverflow.com/questions/67357849/conversion-of-isolationforest-decision-score-to-probability-algorithm
    https://introcs.cs.princeton.edu/java/21function/ErrorFunction.java.html
    '''

    def erf(x):
        # save the sign of x
        sign = 1 if x >= 0 else -1
        x = abs(x)

        # constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return sign * y  # erf(-x) = -erf(x)

    decision_scores = iforest.decision_function(df_)
    probs = np.zeros([df_.shape[0], 2])
    pre_erf_score = (decision_scores - np.mean(decision_scores)) / (np.std(decision_scores) * np.sqrt(2))
    erf_score = np.array([math.erf(x) for x in pre_erf_score])
    probs[:, 1] = erf_score.clip(0, 1).ravel()
    probs[:, 0] = 1 - probs[:, 1]

    '''
    based on score alignment generated in Fraud Model 5.0.0 PBA, page 11 by Nils Carlsson
    ## 1.22643393399164 * score + 0.00197736842965041
    partner preference generally at 0.05 except one partner at 0.15
    '''
    fm_5_scores = (fm_5_scores - 0.00197736842965041) / 1.22643393399164
    fm_pred = fm_5_scores > 0.05
    augmented_pred = (fm_pred | iso_pred)

    precision, recall, _ = precision_recall_curve(labels, fm_5_scores)
    pr_auc = auc(recall, precision)

    if adjustment == 'dominance':
        assert sum(augmented_pred) - sum(fm_pred) == sum(
            np.logical_xor(augmented_pred, fm_pred)), 'number of anomalies added do not add up to expected sum'
        adjusted_fm5_scores = np.where(np.logical_xor(augmented_pred, fm_pred), probs[:, 1], fm_5_scores)

    if adjustment == 'logit_blend':
        clf = LogisticRegression(random_state=42)
        clf.fit(X=np.c_[fm_5_scores, probs[:, 1]], y=labels)
        adjusted_fm5_scores = clf.predict_proba(np.c_[fm_5_scores, probs[:, 1]])
        adjusted_fm5_scores = adjusted_fm5_scores[:, 1]

    precision_adj, recall_adj, _ = precision_recall_curve(labels, adjusted_fm5_scores)
    pr_auc_adj = auc(recall_adj, precision_adj)

    plt.figure(figsize=(12, 5))
    plt.plot(recall, precision, lw=3, alpha=0.4, c='r', label=f'AUCPR {pr_auc:0.5f}')
    plt.plot(recall_adj, precision_adj, lw=3, alpha=0.6, c='b', label=f'AUCPR ADJ {pr_auc_adj:0.5f}')
    plt.legend(prop={'size': 13}, loc=0)
    plt.savefig("./AUC PRECISION RECALL CURVE.png")
    plt.close()

    out = {'fm_5_only_pred': fm_pred, 'fm_5_augmented_pred': augmented_pred,
           'fm_5_only_precision': ((fm_pred == 1) & (labels == 1)).sum() / ((fm_pred == 1)).sum(),
           'augmented_precision': ((augmented_pred == 1) & (labels == 1)).sum() / ((augmented_pred == 1)).sum(),
           'fm_5_only_recall': ((fm_pred == 1) & (labels == 1)).sum() / (labels == 1).sum(),
           'augmented_recall': ((augmented_pred == 1) & (labels == 1)).sum() / (labels == 1).sum(),
           'pr_auc': pr_auc,
           'pr_auc_adj': pr_auc_adj
           }

    return out['augmented_recall'] - out['fm_5_only_recall']