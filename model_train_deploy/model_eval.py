
import pandas as pd
import numpy as np
from numpy import interp
# evaluation imports
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, f1_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, \
    precision_recall_curve
import os
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
import scikitplot as skplt
import shap


class ModelEvaluation():

    def __init__(self):
        pass

    def cv_partner_performance(self, classifier, cv, X, y, X_raw, weights=None):

        partner_perf_list = []

        i = 0
        for train, test in cv.split(X, y):
            if weights is None:
                fold_model = classifier.fit(X.iloc[train], y.iloc[train])
            else:
                model_weights = weights[train]
                fold_model = classifier.fit(X.iloc[train], y.iloc[train], sample_weight=model_weights)

            probas_ = fold_model.predict_proba(X.iloc[test])
            y_score = probas_[:, 1]
            y_test = y.iloc[test]

            overall_scores_df = pd.DataFrame(
                {'ca_id': X_raw[['ca_id']].iloc[test].values.flatten(),
                 'partner': X_raw[['partner']].iloc[test].values.flatten(),
                 'yscore': y_score,
                 'ytrue': y_test
                 })

            ovl_auc_pr = average_precision_score(y_true=y_test, y_score=y_score)
            ovl_auc_roc = roc_auc_score(y_true=y_test, y_score=y_score)
            partner_scores = overall_scores_df.groupby('partner', as_index=False).apply(
                self.model_evaluator).reset_index()
            partner_scores['cv_round'] = i
            partner_scores['ovl_auc_pr'] = ovl_auc_pr
            partner_scores['ovl_auc_roc'] = ovl_auc_roc

            partner_perf_list.append(
                partner_scores[['cv_round', 'partner', 'auc_pr', 'auc_roc', 'ovl_auc_pr', 'ovl_auc_roc']])

            i += 1

        partner_perf_df = pd.concat(partner_perf_list)
        return partner_perf_df

    def model_evaluator(self, dframe, ytrue='ytrue', scores='yscore'):

        total_apps = len(dframe.index)
        actual_fraud = dframe[ytrue].sum()
        auc_roc = roc_auc_score(y_true=dframe[ytrue], y_score=dframe[scores])
        auc_pr = average_precision_score(y_true=dframe[ytrue], y_score=dframe[scores])

        return pd.Series({'total_applications': total_apps, 'actual_fraud': actual_fraud,
                          'auc_roc': auc_roc, 'auc_pr': auc_pr})

    def threshold_evaluator(self, threshold, ytrue, scores):

        ypred = np.where(100 * scores > threshold, 1, 0)
        # using scores to create threshold
        # false positives, true negatives for false positive rate
        true_positives = (ytrue * ypred).sum()
        false_positives = ((1 - ytrue) * ypred).sum()
        false_negatives = (ytrue * (1 - ypred)).sum()
        true_negatives = ((1 - ytrue) * (1 - ypred)).sum()

        # calculating multiple metrics
        total_apps = len(ytrue)
        actual_fraud = ytrue.sum()
        predicted_fraud = ypred.sum()
        precision = precision_score(y_true=ytrue, y_pred=ypred, pos_label=1, zero_division=0)
        recall = recall_score(y_true=ytrue, y_pred=ypred, pos_label=1, zero_division=0)
        false_positive_rate = false_positives / (false_positives + true_negatives)
        fraud_missed_rate = false_negatives / (false_negatives + true_positives)
        net_effect_fraud = true_positives - (0.05 * false_positives) - false_negatives
        f1score = f1_score(y_true=ytrue, y_pred=ypred, pos_label=1)

        return pd.Series({'threshold': threshold, 'total_applications': total_apps, 'actual_fraud': actual_fraud,
                          'predicted_fraud': predicted_fraud,
                          'precision': precision, 'recall': recall, 'f1score': f1score,
                          'false_positive_rate': false_positive_rate, 'fraud_missed_rate': fraud_missed_rate,
                          'net_effect_fraud': net_effect_fraud})

    def create_threshold_evaluation_df(self, fit_model, prediction_data, raw_data):
        scores_df = self.create_scoresdf(fit_model, prediction_data, raw_data)
        counter = range(0, 101)
        test_list = []
        for i in counter:
            test_list.append(
                self.threshold_evaluator(ytrue=scores_df['ytrue'], scores=scores_df['xgb_scores'], threshold=i))

        thresholds_output = pd.DataFrame(test_list)
        return thresholds_output

    def get_rfe_results(self, X_train, y_train, model, results_path, cv_obj, step=5):
        # parallel
        features = list(X_train)
        num_features = []
        model_perf_cv_mean = []
        model_perf_cv_std = []
        logs_dir = os.path.abspath(os.path.join(results_path))

        while len(features) > step:
            train_df = X_train[features]
            total_features = len(list(train_df))
            print("Tuning with {} features now".format(total_features))

            # tuning model
            cv_results = cross_validate(model, train_df, y_train, cv=cv_obj, scoring=('average_precision', 'roc_auc'),
                                        return_estimator=True)

            # collecting cv results
            cv_auc_pr = []
            roc_auc_roc = []
            for trial_run in range(len(cv_results['test_average_precision'])):
                cv_auc_pr.append(cv_results['test_average_precision'][trial_run])
                roc_auc_roc.append(cv_results['test_roc_auc'][trial_run])

            fi_list = []
            for idx, estimator in enumerate(cv_results['estimator']):
                feature_importances = pd.DataFrame([estimator.feature_importances_], columns=list(train_df))
                fi_list.append(feature_importances)

            # appending results
            num_features.append(total_features)
            model_perf_cv_mean.append(cv_results['test_average_precision'].mean())
            model_perf_cv_std.append(statistics.stdev(cv_auc_pr))

            if total_features < 200:
                step = 5
            else:
                pass

            # removing the bottom 'step' features by FI value
            varimp_df = pd.concat(fi_list)
            feature_list_path = os.path.join(logs_dir, "{}_features_list.csv".format(total_features))
            varimp_scores_df = varimp_df.mean(axis=0).to_frame().sort_values(by=0, ascending=False)
            varimp_scores_df.to_csv(feature_list_path)
            varimp_scores_df.drop(varimp_scores_df.tail(step).index, inplace=True)
            features = list(varimp_scores_df.index)

        scores_per_var_group = pd.DataFrame()
        scores_per_var_group['num_features'] = num_features
        scores_per_var_group['auc_pr_cv_mean'] = model_perf_cv_mean
        scores_per_var_group['auc_pr_cv_std'] = model_perf_cv_std

        return scores_per_var_group

    def get_rfe_results_shap(self, X, y, model, results_path, cv_obj, step=100, step_small=10):
        # parallel
        features = list(X)
        num_features = []
        model_perf_cv_mean = []
        model_perf_cv_std = []
        logs_dir = os.path.abspath(os.path.join(results_path))

        while len(features) > step:
            train_df = X[features]
            total_features = len(list(train_df))
            print("Tuning with {} features now".format(total_features))
            cv_auc_pr = []
            roc_auc_roc = []
            fi_list = []
            # tuning model
            for train, test in cv_obj.split(train_df, y):
                Xfold_train = train_df.iloc[train]
                yfold_train = y.iloc[train]
                Xfold_test = train_df.iloc[test]
                yfold_test = y.iloc[test]
                fold_model = model.fit(Xfold_train, yfold_train)

                probas_ = fold_model.predict_proba(Xfold_test)
                y_score = probas_[:, 1]
                auc_pr = average_precision_score(y_true=yfold_test, y_score=y_score)
                auc_roc = roc_auc_score(y_true=yfold_test, y_score=y_score)

                cv_auc_pr.append(auc_pr)
                roc_auc_roc.append(auc_roc)

                # shap on training set
                explainerXGB = shap.TreeExplainer(model=fold_model, data=Xfold_train, feature_dependence="independent",
                                                  model_output="probability")
                shap_values_XGB_test = explainerXGB(Xfold_test)
                avg_shap_values = shap_values_XGB_test.abs.mean(0).values
                feature_importances = pd.DataFrame([avg_shap_values], columns=list(train_df))
                fi_list.append(feature_importances)

            # appending results
            num_features.append(total_features)
            model_perf_cv_mean.append(statistics.mean(cv_auc_pr))
            model_perf_cv_std.append(statistics.stdev(cv_auc_pr))

            if total_features < 200:
                step = step_small
            else:
                pass

            # removing the bottom 'step' features by FI value
            varimp_df = pd.concat(fi_list)
            feature_list_path = os.path.join(logs_dir, "{}_features_list.csv".format(total_features))
            varimp_scores_df = varimp_df.mean(axis=0).to_frame().sort_values(by=0, ascending=False)
            varimp_scores_df.to_csv(feature_list_path)
            varimp_scores_df.drop(varimp_scores_df.tail(step).index, inplace=True)
            features = list(varimp_scores_df.index)

        scores_per_var_group = pd.DataFrame()
        scores_per_var_group['num_features'] = num_features
        scores_per_var_group['auc_pr_cv_mean'] = model_perf_cv_mean
        scores_per_var_group['auc_pr_cv_std'] = model_perf_cv_std

        return scores_per_var_group

    def draw_cv_roc_curve(self, classifier, cv, X, y, title='ROC Curve'):
        """
        Draw a Cross Validated ROC Curve.
        Keyword Args:
            classifier: Classifier Object
            cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
            X: Feature Pandas DataFrame
            y: Response Pandas Series
        Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        """

        fig = plt.figure()
        # Creating ROC Curve with Cross Validation
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc=(1.04, 0))
        return fig

    def draw_cv_pr_curve(self, classifier, cv, X, y, title='PR Curve'):
        """
        Draw a Cross Validated PR Curve.
        Keyword Args:
            classifier: Classifier Object
            cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
            X: Feature Pandas DataFrame
            y: Response Pandas Series

        Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
        """

        fig = plt.figure()
        y_real = []
        y_proba = []

        i = 0
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
            # Compute ROC curve and area the curve
            precision, recall, _ = precision_recall_curve(y.iloc[test], probas_[:, 1])

            # Plotting each individual PR Curve
            plt.plot(recall, precision, lw=1, alpha=0.3,
                     label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

            y_real.append(y.iloc[test])
            y_proba.append(probas_[:, 1])

            i += 1

        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)

        precision, recall, _ = precision_recall_curve(y_real, y_proba)

        plt.plot(recall, precision, color='b',
                 label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
                 lw=2, alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc=(1.04, 0))
        return fig
        # plt.show()

    def plot_training_curve(self, model, X, y, evalset):
        model.fit(X, y, eval_metric='logloss', eval_set=evalset, verbose=False)
        # plot learning curves
        fig = plt.figure()
        results = model.evals_result()
        fig, ax = plt.subplots()
        plt.plot(results['validation_0']['logloss'], label='train')
        plt.plot(results['validation_1']['logloss'], label='test')
        fig.suptitle('Learning Curves for Tuned xG Boost')
        ax.set_xlabel('Boosting Rounds')
        ax.set_ylabel('Logloss error')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        return fig

    def create_scoresdf(self, fit_model, prediction_data, raw_data, **kwargs):
        probas_ = fit_model.predict_proba(prediction_data)
        plots_ytrue = raw_data['dep_var'].copy()
        plots_yscore = probas_[:, 1]

        ids = raw_data['ca_id'].copy()

        scores_df = pd.DataFrame(
            {'ca_id': ids,
             'xgb_scores': plots_yscore,
             'ytrue': plots_ytrue
             })
        scores_df['unscaled_model5_scores'] = (raw_data['fm_5_score'] - 0.00197736842965041) / 1.22643393399164

        for key in kwargs:
            scores_df[key] = raw_data[kwargs[key]].copy()

        return scores_df

    def create_model_comparison_curve(self, df, ytrue, **kwargs):
        avg_rate = df[ytrue].sum() / len(df[ytrue])
        ns_probs = [avg_rate] * len(df[ytrue])
        ns_auc = average_precision_score(df[ytrue], ns_probs)
        fig = plt.figure()
        plt.plot([0, 1], [avg_rate, avg_rate], linestyle='--', label='No Skill (area = %0.3f)' % ns_auc)
        for key in kwargs:
            plot_df = df[[ytrue, key]]
            plot_df.dropna(inplace=True)
            plotlabel = kwargs[key]
            _precision, _recall, _ = precision_recall_curve(plot_df[ytrue], plot_df[key])
            _auc = average_precision_score(plot_df[ytrue], plot_df[key])
            plt.plot(_recall, _precision, marker='.', label=plotlabel + ' (area = %0.3f)' % _auc)

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        # plt.show()
        return fig

    def draw_auc_pr_comparison_curve(self, fit_model, prediction_data, raw_data):
        scores_df = self.create_scoresdf(fit_model, prediction_data, raw_data)
        xg_precision, xg_recall, _ = precision_recall_curve(scores_df['ytrue'], scores_df['xgb_scores'])
        pm_precision, pm_recall, _ = precision_recall_curve(scores_df['ytrue'], scores_df['model5_scores'])

        plots_ytrue = scores_df['ytrue'].copy()
        avg_rate = len(plots_ytrue[plots_ytrue == 1]) / len(plots_ytrue)
        ns_probs = [avg_rate] * len(plots_ytrue)

        ns_auc = average_precision_score(scores_df['ytrue'], ns_probs)
        xg_auc = average_precision_score(scores_df['ytrue'], scores_df['xgb_scores'])
        pm_auc = average_precision_score(scores_df['ytrue'], scores_df['model5_scores'])

        fig = plt.figure()
        plt.plot([0, 1], [avg_rate, avg_rate], linestyle='--', label='No Skill (area = %0.3f)' % ns_auc)
        plt.plot(xg_recall, xg_precision, marker='.', label='xG Boost (area = %0.3f)' % xg_auc)
        plt.plot(pm_recall, pm_precision, marker='.', label='Model 5.0.0 (area = %0.3f)' % pm_auc)

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        return fig

    def draw_ks_chart(self, fit_model, prediction_data, raw_data):
        probas_ = fit_model.predict_proba(prediction_data)
        plots_ytrue = raw_data['dep_var'].copy()
        fig = skplt.metrics.plot_ks_statistic(plots_ytrue, probas_)
        return fig

    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None, scoring='average_precision',
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, scoring=scoring,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt











