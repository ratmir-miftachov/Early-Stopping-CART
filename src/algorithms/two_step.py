import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold


np.random.seed(42)

def esmp(m_stop, X, y_samp, threshold, k_cv):
    dtree_full = DecisionTreeRegressor(random_state=0, max_depth=m_stop)
    dtree_full.fit(X, y_samp)
    path = dtree_full.cost_complexity_pruning_path(X, y_samp)
    ccp_alphas, impurities = path.ccp_alphas[0:-1], path.impurities[0:-1]

    # Filter alphas based on impurity changes
    filtered_alphas = np.array([ccp_alphas[0]])  # Always include the first alpha
    filtered_impurities = np.array([impurities[0]])  # Include the first impurity

    for i in range(1, len(ccp_alphas)):
        impurity_change = impurities[i] - impurities[i - 1]
        if impurity_change >= threshold:
            filtered_alphas = np.append(filtered_alphas, ccp_alphas[i])
            filtered_impurities = np.append(filtered_impurities, impurities[i])

    ccp_alphas = filtered_alphas

    trees = []
    for ccp_alpha in ccp_alphas:
        dtree = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha, max_depth=m_stop)
        dtree.fit(X, y_samp)
        trees.append(dtree)

    #Extract depth of each tree:
    depths_prun = [tree.get_depth() for tree in trees]
    nodes_prun = [tree.get_n_leaves() for tree in trees]
    parameters = {'ccp_alpha':ccp_alphas.tolist()}
    gsearch = GridSearchCV(DecisionTreeRegressor(random_state=0, max_depth=m_stop), parameters, cv=k_cv, scoring='neg_mean_squared_error')
    gsearch.fit(X, y_samp)
    # best alpha
    best_alpha = gsearch.best_params_
    # corresponding depth
    alpha = best_alpha['ccp_alpha']
    position = parameters['ccp_alpha'].index(alpha)
    m_prun = depths_prun[position]
    node_prun = nodes_prun[position]
    # tree corresponding to best alpha
    clf = gsearch.best_estimator_
    # show cv error for each alpha:
    cv_err = -gsearch.cv_results_['mean_test_score']

    res = m_prun
    return res, clf, ccp_alphas
