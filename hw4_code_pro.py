import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    n = x.size
    if n <= 1:
        return np.array([]), np.array([]), None, None

    mask = x[1:] != x[:-1]
    if not np.any(mask):
        return np.array([]), np.array([]), None, None

    thresholds = ((x[1:] + x[:-1]) / 2.0)[mask]

    y1 = (y == 1).astype(float)
    pref1 = np.cumsum(y1)
    total1 = pref1[-1]

    pos = np.nonzero(mask)[0]
    left_n = pos + 1
    right_n = n - left_n
    left_1 = pref1[pos]
    right_1 = total1 - left_1

    p_left = left_1 / left_n
    p_right = right_1 / right_n
    g_left = 1.0 - p_left ** 2 - (1.0 - p_left) ** 2
    g_right = 1.0 - p_right ** 2 - (1.0 - p_right) ** 2
    ginis = -(left_n / n) * g_left - (right_n / n) * g_right

    best = int(np.argmax(ginis))
    return thresholds, ginis, thresholds[best], ginis[best]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        sub_X = np.atleast_2d(np.asarray(sub_X))
        sub_y = np.asarray(sub_y).reshape(-1)

        if sub_y.size == 0 or sub_X.shape[0] != sub_y.shape[0]:
            return
        if sub_X.shape[0] <= 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if self._min_samples_split is not None and sub_y.size < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        p1 = np.mean(sub_y == 1)
        base_gini = -(1.0 - p1 ** 2 - (1.0 - p1) ** 2)

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                col = np.asarray(sub_X[:, feature], dtype=object).reshape(-1)
                col_list = list(col)
                counts = Counter(col_list)
                clicks = Counter([col_list[i] for i in range(len(col_list)) if sub_y[i] == 1])
                ratio = {}
                for key, current_count in counts.items():
                    ratio[key] = clicks.get(key, 0) / current_count
                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in col_list])
            else:
                raise ValueError

            feature_vector = np.asarray(feature_vector).reshape(-1)
            if np.unique(feature_vector).shape[0] <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue
            if gini_best is None or gini > gini_best:
                left_size = int(np.sum(feature_vector < threshold))
                right_size = feature_vector.size - left_size
                if self._min_samples_leaf is not None:
                    if left_size < self._min_samples_leaf or right_size < self._min_samples_leaf:
                        continue
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [k for k, v in categories_map.items() if v < threshold]
                else:
                    raise ValueError

        if feature_best is None or gini_best <= base_gini:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        if self._feature_types[feature] == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
