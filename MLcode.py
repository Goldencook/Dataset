
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, ParameterSampler, LeaveOneOut, learning_curve
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             precision_score, recall_score, classification_report,
                             f1_score, cohen_kappa_score, matthews_corrcoef,
                             roc_curve, precision_score, recall_score, auc)
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap
import joblib
from copy import deepcopy
import os
import matplotlib
matplotlib.use('Qt5agg')  # 设置为非交互式后端

# 设置随机种子以确保可重复性
np.random.seed(42)

def load_and_preprocess(file_path, use_lasso=True, lasso_threshold=0.5, return_selector=False):
    """
    加载并预处理数据

    参数:
    file_path: Excel文件路径
    use_lasso: 是否使用LASSO特征选择
    lasso_threshold: LASSO特征选择阈值
    return_selector: 是否返回特征筛选器

    返回:
    X: 处理后的特征数据
    y: 目标变量
    selector: 特征筛选器（如果return_selector为True）
    """
    df = pd.read_excel(file_path)

    # 定义非特征列
    non_features = ['序号', '姓名', '性别（0是女性，1是男性）', '年龄', '身高', '体重',
                    '0表示帕金森患者，1表示正常人', '2TD/0PIGD',
                    'MDS-UPDRS-I', 'MDS-UPDRS-II', 'MDS-UPDRS-III', 'MDS-UPDRS-IV', 'MDS-UPDRS总分',
                    'H&Y', 'MMSE', 'HAMD-24', 'SS-12', 'MMSE_XIEHE', ]

    # 分离特征和目标变量
    X = df.drop(columns=non_features)
    y = df['2TD/0PIGD'].replace(2, 1)  # 转换为二分类问题

    # 处理缺失值
    X = X.dropna(axis=1, how='all')  # 删除全空列
    X = X.fillna(X.median())  # 用中位数填充剩余缺失值
    y = y.loc[X.index]  # 确保目标变量索引对齐

    # 基础特征筛选 - 移除低方差特征
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    # 保存特征筛选器
    feature_selector = selector

    # LASSO特征选择
    if use_lasso:
        stable_features, _ = select_stable_features_wps(X, y, threshold=lasso_threshold)
        X = X[stable_features]

    if return_selector:
        return X, y, feature_selector
    else:
        return X, y




def get_models():
    """
    定义和配置要比较的机器学习模型

    返回:
    包含模型配置的字典列表
    """
    return [
        # 逻辑回归
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(penalty='l1', C=0.05, solver='saga',
                                        max_iter=1000, class_weight='balanced'),
            'params': {'model__C': [0.001, 0.01, 0.1], 'model__penalty': ['l1']}
        },

        # 随机森林
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(max_features=15, min_impurity_decrease=0.01,
                                            ccp_alpha=0.02, random_state=42, class_weight='balanced'),
            'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
        },

        # 梯度提升
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]}
        },

        # 支持向量机
        {
            'name': 'SVM',
            'model': SVC(probability=True, random_state=42,class_weight='balanced'),
            'params': {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 'auto']}
        },

        # XGBoost
        {
            'name': 'XGBoost',
            'model': XGBClassifier(reg_alpha=0.5, reg_lambda=1, max_depth=4, subsample=0.8,
                                   min_child_weight=5, eval_metric='logloss', scale_pos_weight=3,
                                   random_state=42),
            'params': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
        },

        # CatBoost (表现最好的模型)
        {
            'name': 'CatBoost',
            'model': CatBoostClassifier(
                verbose=0,
                auto_class_weights='Balanced',
                boosting_type='Plain',
                random_state=42
            ),
            'params': {
                'model__iterations': [700, 1000],
                'model__learning_rate': [0.01, 0.03],
                'model__depth': [6],
                'model__l2_leaf_reg': [3, 5],
                'model__random_strength': [1],
                'model__border_count': [64, 128],
                'model__grow_policy': ['Lossguide'],
                'model__min_data_in_leaf': [3, 5],
                'model__early_stopping_rounds': [50]
            }
        },

        # 神经网络
        {
            'name': 'MLP',
            'model': MLPClassifier(max_iter=1000, hidden_layer_sizes=(64, 32), activation='relu',
                                   alpha=0.001, batch_size=16, learning_rate='adaptive',
                                   random_state=42),
            'params': {'model__hidden_layer_sizes': [(50,), (100,)], 'model__alpha': [0.0001, 0.001]}
        }
    ]


def plot_learning_curve(pipeline, X, y, model_name):
    """
    绘制学习曲线诊断过拟合
    """
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("AUC Score")
    plt.title(f"Learning Curve for {model_name}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"Learning_Curve_{model_name}.png", dpi=300)
    plt.close()

    # 分析过拟合程度
    overfitting_gap = train_scores_mean[-1] - test_scores_mean[-1]
    print(f"{model_name} 过拟合程度: {overfitting_gap:.4f}")

    return overfitting_gap




def plot_shap(best_pipeline, X_train_raw, X_test_raw, model_name):
    """
    绘制SHAP特征重要性图

    参数:
    best_pipeline: 训练好的Pipeline对象
    X_train_raw: 原始训练数据
    X_test_raw: 原始测试数据
    model_name: 模型名称
    """
    plt.figure(figsize=(12, 8))
    try:
        # 获取预处理器和模型
        scaler = best_pipeline.named_steps['scaler']
        model = best_pipeline.named_steps['model']

        # 预处理数据
        X_train_processed = pd.DataFrame(scaler.transform(X_train_raw),
                                         columns=X_train_raw.columns)
        X_test_processed = pd.DataFrame(scaler.transform(X_test_raw),
                                        columns=X_test_raw.columns)

        # 选择适合的SHAP解释器
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier,
                              XGBClassifier, CatBoostClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_processed)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_train_processed)
            shap_values = explainer.shap_values(X_test_processed)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_train_processed)
            shap_values = explainer.shap_values(X_test_processed)

        # 统一SHAP值格式
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., 1]

        # 绘制SHAP摘要图
        shap.summary_plot(
            shap_values,
            X_test_processed,
            feature_names=X_test_raw.columns.tolist(),
            plot_type="dot",
            max_display=10,
            show=False
        )

        plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"SHAP_{model_name}.png", dpi=600)
        plt.close()

    except Exception as e:
        print(f"SHAP可视化失败: {str(e)}")


def evaluate_model(pipeline, X, y):
    """
    使用留一交叉验证评估模型性能

    参数:
    pipeline: 模型pipeline
    X: 特征数据
    y: 目标变量

    返回:
    包含评估指标的字典
    """
    cv = LeaveOneOut()
    metrics = {
        'AUC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [],
        'Precision': [], 'F1': [], 'Kappa': [], 'MCC': [], 'Recall': []
    }

    # 存储预测结果
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cloned_pipe = deepcopy(pipeline)
        cloned_pipe.fit(X_train, y_train)

        y_pred = cloned_pipe.predict(X_test)
        y_proba = cloned_pipe.predict_proba(X_test)[:, 1]

        all_y_true.append(y_test.values[0])
        all_y_pred.append(y_pred[0])
        all_y_proba.append(y_proba[0])

    # 计算评估指标
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    metrics['AUC'] = [roc_auc_score(all_y_true, all_y_proba)]
    metrics['Accuracy'] = [accuracy_score(all_y_true, all_y_pred)]

    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()
    metrics['Sensitivity'] = [tp / (tp + fn) if (tp + fn) > 0 else 0]
    metrics['Specificity'] = [tn / (tn + fp) if (tn + fp) > 0 else 0]
    metrics['Precision'] = [precision_score(all_y_true, all_y_pred)]
    metrics['F1'] = [f1_score(all_y_true, all_y_pred)]
    metrics['Kappa'] = [cohen_kappa_score(all_y_true, all_y_pred)]
    metrics['MCC'] = [matthews_corrcoef(all_y_true, all_y_pred)]
    metrics['Recall'] = [recall_score(all_y_true, all_y_pred)]

    # 格式化结果
    cv_metrics = {
        'CV_AUC': f"{np.mean(metrics['AUC']):.4f}",
        'CV_Accuracy': f"{np.mean(metrics['Accuracy']):.4f}",
        'CV_Sensitivity': f"{np.mean(metrics['Sensitivity']):.4f}",
        'CV_Specificity': f"{np.mean(metrics['Specificity']):.4f}",
        'CV_Precision': f"{np.mean(metrics['Precision']):.4f}",
        'CV_F1': f"{np.mean(metrics['F1']):.4f}",
        'CV_Kappa': f"{np.mean(metrics['Kappa']):.4f}",
        'CV_MCC': f"{np.mean(metrics['MCC']):.4f}",
    }

    return cv_metrics, (all_y_true, all_y_proba)


def evaluate_model_robust(pipeline, X, y, n_splits=5, n_repeats=5):
    """
    使用重复分层K折交叉验证进行更稳健的评估，以AUC为核心指标
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    metrics = {
        'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [],
        'F1': [], 'Specificity': []
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 克隆pipeline避免数据泄露
        cloned_pipe = clone(pipeline)
        cloned_pipe.fit(X_train, y_train)

        y_pred = cloned_pipe.predict(X_test)
        y_proba = cloned_pipe.predict_proba(X_test)[:, 1]

        # 计算指标
        metrics['AUC'].append(roc_auc_score(y_test, y_proba))
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['Recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['F1'].append(f1_score(y_test, y_pred, zero_division=0))

        # 计算特异性
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['Specificity'].append(specificity)

    # 返回平均值和标准差
    return {
        'CV_AUC': f"{np.mean(metrics['AUC']):.4f} ± {np.std(metrics['AUC']):.4f}",
        'CV_Accuracy': f"{np.mean(metrics['Accuracy']):.4f} ± {np.std(metrics['Accuracy']):.4f}",
        'CV_Precision': f"{np.mean(metrics['Precision']):.4f} ± {np.std(metrics['Precision']):.4f}",
        'CV_Recall': f"{np.mean(metrics['Recall']):.4f} ± {np.std(metrics['Recall']):.4f}",
        'CV_F1': f"{np.mean(metrics['F1']):.4f} ± {np.std(metrics['F1']):.4f}",
        'CV_Specificity': f"{np.mean(metrics['Specificity']):.4f} ± {np.std(metrics['Specificity']):.4f}"
    }


def select_stable_features_wps(X, y, groups=None, threshold=0.5, n_repeats=16, cv_folds=10,
                               alpha_range=(0.0001, 1), l1_ratio_range=(0.1, 1)):
    """
    使用ElasticNet进行稳定特征选择

    参数:
    X: 特征数据
    y: 目标变量
    threshold: 特征选择频率阈值
    n_repeats: 交叉验证重复次数
    cv_folds: 每次重复的折数
    alpha_range: alpha参数范围
    l1_ratio_range: L1正则化比例范围

    返回:
    稳定特征列表和特征频率
    """
    feature_counts = pd.Series(0, index=X.columns)
    param_grid = {
        'alpha': np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 20),
        'l1_ratio': np.linspace(l1_ratio_range[0], l1_ratio_range[1], 10)
    }

    # 生成随机参数组合
    params = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

    # 分层重复K折交叉验证
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=16, random_state=42)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        # 寻找最优参数
        best_score = -np.inf
        best_params = {}
        for param in params:
            enet = ElasticNet(**param, max_iter=10000, tol=1e-6, selection='random', random_state=42)
            try:
                enet.fit(X_train_scaled, y_train)
                score = enet.score(scaler.transform(X_test), y_test)
                if score > best_score:
                    best_params = param
                    best_score = score
            except:
                continue

        # 使用最优参数筛选特征
        if best_params:
            enet = ElasticNet(**best_params, max_iter=10000)
            enet.fit(X_train_scaled, y_train)
            selector = SelectFromModel(enet, prefit=True)
            selected = selector.get_support()
            feature_counts += selected.astype(int)

    # 计算选择频率
    total_iters = n_repeats * cv_folds
    feature_freq = feature_counts / total_iters

    # 筛选稳定特征
    stable_features = feature_freq[feature_freq >= threshold].index.tolist()

    print(f"稳定特征数量: {len(stable_features)}/{len(X.columns)}")
    print("TOP 20稳定特征:")
    print(feature_freq.sort_values(ascending=False).head(20))

    return stable_features, feature_freq


def plot_journal_style_roc_curves(results_dict, save_path=None):
    """
    绘制期刊风格的ROC曲线对比图

    参数:
    results_dict: 包含各模型结果的字典，格式为 {模型名: (y_true, y_proba)}
    save_path: 图片保存路径（可选）
    """
    # 设置期刊风格
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.5

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 定义颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    line_styles = ['-', '--', '-.', ':'] * 2

    # 绘制各模型的ROC曲线
    for i, (model_name, (y_true, y_proba)) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        # 使用不同的颜色和线型
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]

        ax.plot(fpr, tpr, color=color, linestyle=line_style,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    # 绘制随机猜测线
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.75, label='Random (AUC = 0.500)')

    # 设置坐标轴
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax.legend(loc='lower right', frameon=True, fancybox=True,
              shadow=True, fontsize=12)

    # 设置标题
    ax.set_title('ROC Curves Of The Models',
                 fontsize=16, fontweight='bold', pad=20)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"ROC曲线已保存至: {save_path}")

    plt.show()



def validate_model_on_new_data(model_path, data_path):
    """
    在新数据集上验证保存的模型

    参数:
    model_path: 保存的模型路径
    data_path: 新数据集路径

    返回:
    包含评估指标的字典
    """
    # 加载模型和特征信息
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    model_info = joblib.load(model_path)
    pipeline = model_info['model']
    selected_features = model_info['selected_features']

    # 加载并预处理新数据
    X_new, y_new = load_and_preprocess(data_path, use_lasso=False)  # 关闭LASSO筛选

    # 确保新数据集包含所有需要的特征
    missing_features = set(selected_features) - set(X_new.columns)
    if missing_features:
        print(f"警告: 新数据集中缺少以下特征: {missing_features}")
        # 为缺失的特征添加默认值（0或中位数）
        for feature in missing_features:
            X_new[feature] = 0  # 或者使用 X_new[feature].median() 如果合适

    # 只保留模型需要的特征
    X_new = X_new[selected_features]

    # 预测
    y_pred = pipeline.predict(X_new)
    y_proba = pipeline.predict_proba(X_new)[:, 1]

    # 计算评估指标
    results = {
        'AUC': roc_auc_score(y_new, y_proba),
        'Accuracy': accuracy_score(y_new, y_pred),
        'Precision': precision_score(y_new, y_pred),
        'Recall': recall_score(y_new, y_pred),
        'F1': f1_score(y_new, y_pred),
        'Confusion_Matrix': confusion_matrix(y_new, y_pred)
    }

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_new, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'CatBoost (AUC = {results["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on New Dataset')
    plt.legend(loc='lower right')
    plt.savefig('ROC_New_Dataset.png', dpi=300)
    plt.close()

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(results['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix on New Dataset')
    plt.savefig('Confusion_Matrix_New_Dataset.png', dpi=300)
    plt.close()

    return results


def main_optimized(use_lasso=True, lasso_threshold=0.5):
    """
    主训练流程 - 针对过拟合问题优化，以AUC为核心指标

    参数:
    use_lasso: 是否使用LASSO特征选择
    lasso_threshold: LASSO特征选择阈值
    """
    # 数据准备
    X, y = load_and_preprocess(
        "Data.xlsx",
        use_lasso=use_lasso,
        lasso_threshold=lasso_threshold
    )

    # 打印类别分布
    class_distribution = np.bincount(y)
    print(
        f"类别分布: 0={class_distribution[0]}, 1={class_distribution[1]}, 比例={class_distribution[0] / class_distribution[1]:.2f}:1")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")
    print(f"特征数量: {X_train.shape[1]}")
    print(f"特征名称: {list(X_train.columns)}")

    # 模型训练与评估
    models = get_models()
    results = []
    best_model = None
    best_auc = 0  # 使用AUC作为最佳模型选择标准

    # 初始化ROC数据收集
    cv_roc_data = {}  # 交叉验证ROC数据
    test_roc_data = {}  # 测试集ROC数据
    overfitting_gaps = {}

    for config in models:
        print(f"\n=== 训练 {config['name']} ===")

        # 构建Pipeline - 确保预处理在CV中正确进行
        steps = [
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ]

        pipe = Pipeline(steps)

        # 交叉验证策略 - 使用重复分层K折
        cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

        # 超参数搜索 - 以AUC为核心指标
        grid = GridSearchCV(pipe, config['params'], cv=cv_strategy,
                            scoring='roc_auc', refit=True, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        # 获取最佳pipeline
        best_pipeline = grid.best_estimator_

        # 交叉验证评估 - 使用稳健的评估方法
        cv_metrics, cv_roc_tuple = evaluate_model(best_pipeline, X_train, y_train)
        cv_roc_data[config['name']] = cv_roc_tuple

        # 测试集评估
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_roc_data[config['name']] = (y_test, y_proba)

        # 计算测试集指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        # 计算特异性
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 保存结果
        model_results = {
            'Model': config['name'],
            'Test_Accuracy': f"{accuracy:.4f}",
            'Test_AUC': f"{roc_auc:.4f}",
            'Test_Precision': f"{precision:.4f}",
            'Test_Recall': f"{recall:.4f}",
            'Test_F1': f"{f1:.4f}",
            'Test_Specificity': f"{specificity:.4f}",
            **cv_metrics,
            'Best_Params': str(grid.best_params_)
        }
        results.append(model_results)

        # 绘制学习曲线诊断过拟合
        overfitting_gap = plot_learning_curve(best_pipeline, X_train, y_train, config['name'])
        overfitting_gaps[config['name']] = overfitting_gap

        # 保存最佳模型 (基于测试集AUC)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = best_pipeline

            # 保存模型和特征信息
            model_info = {
                'model': best_model,
                'selected_features': X_train.columns.tolist(),
                'feature_names': list(X_train.columns)
            }
            joblib.dump(model_info, 'best_model_earlystage.pkl')
            print(f"保存最佳模型: {config['name']} (测试AUC = {best_auc:.4f})")

        # SHAP可视化
        plot_shap(best_pipeline, X_train, X_test, config['name'])

        # 使用期刊风格绘制交叉验证的ROC曲线
        print("\n=== 绘制交叉验证ROC曲线 ===")
        plot_journal_style_roc_curves(cv_roc_data, save_path='CV_ROC_Curves_Journal_Style.png')

        # 使用期刊风格绘制测试集的ROC曲线
        print("\n=== 绘制测试集ROC曲线 ===")
        plot_journal_style_roc_curves(test_roc_data, save_path='Test_ROC_Curves_Journal_Style.png')

    # 输出结果
    results_df = pd.DataFrame(results)
    print("\n模型性能对比:")
    print(results_df)

    # 输出过拟合分析
    print("\n过拟合程度分析:")
    for model_name, gap in overfitting_gaps.items():
        print(f"{model_name}: {gap:.4f}")

    # 保存结果到CSV
    results_df.to_csv('optimized_model_comparison_results.csv', index=False)

    return best_model, cv_roc_data, test_roc_data


if __name__ == "__main__":
    # 训练模型并保存最佳模型
    best_model, cv_roc_data, test_roc_data = main_optimized(use_lasso=True, lasso_threshold=0.4)
    # 打印各模型的AUC值
    print("\n=== 交叉验证AUC结果 ===")
    for model_name, (y_true, y_proba) in cv_roc_data.items():
        auc_score = roc_auc_score(y_true, y_proba)
        print(f"{model_name}: AUC = {auc_score:.4f}")

    print("\n=== 测试集AUC结果 ===")
    for model_name, (y_true, y_proba) in test_roc_data.items():
        auc_score = roc_auc_score(y_true, y_proba)
        print(f"{model_name}: AUC = {auc_score:.4f}")

