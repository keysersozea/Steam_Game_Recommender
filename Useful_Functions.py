import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve, cross_validate, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from scipy.stats import skew
import os
import graphviz


def read_csv(csv_names):
    # Get the directory of the current script (Main.py)
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the full path to .csv
    csv_path = []
    csv_path.append(os.path.join(script_dir, csv_names[0]))

    # Read the CSV file
    dataframe = pd.read_csv(csv_path[0], sep=";")

    return dataframe

def check_df(dataframe, head=5):
    print("########## shape #############")
    print(dataframe.shape)

    print("########## dtype #############")
    print(dataframe.dtypes)

    print("########## head #############")
    print(dataframe.head(head))

    print("########## tail #############")
    print(dataframe.tail())

    print("########## NA #############")
    print(dataframe.isnull().sum())

    print("########## Quantiles #############")
    (print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T))

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def catch_useless_Bincols(dataframe):
    useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and
                    (dataframe[col].value_counts() / len(dataframe) < 0.01).any(axis=None)]
    return useless_cols


######################################################
# Summary
######################################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
# Değişkenlerin içindeki sınıfların frekansını ve bu frekansın toplama oranını gösteren fonksiyon


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
# İstenen değişkenlerin birbiriyle olan korelasyonunu görselleştiren fonksiyon

def check_skewness(df, col):
    col = "hours"
    skewness = skew(df[col].dropna())
    print(f"Çarpıklık değeri ({col}): {skewness}")
    if skewness > 1:
        print(f"  variable {col}: right skew")


######################################################
# Target Analysis
######################################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")



######################################################
# Outlier
######################################################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        print(f"Variable: {col_name}")
        print("True")
        return True
    else:
        print(f"Variable: {col_name}")
        print("False")
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
# threshold dışında kalan verileri dataframe'den siler


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # Alt thresholddan küçük değerleri thresholda eşitler
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # Üst thresholddan büyük değerleri thresholda eşitler
# threshold dışında kalan verilerin yerine thresholdu yerleştirir





######################################################
# Missing Values
######################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
# Eksik değerlerin sayısını ve oranını columnlara göre veren func

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)   # Eksik değerleri binary temsil etti

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:    # Eksik olup olmamanın bağımlı değişkene etkisini gözlem
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
# Değerin esik olup olmamasının bağımlı değişkene etkisini gözlemlemyi sağlar




######################################################
# Encoding
######################################################


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype="int64")
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
# cat_summary fonksiyonuna, incelenen değişkendeki sınıflara göre target meanını gösteren bölüm eklenmiş fonksiyon

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    # Kategorik değişkense, içindeki sınıfların biri bile thresholdun altında frekansa sahipse o değişkeni listeye ekler

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    # Bu listedeki değişkenleri tek tek dolaşır
    # Her bir değişkenin içiçndeki sınıfların frekans oranını tmp df'ine aktarır
    # Frekansı threhsoldun altındaki sınıfların isimlerini başka bir df'e aktarır(rare_labels)
    # Asıl dataframe'de belirlenen sınıfların isimleri 'Rare' olarak değiştirilir.
    # Bu işlemi her bir 'rare_column' için yapar
    return temp_df



######################################################
# Hyperparameter Optimization
######################################################

"""classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]"""
def hyperparameter_optimization(classifiers, X, y, cv=3, search_method = RandomizedSearchCV,scoring="roc_auc", n_iter=50):
    # classifier ile denenecek modeller, model isimleri ve hiperparametre optimizasyonu
    # için denenecek değer aralıkları fonksiyona verilir.
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        # Burada Randomized ya da
        rs_best = search_method(classifier, params, cv=cv, n_iter=n_iter, n_jobs=-1, verbose=0).fit(X, y)
        final_model = classifier.set_params(**rs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models
# classifier ile denenecek modeller, model isimleri ve hiperparametre optimizasyonu
# için denenecek değer aralıkları fonksiyona verilir.


######################################################
# Voting Classifier
######################################################

def voting_classifier(best_models, X, y, votingModels):
    """best_models = {name1: final_model1,
                   name2: final_model2,...}"""
    # voting models = [name1, name2, ..., name2]
    # voting_models voting'e katılacak algoritmaların isimlerini tutan liste
    print("Voting Classifier...")

    estimators = []

    for model_name in votingModels:
        estimators.append((model_name, best_models[model_name]))

    # Oylama sınıflandırıcısını oluşturma
    voting_clf = VotingClassifier(estimators=estimators,
        voting='soft'
    ).fit(X, y)

    # Modelin çapraz doğrulama sonuçlarını hesaplama
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1_macro", "roc_auc"], n_jobs=-1)

    # Performans metriklerini yazdırma
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1 Macro Score: {cv_results['test_f1_macro'].mean()}")
    print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")

    return voting_clf



######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


def plot_importance(model, features, num=True,save=False):
    if num == True: # True ise bütün feature'ları gösterir, numara girildiyse numara kadar gösterir
        num = len(features)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
# test score vs train score grafiği (seçilen hiperparametre değişimine göre)

