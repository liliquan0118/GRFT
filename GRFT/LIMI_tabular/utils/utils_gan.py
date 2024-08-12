import os,sys
sys.path.append("/data/c/liliquan/fairness_testing/Latent_Imitator/LIMI_tabular")
from os import path
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from preprocessing import pre_compas_scores
from preprocessing import pre_lsac
import numpy as np
class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 12
    input_bounds=pre_census_income.constraint

    # # the valid religion of each feature
    # input_bounds = []
    # input_bounds.append([1, 9])  # age
    # input_bounds.append([0, 7])  # workclass
    # # input_bounds.append([0, 39])  # 69 for THEMIS  fnlwgt
    # input_bounds.append([0, 15])  # education
    # input_bounds.append([0, 6])  # marital_status
    # input_bounds.append([0, 13])  # occupation
    # input_bounds.append([0, 5])  # relationship
    # input_bounds.append([0, 4])  # race
    # input_bounds.append([0, 1])  #  sex
    # input_bounds.append([0, 99])  # capital_gain
    # input_bounds.append([0, 39])  # capital_loss
    # input_bounds.append([0, 99])  # hours_per_week
    # input_bounds.append([0, 39])  # native_country
    # input_bounds_size = []
    # for x in input_bounds:
    #     input_bounds_size.append(x[1] - x[0])
    # the name of each feature
    # feature_name = [
    #     "age",  # a continuous
    #     "workclass",  # b
    #     "fnlwgt",  # c continuous
    #     "education",  # d
    #     "marital_status",  # e
    #     "occupation",  # f
    #     "relationship",  # g
    #     "race",  # h
    #     "sex",  # i
    #     "capital_gain",  # j continuous
    #     "capital_loss",  # k continuous
    #     "hours_per_week",  # l continuous
    #     "native_country",  # m
    # ]
    # feature_name = [
    #     "age",  # a continuous
    #     "workclass",  # b
    #     "education",  # d
    #     "marital_status",  # e
    #     "occupation",  # f
    #     "relationship",  # g
    #     "race",  # h
    #     "sex",  # i
    #     "capital_gain",  # j continuous
    #     "capital_loss",  # k continuous
    #     "hours_per_week",  # l continuous
    #     "native_country",  # m
    # ]
    # # the name of each class
    # class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    discrete_columns = [
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
    ]
class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 24

    # # the valid religion of each feature
    # input_bounds = []
    # input_bounds.append([0, 3])  # a
    # input_bounds.append([1, 80])  # b
    # input_bounds.append([0, 4])  # c
    # input_bounds.append([0, 10])  # d
    # input_bounds.append([1, 200])  # e
    # input_bounds.append([0, 4])  # f
    # input_bounds.append([0, 4])  # g
    # input_bounds.append([1, 4])  # h
    # input_bounds.append([0, 1])  # i
    # input_bounds.append([0, 2])  # j
    # input_bounds.append([1, 4])  # k
    # input_bounds.append([0, 3])  # l
    # input_bounds.append([1, 8])  # m
    # input_bounds.append([0, 2])  # n
    # input_bounds.append([0, 2])  # o
    # input_bounds.append([1, 4])  # p
    # input_bounds.append([0, 3])  # q
    # input_bounds.append([1, 2])  # r
    # input_bounds.append([0, 1])  # s
    # input_bounds.append([0, 1])  # t


    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 4])  # a
    input_bounds.append([1, 4])  # b
    input_bounds.append([0, 4])  # c
    input_bounds.append([1, 4])  # d
    input_bounds.append([1, 5])  # e
    input_bounds.append([1, 5])  # f
    input_bounds.append([1, 4])  # g
    input_bounds.append([1, 4])  # h
    input_bounds.append([1, 4])  # i
    input_bounds.append([1, 4])  # j
    input_bounds.append([1, 3])  # k
    input_bounds.append([1, 4])  # l
    input_bounds.append([1, 2])  # m
    input_bounds.append([1, 2])  # n
    input_bounds.append([1, 2])  # o
    input_bounds.append([0, 1])  # p
    input_bounds.append([0, 1])  # q
    input_bounds.append([0, 1])  # r
    input_bounds.append([0, 1])  # s
    input_bounds.append([0, 1])  # t
    input_bounds.append([0, 1])  # q
    input_bounds.append([0, 1])  # r
    input_bounds.append([0, 1])  # s
    input_bounds.append([0, 1])  # t


    # the name of each feature
    feature_name = [
        "checking_status",  # a
        "duration",  # b continuous
        "credit_history",  # c
        "purpose",  # d
        "credit_amount",  # e continuous
        "savings_status",  # f
        "employment",  # g
        "installment_commitment",  # h continuous
        "sex",  # i
        "other_parties",  # j
        "residence",  # k continuous
        "property_magnitude",  # l
        "age",  # m continuous
        "other_payment_plans",  # n
        "housing",  # o
        "existing_credits",  # p continuous
        "job",  # q
        "num_dependents",  # r continuous
        "own_telephone",  # s
        "foreign_worker",  # t
    ]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23
    ]
    discrete_columns = [
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
    ]


class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # # the valid religion of each feature
    # input_bounds = []
    # input_bounds.append([1, 9])
    # input_bounds.append([0, 11])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 3])
    # input_bounds.append([0, 1])
    # input_bounds.append([-20, 179])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 2])
    # input_bounds.append([1, 31])
    # input_bounds.append([0, 11])
    # input_bounds.append([0, 99])
    # input_bounds.append([1, 63])
    # input_bounds.append([-1, 39])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 3])

       # the valid religion of each feature
    input_bounds = pre_bank_marketing.constraint
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 10])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 1])
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([1, 3])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 2])

    # the name of each feature
    # feature_name = [
    #     "age",  # a continuous
    #     "job",  # b
    #     "marital",  # c
    #     "education",  # d
    #     "default",  # e
    #     "balance",  # f continuous
    #     "housing",  # g
    #     "loan",  # h
    #     "contact",  # i
    #     "day",  # j
    #     "month",  # k
    #     "duration",  # l continuous
    #     "campaign",  # m continuous
    #     "pdays",  # n continuous
    #     "previous",  # o continuous
    #     "poutcome",  # p
    # ]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    discrete_columns = [
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q"
    ]

class compas:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = len(pre_compas_scores.X[0])

    # # the valid religion of each feature
    # input_bounds = []
    # input_bounds.append([1, 9])
    # input_bounds.append([0, 11])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 3])
    # input_bounds.append([0, 1])
    # input_bounds.append([-20, 179])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 2])
    # input_bounds.append([1, 31])
    # input_bounds.append([0, 11])
    # input_bounds.append([0, 99])
    # input_bounds.append([1, 63])
    # input_bounds.append([-1, 39])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 3])

       # the valid religion of each feature
    input_bounds = pre_compas_scores.constraint
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 10])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 1])
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([1, 3])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 2])

    # the name of each feature
    feature_name = [
        "age",  # a continuous
        "job",  # b
        "marital",  # c
        "education",  # d
        "default",  # e
        "balance",  # f continuous
        "housing",  # g
        "loan",  # h
        "contact",  # i
    ]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = np.arange(0,params)

    discrete_columns = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h"
        
    ]

class lsac:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = len(pre_lsac.X[0])

    # # the valid religion of each feature
    # input_bounds = []
    # input_bounds.append([1, 9])
    # input_bounds.append([0, 11])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 3])
    # input_bounds.append([0, 1])
    # input_bounds.append([-20, 179])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 2])
    # input_bounds.append([1, 31])
    # input_bounds.append([0, 11])
    # input_bounds.append([0, 99])
    # input_bounds.append([1, 63])
    # input_bounds.append([-1, 39])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 3])

       # the valid religion of each feature
    input_bounds = pre_lsac.constraint
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 10])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 2])
    # input_bounds.append([0, 1])
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([0, 1])
    # input_bounds.append([1, 3])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([1, 4])
    # input_bounds.append([0, 2])

    # # the name of each feature
    # feature_name = [
    #     "age",  # a continuous
    #     "job",  # b
    #     "marital",  # c
    #     "education",  # d
    #     "default",  # e
    #     "balance",  # f continuous
    #     "housing",  # g
    #     "loan",  # h
    #     "contact",  # i
    #     "day",  # j
    #     "month",  # k
    #     "duration",  # l continuous
    #     "campaign",  # m continuous
    #     "pdays",  # n continuous
    #     "previous",  # o continuous
    #     "poutcome",  # p
    # ]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = np.arange(0,params)

    discrete_columns = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
    ]
class meps:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 40

    # the valid religion of each feature
    input_bounds = [
        [0, 3],
        [0, 85],
        [0, 1],
        [0, 1],
        [0, 9],
        [0, 3],
        [0, 3],
        [0, 3],
        [0, 5],
        [0, 5],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 3],
        [0, 1],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [-9, 70],
        [-9, 75],
        [-9, 24],
        [0, 7],
        [0, 4],
        [0, 4],
        [0, 2],
    ]
    # the name of each feature
    feature_name = [
        'REGION',
        'AGE',
        'SEX',
        'RACE',
        'MARRY',
        'FTSTU',
        'ACTDTY',
        'HONRDC',
        'RTHLTH',
        'MNHLTH',
        'CHDDX',
        'ANGIDX',
        'MIDX',
        'OHRTDX',
        'STRKDX',
        'EMPHDX',
        'CHBRON',
        'CHOLDX',
        'CANCERDX',
        'DIABDX',
        'JTPAIN',
        'ARTHDX',
        'ARTHTYPE',
        'ASTHDX',
        'ADHDADDX',
        'PREGNT',
        'WLKLIM',
        'ACTLIM',
        'SOCLIM',
        'COGLIM',
        'DFHEAR42',
        'DFSEE42',
        'ADSMOK42',
        'PCS42',
        'MCS42',
        'K6SUM42',
        'PHQ242',
        'EMPST',
        'POVCAT',
        'INSCOV',
    ]

    sens_name = {3: 'SEX'}
    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = range(0, 40)

    discrete_columns = [
        'REGION',
        'SEX',
        'MARRY',
        'FTSTU',
        'ACTDTY',
        'HONRDC',
        'RTHLTH',
        'MNHLTH',
        'CHDDX',
        'ANGIDX',
        'MIDX',
        'OHRTDX',
        'STRKDX',
        'EMPHDX',
        'CHBRON',
        'CHOLDX',
        'CANCERDX',
        'DIABDX',
        'JTPAIN',
        'ARTHDX',
        'ARTHTYPE',
        'ASTHDX',
        'ADHDADDX',
        'PREGNT',
        'WLKLIM',
        'ACTLIM',
        'SOCLIM',
        'COGLIM',
        'DFHEAR42',
        'DFSEE42',
        'ADSMOK42',
        'PHQ242',
        'EMPST',
        'POVCAT',
        'INSCOV',
    ]


def make_dir(pathname):
    if not path.isdir(pathname):
        os.makedirs(pathname, exist_ok=True)
