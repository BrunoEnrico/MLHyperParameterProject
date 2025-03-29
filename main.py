from data_processing import DataProcessing
from machine_learning import MachineLearning
import numpy as np

class Main:
    def __init__(self):
        self.data = DataProcessing()
        self.machine = MachineLearning()

    def process(self):
        csv_data_original = self.data.get_csv_data("machine-learning-carros-simulacao.csv")
        data = self.data.drop_dataframe_columns(csv_data_original, "Unnamed: 0", axis=1)

        sorted_df = self.data.sort_dataframe(data, "vendido", ascending=True)
        x_data, y_data = self.data.split_x_y_dataframe(sorted_df, ["preco", "idade_do_modelo", "km_por_ano"],
                                                                   "vendido")
        print(sorted_df.head())


        dummy_classifier = self.machine.get_dummy_classifier()
        cross_validate = self.machine.get_cross_validate(dummy_classifier, x_data, y_data,
                                                         cv = 10, return_train_score=False)

        mean = self.machine.get_cross_validate_mean(cross_validate, "test_score")
        std = self.machine.get_std_cross_validate(cross_validate, "test_score")

        decision_tree = self.machine.get_decision_tree_classifier(max_depth=2)
        cross_validate = self.machine.get_cross_validate(decision_tree, x_data, y_data,
                                                         cv=10, return_train_score=False)
        mean = self.machine.get_cross_validate_mean(cross_validate, "test_score")
        std = self.machine.get_std_cross_validate(cross_validate, "test_score")


        result = np.random.randint(-2, 2, size=10000)
        data["modelo"] = result
        data["modelo"] = data["idade_do_modelo"] + abs(data["modelo"].min() + 1)

        group_k_fold = self.machine.get_groups_k_fold(n_splits = 10)
        model = self.machine.get_decision_tree_classifier(max_depth=2)
        results = self.machine.get_cross_validate(model, x_data, y_data, cv=group_k_fold, groups = data["modelo"],
                                                  return_train_score=False)
        #self.machine.print_results(results, "test_score")

        scaler = self.machine.get_standard_scaler()
        svc = self.machine.get_svc()

        pipeline = self.machine.get_pipeline([('transformer', scaler), ('classifier', svc)])

        group_k_fold = self.machine.get_groups_k_fold(n_splits=10)
        results = self.machine.get_cross_validate(pipeline, x_data, y_data, cv=group_k_fold, groups=data["modelo"],
                                                  return_train_score=False)
        self.machine.print_results(results, "test_score")


if __name__ == '__main__':
    main = Main()
    main.process()