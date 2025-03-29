from data_processing import DataProcessing
from machine_learning import MachineLearning

class Main:
    def __init__(self):
        self.data = DataProcessing()
        self.machine = MachineLearning()

    def process(self):
        csv_data_original = self.data.get_csv_data("machine-learning-carros-simulacao.csv")
        data = self.data.drop_dataframe_columns(csv_data_original, "Unnamed: 0", axis=1)
        print(data.head())

        sorted_df = self.data.sort_dataframe(data, "vendido", ascending=True)
        x_data, y_data = self.data.split_x_y_dataframe(sorted_df, ["preco", "idade_do_modelo", "km_por_ano"],
                                                                   "vendido")
        print(sorted_df.head())


if __name__ == '__main__':
    main = Main()
    main.process()