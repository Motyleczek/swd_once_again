####
# imports
####








####


# Michał Motyl 401943

# import main
# import topsis
# import rms

# Every GUI app must have exactly one instance of QApplication.
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PopupWindow(QDialog):
    def __init__(self, num_criteria):
        super().__init__()
        self.setWindowTitle('Popup Window')
        self.setGeometry(400, 400, 400, 150)

        self.combo_boxes = []
        self.input_fields1 = []
        self.input_fields2 = []
        layout = QVBoxLayout()

        
        for i in range(num_criteria):
            # Create a horizontal layout for each row
            row_layout = QHBoxLayout()

            # Title for the criterion
            title_label1 = QLabel(f'Kryterium {i + 1}')
            row_layout.addWidget(title_label1)

            # Combo box for selecting an option
            combo_box = QComboBox()
            combo_box.addItem("Rozklad Normalny")
            combo_box.addItem("Rozklad Gausa")
            combo_box.addItem("Rozloz nogi malenki")
            self.combo_boxes.append(combo_box)
            row_layout.addWidget(combo_box)

            title_label2 = QLabel(f'parametr µ {i + 1}')
            row_layout.addWidget(title_label2)

            # Input field for entering parameters 
            input_field1 = QLineEdit()
            self.input_fields1.append(input_field1)
            row_layout.addWidget(input_field1)

            title_label3 = QLabel(f'parametr σ {i + 1}')
            row_layout.addWidget(title_label3)

            input_field2 = QLineEdit()
            self.input_fields2.append(input_field2)
            row_layout.addWidget(input_field2)

            layout.addLayout(row_layout)

        accept_button = QPushButton('Accept')
        accept_button.clicked.connect(self.accept)
        layout.addWidget(accept_button)

        self.setLayout(layout)

    def get_selected_values(self):
        selected_values = []
        for i in range(len(self.combo_boxes)):
            combo_box_value = self.combo_boxes[i].currentText()
            input_value1 = self.input_fields1[i].text()
            input_value2 = self.input_fields2[i].text()
            selected_values.append((combo_box_value, input_value1, input_value2))
        return selected_values

def compare(y, x):
    # realizuje porównanie y <= x
    truth_tab = []
    n = len(y)
    for i in range(n):
        truth_tab.append(y[i] <= x[i])
    summary = sum(truth_tab)
    if summary == n:
        return True
    else:
        return False

class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.setWindowTitle('SWD GUI')
        self.originalPalette = QApplication.palette()
        
        # tutaj zapisujemy co potrzeba
        self.data = None
        self.result = None
        self.res_type = None

        # tworzymy tego fusion boxa z gotowych już stylów
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        # dajemy mu odpowiednie labele
        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        # tutaj sie faktycznie tworzy to gui wedle tego co zdefiniowane jest niżej
        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()

        # tutaj co sie dzieje na aktywacji guzików
        styleComboBox.activated[str].connect(self.changeStyle)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)

        self.mainLayout = QGridLayout()
        self.mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        self.mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        self.mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        self.mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        self.mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        self.mainLayout.setRowStretch(1, 1)
        # self.mainLayout.setRowStretch(2, 1)
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 1)
        self.setLayout(self.mainLayout)

        self.setWindowTitle("Styles")
        self.changeStyle('Fusion')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        QApplication.setPalette(self.originalPalette)

    # TODO: ta funkcja powinna otwierać okienko popup do generacji danych, wtedy po wprowadzeniu wyborów
    # będziemy generować dane --> zmienić nazwe na data input zamiast import
    # dodac mozliwosc generacji zeby rozne kryteria mogly miec rózne rozkłady
    def data_import(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Data to import",
                                                  "",
                                                  "All Files (*);;CSV (*.csv)",
                                                  options=options)
        if len(fileName) != 0:
            self.data = pd.read_csv(fileName, sep=';')
        print(self.data)

        # uzupełnienie stworzonej wcześniej tabeli
        self.createBottomLeftTabWidget()
        self.mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)

    # TODO: trzeba to podciągnąć az do 4 kryterii
    def visualise_result(self):
        if self.result is None:
            return None
        if not isinstance(self.result, np.ndarray):
            self.result = np.array(self.result)
        self.createBottomRightGroupBox()
        self.mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        print('visualization')

    def owd_v1(self):
        if self.data is None:
            return None

        self.res_type = 'OWD bez filtracji'
        X = self.data.to_numpy().tolist()
        n = len(X)
        P = []

        for i in range(n):
            if len(X) == 0:
                break
            X_alter = X[:]
            m = len(X_alter)
            Y = X[0]
            fl = 0
            for j in range(1, m):
                if compare(Y, X[j]):
                    X_alter.remove(X[j])
                elif compare(X[j], Y):
                    X_alter.remove(Y)
                    Y = X[j]
                    fl = 1

            if Y not in P:
                P.append(Y)
            if fl == 0:
                X_alter.remove(Y)
            X = X_alter[:]

        print('owd 1')
        self.result = P

    def owd_v2(self):
        if self.data is None:
            return None

        self.res_type = 'OWD z filtracją'
        X = self.data.to_numpy().tolist()
        n = len(X)
        P = []

        for i in range(n):
            if len(X) == 0:
                break
            X_alter = X[:]
            m = len(X_alter)
            Y = X[0]
            for j in range(1, m):
                if compare(Y, X[j]):
                    X_alter.remove(X[j])
                elif compare(X[j], Y):
                    X_alter.remove(Y)
                    Y = X[j]

            if Y not in P:
                P.append(Y)

            # poniżej bez kopii [:] listy iteracja po liście nam przeskoczy elementy po usunięciu,
            # dlatego musimy tej kopii użyć
            for elem in X_alter[:]:
                if compare(Y, elem):
                    X_alter.remove(elem)

            if Y in X_alter:
                X_alter.remove(Y)
            X = X_alter[:]

        self.result = P
        print('owd 2')

    def owd_v3(self):
        if self.data is None:
            return None

        self.res_type = 'OWD punkt idealny'
        X = self.data.to_numpy()
        x = self.data.to_numpy().tolist()
        X_list = x[:]
        X_altered = X_list[:]
        n = len(X)
        P = []

        # znajdujemy współrzędne naszego punktu idealnego:
        # numpy to znacząco ułatwia, zakładamy tylko że podane dane na wejście
        # są w odpowiedniej formie (lista krotek)
        xmin = np.min(X, axis=0)

        # dummy d:
        d = np.array([np.nan, np.nan])

        for i in range(n):
            distance_sq = (np.linalg.norm(xmin-X[i, :]))**2
            if i == 0:
                d = np.array([i, distance_sq])
            else:
                d = np.vstack([d, [i, distance_sq]])
        d_sorted = d[np.argsort(d[:, 1])]

        M = n
        m = 0
        while m <= M:
            X_temporary_list = X_altered[:]
            X_jm = X_list[int(d_sorted[m, 0])]

            # poniżej skipujemy jeden punkt, jak już go usunęliśmy, bez zmniejszania M
            if X_jm not in X_altered:
                m += 1
                continue

            # usuń z X wszystkie X(i) takie, że X(J(m))≤X(i);
            for elem in X_altered:
                if compare(X_jm, elem):
                    X_temporary_list.remove(elem)

            # usuń X(J(m)) z X;
            if X_jm in X_temporary_list:
                X_temporary_list.remove(X_jm)

            # dodaj X(J(m)) do listy punktów niezdominowanych P;
            P.append(X_jm)

            # aktualizacja zmiennych
            X_altered = X_temporary_list[:]
            M -= 1
            m += 1

        self.result = P
        print('owd 3')

    def topsis(self):
        """
        Function does topsis algorithm on a given set of points, where the ideal point is the minimal value
        from all vector coordinates from the given points

        :param points: List[tuple[Union(int, float)]
        :param weights: List[Union(float, int)] - must be same length as the numer of criterion of points
        :return: np.ndarray - last column - calculated rating, second to last - distance (in the given norm)
                 to the non-ideal point, third to last - distance to the ideal point
        """
        if self.data is None:
            return None

        self.res_type = 'TOPSIS'
        np_points_orig = self.data.to_numpy()
        n = np_points_orig.shape[1]
        weights = [1/n for i in range(n)]

        # criterion rescaling:
        np_points = np_points_orig.copy()
        for i in range(np_points_orig.shape[1]):
            np_points[:, i] = np_points[:, i] / np.sqrt(np.sum(np_points[:, i])**2)

        # adding weights:
        np_points = np_points * weights

        # ideal point:
        p_ideal = np.min(np_points, axis=0)

        # non-ideal:
        p_non_ideal = np.max(np_points)

        # adding necessary columns for further calculations:
        ideal_dist = np.array([])
        non_ideal_dist = np.array([])

        # calculating distace with given norm:

        for i in range(np_points.shape[0]):
            ideal_dist = np.append(ideal_dist, np.linalg.norm(np_points[i, :] - p_ideal))
            non_ideal_dist = np.append(non_ideal_dist, np.linalg.norm(np_points[i, :] - p_non_ideal))


        # necessary calculations:
        ranking = non_ideal_dist/(non_ideal_dist + ideal_dist)
        ranking = np.reshape(ranking, (len(ranking), 1))

        np_points_orig = np.concatenate((np_points_orig, ranking), axis=1)
        p_ordered = np_points_orig[np.argsort(np_points_orig[:, -1])][::-1, :]

        self.result = p_ordered[0:2, :-1]
        print('topsisss')

    #TODO
    def rms(self):
        print('rms - nie wiem czy algorytm jest dobry więc nie implementuje')

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Wybór algorytmu")

        pushButton1 = QPushButton("OWD bez filtracji")
        pushButton2 = QPushButton("OWD filtracja")
        pushButton3 = QPushButton("OWD punkt idealny")
        # pushButton4 = QPushButton("Topsis")
        # pushButton5 = QPushButton("RMS")

        pushButton1.setChecked(False)
        pushButton2.setChecked(False)
        pushButton3.setChecked(False)
        # pushButton4.setChecked(False)
        # pushButton5.setChecked(False)

        pushButton1.clicked.connect(self.owd_v1)
        pushButton2.clicked.connect(self.owd_v2)
        pushButton3.clicked.connect(self.owd_v3)
        # pushButton4.clicked.connect(self.topsis)
        # pushButton5.clicked.connect(self.rms)

        layout = QVBoxLayout()
        layout.addWidget(pushButton1)
        layout.addWidget(pushButton2)
        layout.addWidget(pushButton3)
        # layout.addWidget(pushButton4)
        # layout.addWidget(pushButton5)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)

    
    # Modify createTopRightGroupBox method
    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Funkcjonalności")

        defaultComboBox1 = QComboBox()
        defaultComboBox1.addItem("2 kryteria")
        defaultComboBox1.addItem("3 kryteria")
        defaultComboBox1.addItem("4 kryteria")
        defaultComboBox1.addItem("6 kryteria")
        defaultComboBox1.addItem("10 kryteria")  # Add more options as needed

        defaultComboBox2 = QComboBox()
        defaultComboBox2.addItem("10 punktów")
        defaultComboBox2.addItem("50 punktów")
        defaultComboBox2.addItem("100 punktów")
        defaultComboBox2.addItem("1000 punktów")

        defaultPushButton1 = QPushButton("Import danych")
        defaultPushButton3 = QPushButton("Generacja danych")
        defaultPushButton2 = QPushButton("Wizualicja na wykresie")

        defaultPushButton1.setDefault(False)
        defaultPushButton2.setDefault(False)
        defaultPushButton3.setDefault(False)

        defaultPushButton1.clicked.connect(self.data_import)
        defaultPushButton2.clicked.connect(self.visualise_result)
        defaultPushButton3.clicked.connect(lambda: self.show_popup([defaultComboBox1, defaultComboBox2]))

        layout = QVBoxLayout()
        layout.addWidget(defaultComboBox1)
        layout.addWidget(defaultComboBox2)
        layout.addWidget(defaultPushButton3)
        layout.addWidget(defaultPushButton1)
        layout.addWidget(defaultPushButton2)

        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def show_popup(self, criteria_combo):
        
        criteria_combo2 = criteria_combo[1]
        criteria_option = criteria_combo2.currentText()
        num_of_records = int(criteria_option.split()[0])
        
        criteria_combo1 = criteria_combo[0]
        criteria_option = criteria_combo1.currentText()
        num_criteria = int(criteria_option.split()[0])
        
        popup = PopupWindow(num_criteria)
        result = popup.exec_()
        if result == QDialog.Accepted:
            selected_values = popup.get_selected_values()
            print("Selected values:", selected_values)
            
            # generacje oraz zapis do dataframe jako self.data
            criterions = len(selected_values)
            generated_criterions = []
            columns_names = []
            
            for i in range(criterions):
                column_name = f"Criterion_{i+1}"
                columns_names.append(column_name)
                type_of_dist, mu, sigma = selected_values[i]
                mu = float(mu)
                sigma = float(sigma)
                
                if type_of_dist == "Rozklad Normalny":
                    gen_krit = np.random.normal(mu, sigma, num_of_records)
                elif type_of_dist == "Rozklad Gausa": 
                    gen_krit = np.random.normal(mu, sigma, num_of_records)
                else:
                    raise ValueError('upsi poopsi ni e taki rozklad')
                
                generated_criterions.append(gen_krit)
            
            np_gen_crit = np.array(generated_criterions).transpose()
            self.data = pd.DataFrame(np_gen_crit, columns=columns_names)

            print(self.data)
            self.createBottomLeftTabWidget()
            self.mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
             

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QTabWidget()
        self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Preferred,
                                               QSizePolicy.Ignored)
        tab1 = QWidget()
        if self.data is not None:
            hight, width = self.data.shape
            tableWidget = QTableWidget(hight, width)
            horHeaders = []
            i = 0
            for column in self.data.columns:
                horHeaders.append(column)
                m = 0
                for elem in self.data[column]:
                    # tutaj trzeba rzutować na str, i tak tylko to do wyświetlania w tabeli to nic sie nie zepsuje
                    newitem = QTableWidgetItem(str(elem))
                    tableWidget.setItem(m, i, newitem)
                    m += 1
                i += 1
            tableWidget.setHorizontalHeaderLabels(horHeaders)
        else:
            tableWidget = QTableWidget(10, 10)

        tab1hbox = QHBoxLayout()
        tab1hbox.setContentsMargins(5, 5, 5, 5)
        tab1hbox.addWidget(tableWidget)
        tab1.setLayout(tab1hbox)

        self.bottomLeftTabWidget.addTab(tab1, "&Wprowadzone dane")

    def createBottomRightGroupBox(self):
        if self.result is None:
            self.bottomRightGroupBox = QGroupBox("Wizualizacja wyników")
            canvas = FigureCanvas(plt.Figure(figsize=(10, 5)))
        else:
            self.bottomRightGroupBox = QGroupBox("Wizualizacja wyników")
            canvas = FigureCanvas(plt.Figure(figsize=(5, 5)))
            axs = canvas.figure.add_subplot(projection='3d')
            # tutaj trzeba poprawnie zrobić plot danych żeby się wyświetlały z result
            # najlepszym podejściem byłoby zrobić do tego osobną funkcje klasową, ale z braku
            # czasu podejście sub-optymalne będzie wystarczające
            axs.set_title('Wyniki, dwie pierwsze zmienne, {}'.format(self.res_type))
            
            # TODO: tutaj jest druga część visualise_result: chyba tak to zostawimy byleby zadziałało
            import matplotlib.cm as cmx
            import matplotlib
            
            if self.data.shape[1] == 2:
                y_label = self.data.columns[1]
                x_label = self.data.columns[0]
                # axs.set_ylabel(y_label)
                # axs.set_xlabel(x_label)
                axs.grid()
                axs.scatter(self.result[:, 0], self.result[:, 1], [0 for i in range(len(self.result[:, 0]))])
            elif self.data.shape[1] == 3:
                axs.grid()
                axs.scatter(self.result[:, 0], self.result[:, 1], self.result[:, 2])
            else:
                axs.grid()
                cm = plt.get_cmap('jet')
                cNorm = matplotlib.colors.Normalize(vmin=min(self.result[:, 3]), vmax=max(self.result[:, 3]))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                axs.scatter(self.result[:, 0], self.result[:, 1], self.result[:, 2], c=scalarMap.to_rgba(self.result[:, 3]))
                scalarMap.set_array(self.result[:, 3])
                canvas.figure.colorbar(scalarMap,label='Test')
            ######
            
        layout = QGridLayout()
        layout.addWidget(canvas)
        self.bottomRightGroupBox.setLayout(layout)


# class PopupWindow(QDialog):
#     def __init__(self, num_buttons):
#         super().__init__()

#         self.initUI(num_buttons)

#     def initUI(self, num_buttons):
#         self.setWindowTitle('Popup Window')
#         self.setGeometry(300, 300, 300, 100)

#         layout = QVBoxLayout()
#         for i in range(num_buttons):
#             button = QPushButton(f'Button {i + 1}')
#             layout.addWidget(button)
#         # Create a combo box and add items
#         self.combo_box = QComboBox()
#         self.combo_box.addItem("Option 1")
#         self.combo_box.addItem("Option 2")
#         self.combo_box.addItem("Option 3")
#         layout.addWidget(self.combo_box)

#         # Create a button to close the popup
#         close_button = QPushButton('Close')
#         close_button.clicked.connect(self.close)
#         layout.addWidget(close_button)

#         self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication([])
    gallery = WidgetGallery()
    gallery.show()
    app.exec()

# testowanie różnej ilosci w zbiorach
# testowanie po 2, 3, 6 cech


# mój pomysł jest taki ze znajde to zajebane wiki i zajebiemy projekt do matlaba
# bo normalnie bym napisał w pythonie, ale ten przypadek ciągły to mnie rozjebie