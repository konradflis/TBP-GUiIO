"""
GUI implementation for PSO, PSO2 and ABC algorithms, using PyQt6.
"""
import sys
from PyQt6.QtWidgets import QMainWindow, QApplication # pylint: disable=no-name-in-module
from PyQt6.QtCore import QCoreApplication, QTranslator #  pylint: disable=no-name-in-module
import matplotlib.pyplot as plt
from gui_files.TBP_visualisation import Ui_MainWindow
from gui_files.user_inputs import UserInputs
from gui_files.visualisation import Visualisation
from sources import abc_alg, pso

class App(QMainWindow, UserInputs, Visualisation):
    # pylint: disable=R0902, R0903, R0913, R0917
    """
    Defines all the GUI elements.
    """
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()  # Create an instance of the UI class
        self.ui.setupUi(self)
        self.orbit_table = None
        self.canvas = None
        self.plot_properties_list = None
        self.translator = QTranslator()
        self.language_version = "PL"

        self.set_user_inputs_ui(self.ui)
        self.set_visualisation_ui(self.ui)

        self.introduction_logic()
        self.pso_logic()
        self.pso2_logic()
        self.abc_logic()

        self.ui.PSOstartButton.clicked.connect(self.button_clicked_pso)
        self.ui.PSO2startButton.clicked.connect(self.button_clicked_pso2)
        self.ui.ABCstartButton.clicked.connect(self.button_clicked_abc)

        self.ui.comboBoxLanguage.currentIndexChanged.connect(
            self.combobox_language_selected)

    def combobox_language_selected(self, index):
        """
        Combobox - response to language selection. Loads the English translation file if needed.
        """
        if index: # index is set to 1 only if EN language is selected
            translation_file = "en_translation.qm"
            if self.translator.load(translation_file):
                QCoreApplication.installTranslator(self.translator)
                self.ui.retranslateUi(self)
                self.setWindowTitle("Three-body problem - orbit visualisation")
                self.language_version = "EN"
                self.refresh_widgets()

        else: # default PL
            QCoreApplication.removeTranslator(self.translator)
            self.ui.retranslateUi(self)
            self.setWindowTitle("Problem trzech ciał - wizualizacja trajektorii")
            self.language_version = "PL"
            self.refresh_widgets()


    def introduction_logic(self):
        """
        Actions to be taken when the app is started.
        """
        self.setWindowTitle("Problem trzech ciał - wizualizacja trajektorii")
        self.show()

        self.ui.orbitComboBox.currentIndexChanged.connect(
            self.orbit_combobox_selected)
        self.orbit_show_table()
        self.ui.introTextEdit.setReadOnly(True)
        self.ui.introTextEdit.setStyleSheet("QTextEdit { border: none; }")


    def button_clicked_pso(self):
        """
        Starts the PSO algorithm as a response to clicking the button.
        """
        self.ui.outputLabel.setVisible(True)
        if not self.ui.PSOinertiaCheckBox.isChecked():
            self.optional_pso.inertia_setter = 0
        if not self.ui.PSOvelocityCheckBox.isChecked():
            self.optional_pso.c_setter = 0

        self.plot_properties_list = pso.pso(
            self.mandatory_pso,
            self.optional_pso
        )

        self.plotting_charts("PSO")
        self.ui.outputLabel.setVisible(False)
        self.show_results("PSO")
        self.refresh_widgets()

    def button_clicked_pso2(self):
        """
        Starts the PSO2 algorithm as a response to clicking the button.
        """
        if not self.ui.PSO2inertiaCheckBox1.isChecked():
            self.optional_pso2_1.inertia_setter = 0
        if not self.ui.PSO2inertiaCheckBox2.isChecked():
            self.optional_pso2_2.inertia_setter = 0
        if not self.ui.PSO2velocityCheckBox1.isChecked():
            self.optional_pso2_1.c_setter = 0
        if not self.ui.PSO2velocityCheckBox2.isChecked():
            self.optional_pso2_2.c_setter = 0

        self.plot_properties_list = pso.pso(
            self.mandatory_pso2_1,
            self.optional_pso2_1
        )

        self.optional_pso2_2.if_best_velocity = 1
        self.optional_pso2_2.best_velocity = self.plot_properties_list[1].global_best_state[3:]
        self.plot_properties_list = pso.pso(
            self.mandatory_pso2_2,
            self.optional_pso2_2
        )
        self.plotting_charts("PSO2")
        self.show_results("PSO2")
        self.refresh_widgets()

    def button_clicked_abc(self):
        """
        Starts the ABC algorithm as a response to clicking the button.
        """
        self.plot_properties_list = abc_alg.abc_alg(
            self.mandatory_abc,
            self.optional_abc
        )

        self.plotting_charts("ABC")
        self.show_results("ABC")
        self.refresh_widgets()


if __name__ == '__main__':
    app = QApplication(sys.argv)


    plt.rc('font', size=7)
    plt.rc('axes', titlesize=7)
    plt.rc('axes', labelsize=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=7)
    plt.rc('figure', titlesize=7)

    with open("../Combinear.qss", "r", encoding='utf-8') as file:
        stylesheet = file.read()
    app.setStyleSheet(stylesheet)

    login_window = App()
    sys.exit(app.exec())
