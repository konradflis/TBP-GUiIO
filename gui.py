# pylint: disable=no-name-in-module
"""
GUI implementation for PSO, PSO2 and ABC algorithms, using PyQt6.
"""
import sys

from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import (QMainWindow, QApplication,
                             QLineEdit)
from PyQt6.QtCore import (QCoreApplication, QTranslator,
                          QLocale, QTimer)
import matplotlib.pyplot as plt
from gui_files.TBP_visualisation import Ui_MainWindow
from gui_files.user_inputs import UserInputs
from gui_files.visualisation import Visualisation
from sources import abc_alg, pso, genetic_alg

class App(QMainWindow, UserInputs, Visualisation):
    # pylint: disable=R0902, R0903, R0913, R0917
    """
    Defines all the GUI elements.
    """
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.orbit_table = None
        self.plot_properties_list = None
        self.translator = QTranslator()
        self.language_version = "PL"

        self.set_user_inputs_ui(self.ui)
        self.set_visualisation_ui(self.ui)

        self.set_validations()
        self.introduction_logic()
        self.general_settings()
        self.pso_logic()
        self.pso2_logic()
        self.abc_logic()
        self.gen_logic()

        self.ui.PSOstartButton.clicked.connect(self.button_clicked_pso)
        self.ui.PSO2startButton.clicked.connect(self.button_clicked_pso2)
        self.ui.ABCstartButton.clicked.connect(self.button_clicked_abc)
        self.ui.GENstartButton.clicked.connect(self.button_clicked_gen)

        self.ui.comboBoxLanguage.currentIndexChanged.connect(
            self.combobox_language_selected)

    def set_validations(self):
        """
        Configures the validations used for user manual input fields.
        PyQt cannot assess correctly if number is bigger than the set limit (for instance,
        it lets user insert 99 when the limit is 50). This method should block it.
        """
        for obj_name, limitation in self.validations.dictionary.items():
            validated_field = self.findChild(QLineEdit, obj_name)
            parent_object_name, child_name = limitation.mapped_attribute.rsplit(".", 1)
            validated_field.setPlaceholderText(str(getattr(getattr(self, parent_object_name),
                                                       child_name, limitation.min_value)))
            if limitation.expected_type is int:
                int_validator = QIntValidator()
                validated_field.setValidator(int_validator)

            #Validation limits will be overridden in self.additional_validation()
            if limitation.expected_type is float:
                double_validator = QDoubleValidator(bottom=0, top=999, decimals=4)
                double_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                double_validator.setLocale(QLocale(QLocale.Language.English,
                                                   QLocale.Country.UnitedStates))
                validated_field.setValidator(double_validator)

            #Connections: created once, in a loop. Lambda function passes all necessary information.
            validated_field.editingFinished.connect(lambda field=validated_field,
                                                           obj_name=parent_object_name,
                                                           attr=child_name,
                                                           limits=limitation:
                                                           self.additional_validation(field,
                                                                                      obj_name,
                                                                                      attr,
                                                                                      limits))

    def additional_validation(self,
                              validated_field,
                              parent_object_name,
                              child_name,
                              limitation):
        """
        PyQt cannot assess correctly if number is bigger than the set limit (for instance,
        it lets user insert 99 when the limit is 50). This method should block it.
        :param validated_field: the user input field
        :param parent_object_name: the group name of a given parameter (e.g. "mandatory_pso")
        :param child_name: the parameter name (e.g. "max_iterations")
        :param limitation: ValidatedElement() object containing limits for each field
        """
        value = limitation.expected_type(validated_field.text())
        #If within limits, set the value and make the field background green
        if limitation.min_value <= value <= limitation.max_value:
            setattr(getattr(self, parent_object_name), child_name, value)
            validated_field.setStyleSheet("")
            validated_field.setStyleSheet("background-color: #81a17a; "
                                          "color: black;")
            QTimer.singleShot(500, lambda: validated_field.setStyleSheet(""))
        #If exceeds limits, print an error message and make the field background red.
        else:
            self.show_error(f"Invalid input: value must be of {limitation.expected_type} type "
                            f"between {limitation.min_value} and {limitation.max_value}.")
            validated_field.clear()
            previous_value = getattr(getattr(self, parent_object_name),
                                     child_name, limitation.min_value)
            validated_field.setText(str(previous_value))
            validated_field.setFocus()
            validated_field.setStyleSheet("background-color: #80464e; "
                                          "color: black;")
            QTimer.singleShot(500, lambda: validated_field.setStyleSheet(""))

    def show_error(self, message):
        """
        Prints an error message in the status bar.
        :param message: error message to be displayed
        """
        self.statusBar().showMessage(message, 5000)

    def combobox_language_selected(self, index):
        """
        Combobox - response to language selection. Loads the English translation file if needed.
        """
        if index: # index is set to 1 only if EN language is selected
            translation_file = "gui_files/en_translation.qm"
            if self.translator.load(translation_file):
                QCoreApplication.installTranslator(self.translator)
                self.ui.retranslateUi(self)
                self.setWindowTitle("Three-body problem - orbit visualisation")
                self.language_version = "EN"

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

        if not self.ui.PSOinertiaCheckBox.isChecked():
            self.optional_pso.inertia_setter = 0
        if not self.ui.PSOvelocityCheckBox.isChecked():
            self.optional_pso.c_setter = 0

        self.plot_properties_list = pso.pso(
            self.mandatory_pso,
            self.optional_pso
        )

        self.plotting_charts("PSO", self.settings)
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
        self.plotting_charts("PSO2", self.settings)
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

        self.plotting_charts("ABC", self.settings)
        self.show_results("ABC")
        self.refresh_widgets()

    def button_clicked_gen(self):
        """
        Starts the genetic algorithm as a response to clicking the button.
        """
        ga = genetic_alg.GeneticAlgorithm(self.mandatory_gen.population_size,
                                          self.mandatory_gen.max_generations,
                                          self.mandatory_gen.mutation_rate,
                                          self.mandatory_gen.crossover_rate,)
        self.plot_properties_list = ga.run()

        self.plotting_charts("GEN", self.settings)
        self.show_results("GEN")
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

    with open("./Combinear.qss", "r", encoding='utf-8') as file:
        stylesheet = file.read()
    app.setStyleSheet(stylesheet)

    login_window = App()
    sys.exit(app.exec())
