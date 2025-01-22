from PyQt6.QtWidgets import QMainWindow, QApplication, QTableWidgetItem, QHeaderView
from PyQt6 import uic
from PyQt6.QtCore import Qt
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from Sources import abc_alg, pso, plot_functions, data_load


class MplCanvas(FigureCanvas):
    """
    Chart configuration code. Source: https://www.pythonguis.com/tutorials/plotting-matplotlib/
    """

    def __init__(self, width=5, height=5, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class App(QMainWindow):
    """
    Defines all the GUI elements.
    """

    def __init__(self):
        super().__init__()
        self.orbitTable = None
        self.ui = uic.loadUi('TBP_visualisation.ui', self)
        """HELPERS"""
        self.plot_properties_list = None
        self.results = None
        self.filepath = "../Orbits/L2_7days.txt"

        """ENTRY FIELDS:"""
        self.max_iterations = None
        self.max_iterations_1 = None
        self.max_iterations_2 = None
        self.population_size = None
        self.population_size_1 = None
        self.population_size_2 = None
        self.inertia = None
        self.inertia_1 = None
        self.inertia_2 = None
        self.c1 = None
        self.c1_1 = None
        self.c1_2 = None
        self.c2 = None
        self.c2_1 = None
        self.c2_2 = None
        self.number_of_measurements = None
        self.number_of_measurements_1 = None
        self.number_of_measurements_2 = None
        self.stop_inertia = None
        self.stop_inertia_1 = None
        self.stop_inertia_2 = None
        self.opt_inertia_setter = None
        self.opt_inertia_setter_1 = None
        self.opt_inertia_setter_2 = None
        self.opt_n_setter = None
        self.opt_n_setter_1 = None
        self.opt_n_setter_2 = None
        self.opt_multistart = None
        self.opt_number_of_multistarts = 1
        self.opt_if_best_velocity_mod = None
        self.opt_best_velocity = None

        self.employee_phase_neighbours = None
        self.onlooker_phase_neighbours = None
        self.inactive_cycles_limit = None
        self.opt_inactive_cycles_modification = 0
        self.neighbours_pos_limit = None
        self.neighbours_vel_limit = None
        self.probability_distribution_setter = 0
        self.generating_method = 0
        self.neighbourhood_type = 0
        self.neigh_percent = None
        self.dim_probability = None

        self.setWindowTitle("Problem trzech ciał - wizualizacja trajektorii")
        self.show()

        """ACTIONS - INTRODUCTION """
        self.ui.orbitComboBox.currentIndexChanged.connect(
            self.orbit_combobox_selected)
        self.orbit_show_table()
        self.ui.introTextEdit.setReadOnly(True)
        self.ui.introTextEdit.setStyleSheet("QTextEdit { border: none; }")

        """ACTIONS - PART 1 - PSO"""
        self.ui.outputLabel.setVisible(False)
        self.ui.PSOstopInertia.setEnabled(False)
        self.ui.PSOinertiaComboBox.setEnabled(False)
        self.ui.PSOvelocityComboBox.setEnabled(False)
        self.ui.PSOmaxIterations.editingFinished.connect(lambda: setattr(
            self, 'max_iterations', int(self.ui.PSOmaxIterations.text())))
        self.ui.PSOpopulationSize.editingFinished.connect(lambda: setattr(
            self, 'population_size', int(self.ui.PSOpopulationSize.text())))
        self.ui.PSOinertia.editingFinished.connect(lambda: setattr(
            self, 'inertia', float(self.ui.PSOinertia.text())))
        self.ui.PSOc1.editingFinished.connect(
            lambda: setattr(
                self, 'c1', float(
                    self.ui.PSOc1.text())))
        self.ui.PSOc2.editingFinished.connect(
            lambda: setattr(
                self, 'c2', float(
                    self.ui.PSOc2.text())))
        self.ui.PSOnumberOfMeasurements.editingFinished.connect(lambda: setattr(
            self, 'number_of_measurements', int(self.ui.PSOnumberOfMeasurements.text())))
        self.ui.PSOstopInertia.editingFinished.connect(lambda: setattr(
            self, 'stop_inertia', float(self.ui.PSOstopInertia.text())))
        self.ui.PSOinertiaComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_PSO_inertia_selected(index, "PSO"))
        self.ui.PSOvelocityComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_PSO_n_selected(index, "PSO"))
        self.ui.PSOinertiaCheckBox.stateChanged.connect(
            lambda checked: self.checkbox_PSO_inertia_selected(checked, "PSO"))
        self.ui.PSOvelocityCheckBox.stateChanged.connect(
            lambda checked: self.checkbox_PSO_n_selected(checked, "PSO"))
        self.ui.PSOstartButton.clicked.connect(self.button_clicked_PSO)

        """ACTIONS - PART 2 - 2 PSOs"""
        self.ui.PSO21stopInertia.setEnabled(False)
        self.ui.PSO22stopInertia.setEnabled(False)
        self.ui.PSO21inertiaComboBox.setEnabled(False)
        self.ui.PSO22inertiaComboBox.setEnabled(False)
        self.ui.PSO21velocityComboBox.setEnabled(False)
        self.ui.PSO22velocityComboBox.setEnabled(False)
        self.ui.PSO2maxIterations1.editingFinished.connect(lambda: setattr(
            self, 'max_iterations_1', int(self.ui.PSO2maxIterations1.text())))
        self.ui.PSO2maxIterations2.editingFinished.connect(lambda: setattr(
            self, 'max_iterations_2', int(self.ui.PSO2maxIterations2.text())))
        self.ui.PSO2populationSize1.editingFinished.connect(lambda: setattr(
            self, 'population_size_1', int(self.ui.PSO2populationSize1.text())))
        self.ui.PSO2populationSize2.editingFinished.connect(lambda: setattr(
            self, 'population_size_2', int(self.ui.PSO2populationSize2.text())))
        self.ui.PSO2inertia1.editingFinished.connect(lambda: setattr(
            self, 'inertia_1', float(self.ui.PSO2inertia1.text())))
        self.ui.PSO2inertia2.editingFinished.connect(lambda: setattr(
            self, 'inertia_2', float(self.ui.PSO2inertia2.text())))
        self.ui.PSO2c11.editingFinished.connect(
            lambda: setattr(
                self, 'c1_1', float(
                    self.ui.PSO2c11.text())))
        self.ui.PSO2c21.editingFinished.connect(
            lambda: setattr(
                self, 'c2_1', float(
                    self.ui.PSO2c21.text())))
        self.ui.PSO2c12.editingFinished.connect(
            lambda: setattr(
                self, 'c1_2', float(
                    self.ui.PSO2c12.text())))
        self.ui.PSO2c22.editingFinished.connect(
            lambda: setattr(
                self, 'c2_2', float(
                    self.ui.PSO2c22.text())))
        self.ui.PSO2numberOfMeasurements1.editingFinished.connect(
            lambda: setattr(
                self, 'number_of_measurements_1', int(
                    self.ui.PSO2numberOfMeasurements1.text())))
        self.ui.PSO2numberOfMeasurements2.editingFinished.connect(
            lambda: setattr(
                self, 'number_of_measurements_2', int(
                    self.ui.PSO2numberOfMeasurements2.text())))
        self.ui.PSO2multistart2.editingFinished.connect(
            lambda: setattr(
                self, 'opt_number_of_multistarts', int(
                    self.ui.PSO2multistart2.text())))
        self.ui.PSO2multistartCheckBox2.toggled.connect(self.multistart_setter)

        self.ui.PSO21stopInertia.editingFinished.connect(lambda: setattr(
            self, 'stop_inertia_1', float(self.ui.PSO21stopInertia.text())))
        self.ui.PSO22stopInertia.editingFinished.connect(lambda: setattr(
            self, 'stop_inertia_2', float(self.ui.PSO22stopInertia.text())))
        self.ui.PSO21inertiaComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_PSO_inertia_selected(index, "PSO21"))
        self.ui.PSO22inertiaComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_PSO_inertia_selected(index, "PSO22"))
        self.ui.PSO21velocityComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_PSO_n_selected(index, "PSO21"))
        self.ui.PSO22velocityComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_PSO_n_selected(index, "PSO22"))
        self.ui.PSO2inertiaCheckBox1.toggled.connect(
            lambda checked: self.checkbox_PSO_inertia_selected(
                checked, "PSO21"))
        self.ui.PSO2inertiaCheckBox2.toggled.connect(
            lambda checked: self.checkbox_PSO_inertia_selected(
                checked, "PSO22"))
        self.ui.PSO2velocityCheckBox1.toggled.connect(
            lambda checked: self.checkbox_PSO_n_selected(checked, "PSO21"))
        self.ui.PSO2velocityCheckBox2.toggled.connect(
            lambda checked: self.checkbox_PSO_n_selected(checked, "PSO22"))
        self.ui.PSO2startButton.clicked.connect(self.button_clicked_PSO2)

        """ACTIONS 3 - ABC"""
        self.ui.ABCneighPercent.setEnabled(False)
        self.ui.ABCdimProbability.setEnabled(False)
        self.ui.ABCmaxIterations.editingFinished.connect(lambda: setattr(
            self, 'max_iterations', int(self.ui.ABCmaxIterations.text())))
        self.ui.ABCpopulationSize.editingFinished.connect(lambda: setattr(
            self, 'population_size', int(self.ui.ABCpopulationSize.text())))
        self.ui.ABCnumberOfMeasurements.editingFinished.connect(lambda: setattr(
            self, 'number_of_measurements', int(self.ui.ABCnumberOfMeasurements.text())))
        self.ui.ABCneighboursFirst.editingFinished.connect(
            lambda: setattr(
                self, 'employee_phase_neighbours', int(
                    self.ui.ABCneighboursFirst.text())))
        self.ui.ABCneighboursSecond.editingFinished.connect(
            lambda: setattr(
                self, 'onlooker_phase_neighbours', int(
                    self.ui.ABCneighboursSecond.text())))
        self.ui.ABCplaceLimits.editingFinished.connect(
            lambda: setattr(
                self, 'neighbours_pos_limit', float(
                    self.ui.ABCplaceLimits.text())))
        self.ui.ABCvelocityLimit.editingFinished.connect(
            lambda: setattr(
                self, 'neighbours_vel_limit', float(
                    self.ui.ABCvelocityLimit.text())))
        self.ui.ABCinactiveCycles.editingFinished.connect(lambda: setattr(
            self, 'inactive_cycles_limit', int(self.ui.ABCinactiveCycles.text())))
        self.ui.ABCneighPercent.editingFinished.connect(lambda: setattr(
            self, 'neigh_percent', float(self.ui.ABCneighPercent.text())))
        self.ui.ABCdimProbability.editingFinished.connect(lambda: setattr(
            self, 'dim_probability', float(self.ui.ABCdimProbability.text())))

        self.ui.ABCneighbourhoodTypeComboBox.currentIndexChanged.connect(
            self.combobox_generating_method)
        self.ui.ABCneighbourhoodDimComboBox.currentIndexChanged.connect(
            self.combobox_dimensions_method)
        self.ui.ABCwheelComboBox.currentIndexChanged.connect(
            self.combobox_wheel_method)
        self.ui.ABCinactiveCyclesCheckBox.toggled.connect(
            self.inactive_cycles_mod)
        self.ui.ABCstartButton.clicked.connect(self.button_clicked_ABC)

    def get_text_value(self, input_box):
        return input_box.text()

    def orbit_combobox_selected(self, index):
        """
        Uses the orbit specified by the index.
        :param index: number of orbit, based on orbits table.
        """
        orbits = [
            '../Orbits/L2_7days.txt',
            '../Orbits/L2_35.txt',
            '../Orbits/L2_821.txt',
            '../Orbits/L2_1524.txt',
            '../Orbits/ID_16.txt']
        self.filepath = orbits[index]

    def orbit_show_table(self):
        """
        Presents the testing orbits as a friendly table with basic information
        (initial point in space of positions and velocities, period)
        """
        self.ui.orbitTable.setRowCount(5)
        self.ui.orbitTable.setColumnCount(7)
        self.ui.orbitTable.setHorizontalHeaderLabels(
            ["x", "y", "z", "vx", "vy", "vz", "Okres"])
        self.ui.orbitTable.setVerticalHeaderLabels(["0", "1", "2", "3", "4"])

        table_content = np.array([
            [1.0797E+0, -1.2056E-27, 2.0235E-1, 1.0013E-14, -1.9759E-1, -1.8507E-14, 2.3350E+0],
            [1.1468E+0, -1.5968E-28, 1.5227E-1, 4.2266E-15, -2.2022E-1, -6.8196E-15, 3.1678E+0],
            [1.1809E+0, -2.5445E-26, 1.0295E-4, 3.3765E-15, -1.5586E-1, 5.5264E-18, 3.4155E+0],
            [1.0066E+0, 1.1068E-27, 1.6831E-1, -2.0119E-12, -6.6877E-2, 2.7331E-11, 1.2975E+0],
            [1.0297E+0, 2.0117E-27, 1.8694E-1, -5.5856E-14, -1.1944E-1, 9.8040E-14, 1.6130E+0]])

        for row in range(5):
            for col in range(7):
                item = QTableWidgetItem(
                    str(np.round(float(table_content[row, col]), 4)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.ui.orbitTable.setItem(row, col, item)

        self.ui.orbitTable.resizeColumnsToContents()
        self.ui.orbitTable.resizeRowsToContents()

        for row in range(5):
            self.ui.orbitTable.setRowHeight(row, 29)
        self.ui.orbitTable.horizontalHeader().setMaximumHeight(30)
        self.ui.orbitTable.verticalHeader().setMaximumWidth(80)

        uniform_column_width = 50
        self.ui.orbitTable.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed)

        for col in range(self.ui.orbitTable.columnCount()):
            self.orbitTable.setColumnWidth(col, uniform_column_width)

    """
    Functions of type 'button_clicked_<alg>' define the way each algorithm is started after choosing the 'Start' button.
    self.plot_properties_list is a vector of result data that is later used for visualisation.
    """

    def button_clicked_PSO(self):
        self.ui.outputLabel.setVisible(True)
        if not self.ui.PSOinertiaCheckBox.isChecked():
            self.opt_inertia_setter = 0
        if not self.ui.PSOvelocityCheckBox.isChecked():
            self.opt_n_setter = 0

        self.plot_properties_list = pso.pso(
            self.max_iterations,
            self.population_size,
            self.number_of_measurements,
            self.inertia,
            self.c1,
            self.c2,
            self.opt_inertia_setter,
            self.stop_inertia,
            self.opt_n_setter,
            orbit_filepath=self.filepath)

        self.plotting_charts("")
        self.ui.outputLabel.setVisible(False)

        self.show_results("PSO")

    def button_clicked_PSO2(self):
        if not self.ui.PSO2inertiaCheckBox1.isChecked():
            self.opt_inertia_setter_1 = 0
        if not self.ui.PSO2inertiaCheckBox2.isChecked():
            self.opt_inertia_setter_2 = 0
        if not self.ui.PSO2velocityCheckBox1.isChecked():
            self.opt_n_setter_1 = 0
        if not self.ui.PSO2velocityCheckBox2.isChecked():
            self.opt_n_setter_2 = 0

        self.plot_properties_list = pso.pso(
            self.max_iterations_1,
            self.population_size_1,
            self.number_of_measurements_1,
            self.inertia_1,
            self.c1_1,
            self.c2_1,
            self.opt_inertia_setter_1,
            self.stop_inertia_1,
            self.opt_n_setter_1,
            orbit_filepath=self.filepath)

        best_velocity = self.plot_properties_list[1].global_best_state[3:]
        self.plot_properties_list = pso.pso(
            self.max_iterations_2,
            self.population_size_2,
            self.number_of_measurements_2,
            self.inertia_2,
            self.c1_2,
            self.c2_2,
            self.opt_inertia_setter_2,
            self.stop_inertia_2,
            self.opt_n_setter_2,
            1,
            best_velocity,
            self.opt_multistart,
            self.opt_number_of_multistarts,
            orbit_filepath=self.filepath)
        self.plotting_charts(2)
        self.show_results("PSO2")

    def button_clicked_ABC(self):

        self.plot_properties_list = abc_alg.abc_alg(
            self.max_iterations,
            self.population_size,
            self.employee_phase_neighbours,
            self.onlooker_phase_neighbours,
            self.number_of_measurements,
            self.neighbours_pos_limit,
            self.neighbours_vel_limit,
            self.inactive_cycles_limit,
            self.opt_inactive_cycles_modification,
            self.probability_distribution_setter,
            self.generating_method,
            self.neighbourhood_type,
            self.neigh_percent,
            self.dim_probability,
            orbit_filepath=self.filepath)
        self.plotting_charts(3)
        self.show_results("ABC")

    def plotting_charts(self, method):
        """
        Creates the set of plots for each algorithm.
        :param method: value specific for each algorithm (PSO/PSO2/ABC)
        :return: None
        """
        if method != 3:
            params = pso.final_plot(*self.plot_properties_list)
        else:
            params = abc_alg.final_plot(*self.plot_properties_list)

        self.results = plot_functions.present_results(
            self.plot_properties_list[3],
            self.plot_properties_list[1].global_best_state)

        fig = plot_functions.plot_propagated_trajectories(*params[:4])
        self.canvas = FigureCanvas(fig)
        getattr(
            self.ui,
            f"PSOgridOrbit{method}").addWidget(
            self.canvas,
            0,
            0,
            1,
            1)

        self.plot_properties_list[1].convert_to_metric_units()
        self.plot_properties_list[3] = data_load.convert_to_metric_units(
            self.plot_properties_list[3])

        fig1, fig2 = plot_functions.dim3_scatter_plot(
            params[4], params[5], self.plot_properties_list[3], self.plot_properties_list[1].global_best_state)
        self.canvas = FigureCanvas(fig1)
        getattr(
            self.ui,
            f"PSOgridPosition{method}").addWidget(
            self.canvas,
            0,
            0,
            1,
            1)
        self.canvas = FigureCanvas(fig2)
        getattr(
            self.ui,
            f"PSOgridVelocity{method}").addWidget(
            self.canvas,
            0,
            0,
            1,
            1)

        fig = plot_functions.plot_global_best_scores(
            self.plot_properties_list[5], self.plot_properties_list[4])
        fig.tight_layout(pad=6.0)
        self.canvas = FigureCanvas(fig)
        self.canvas.draw()
        getattr(
            self.ui,
            f"PSOgridError{method}").addWidget(
            self.canvas,
            0,
            0,
            1,
            1)

    """
    Functions of type combobox/checkbox_<alg>_<element>_<action> and similar attach an action to each interface element.
    :param index: indicates the method triggered by given interface element
    :param method: indicates the algorithm that will be used with the element
    :param checked: indicates the checkbox state
    """

    def combobox_PSO_inertia_selected(self, index, method):

        getattr(self.ui, f"{method}stopInertia").setEnabled(
            index == 1)  # if index is 1, setEnabled(True)

        if method == "PSO":
            self.opt_inertia_setter = index
        elif method == "PSO21":
            self.opt_inertia_setter_1 = index
        elif method == "PSO22":
            self.opt_inertia_setter_2 = index
        else:
            pass

    def combobox_PSO_n_selected(self, index, method):

        if method == "PSO":
            self.opt_n_setter = index
        if method == "PSO21":
            self.opt_n_setter_1 = index
        if method == "PSO22":
            self.opt_n_setter_2 = index

    def checkbox_PSO_inertia_selected(self, checked, method):
        if checked:
            getattr(self.ui, f"{method}inertiaComboBox").setEnabled(True)
        else:
            getattr(self.ui, f"{method}inertiaComboBox").setEnabled(False)
            getattr(self.ui, f"{method}inertiaComboBox").setCurrentIndex(0)

    def checkbox_PSO_n_selected(self, checked, method):
        if checked:
            getattr(self.ui, f"{method}velocityComboBox").setEnabled(True)
        else:
            getattr(self.ui, f"{method}velocityComboBox").setEnabled(False)
            getattr(self.ui, f"{method}velocityComboBox").setCurrentIndex(0)

    def multistart_setter(self, checked):
        self.opt_multistart = 1 if checked else 0

    def inactive_cycles_mod(self, checked):
        self.opt_inactive_cycles_modification = 1 if checked else 0

    def combobox_generating_method(self, index):
        self.generating_method = index
        self.ui.ABCneighPercent.setEnabled(
            index != 0)  # if index =! 0, set True

    def combobox_dimensions_method(self, index):
        self.neighbourhood_type = index
        self.ui.ABCdimProbability.setEnabled(
            index != 0)  # if index =! 0, set True

    def combobox_wheel_method(self, index):
        self.probability_distribution_setter = index

    def show_results(self, method):
        """
        Presents results of each algorithm as a table.
        The loops' sizes and table's dimensions are set arbitrary based on the author's design.
        :param method: value specific for each algorithm
        :return: None
        """
        getattr(self.ui, f"{method}resultTable").setRowCount(5)
        getattr(self.ui, f"{method}resultTable").setColumnCount(6)
        getattr(self.ui, f"{method}resultTable").setHorizontalHeaderLabels(
            ["x", "y", "z", "vx", "vy", "vz"])
        getattr(self.ui, f"{method}resultTable").setVerticalHeaderLabels(
            ["cel", "wynik", "|różnica|", "|odległość|", "fun. celu"])

        for row in range(3):
            for col in range(6):
                if 0 <= col < 3:
                    item = QTableWidgetItem(
                        str(np.round(float(self.results[row, col]), 2)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    getattr(
                        self.ui, f"{method}resultTable").setItem(
                        row, col, item)
                elif col >= 3:
                    item = QTableWidgetItem(
                        str(np.round(float(self.results[row, col]), 4)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    getattr(
                        self.ui, f"{method}resultTable").setItem(
                        row, col, item)

        getattr(self.ui, f"{method}resultTable").setSpan(3, 0, 1, 3)
        getattr(self.ui, f"{method}resultTable").setSpan(3, 3, 1, 3)
        getattr(self.ui, f"{method}resultTable").setSpan(4, 0, 1, 6)

        item = QTableWidgetItem(str(np.round(float(self.results[3, 0]), 5)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        getattr(self.ui, f"{method}resultTable").setItem(3, 0, item)

        item = QTableWidgetItem(str(np.round(float(self.results[3, 3]), 5)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        getattr(self.ui, f"{method}resultTable").setItem(3, 3, item)

        item = QTableWidgetItem(
            str(np.round(float(self.plot_properties_list[1].global_best_score), 10)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        getattr(self.ui, f"{method}resultTable").setItem(4, 0, item)

        getattr(self.ui, f"{method}resultTable").resizeColumnsToContents()
        getattr(self.ui, f"{method}resultTable").resizeRowsToContents()
        getattr(
            self.ui,
            f"{method}resultTable").horizontalHeader().setSectionResizeMode(
            getattr(
                self.ui,
                f"{method}resultTable").columnCount() -
            1,
            QHeaderView.ResizeMode.Stretch)
        for row in range(5):
            getattr(self.ui, f"{method}resultTable").setRowHeight(row, 12)
        getattr(
            self.ui,
            f"{method}resultTable").horizontalHeader().setMaximumHeight(28)
        getattr(
            self.ui,
            f"{method}resultTable").verticalHeader().setMaximumWidth(75)


if __name__ == '__main__':
    """
    General design settings + GUI start.
    """
    app = QApplication(sys.argv)

    plt.rc('font', size=7)
    plt.rc('axes', titlesize=7)
    plt.rc('axes', labelsize=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=7)
    plt.rc('figure', titlesize=7)

    with open("../Combinear.qss", "r") as file:
        stylesheet = file.read()
    app.setStyleSheet(stylesheet)

    login_window = App()
    sys.exit(app.exec())
