"""
Classes handling visualisation (plotting charts and showing tables).
"""
from PyQt6.QtWidgets import QTableWidgetItem, QHeaderView # pylint: disable=no-name-in-module
from PyQt6.QtCore import Qt # pylint: disable=no-name-in-module
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sources import plot_functions, data_load, common_elements
from sources.data_structures import Translations

class MplCanvas(FigureCanvas):
    """
    Chart configuration code. Source: https://www.pythonguis.com/tutorials/plotting-matplotlib/
    """
    def __init__(self, width=5, height=5, dpi=300):
        self.canvas = None
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class Visualisation:
    """
    Class handling visualisation of the results.
    """
    def __init__(self, app):
        self.app = app
        self.ui = None
        self.results = None
        self.canvas = None
        self._plot_properties_list = None
        self._language_version = "PL"

    @property
    def plot_properties_list(self):
        """Passes plot_properties_list from gui.py"""
        return self._plot_properties_list

    @plot_properties_list.setter
    def plot_properties_list(self, value):
        """Setter for plot_properties_list"""
        self._plot_properties_list = value

    @property
    def language_version(self):
        """Passes language_version from gui.py"""
        return self._language_version

    @language_version.setter
    def language_version(self, value):
        """Setter for language_version"""
        self._language_version = value

    def set_visualisation_ui(self, ui):
        """Passes UI item from gui.py"""
        self.ui = ui

    def orbit_show_table(self):
        """
        Presents the testing orbits as a friendly table with basic information
        (initial point in space of positions and velocities, period)
        """
        self.ui.orbit_table.setRowCount(5)
        self.ui.orbit_table.setColumnCount(7)
        self.ui.orbit_table.setHorizontalHeaderLabels(
            ["x", "y", "z", "vx", "vy", "vz", "Okres"])
        self.ui.orbit_table.setVerticalHeaderLabels(["0", "1", "2", "3", "4"])

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
                self.ui.orbit_table.setItem(row, col, item)

        self.ui.orbit_table.resizeColumnsToContents()
        self.ui.orbit_table.resizeRowsToContents()

        for row in range(5):
            self.ui.orbit_table.setRowHeight(row, 29)
        self.ui.orbit_table.horizontalHeader().setMaximumHeight(30)
        self.ui.orbit_table.verticalHeader().setMaximumWidth(80)

        uniform_column_width = 50
        self.ui.orbit_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed)

        for col in range(self.ui.orbit_table.columnCount()):
            self.ui.orbit_table.setColumnWidth(col, uniform_column_width)

    def plotting_charts(self, method, settings):
        """
        Creates the set of plots for each algorithm.
        :param settings: general trajectory plot settings like density and number of periods.
        :param method: value specific for each algorithm (PSO/PSO2/ABC)
        :return: None
        """
        params = common_elements.final_plot(*self._plot_properties_list[:4], settings)

        self.results = plot_functions.present_results(
            self._plot_properties_list[3],
            self._plot_properties_list[1].global_best_state)

        fig_orbits = plot_functions.plot_propagated_trajectories(*params[:4])

        self._plot_properties_list[1].convert_to_metric_units()
        self._plot_properties_list[3] = data_load.convert_to_metric_units(
            self._plot_properties_list[3])

        fig_pos, fig_vel = plot_functions.dim3_scatter_plot(
            params[4], params[5], self._plot_properties_list[3],
            self._plot_properties_list[1].global_best_state)

        fig_obj = plot_functions.plot_global_best_scores(
            self._plot_properties_list[5], self._plot_properties_list[4])
        fig_obj.tight_layout(pad=6.0)

        plot_dictionary = {"gridOrbit" : fig_orbits,
                           "gridPosition" : fig_pos,
                           "gridVelocity" : fig_vel,
                           "gridError" : fig_obj}

        for plot_type, figure in plot_dictionary.items():
            self.canvas = FigureCanvas(figure)
            self.canvas.draw()
            #Clearing the old plots, after an algorithm is restarted
            layout = getattr(self.ui, f"{plot_type}{method}")
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            layout.addWidget(self.canvas, 0, 0, 1, 1)


    def refresh_widgets(self):
        """
        Refreshes the widgets that are not translated within .qm files. Uses a nested dictionary
        defined in data_structures.py.
        :return:
        """
        for method in ["PSO", "PSO2", "ABC"]:
            #If number of rows > 0, this element must have been initialised,
            # and thus it should be modified
            result_table = getattr(self.ui, f"{method}resultTable")
            if result_table.rowCount() > 0:
                #Choose the table's header translation based on language.
                result_table.setVerticalHeaderLabels(
                    Translations().get_translation("Table", self._language_version))
                for plot_type in ["gridOrbit", "gridPosition", "gridVelocity"]:
                    #For all the scatter-plot widgets, find a translation
                    # and update the legend with draw()
                    grid_plot = getattr(self.ui, f"{plot_type}{method}").itemAt(0).widget()
                    grid_plot.figure.gca().legend(
                        Translations().get_translation("Plot", self._language_version,
                                                       plot_type), loc='upper left')
                    grid_plot.draw()

                #Grid plot for obj. fun. is not scatter-like, so a separate logic is used below
                translations = Translations().get_translation("Plot",
                                                              self._language_version, "gridError")
                grid_plot_objective_fun = getattr(self.ui, f"gridError{method}").itemAt(0).widget()
                grid_plot_objective_fun.figure.gca().set_xlabel(translations[0])
                grid_plot_objective_fun.figure.gca().set_ylabel(translations[1])
                grid_plot_objective_fun.draw()


    def show_results(self, method):
        """
        Presents results of each algorithm as a table.
        The loops' sizes and table's dimensions are set arbitrary based on the author's design.
        :param method: value specific for each algorithm
        """
        result_table = getattr(self.ui, f"{method}resultTable")
        result_table.setRowCount(5)
        result_table.setColumnCount(6)
        result_table.setHorizontalHeaderLabels(
            ["x", "y", "z", "vx", "vy", "vz"])

        result_table.setVerticalHeaderLabels(
            Translations().get_translation("Table", self._language_version))

        for row in range(3):
            for col in range(6):
                if 0 <= col < 3:
                    item = QTableWidgetItem(
                        str(np.round(float(self.results[row, col]), 2)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    result_table.setItem(row, col, item)
                elif col >= 3:
                    item = QTableWidgetItem(
                        str(np.round(float(self.results[row, col]), 4)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    result_table.setItem(row, col, item)

        result_table.setSpan(3, 0, 1, 3)
        result_table.setSpan(3, 3, 1, 3)
        result_table.setSpan(4, 0, 1, 6)

        item = QTableWidgetItem(str(np.round(float(self.results[3, 0]), 5)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        result_table.setItem(3, 0, item)

        item = QTableWidgetItem(str(np.round(float(self.results[3, 3]), 5)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        result_table.setItem(3, 3, item)

        item = QTableWidgetItem(
            str(np.round(float(self._plot_properties_list[1].global_best_score), 10)))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        result_table.setItem(4, 0, item)

        result_table.resizeColumnsToContents()
        result_table.resizeRowsToContents()
        result_table.horizontalHeader().setSectionResizeMode(
            result_table.columnCount() - 1, QHeaderView.ResizeMode.Stretch)
        for row in range(5):
            result_table.setRowHeight(row, 12)
        result_table.horizontalHeader().setMaximumHeight(28)
        result_table.verticalHeader().setMaximumWidth(75)
