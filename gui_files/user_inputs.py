"""
Class handling the fields/checkboxes/comboboxes that are entered or selected by the user.
"""
from sources.data_structures import (MandatorySettingsPSO, MandatorySettingsABC,
                                     OptionalSettingsPSO, OptionalSettingsABC,
                                     PlotSettings, Validations)

class UserInputs:
    """
    Class defining actions based on user inputs.
    """
    # pylint: disable=R0902, R0903, R0913, R0917
    def __init__(self):
        self.ui = None
        # ENTRY FIELDS - defines mandatory and optional fields:
        self.mandatory_pso = MandatorySettingsPSO()
        self.mandatory_pso2_1 = MandatorySettingsPSO()
        self.mandatory_pso2_2 = MandatorySettingsPSO()
        self.mandatory_abc = MandatorySettingsABC()
        self.optional_pso = OptionalSettingsPSO()
        self.optional_pso2_1 = OptionalSettingsPSO()
        self.optional_pso2_2 = OptionalSettingsPSO()
        self.optional_abc = OptionalSettingsABC()
        self.settings = PlotSettings()
        self.validations = Validations()
        self.filepath = "../orbits/L2_7days.txt"

    def set_user_inputs_ui(self, ui):
        """Passes UI item from gui.py"""
        self.ui = ui

    def get_text_value(self, input_box):
        """
        Get the text value that was entered - it will be used as an algorithm's paramater value.
        :param input_box: input field object
        """
        return input_box.text()

    def general_settings(self):
        """
        Combines the actions related to settings.
        """
        self.ui.multiplePeriods.setEnabled(False) #by default, this option is disabled
        self.ui.onlyMeasurementsButton.clicked.connect(self.radiobutton_plot_type_clicked)
        self.ui.densePlotButton.clicked.connect(self.radiobutton_plot_type_clicked)
        self.ui.multiplePeriods.editingFinished.connect(lambda: setattr(
            self.settings, 'periods', int(self.ui.multiplePeriods.text())))

    def input_validation(self):
        """Validates the user inputs. If it's not valid, it asks the user to try again."""
        validation_input = self.sender()
        try:
            self.validations.dictionary[validation_input.objectName()]
        except KeyError as key_error:
            raise KeyError(f"GUI element {validation_input.objectName()} "
                           f"not defined in validation dictionary") from key_error
        validation_data = validation_input.text()
        if validation_data.isdigit():
            validation_data = int(validation_data)
        else:
            try:
                validation_data = float(validation_data)
            except ValueError:
                pass
        #TBA

    def pso_logic(self):
        """
        Combines the actions related to basic PSO implementation.
        """
        self.ui.outputLabel.setVisible(False)
        self.ui.PSOstopInertia.setEnabled(False)
        self.ui.PSOinertiaComboBox.setEnabled(False)
        self.ui.PSOvelocityComboBox.setEnabled(False)
        #self.ui.PSOmaxIterations.editingFinished.connect(lambda: setattr(
        #    self.mandatory_pso, 'max_iterations', int(self.ui.PSOmaxIterations.text())))
        self.ui.PSOmaxIterations.editingFinished.connect(self.input_validation)
        self.ui.PSOpopulationSize.editingFinished.connect(lambda: setattr(
            self.mandatory_pso, 'population_size', int(self.ui.PSOpopulationSize.text())))
        self.ui.PSOinertia.editingFinished.connect(lambda: setattr(
            self.mandatory_pso, 'inertia', float(self.ui.PSOinertia.text())))
        self.ui.PSOc1.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso, 'c1', float(
                    self.ui.PSOc1.text())))
        self.ui.PSOc2.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso, 'c2', float(
                    self.ui.PSOc2.text())))
        self.ui.PSOnumberOfMeasurements.editingFinished.connect(lambda: setattr(
            self.mandatory_pso, 'number_of_measurements',
            int(self.ui.PSOnumberOfMeasurements.text())))
        self.ui.PSOstopInertia.editingFinished.connect(lambda: setattr(
            self.optional_pso, 'stop_inertia', float(self.ui.PSOstopInertia.text())))

        self.ui.PSOinertiaComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_pso_inertia_selected(index, "PSO"))
        self.ui.PSOvelocityComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_pso_n_selected(index, "PSO"))
        self.ui.PSOinertiaCheckBox.stateChanged.connect(
            lambda checked: self.checkbox_pso_inertia_selected(checked, "PSO"))
        self.ui.PSOvelocityCheckBox.stateChanged.connect(
            lambda checked: self.checkbox_pso_n_selected(checked, "PSO"))



    def pso2_logic(self):
        """
        Combines the actions related to two-stage PSO.
        """
        self.ui.PSO21stopInertia.setEnabled(False)
        self.ui.PSO22stopInertia.setEnabled(False)
        self.ui.PSO21inertiaComboBox.setEnabled(False)
        self.ui.PSO22inertiaComboBox.setEnabled(False)
        self.ui.PSO21velocityComboBox.setEnabled(False)
        self.ui.PSO22velocityComboBox.setEnabled(False)
        self.ui.PSO2maxIterations1.editingFinished.connect(lambda: setattr(
            self.mandatory_pso2_1, 'max_iterations', int(self.ui.PSO2maxIterations1.text())))
        self.ui.PSO2maxIterations2.editingFinished.connect(lambda: setattr(
            self.mandatory_pso2_2, 'max_iterations', int(self.ui.PSO2maxIterations2.text())))
        self.ui.PSO2populationSize1.editingFinished.connect(lambda: setattr(
            self.mandatory_pso2_1, 'population_size', int(self.ui.PSO2populationSize1.text())))
        self.ui.PSO2populationSize2.editingFinished.connect(lambda: setattr(
            self.mandatory_pso2_2, 'population_size', int(self.ui.PSO2populationSize2.text())))
        self.ui.PSO2inertia1.editingFinished.connect(lambda: setattr(
            self.mandatory_pso2_1, 'inertia', float(self.ui.PSO2inertia1.text())))
        self.ui.PSO2inertia2.editingFinished.connect(lambda: setattr(
            self.mandatory_pso2_2, 'inertia', float(self.ui.PSO2inertia2.text())))
        self.ui.PSO2c11.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso2_1, 'c1', float(
                    self.ui.PSO2c11.text())))
        self.ui.PSO2c21.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso2_1, 'c2', float(
                    self.ui.PSO2c21.text())))
        self.ui.PSO2c12.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso2_2, 'c1', float(
                    self.ui.PSO2c12.text())))
        self.ui.PSO2c22.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso2_2, 'c2', float(
                    self.ui.PSO2c22.text())))
        self.ui.PSO2numberOfMeasurements1.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso2_1, 'number_of_measurements', int(
                    self.ui.PSO2numberOfMeasurements1.text())))
        self.ui.PSO2numberOfMeasurements2.editingFinished.connect(
            lambda: setattr(
                self.mandatory_pso2_2, 'number_of_measurements', int(
                    self.ui.PSO2numberOfMeasurements2.text())))
        self.ui.PSO2multistart2.editingFinished.connect(
            lambda: setattr(
                self.optional_pso2_2, 'number_of_multistarts', int(
                    self.ui.PSO2multistart2.text())))
        self.ui.PSO2multistartCheckBox2.toggled.connect(self.multistart_setter)

        self.ui.PSO21stopInertia.editingFinished.connect(lambda: setattr(
            self.optional_pso2_1, 'stop_inertia', float(self.ui.PSO21stopInertia.text())))
        self.ui.PSO22stopInertia.editingFinished.connect(lambda: setattr(
            self.optional_pso2_2, 'stop_inertia', float(self.ui.PSO22stopInertia.text())))
        self.ui.PSO21inertiaComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_pso_inertia_selected(index, "PSO21"))
        self.ui.PSO22inertiaComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_pso_inertia_selected(index, "PSO22"))
        self.ui.PSO21velocityComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_pso_n_selected(index, "PSO21"))
        self.ui.PSO22velocityComboBox.currentIndexChanged.connect(
            lambda index: self.combobox_pso_n_selected(index, "PSO22"))
        self.ui.PSO2inertiaCheckBox1.toggled.connect(
            lambda checked: self.checkbox_pso_inertia_selected(
                checked, "PSO21"))
        self.ui.PSO2inertiaCheckBox2.toggled.connect(
            lambda checked: self.checkbox_pso_inertia_selected(
                checked, "PSO22"))
        self.ui.PSO2velocityCheckBox1.toggled.connect(
            lambda checked: self.checkbox_pso_n_selected(checked, "PSO21"))
        self.ui.PSO2velocityCheckBox2.toggled.connect(
            lambda checked: self.checkbox_pso_n_selected(checked, "PSO22"))


    def abc_logic(self):
        """
        Combines the actions related to ABC algorithm.
        """
        # ACTIONS 3 - ABC
        self.ui.ABCneighPercent.setEnabled(False)
        self.ui.ABCdimProbability.setEnabled(False)
        self.ui.ABCmaxIterations.editingFinished.connect(lambda: setattr(
            self.mandatory_abc, 'max_iterations', int(self.ui.ABCmaxIterations.text())))
        self.ui.ABCpopulationSize.editingFinished.connect(lambda: setattr(
            self.mandatory_abc, 'population_size', int(self.ui.ABCpopulationSize.text())))
        self.ui.ABCnumberOfMeasurements.editingFinished.connect(lambda: setattr(
            self.mandatory_abc, 'number_of_measurements',
            int(self.ui.ABCnumberOfMeasurements.text())))
        self.ui.ABCneighboursFirst.editingFinished.connect(
            lambda: setattr(
                self.mandatory_abc, 'employee_phase_neighbours', int(
                    self.ui.ABCneighboursFirst.text())))
        self.ui.ABCneighboursSecond.editingFinished.connect(
            lambda: setattr(
                self.mandatory_abc, 'onlooker_phase_neighbours', int(
                    self.ui.ABCneighboursSecond.text())))
        self.ui.ABCplaceLimits.editingFinished.connect(
            lambda: setattr(
                self.mandatory_abc, 'neighbours_pos_limits', float(
                    self.ui.ABCplaceLimits.text())))
        self.ui.ABCvelocityLimit.editingFinished.connect(
            lambda: setattr(
                self.mandatory_abc, 'neighbours_vel_limits', float(
                    self.ui.ABCvelocityLimit.text())))
        self.ui.ABCinactiveCycles.editingFinished.connect(lambda: setattr(
            self.mandatory_abc, 'inactive_cycles_limit', int(self.ui.ABCinactiveCycles.text())))
        self.ui.ABCneighPercent.editingFinished.connect(lambda: setattr(
            self.optional_abc, 'neigh_percent', float(self.ui.ABCneighPercent.text())))
        self.ui.ABCdimProbability.editingFinished.connect(lambda: setattr(
            self.optional_abc, 'dim_probability', float(self.ui.ABCdimProbability.text())))
        self.ui.ABCneighbourhoodTypeComboBox.currentIndexChanged.connect(
            self.combobox_generating_method)
        self.ui.ABCneighbourhoodDimComboBox.currentIndexChanged.connect(
            self.combobox_dimensions_method)
        self.ui.ABCwheelComboBox.currentIndexChanged.connect(
            self.combobox_wheel_method)
        self.ui.ABCinactiveCyclesCheckBox.toggled.connect(
            self.inactive_cycles_mod)

    def radiobutton_plot_type_clicked(self):
        if self.ui.onlyMeasurementsButton.isChecked():
            self.settings.density = 0
            self.ui.multiplePeriods.setText("1")
            self.ui.multiplePeriods.setEnabled(False)
        else:
            self.ui.multiplePeriods.setEnabled(True)
            self.settings.density = 1

    def combobox_pso_inertia_selected(self, index, method):
        """
        Combobox response definition.
        :param index: indicates the method triggered by given interface element
        :param method: indicates the algorithm that will be used with the element
        """

        getattr(self.ui, f"{method}stopInertia").setEnabled(
            index == 1)  # if index is 1, setEnabled(True)

        if method == "PSO":
            self.optional_pso.inertia_setter = index
        elif method == "PSO21":
            self.optional_pso2_1.inertia_setter = index
        elif method == "PSO22":
            self.optional_pso2_2.inertia_setter = index
        else:
            pass

    def combobox_pso_n_selected(self, index, method):
        """
        Combobox response definition.
        :param index: indicates the method triggered by given interface element
        :param method: indicates the algorithm that will be used with the element
        """
        if method == "PSO":
            self.optional_pso.c_setter = index
        if method == "PSO21":
            self.optional_pso2_1.c_setter = index
        if method == "PSO22":
            self.optional_pso2_2.c_setter = index

    def checkbox_pso_inertia_selected(self, checked, method):
        """
        Checkbox response definition.
        :param checked: indicates if checkbox is clicked
        :param method: indicates the algorithm that will be used with the element
        """
        if checked:
            getattr(self.ui, f"{method}inertiaComboBox").setEnabled(True)
        else:
            getattr(self.ui, f"{method}inertiaComboBox").setEnabled(False)
            getattr(self.ui, f"{method}inertiaComboBox").setCurrentIndex(0)

    def checkbox_pso_n_selected(self, checked, method):
        """
        Checkbox response definition.
        :param checked: indicates if checkbox is clicked
        :param method: indicates the algorithm that will be used with the element
        """
        if checked:
            getattr(self.ui, f"{method}velocityComboBox").setEnabled(True)
        else:
            getattr(self.ui, f"{method}velocityComboBox").setEnabled(False)
            getattr(self.ui, f"{method}velocityComboBox").setCurrentIndex(0)

    def multistart_setter(self, checked):
        """
        If multistart modification is selected, set the optional algorithm's parameter.
        :param checked: indicates if checkbox is clicked
        """
        self.optional_pso2_2.multistart = 1 if checked else 0

    def inactive_cycles_mod(self, checked):
        """
        If inactive cycles modification is selected, set the optional algorithm's parameter.
        :param checked: indicates if checkbox is clicked
        """
        self.optional_abc.inactive_cycles_setter = 1 if checked else 0

    def combobox_generating_method(self, index):
        """
        If generating method modification is selected, set the optional algorithm's parameter.
        :param index: indicates the index of modification
        """
        self.optional_abc.generating_method = index
        self.ui.ABCneighPercent.setEnabled(
            index != 0)  # if index =! 0, set True

    def combobox_dimensions_method(self, index):
        """
        If dimensions modification is selected, set the optional algorithm's parameter.
        :param index: indicates the index of modification
        """
        self.optional_abc.neighbourhood_type = index
        self.ui.ABCdimProbability.setEnabled(
            index != 0)  # if index =! 0, set True

    def combobox_wheel_method(self, index):
        """
        If probability wheel modification is selected, set the optional algorithm's parameter.
        :param index: indicates the index of modification
        """
        self.optional_abc.probability_distribution_setter = index

    def orbit_combobox_selected(self, index):
        """
        Uses the orbit specified by the index.
        :param index: number of orbit, based on orbits table.
        """
        orbits = [
            '../orbits/L2_7days.txt',
            '../orbits/L2_35.txt',
            '../orbits/L2_821.txt',
            '../orbits/L2_1524.txt',
            '../orbits/ID_16.txt']
        self.filepath = orbits[index]
        self.optional_pso.orbit_filepath = self.filepath
        self.optional_pso2_1.orbit_filepath = self.filepath
        self.optional_pso2_2.orbit_filepath = self.filepath
        self.optional_abc.orbit_filepath = self.filepath
