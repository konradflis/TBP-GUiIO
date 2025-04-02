"""
Class handling the fields/checkboxes/comboboxes that are entered or selected by the user.
"""
from pathlib import Path
from sources.data_structures import (MandatorySettingsPSO, MandatorySettingsABC,
                                     OptionalSettingsPSO, OptionalSettingsABC,
                                     PlotSettings, Validations, MandatorySettingsGEN,
                                     OptionalSettingsGEN, MandatorySettingsFA, OptionalSettingsFA)

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
        self.mandatory_gen = MandatorySettingsGEN()
        self.optional_gen = OptionalSettingsGEN()
        self.mandatory_fa = MandatorySettingsFA()
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

    def pso_logic(self):
        """
        Combines the actions related to basic PSO implementation.
        """
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

    #def fa_logic(self):

        """
        Combines the actions related to FA algorithm.
        """
        
            


    def abc_logic(self):
        """
        Combines the actions related to ABC algorithm.
        """
        # ACTIONS 3 - ABC
        self.ui.ABCneighPercent.setEnabled(False)
        self.ui.ABCdimProbability.setEnabled(False)
        self.ui.ABCneighbourhoodTypeComboBox.currentIndexChanged.connect(
            self.combobox_generating_method)
        self.ui.ABCneighbourhoodDimComboBox.currentIndexChanged.connect(
            self.combobox_dimensions_method)
        self.ui.ABCwheelComboBox.currentIndexChanged.connect(
            self.combobox_wheel_method)
        self.ui.ABCinactiveCyclesCheckBox.toggled.connect(
            self.inactive_cycles_mod)
        
    def gen_logic(self):
        """
        Combines the actions related to GEN algorithm.
        """
        self.ui.GEN_mutations.currentIndexChanged.connect(self.combobox_mutate)
        self.ui.GEN_selection.currentIndexChanged.connect(self.combobox_select_parent)

    def radiobutton_plot_type_clicked(self):
        """
        Controls the radiobutton behaviour when choosing plot properties.
        """
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

    def combobox_mutate(self, index):
        self.optional_gen.mutate_opt = index

    def combobox_select_parent(self, index):
        self.optional_gen.select_parent_opt = index

    def orbit_combobox_selected(self, index):
        """
        Uses the orbit specified by the index.
        :param index: number of orbit, based on orbits table.
        """
        orbits = [
            Path(__file__).resolve().parent.parent / "orbits" / "L2_7days.txt",
            Path(__file__).resolve().parent.parent / "orbits" / "L2_35.txt",
            Path(__file__).resolve().parent.parent / "orbits" / "L2_821.txt",
            Path(__file__).resolve().parent.parent / "orbits" / "L2_1524.txt",
            Path(__file__).resolve().parent.parent / "orbits" / "ID_16.txt"]
        self.filepath = orbits[index]
        self.optional_pso.orbit_filepath = self.filepath
        self.optional_pso2_1.orbit_filepath = self.filepath
        self.optional_pso2_2.orbit_filepath = self.filepath
        self.optional_abc.orbit_filepath = self.filepath
