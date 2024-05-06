import logging
import time
from typing import Dict, List, Optional

import pandas as pd
from httpx import RequestError
from pandas import DataFrame
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QDockWidget, QMainWindow, QTabWidget, QWidget

from ert.gui.ertwidgets import showWaitCursorWhileWaiting
from ert.gui.plottery import PlotConfig, PlotContext
from ert.gui.plottery.plots.cesp import CrossEnsembleStatisticsPlot
from ert.gui.plottery.plots.distribution import DistributionPlot
from ert.gui.plottery.plots.ensemble import EnsemblePlot
from ert.gui.plottery.plots.gaussian_kde import GaussianKDEPlot
from ert.gui.plottery.plots.histogram import HistogramPlot
from ert.gui.plottery.plots.statistics import StatisticsPlot
from ert.gui.plottery.plots.std_dev import StdDevPlot
from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition

from .customize import PlotCustomizer
from .data_type_keys_widget import DataTypeKeysWidget
from .plot_api import PlotApi
from .plot_ensemble_selection_widget import EnsembleSelectionWidget
from .plot_widget import PlotWidget

CROSS_ENSEMBLE_STATISTICS = "Cross ensemble statistics"
DISTRIBUTION = "Distribution"
GAUSSIAN_KDE = "Gaussian KDE"
ENSEMBLE = "Ensemble"
HISTOGRAM = "Histogram"
STATISTICS = "Statistics"
STD_DEV = "Std Dev"

logger = logging.getLogger(__name__)

from qtpy.QtCore import QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


def open_error_dialog(title: str, content: str):
    qd = QDialog()
    qd.setModal(True)
    qd.setSizeGripEnabled(True)
    layout = QVBoxLayout()
    top_layout = QHBoxLayout()
    top_layout.addWidget(QLabel(title))
    copy_button = QPushButton("")
    copy_button.setMinimumHeight(35)
    copy_button.setMaximumWidth(100)
    top_layout.addWidget(copy_button)

    restore_timer = QTimer()

    def restore_text() -> None:
        copy_button.setIcon(QIcon("img:copy.svg"))

    restore_text()

    def copy_text() -> None:
        QApplication.clipboard().setText(content)
        copy_button.setIcon(QIcon("img:check.svg"))

        restore_timer.start(1000)

    copy_button.clicked.connect(copy_text)
    restore_timer.timeout.connect(restore_text)
    layout.addLayout(top_layout)

    text = QTextEdit()
    text.setText(content)
    text.setReadOnly(True)
    layout.addWidget(text)
    qd.setLayout(layout)
    QApplication.restoreOverrideCursor()
    qd.exec()


class PlotWindow(QMainWindow):
    def __init__(self, config_file, parent):
        QMainWindow.__init__(self, parent)
        t = time.perf_counter()

        logger.info("PlotWindow __init__")
        self.setMinimumWidth(850)
        self.setMinimumHeight(650)
        self.setWindowTitle(f"Plotting - {config_file}")
        self.activateWindow()
        self._preferred_ensemble_x_axis_format = PlotContext.INDEX_AXIS
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._api = PlotApi()
            self._key_definitions = self._api.all_data_type_keys()
        except (RequestError, TimeoutError) as e:
            logger.exception(e)
            open_error_dialog("Request failed", str(e))
            self._key_definitions = []
        QApplication.restoreOverrideCursor()

        self._plot_customizer = PlotCustomizer(self, self._key_definitions)

        self._plot_customizer.settingsChanged.connect(self.keySelected)

        self._central_tab = QTabWidget()

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(central_layout)

        central_layout.addWidget(self._central_tab)

        self.setCentralWidget(central_widget)

        self._plot_widgets: List[PlotWidget] = []

        self.addPlotWidget(ENSEMBLE, EnsemblePlot())
        self.addPlotWidget(STATISTICS, StatisticsPlot())
        self.addPlotWidget(HISTOGRAM, HistogramPlot())
        self.addPlotWidget(GAUSSIAN_KDE, GaussianKDEPlot())
        self.addPlotWidget(DISTRIBUTION, DistributionPlot())
        self.addPlotWidget(CROSS_ENSEMBLE_STATISTICS, CrossEnsembleStatisticsPlot())
        self.addPlotWidget(STD_DEV, StdDevPlot())
        self._central_tab.currentChanged.connect(self.currentTabChanged)
        self._prev_tab_widget = None

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ensembles = self._api.get_all_ensembles()
        except (RequestError, TimeoutError) as e:
            logger.exception(e)
            open_error_dialog("Request failed", str(e))
            ensembles = []
        QApplication.restoreOverrideCursor()

        ensemble_names: List[str] = [
            ensemble.name for ensemble in ensembles if not ensemble.hidden
        ]

        self._data_type_keys_widget = DataTypeKeysWidget(self._key_definitions)
        self._data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self._data_type_keys_widget)
        self._ensemble_selection_widget = EnsembleSelectionWidget(ensemble_names)

        self._ensemble_selection_widget.ensembleSelectionChanged.connect(
            self.keySelected
        )
        self.addDock("Plot ensemble", self._ensemble_selection_widget)

        self._data_type_keys_widget.selectDefault()

        logger.info(f"PlotWindow __init__ done. time={time.perf_counter() -t}")

    @Slot(int)
    def currentTabChanged(self, index) -> None:
        self.updatePlot()

    @Slot(int)
    def layerIndexChanged(self, index) -> None:
        self.updatePlot(index)

    def updatePlot(self, layer: Optional[int] = None) -> None:
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        key = key_def.key

        plot_widget = self._central_tab.currentWidget()

        if plot_widget._plotter.dimensionality == key_def.dimensionality:
            ensembles = self._ensemble_selection_widget.getPlotEnsembleNames()
            ensemble_to_data_map: Dict[str, pd.DataFrame] = {}
            for ensemble in ensembles:
                try:
                    ensemble_to_data_map[ensemble] = self._api.data_for_key(
                        ensemble, key
                    )
                except (RequestError, TimeoutError) as e:
                    logger.exception(e)
                    open_error_dialog("Request failed", f"{e}")

            observations = None
            if key_def.observations and ensembles:
                try:
                    observations = self._api.observations_for_key(ensembles[0], key)
                except (RequestError, TimeoutError) as e:
                    logger.exception(e)
                    open_error_dialog("Request failed", f"{e}")

            std_dev_images: Dict[str, bytes] = {}
            if "FIELD" in key_def.metadata["data_origin"]:
                plot_widget.showLayerWidget.emit(True)

                layers = key_def.metadata["nz"]
                plot_widget.updateLayerWidget.emit(layers)

                if layer is None:
                    plot_widget.resetLayerWidget.emit()
                    layer = 0

                for ensemble in ensembles:
                    try:
                        std_dev_images[ensemble] = self._api.std_dev_for_parameter(
                            key, ensemble, layer
                        )
                    except (RequestError, TimeoutError) as e:
                        logger.exception(e)
                        open_error_dialog("Request failed", f"{e}")
            else:
                plot_widget.showLayerWidget.emit(False)

            plot_config = PlotConfig.createCopy(self._plot_customizer.getPlotConfig())
            plot_context = PlotContext(plot_config, ensembles, key, layer)

            # Check if key is a history key.
            # If it is it already has the data it needs
            if str(key).endswith("H") or "H:" in str(key):
                plot_context.history_data = DataFrame()
            else:
                try:
                    plot_context.history_data = self._api.history_data(
                        key,
                        plot_context.ensembles(),
                    )

                except (RequestError, TimeoutError) as e:
                    logger.exception(e)
                    open_error_dialog("Request failed", f"{e}")
                    plot_context.history_data = None

            plot_context.log_scale = key_def.log_scale

            for data in ensemble_to_data_map.values():
                data = data.T

                if not data.empty and data.index.inferred_type == "datetime64":
                    self._preferred_ensemble_x_axis_format = PlotContext.DATE_AXIS
                    break

            self._updateCustomizer(plot_widget, self._preferred_ensemble_x_axis_format)

            plot_widget.updatePlot(
                plot_context, ensemble_to_data_map, observations, std_dev_images
            )

    def _updateCustomizer(self, plot_widget: PlotWidget, preferred_x_axis_format: str):
        x_axis_type = PlotContext.UNKNOWN_AXIS
        y_axis_type = PlotContext.UNKNOWN_AXIS

        if plot_widget.name in [ENSEMBLE, STATISTICS]:
            x_axis_type = preferred_x_axis_format
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name in [DISTRIBUTION, CROSS_ENSEMBLE_STATISTICS]:
            y_axis_type = PlotContext.VALUE_AXIS
        elif plot_widget.name == HISTOGRAM:
            x_axis_type = PlotContext.VALUE_AXIS
            y_axis_type = PlotContext.COUNT_AXIS
        elif plot_widget.name == GAUSSIAN_KDE:
            x_axis_type = PlotContext.VALUE_AXIS
            y_axis_type = PlotContext.DENSITY_AXIS

        self._plot_customizer.setAxisTypes(x_axis_type, y_axis_type)

    def getSelectedKey(self) -> Optional[PlotApiKeyDefinition]:
        return self._data_type_keys_widget.getSelectedItem()

    def addPlotWidget(self, name, plotter, enabled=True):
        plot_widget = PlotWidget(name, plotter)
        plot_widget.customizationTriggered.connect(self.toggleCustomizeDialog)
        plot_widget.layerIndexChanged.connect(self.layerIndexChanged)

        index = self._central_tab.addTab(plot_widget, name)
        self._plot_widgets.append(plot_widget)
        self._central_tab.setTabEnabled(index, enabled)

    def addDock(
        self,
        name,
        widget,
        area=Qt.LeftDockWidgetArea,
        allowed_areas=Qt.AllDockWidgetAreas,
    ):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName(f"{name}Dock")
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )

        self.addDockWidget(area, dock_widget)
        return dock_widget

    @showWaitCursorWhileWaiting
    def keySelected(self):
        key_def = self.getSelectedKey()
        if key_def is None:
            return
        self._plot_customizer.switchPlotConfigHistory(key_def)

        available_widgets = [
            widget
            for widget in self._plot_widgets
            if widget._plotter.dimensionality == key_def.dimensionality
        ]

        current_widget = self._central_tab.currentWidget()

        # Enabling/disabling tab triggers the
        # currentTabChanged event which also triggers
        # the updatePlot, which is slow and redundant.
        # Therefore, we disable this signal because this
        # part is only supposed to set which tabs are
        # enabled according to the available widgets.
        self._central_tab.currentChanged.disconnect()
        for plot_widget in self._plot_widgets:
            self._central_tab.setTabEnabled(
                self._central_tab.indexOf(plot_widget), plot_widget in available_widgets
            )
        self._central_tab.currentChanged.connect(self.currentTabChanged)

        # Remember which tab widget was selected when switching between
        # both same and different data-types.
        if current_widget in available_widgets:
            self._central_tab.setCurrentWidget(current_widget)
        else:
            if self._prev_tab_widget is None:
                self._central_tab.setCurrentWidget(available_widgets[0])
            else:
                self._central_tab.setCurrentWidget(self._prev_tab_widget)
            self._prev_tab_widget = current_widget

        self.updatePlot()

    def toggleCustomizeDialog(self):
        self._plot_customizer.toggleCustomizationDialog()
