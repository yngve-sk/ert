from enum import IntEnum
from typing import Any, List
from uuid import UUID

import humanize
from qtpy.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    Slot,
)
from qtpy.QtWidgets import QApplication

from ert.storage import Ensemble, Experiment, Storage


class _Column(IntEnum):
    NAME = 0
    TIME = 1
    TYPE = 2


_NUM_COLUMNS = max(_Column).value + 1
_COLUMN_TEXT = {
    0: "Name",
    1: "Created at",
    2: "Type",
}


class RealizationModel:
    def __init__(self, realization: int, parent: Any) -> None:
        self._parent = parent
        self._name = f"Realization {realization}"
        self._ensemble_id = parent._id
        self._realization = realization
        self._id = f"{parent._id}_{realization}"

    @property
    def ensemble_id(self) -> UUID:
        return self._ensemble_id

    @property
    def realization(self) -> int:
        return self._realization

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole and col == _Column.NAME:
            return self._name

        return None


class EnsembleModel:
    def __init__(self, ensemble: Ensemble, parent: Any):
        self._parent = parent
        self._name = ensemble.name
        self._id = ensemble.id
        self._start_time = ensemble.started_at
        self._children: List[RealizationModel] = []

    def add_realization(self, realization: RealizationModel) -> None:
        self._children.append(realization)

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == _Column.NAME:
                return self._name
            if col == _Column.TIME:
                return humanize.naturaltime(self._start_time)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == _Column.TIME:
                return str(self._start_time)

        return None


class ExperimentModel:
    def __init__(self, experiment: Experiment, parent: Any):
        self._parent = parent
        self._id = experiment.id
        self._name = experiment.name
        self._experiment_type = experiment.metadata.get("ensemble_type")
        self._children: List[EnsembleModel] = []

    def add_ensemble(self, ensemble: EnsembleModel) -> None:
        self._children.append(ensemble)

    def row(self) -> int:
        if self._parent:
            return self._parent._children.index(self)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == _Column.NAME:
                return self._name
            if col == _Column.TIME:
                return (
                    humanize.naturaltime(self._children[0]._start_time)
                    if self._children
                    else "None"
                )
            if col == _Column.TYPE:
                return self._experiment_type or "None"
        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == _Column.TYPE and not self._experiment_type:
                qapp = QApplication.instance()
                assert isinstance(qapp, QApplication)
                return qapp.palette().mid()

        return None


class StorageModel(QAbstractItemModel):
    def __init__(self, storage: Storage):
        super().__init__(None)
        self._children: List[ExperimentModel] = []
        self._load_storage(storage)

    @Slot(Storage)
    def reloadStorage(self, storage: Storage) -> None:
        print("self.beginResetModel()")
        self.beginResetModel()
        print("self._load_storage(storage)")
        self._load_storage(storage)
        print("self.endResetModel()")
        self.endResetModel()

    @Slot()
    def add_experiment(self, experiment: ExperimentModel) -> None:
        idx = QModelIndex()
        self.beginInsertRows(idx, 0, 0)
        self._children.append(experiment)
        self.endInsertRows()

    def _load_storage(self, storage: Storage) -> None:
        self._children = []
        for experiment in storage.experiments:
            ex = ExperimentModel(experiment, self)
            for ensemble in experiment.ensembles:
                ens = EnsembleModel(ensemble, ex)
                ex.add_ensemble(ens)
                for realization in range(ensemble.ensemble_size):
                    ens.add_realization(RealizationModel(realization, ens))

            self._children.append(ex)

    @staticmethod
    def columnCount(parent: QModelIndex) -> int:
        return _NUM_COLUMNS

    def rowCount(self, parent: QModelIndex) -> int:
        if parent.isValid():
            if isinstance(parent.internalPointer(), RealizationModel):
                return 0
            return len(parent.internalPointer()._children)
        else:
            return len(self._children)

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parentItem = child_item._parent

        if parentItem == self:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    @staticmethod
    def headerData(section: int, orientation: int, role: int) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        return _COLUMN_TEXT[_Column(section)]

    @staticmethod
    def data(index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        return index.internalPointer().data(index, role)

    def index(self, row: int, column: int, parent: QModelIndex) -> QModelIndex:
        parentItem = parent.internalPointer() if parent.isValid() else self
        try:
            childItem = parentItem._children[row]
        except KeyError:
            childItem = None
        if childItem:
            return self.createIndex(row, column, childItem)
        return QModelIndex()
