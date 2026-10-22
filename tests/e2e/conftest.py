"""E2E test configuration.

Provides stubs for PySide6 when not installed, so bridge tests
can be verified without the full Qt dependency.
"""

import sys
import types

try:
    from PySide6.QtCore import QObject, Signal, Slot  # noqa: F401  (probe: real Qt vs stubs)
except ImportError:
    # Minimal stubs so bridge module can be imported for testing
    class _QObjectStub:
        pass

    def _slot(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    def _signal(*args, **kwargs):
        class SignalDescriptor:
            def __init__(self, *types):
                self.types = types

            def __get__(self, obj, objtype=None):
                return _SignalInstance()

            def emit(self, *args):
                pass

        return SignalDescriptor(*args)

    class _SignalInstance:
        def emit(self, *args):
            pass

        def connect(self, f):
            pass

    pyside6 = types.ModuleType("PySide6")
    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.QObject = _QObjectStub
    qtcore.Signal = _signal
    qtcore.Slot = _slot
    pyside6.QtCore = qtcore
    sys.modules["PySide6"] = pyside6
    sys.modules["PySide6.QtCore"] = qtcore

    # Also stub QWebChannel
    qwebchannel = types.ModuleType("PySide6.QtWebChannel")
    qwebchannel.QWebChannel = _QObjectStub
    sys.modules["PySide6.QtWebChannel"] = qwebchannel

    # Stub QWebEngineWidgets
    qwebengine = types.ModuleType("PySide6.QtWebEngineWidgets")
    qwebengine.QWebEngineView = _QObjectStub
    qwebengine.QWebEngineSettings = _QObjectStub
    sys.modules["PySide6.QtWebEngineWidgets"] = qwebengine

    # Stub QWebEngineCore
    qwebenginecore = types.ModuleType("PySide6.QtWebEngineCore")
    qwebenginecore.QWebEnginePage = _QObjectStub
    qwebenginecore.QWebEngineSettings = _QObjectStub
    sys.modules["PySide6.QtWebEngineCore"] = qwebenginecore

    # Stub QtGui
    QtGui = types.ModuleType("PySide6.QtGui")
    QtGui.QIcon = _QObjectStub
    sys.modules["PySide6.QtGui"] = QtGui

    # Stub QtWidgets
    QtWidgets = types.ModuleType("PySide6.QtWidgets")
    QtWidgets.QApplication = _QObjectStub
    QtWidgets.QMainWindow = _QObjectStub
    QtWidgets.QSystemTrayIcon = _QObjectStub
    QtWidgets.QMenu = _QObjectStub
    sys.modules["PySide6.QtWidgets"] = QtWidgets
