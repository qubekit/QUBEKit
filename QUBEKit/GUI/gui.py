from QUBEKit.ligand import Ligand

import os
import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView

import qdarkstyle


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, molecule=None, parent=None):
        super(MainWindow, self).__init__(parent)

        # get the current directory
        self.cwd = os.getcwd()

        self.layout = QtWidgets.QVBoxLayout()

        self.tabs = QtWidgets.QTabWidget()
        self.ligand_tab = LigandTab(molecule_file=molecule)

        self.tabs.addTab(self.ligand_tab, "Ligand")

        self.layout.addWidget(self.tabs)

        # Central widget
        central = QtWidgets.QWidget()
        central.setLayout(self.layout)
        self.setCentralWidget(central)


class LigandTab(QtWidgets.QWidget):

    def __init__(self, molecule_file=None, parent=None):
        super(LigandTab, self).__init__(parent)

        # Try to load the molecule if we have been passed a name
        if molecule_file is not None:
            self.molecule = Ligand(molecule_file)
        else:
            self.molecule = None

        # Set the main layout
        self.layout = QtWidgets.QVBoxLayout()

        # Add the main label
        self.main_label = QtWidgets.QLabel("QUBEKit Ligand setup")
        self.main_label.setFont(QtGui.QFont("Aerial", 16, QtGui.QFont.Bold))

        if self.molecule is None:
            self.ligand_name = QtWidgets.QLabel('Load molecule .pdb .mol2 file')
        else:
            self.ligand_name = QtWidgets.QLabel(f'{self.molecule.name}')

        # Add the file loader button
        self.file_button = QtWidgets.QPushButton('Load Molecule')
        self.file_button.clicked.connect(self.load_molecule)

        # Put the label and Button in there own box
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.ligand_name)
        top_row.addWidget(self.file_button)

        if molecule_file is not None:
            self.viewer = Viewer(self.molecule.filename)
        else:
            self.viewer = Viewer()

        # Add representation settings for the ligand
        self.representation_label = QtWidgets.QLabel('Representation')
        self.representation = QtWidgets.QComboBox()
        self.representation.addItems(['Licorice', 'hyperball', 'spheres', 'partialCharge', 'ball+stick', 'spacefill'])
        # Add text change logic
        self.representation.currentTextChanged.connect(self.change_rep)

        # Add own box for layout
        repersentation = QtWidgets.QHBoxLayout()
        repersentation.addWidget(self.representation_label)
        repersentation.addWidget(self.representation)

        # Add a surface control box
        self.surface_group = QtWidgets.QGroupBox('Surface Controls')
        self.surface_group.setCheckable(True)
        self.surface_label = QtWidgets.QLabel('Surface file')
        self.surface_file = QtWidgets.QPushButton('Load surface file')
        self.surface_file.clicked.connect(self.load_surface)
        # self.find

        # Set the master layout
        self.layout.addWidget(self.main_label)
        self.layout.addLayout(top_row)
        self.layout.addWidget(self.viewer.view)
        self.layout.addLayout(repersentation)

        self.setLayout(self.layout)

    def load_molecule(self):
        """Load the molecule into the gui and make an instance of the Ligand class."""

        # Open up the file explorer
        filename = self.load_file(['pdb', 'mol2', 'mol', 'sdf'])
        if '.pdb' in filename or '.mol2' in filename or 'mol' in filename:
            # Instance the QUBEKit class
            self.molecule = Ligand(filename)
            self.viewer.load_molecule(filename)
            self.ligand_name.setText(f'{self.molecule.name}')

    def load_surface(self):
        """Check if we have a valid surface file then load it into the viewer."""

        filename = self.load_file(['cube'])

    def load_file(self, types):
        """Load a file name through pyqt5"""

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        # Make the file type selector string
        file_types = 'All Files (*);;'
        for file_type in types:
            file_types += f'{str(file_type).upper()} Files (*.{file_type});;'

        # Get the file name from the window
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "QUBEKit File Selector", "", file_types,
                                                            options=options)

        return filename

    def change_rep(self, representation):
        """Change the representation of the ligand."""
        # Pass the representation to the viewer
        self.viewer.change_view(representation)


class Viewer:

    def __init__(self, molecule_file=None):

        self.molecule_file = molecule_file
        self.view = QWebEngineView()
        self.view.setPage(WebEnginePage(self.view))
        self.view.loadFinished.connect(self.on_load_finish)
        self.html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "view.html"))
        self.local_url = QUrl.fromLocalFile(self.html_path)
        self.view.load(self.local_url)

    def on_load_finish(self, ok):
        if ok and self.molecule_file is not None:
            file = os.path.abspath(self.molecule_file)
            self.load_molecule(file)

    def ready(self, return_value):
        print(return_value)

    def change_view(self, representation):
        print(representation)
        self.view.page().runJavaScript(f'ChangeView("{representation}")', self.ready)

    def load_molecule(self, file):
        self.view.page().runJavaScript(f'moleculeLoader("{file}")', self.ready)


def main():
    sys.argv.append("--disable-web-security")
    app = QtWidgets.QApplication(sys.argv)
    if '.pdb' in sys.argv[1] or '.mol2' in sys.argv[1]:
        molecule_name = sys.argv[1]
        gui = MainWindow(molecule=molecule_name)
    else:
        gui = MainWindow()
    gui.setWindowTitle('QUBEKit-gui')
    # TODO make sure this is the correct error
    try:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except ReferenceError:
        pass
    gui.show()
    sys.exit(app.exec_())


# The molecule file is the last sys argument
# Now need to pass it into the web java script

class WebEnginePage(QWebEnginePage):
    """Class to overwirte the java script console log so it prints to terminal for debugging"""

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print("javaScriptConsoleMessage: ", level, message, lineNumber, sourceID)

