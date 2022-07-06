from fileinput import filename
import sys
sys.path.append('..')
import os.path
import design
from PyQt5 import uic, QtWidgets, QtPrintSupport, QtCore, sip
from PyQt5.QtCore import QSizeF, QDateTime
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import QApplication, QFileDialog
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mass import MassSpectrum
from mass import VanKrevelen
from mass import ErrorTable
from mass import Tmds
from brutto_generator import brutto_gen
from matplotlib_venn import venn2
import copy

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.spec = MassSpectrum()
        self.spec2 = MassSpectrum()
        self.etalon = MassSpectrum()
        self.err = ErrorTable()
        self.squares = pd.DataFrame()
        self.tmds = Tmds()
        self.elems = {'C':(1, 60),'H':(0,100),'O':(0,60), 'N':(0,3), 'S':(0,2)}
        self.gdf = pd.DataFrame()

        self.load_sep.setText('')
        self.load_mz.setText('')
        self.load_intensity.setText('')
        self.load_min_intens.setText('')
        self.load_max_intens.setText('')
        self.load_min_mz.setText('')
        self.load_max_mz.setText('')
        self.assign_element.setText('')
        self.assign_isotope.setText('')
        self.assign_range_max.setText('')
        self.assign_range_min.setText('')
        self.assign_error_ppm.setText('0.5')
        self.recal_range_max.setText('')
        self.recal_range_min.setText('')
        self.vk_mark_element.setText('')
        self.tmds_p.setText('')

        self.assign_neg.setChecked(True)
        self.vk_S_box.setChecked(True)
        self.vk_N_box.setChecked(True)


        self.load_spectrum.clicked.connect(self.load_spectrum_)
        self.save_spectrum.clicked.connect(self.save_spectrum_)
        self.print_spec.clicked.connect(self.print_spec_)
        self.assign.clicked.connect(self.assign_)
        self.add_element.clicked.connect(self.add_element_)
        self.reset_element.clicked.connect(self.reset_element_)
        self.show_assign_error.clicked.connect(self.show_assign_error_)
        self.recallibrate.clicked.connect(self.recallibrate_)
        self.load_etalon.clicked.connect(self.load_etalon_)
        self.calc_recal_by_etalon.clicked.connect(self.calc_recal_by_etalon_)
        self.calc_self_recal_by_assign.clicked.connect(self.calc_self_recal_by_assign_)
        self.calc_self_recal_by_mdiff.clicked.connect(self.calc_self_recal_by_mdiff_)
        self.load_error.clicked.connect(self.load_error_)
        self.save_error.clicked.connect(self.save_error_)
        self.show_error.clicked.connect(self.show_error_)
        self.plot_spectrum.clicked.connect(self.plot_spectrum_)
        self.plot_vk.clicked.connect(self.plot_vk_)
        self.density_plot.clicked.connect(self.density_plot_)
        self.square_density_plot.clicked.connect(self.square_density_plot_)
        self.save_square.clicked.connect(self.save_square_)
        self.load_second_spec.clicked.connect(self.load_second_spec_)
        self.operate.clicked.connect(self.operate_)
        self.range.clicked.connect(self.range_)
        self.generate_gdf.clicked.connect(self.generate_gdf_)
        self.err_extra.clicked.connect(self.err_extra_)
        self.gen_tmds.clicked.connect(self.gen_tmds_)
        self.assign_tmds.clicked.connect(self.assign_tmds_)

    def addText(self,text):

        print(text)
        self.textBrowser.setPlainText(text)
        
    def load_spectrum_(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Open File')
        if file:
            self.addText(file)

        sep = self.load_sep.text()
        if sep == 'tab':
            sep = '\t'
        elif sep == '':
            sep = ','

        mz_head = self.load_mz.text()
        if mz_head == '':
            mz_head = 'mass'

        intens_head = self.load_intensity.text()
        if intens_head == '':
            intens_head = 'intensity'

        intens_min = self.load_min_intens.text()
        if intens_min != '':
            intens_min = int(intens_min)
        else:
            intens_min = None

        intens_max = self.load_max_intens.text()
        if intens_max != '':
            intens_max = int(intens_max)
        else:
            intens_max = None

        mz_min = self.load_min_mz.text()
        if mz_min != '':
            mz_min = int(mz_min)
        else:
            mz_min = None

        mz_max = self.load_max_mz.text()
        if mz_max != '':
            mz_max = int(mz_max)
        else:
            mz_max = None

        if self.load_new.isChecked():
            new = True
        else:
            new = False

        try:
            self.spec = MassSpectrum().load(filename=file,
                                mapper={mz_head:'mass', intens_head:'intensity'},
                                take_only_mz=new,
                                sep=sep,
                                intens_min=intens_min,
                                intens_max=intens_max,
                                mass_min=mz_min,
                                mass_max=mz_max
                                )
        except Exception:
            self.addText(traceback.format_exc())

    def save_spectrum_(self):
        try:
            file, _ = QFileDialog.getSaveFileName(self, 'Save File', "Text file *.txt")
            self.addText(f'save {file}')
            self.spec.save(file)
        except Exception:
            self.addText(traceback.format_exc())

    def print_spec_(self):
        self.addText(str(self.spec.table))

    def assign_(self):
    
        ppm = float(self.assign_error_ppm.text())
        
        if self.assign_neg.isChecked():
            sign = '-'
        else:
            sign = '+'

        if len(self.gdf) == 0:
            self.gdf = brutto_gen(elems = self.elems)

        try:
            self.spec = self.spec.assign(generated_bruttos_table=self.gdf, rel_error=ppm, sign=sign)
            self.addText('assigned')
        except Exception:
            self.addText(traceback.format_exc())

    def add_element_(self):
        if self.elems == {'C':(1, 60),'H':(0,100),'O':(0,60), 'N':(0,3), 'S':(0,2)}:
            self.elems = {}

        ele = self.assign_element.text()
        isotop = self.assign_isotope.text()
        if isotop != '':
            ele = ele + '_' + isotop

        r_min = int(self.assign_range_min.text())
        r_max = int(self.assign_range_max.text()) + 1

        self.elems[ele]=(r_min, r_max)
       
        t = ''
        for i in self.elems:
            t = t + f'{i}:{self.elems[i][0]}-{self.elems[i][1]-1}\n'
        self.addText(t)

    def generate_gdf_(self):
        
        try:
            self.addText('gen gdf')
            self.gdf = brutto_gen(elems = self.elems)
            self.addText(str(self.gdf))
        except Exception:
            self.addText(traceback.format_exc())

    def gen_tmds_(self):

        try:
            p = self.tmds_p.text()
            if p == '':
                p = 0.2
            else:
                p = float(p)
            self.tmds = Tmds().calc(self.spec, p=p) #by varifiy p-value we can choose how much mass-diff we will take
            if len(self.gdf) > 0:
                self.tmds = self.tmds.assign(self.gdf)
            else:
                self.tmds = self.tmds.assign()
            self.tmds = self.tmds.calculate_mass()
            self.addText(str(self.tmds.table))
        
        except Exception:
            self.addText(traceback.format_exc())

    def assign_tmds_(self):

        try:
            self.addText('it is takes a lot of time')
            self.spec = self.spec.assign_by_tmds(self.tmds)
            self.addText('done')
        
        except Exception:
            self.addText(traceback.format_exc())

    def reset_element_(self):
        
        try:
            self.elems = {'C':(1, 60),'H':(0,100),'O':(0,60), 'N':(0,3), 'S':(0,2)}
            t = ''
            self.gdf = pd.DataFrame()
            for i in self.elems:
                t = t + f'{i}:{self.elems[i][0]}-{self.elems[i][1]-1}\n'
            self.addText(t)
        except Exception:
            self.addText(traceback.format_exc())

    def show_assign_error_(self):
        
        try:
            if self.assign_neg.isChecked():
                sign = '-'
            else:
                sign = '+'
            self.spec = self.spec.calculate_mass()
            self.spec = self.spec.calculate_error(sign=sign)
            self.spec.show_error()
            plt.show(block=False)
            self.addText('show assign error')
        except Exception:
            self.addText(traceback.format_exc())

    def recallibrate_(self):
        
        try:
            self.spec = self.spec.recallibrate(self.err)
            self.addText('recallibrate')
        except Exception:
            self.addText(traceback.format_exc())

    def load_etalon_(self):
        
        file, _ = QFileDialog.getOpenFileName(self, 'Open File')
        if file:
            self.addText(file)

        try:
            self.etalon = MassSpectrum().load(filename=file,
                                mapper={'mass':'mass', 'intensity':'intensity'},
                                take_only_mz=False,
                                sep=','
                                )
            self.addText('load etalon')
        except Exception:
            self.addText(traceback.format_exc())

    def calc_recal_by_etalon_(self):
        
        try:
            self.err = ErrorTable().etalon_error(spec=self.spec, etalon=self.etalon)
            self.addText('calc_recal_by_etalon')
            plt.show(block=False)
        except Exception:
            self.addText(traceback.format_exc())

    def calc_self_recal_by_assign_(self):

        try:
            if self.assign_neg.isChecked():
                sign = '-'
            else:
                sign = '+'
            self.err = ErrorTable().assign_error(self.spec, sign=sign)
            self.addText('calc_self_recal_by_assign')
            plt.show(block=False)
        except Exception:
            self.addText(traceback.format_exc())

    def calc_self_recal_by_mdiff_(self):

        try:
            self.err = ErrorTable().massdiff_error(self.spec)
            self.addText('calc_self_recal_by_mdiff')
            plt.show(block=False)
        except Exception:
            self.addText(traceback.format_exc())

    def load_error_(self):
        
        file, _ = QFileDialog.getOpenFileName(self, 'Open File')
        if file:
            self.addText(file)

        try:
            error = pd.read_csv(file)
            self.err = ErrorTable(error)
            self.addText('load_error')
        except Exception:
            self.addText(traceback.format_exc())

    def save_error_(self):

        try:
            file, _ = QFileDialog.getSaveFileName(self, 'Save File', "Text file *.txt")
            self.addText(f'save {file}')
            self.err.table.to_csv(file, index=False)
        except Exception:
            self.addText(traceback.format_exc())        
    
    def show_error_(self):
        
        try:
            self.err.show_error()
            self.addText('show_error')
            plt.show(block=False)
        except Exception:
            self.addText(traceback.format_exc()) 

    def range_(self):

        try:
            r_min = int(self.recal_range_min.text())
            r_max = int(self.recal_range_max.text())
            self.err.table = self.err.table.loc[(self.err.table['mass'] > r_min) & ((self.err.table['mass'] < r_max))]
            self.err = self.err.extrapolate(ranges=(r_min,r_max))
            self.addText('range error')
        except Exception:
            self.addText(traceback.format_exc()) 

    def err_extra_(self):
        
        try:
            r_min = int(self.spec['mass'].min())
            r_max = int(self.spec['mass'].max())
            self.err = self.err.extrapolate(ranges=(r_min,r_max))
            self.addText('err_extra')
        except Exception:
            self.addText(traceback.format_exc()) 
    
    def plot_spectrum_(self):
        
        try:
            self.addText('plot_spectrum')
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            self.spec.draw(ax=ax)
            plt.show(block=False)
            self.addText('plot_spectrum')
        except Exception:
            self.addText(traceback.format_exc())
    
    def plot_vk_(self):

        try:
            if self.vk_S_box.isChecked():
                sul = True
            else:
                sul = False

            if self.vk_N_box.isChecked():
                nit = True
            else:
                nit = False

            mark_elemnt = self.vk_mark_element.text()
            if mark_elemnt == '':
                mark_elemnt = None

            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            vk = VanKrevelen(self.spec.table).draw_scatter(ax=ax, sulphur=sul, nitrogen=nit, mark_elem=mark_elemnt)
            plt.show(block=False)
            self.addText('plot_vk')
        except Exception:
            self.addText(traceback.format_exc())

    def density_plot_(self):
            
        try:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            vk = VanKrevelen(self.spec.table).draw_density()
            plt.show(block=False)
            self.addText('density_plot')
        except Exception:
            self.addText(traceback.format_exc())

    def square_density_plot_(self):
        
        try:
            vk = VanKrevelen(self.spec.table)
            self.squares = vk.squares()
            self.addText(str(self.squares))
            plt.show(block=False)
        except Exception:
            self.addText(traceback.format_exc())

    def save_square_(self):
        try:
            file, _ = QFileDialog.getSaveFileName(self, 'Save File', "Text file *.csv")
            self.addText(f'save {file}')
            self.squares.to_csv(file, index=False)
        except Exception:
            self.addText(traceback.format_exc())  

    def load_second_spec_(self):
        try:
            file, _ = QFileDialog.getOpenFileName(self, 'Open File')
            if file:
                self.addText(file)

            self.spec2 = MassSpectrum().load(filename=file,
                                mapper={'mass':'mass', 'intensity':'intensity'},
                                take_only_mz=False,
                                sep=','
                                )
        except Exception:
            self.addText(traceback.format_exc())
        
    def operate_(self):

        try:
            op = self.arifm_box.currentText()

            if "calculated_mass" not in self.spec.table:
                e = copy.deepcopy(self.spec.calculate_mass())
            else:
                e = copy.deepcopy(self.spec)
            if "calculated_mass" not in self.spec2.table:
                s = copy.deepcopy(self.spec2.calculate_mass())
            else:
                s = copy.deepcopy(self.spec2)

            a = e.table.dropna()
            b = s.table.dropna()
            
            a = e.table['calculated_mass'].dropna().values
            b = s.table['calculated_mass'].dropna().values

            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            venn2([set(a), set(b)])
            plt.show(block=False)

            if op == 'and':
                self.spec = self.spec & self.spec2
                self.addText('operate')
            elif op == '+':
                self.spec = self.spec + self.spec2
                self.addText('operate')
            elif op == '-':
                self.spec = self.spec - self.spec2
                self.addText('operate')
            elif op == 'xor':
                self.spec = self.spec ^ self.spec2
                self.addText('operate')
            elif op == 'metric':
                jaccard = len(self.spec & self.spec2)/len(self.spec + self.spec2)
                tonimoto = len(self.spec & self.spec2)/(len(self.spec - self.spec2) + len(self.spec2 - self.spec) + len(self.spec & self.spec2))
                self.addText(f'Jaccard, {jaccard}\nTanimoto, {tonimoto}')

        except Exception:
            self.addText(traceback.format_exc())


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()