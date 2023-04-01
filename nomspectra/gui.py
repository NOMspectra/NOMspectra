#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    nomspectra is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nomspectra is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nomspectra.  If not, see <http://www.gnu.org/licenses/>.

import copy
import traceback
import os
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

from .gui_design import Ui_MainWindow
from .gui_dialog import Ui_Dialog
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDialog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2, venn3
from nomspectra.spectrum import Spectrum 
from nomspectra.spectra import SpectrumList
from nomspectra.recal import ErrorTable, recallibrate
from nomspectra.diff import Tmds, assign_by_tmds
from nomspectra.brutto import brutto_gen
import nomspectra.draw as draw

here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, "__version__.py")) as f:
    exec(f.read(), version)

about =f'''nomspectra. Version {version["__version__"]}

Graphical user interface for nomspectra package: https://github.com/nomspectra/nomspectra

Tutorial: https://nomspectra.readthedocs.io/en/latest/gui_tutorial.html

Distributed under license GPLv3 http://www.gnu.org/licenses/
'''

def_colors = ['blue','red','green','orange','purple','brown','pink','gray','olive','cyan']

class ListDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, spec = None):
        super().__init__()
        self.setupUi(self)

        self.spec = spec
        
        self.Name.setText(self.spec.metadata['name'])
        self.Color.setText(self.spec.metadata['color'])
        self.Alpha.setText(str(self.spec.metadata['alpha']))

        self.OK.clicked.connect(self.ok_)
    
    def ok_(self):
        self.spec.metadata['name'] = self.Name.text()
        self.spec.metadata['color'] = self.Color.text()
        self.spec.metadata['alpha'] = float(self.Alpha.text())
        self.accept()

class App(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.addText(about)

        self.spec = Spectrum()
        self.back = Spectrum()
        self.etalon = Spectrum()
        self.err = ErrorTable()
        self.squares = pd.DataFrame()
        self.tmds = Tmds()
        self.elems = {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}
        self.temp_element = ''
        self.gdf_elems = {}
        self.gdf = pd.DataFrame()
        self.specs_list = SpectrumList()
        self.temp_list = SpectrumList()
        self.plot_color = []
        self.plot_alpha = []
        self.path_img = None
        self.temp_name_img = 1
        self.temp_df = pd.DataFrame()

        self.listWidget.itemDoubleClicked.connect(self.list_double_clicked_)
        self.refresh_list_2()

        self.load_sep.setText('')
        self.dpi_line.setText('75')
        self.size_line.setText('4')
        self.size_line_2.setText('4')
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
        self.cahrge_max.setText('1')
        self.recal_range_max.setText('')
        self.recal_range_min.setText('')
        self.tmds_p.setText('')

        self.rules.setChecked(True)
        self.tmds_c13.setChecked(True)
        self.save_box.setChecked(False)

        self.load_spectrum.clicked.connect(self.load_spectrum_)
        self.save_spectrum.clicked.connect(self.save_spectrum_)
        self.print_spec.clicked.connect(self.print_spectrum_)
        self.assign.clicked.connect(self.assign_)
        self.add_element.clicked.connect(self.add_element_)
        self.remove_element.clicked.connect(self.remove_element_)
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
        self.remove_C_13.clicked.connect(self.remove_C_13_)
        self.duplicates.clicked.connect(self.duplicates_)
        self.plot_spectrum.clicked.connect(self.plot_spectrum_)
        self.plot_vk.clicked.connect(self.plot_van_krevelen)
        self.range.clicked.connect(self.range_)
        self.err_extra.clicked.connect(self.extrapolate_)
        self.generate_gdf.clicked.connect(self.generate_gdf_)
        self.gen_tmds.clicked.connect(self.gen_tmds_)
        self.assign_tmds.clicked.connect(self.assign_by_tmds_)
        self.load_background.clicked.connect(self.load_background_)
        self.remove_background.clicked.connect(self.remove_background_)
        self.multi_load.clicked.connect(self.multi_load_)
        self.add.clicked.connect(self.add_bufer_)
        self.save_spectrum_2.clicked.connect(self.save_)
        self.remove.clicked.connect(self.remove_)
        self.add_2.clicked.connect(self.add_operate)
        self.sub.clicked.connect(self.sub_operate)
        self.and_.clicked.connect(self.and_operate)
        self.xor.clicked.connect(self.xor_operate)
        self.int_sub.clicked.connect(self.int_sub_operate)
        self.venn.clicked.connect(self.venn_operate)
        self.check_all.clicked.connect(self.check_all_)
        self.uncheck_all.clicked.connect(self.uncheck_all_)
        self.up.clicked.connect(self.up_)
        self.down.clicked.connect(self.down_)
        self.calc_all.clicked.connect(self.calculate_)
        self.count.clicked.connect(self.count_)
        self.normalize.clicked.connect(self.normalize_)
        self.cut.clicked.connect(self.cut_)
        self.classes.clicked.connect(self.classes_)
        self.square_density_plot.clicked.connect(self.squares_)
        self.plot_matrix.clicked.connect(self.simmilarity_)
        self.plot_dbe.clicked.connect(self.dbe_vs_no)
        self.save_csv.clicked.connect(self.save_csv_)
        self.plot_spectrum_2.clicked.connect(self.spectrum_)
        self.scatter.clicked.connect(self.scatter_)
        self.density.clicked.connect(self.density_)
        self.density_plot.clicked.connect(self.scatter_dens)
        self.path.clicked.connect(self.path_)

    def addText(self,text):

        self.textBrowser.setPlainText(text)
        
    def load_spectrum_(self):

        try:
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

            if file != "":
                self.spec = Spectrum().read_csv(filename=file,
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
            file, _ = QFileDialog.getSaveFileName(self, 'Save File', f"{self.spec.metadata['name']}.txt")
            if file!= "":
                self.addText(f'save {file}')
                self.spec.to_csv(file)
        except Exception:
            self.addText(traceback.format_exc())

    def print_spectrum_(self):
        try:
            self.addText(str(self.spec.table))
        except Exception:
            self.addText(traceback.format_exc())

    def plot_spectrum_(self):
        
        try:
            fig, ax = plt.subplots(figsize=(4,4), dpi=75)
            draw.spectrum(self.spec, ax=ax)
            fig.tight_layout()
            plt.show(block=False)
            self.addText('plot_spectrum')
        
        except Exception:
            self.addText(traceback.format_exc())
    
    def plot_van_krevelen(self):

        try:
            fig, ax = plt.subplots(figsize=(4,4), dpi=75)
            draw.vk(self.spec, ax=ax)
            fig.tight_layout()
            plt.show(block=False)
            self.addText('plot_vk')
        
        except Exception:
            self.addText(traceback.format_exc())

    def assign_(self):
        
        try:
            ppm = float(self.assign_error_ppm.text())
            charge = int(self.cahrge_max.text())

            mod = self.mode.currentText()

            if mod == 'negative':
                sign = '-'
            else:
                sign = '+'
            
            if self.rules.isChecked():
                rul = True
            else:
                rul = False

            if self.elems != self.gdf_elems:
                self.gdf_elems = copy.deepcopy(self.elems)
                self.gdf = brutto_gen(elems = self.elems, rules=rul)

            self.spec = self.spec.assign(generated_bruttos_table=self.gdf, rel_error=ppm, sign=sign, charge_max=charge)
            self.addText('assigned')
        except Exception:
            self.addText(traceback.format_exc())

    def show_assign_error_(self):
        
        try:
            mod = self.mode.currentText()

            if mod == 'negative':
                sign = '-'
            else:
                sign = '+'
            self.spec = self.spec.calc_mass()
            self.spec = self.spec.calc_error(sign=sign)
            draw.show_error(self.spec)
            plt.show(block=False)
            self.addText('show assign error')
        
        except Exception:
            self.addText(traceback.format_exc())

    def remove_C_13_(self):
        
        try:
            self.spec = self.spec.filter_by_C13(remove=True)
            self.addText('peaks that dont have 13C isotope peak are removed')
        
        except Exception:
            self.addText(traceback.format_exc())

    def duplicates_(self):
        
        try:
            self.spec = self.spec.merge_duplicates()
            self.addText('duplicates are merged')
        
        except Exception:
            self.addText(traceback.format_exc()) 

    def list_2_clicked_(self):

        try:
            value = str(self.listWidget_2.currentItem().text())
            self.temp_element = value.split(" ")[0]

        except Exception:
            self.addText(traceback.format_exc())

    def refresh_list_2(self):
        try:
            self.listWidget_2.clear()

            for i, el in enumerate(self.elems):
                self.listWidget_2.insertItem(i, f'{el} [{self.elems[el][0]},{self.elems[el][1]-1}]')

        except Exception:
            self.addText(traceback.format_exc())

    def add_element_(self):

        try:
            ele = self.assign_element.text()
            isotop = self.assign_isotope.text()
            if isotop != '':
                ele = ele + '_' + isotop

            r_min = int(self.assign_range_min.text())
            r_max = int(self.assign_range_max.text()) + 1

            self.elems[ele]=(r_min, r_max)
        
            self.refresh_list_2()
        
        except Exception:
            self.addText(traceback.format_exc())
    
    def remove_element_(self):

        try:
            value = str(self.listWidget_2.currentItem().text())
            self.temp_element = value.split(" ")[0]
            del self.elems[self.temp_element]
            self.refresh_list_2()
        
        except Exception:
            self.addText(traceback.format_exc())

    def reset_element_(self):
        
        try:
            self.elems = {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}
            self.refresh_list_2()
        
        except Exception:
            self.addText(traceback.format_exc())

    def generate_gdf_(self):
        
        try:
            if self.rules.isChecked():
                rul = True
            else:
                rul = False
                
            self.addText('gen gdf')
            self.gdf = brutto_gen(elems = self.elems, rules = rul)
            self.addText(str(self.gdf))

        except Exception:
            self.addText(traceback.format_exc())

    def gen_tmds_(self):

        try:
            p = self.tmds_p.text()
            n = self.max_tmds.text()

            if self.tmds_c13.isChecked():
                c13 = True
            else:
                c13 = False

            if n == '':
                n = None
            else:
                n = int(n)

            if p == '':
                p = 0.2
            else:
                p = float(p)
                
            self.tmds = Tmds(self.spec).calc(p=p, C13_filter=c13)

            if self.rules.isChecked():
                rul = True
            else:
                rul = False

            if self.elems != self.gdf_elems:
                self.gdf_elems = copy.deepcopy(self.elems)
                self.gdf = brutto_gen(elems = self.elems, rules=rul)

            self.tmds = self.tmds.assign(self.gdf, max_num=n)
            self.tmds = self.tmds.calc_mass()

            draw.spectrum(self.tmds)
            plt.show(block=False)

            self.addText(str(self.tmds.table))
        
        except Exception:
            self.addText(traceback.format_exc())

    def assign_by_tmds_(self):

        try:
            p = self.tmds_p.text()
            n = self.max_tmds.text()

            if self.tmds_c13.isChecked():
                c13 = True
            else:
                c13 = False

            if n == '':
                n = None
            else:
                n = int(n)

            if p == '':
                p = 0.2
            else:
                p = float(p)

            self.addText('it is takes a lot of time')
            self.spec = assign_by_tmds(spec=self.spec, tmds_spec=self.tmds, p=p, max_num=n, C13_filter=c13)
            self.addText('done')
        
        except Exception:
            self.addText(traceback.format_exc())

    def recallibrate_(self):
        
        try:
            mod = self.mode.currentText()

            if mod == 'negative':
                sign = '-'
            else:
                sign = '+'
            self.spec = recallibrate(self.spec, self.err, mode=sign)
            self.addText('recallibrate')
        
        except Exception:
            self.addText(traceback.format_exc())

    def load_etalon_(self):
        
        try:
            file, _ = QFileDialog.getOpenFileName(self, 'Open File')
            if file!="":
                self.addText(file)
                self.etalon = Spectrum.read_csv(filename=file,
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
            mod = self.mode.currentText()

            if mod == 'negative':
                sign = '-'
            else:
                sign = '+'
            self.err = ErrorTable().assign_error(self.spec, mode=sign)
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
        
        try:
            file, _ = QFileDialog.getOpenFileName(self, 'Open File')
            if file != "":
                self.addText(file)
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

    def extrapolate_(self):
        
        try:
            r_min = int(self.spec['mass'].min())
            r_max = int(self.spec['mass'].max())
            self.err = self.err.extrapolate(ranges=(r_min,r_max))
            self.addText('err_extra')
        
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

    def load_background_(self):
        try:
            file, _ = QFileDialog.getOpenFileName(self, 'Open File')
            if file:
                self.addText(file)

            self.back = Spectrum.read_csv(filename=file,
                                mapper={'mass':'mass', 'intensity':'intensity'},
                                take_only_mz=False,
                                sep=','
                                )
        
        except Exception:
            self.addText(traceback.format_exc())

    def remove_background_(self):
        try:
            temp = self.spec.metadata
            self.spec = self.spec - self.back
            self.spec.metadata = temp
            self.spec.metadata.add({f'background_removed':{self.back.metadata['name']}})
            self.addText('background is removed')
        
        except Exception:
            self.addText(traceback.format_exc())

    def add_bufer_(self):

        try:
            self.specs_list.append(copy.deepcopy(self.spec))
            self.listWidget.insertItem(self.listWidget.count(), self.specs_list[-1].metadata['name'])
            self.listWidget.item(self.listWidget.count()-1).setCheckState(0)
            self.listWidget.setCurrentRow(self.listWidget.count()-1)
            self.specs_list[-1].metadata['color'] = def_colors[len(self.specs_list)%10]
            self.specs_list[-1].metadata['alpha'] = 0.2
            
        except Exception:
            self.addText(traceback.format_exc())

    def multi_load_(self):
        try:
            files, _ = QFileDialog.getOpenFileNames(self, 'Open File') 
            if files:
                init_len = len(self.specs_list)
                for i, file in enumerate(files):
                    self.specs_list.append(Spectrum.read_csv(file))
                    self.listWidget.insertItem(i+init_len, self.specs_list[-1].metadata['name'])
                    self.listWidget.item(i+init_len).setCheckState(0)
                    self.specs_list[-1].metadata['color'] = def_colors[len(self.specs_list)%10]
                    self.specs_list[-1].metadata['alpha'] = 0.2

            self.addText(f'{len(files)} spectra are loaded')

        except Exception:
            self.addText(traceback.format_exc())

    def save_(self):

        try:
            file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

            if 'out' not in os.listdir(file):
                path = os.path.join(file, "out")
                os.mkdir(path)

            text =''
            for i, spec in enumerate(self.specs_list):
                if self.listWidget.item(i).checkState() == 2:
                    text += f'{self.specs_list[i].metadata["name"]}\n'
                    spec.to_csv(f'{file}/out/{spec.metadata["name"]}.txt')

            self.addText(f'Spectrum are saved in {file}/out\n{text}')
        
        except Exception:
            self.addText(traceback.format_exc())
            
    def remove_(self):

        try:
            text =''

            i = 0
            while i!= self.listWidget.count():
                if self.listWidget.item(i).checkState() == 2:
                    text += f'{self.specs_list[i].metadata["name"]}\n'
                    del self.specs_list[i]
                    self.listWidget.takeItem(i)
                else:
                    i += 1
            
            self.addText(f'Spectrum are removed\n{text}')
        
        except Exception:
            self.addText(traceback.format_exc())

    def add_operate(self):
        self.operate(op='add')
    
    def sub_operate(self):
        self.operate(op='sub')

    def and_operate(self):
        self.operate(op='and')

    def xor_operate(self):
        self.operate(op='xor')

    def int_sub_operate(self):
        self.operate(op='int_sub')

    def operate(self, op):

        try:
            self.form_temp_list()
            if len(self.temp_list) < 2:
                raise Exception('Operation requires at least two spectra')

            obj = copy.deepcopy(self.temp_list)
            for i, spec in enumerate(obj):
                obj[i] = spec.calc_mass()
                obj[i].table = obj[i].table.dropna()

            temp_spec = obj[0]
            if op == 'and':
                for i in range(1, len(obj)):
                    temp_spec = temp_spec & obj[i]
                self.addText('and')
            elif op == 'add':
                for i in range(1, len(obj)):
                    temp_spec = temp_spec + obj[i]
                self.addText('add')
            elif op == 'sub':
                for i in range(1, len(obj)):
                    temp_spec = temp_spec - obj[i]
                self.addText('sub')
            elif op == 'xor':
                for i in range(1, len(obj)):
                    temp_spec = temp_spec ^ obj[i]
                self.addText('xor')
            elif op == 'int_sub':
                for i in range(1, len(obj)):
                    temp_spec = temp_spec.intens_sub(obj[i])
                self.addText('int_sub')

            self.specs_list.append(copy.deepcopy(temp_spec))
            self.listWidget.insertItem(self.listWidget.count(), self.specs_list[-1].metadata['name'])
            self.listWidget.item(self.listWidget.count()-1).setCheckState(0)
            self.listWidget.setCurrentRow(self.listWidget.count()-1)
            self.specs_list[-1].metadata['color'] = def_colors[len(self.specs_list)%10]
            self.specs_list[-1].metadata['alpha'] = 0.2

        except Exception:
            self.addText(traceback.format_exc())

    def venn_operate(self):
        
        try:
            self.form_temp_list()

            obj = copy.deepcopy(self.temp_list)
            sets = []
            for i, spec in enumerate(obj):
                obj[i] = spec.calc_mass()
                obj[i].table = obj[i].table.dropna()
                sets.append(obj[i].table['calc_mass'].dropna().values)

            fig, ax = self.get_fig_ax()
            labels = (self.temp_list[i].metadata['name'] for i in range(len(self.temp_list)))                
            
            if len(self.temp_list) == 2:                    
                venn2([set(sets[0]), set(sets[1])], set_labels=labels, ax=ax)

            elif len(self.temp_list) == 3:
                venn3([set(sets[0]), set(sets[1]), set(sets[2])], set_labels=labels, ax=ax)

            else:
                raise Exception(f"This can be done for two or three spectra, not for {len(self.temp_list)}")
            
            self.addText('venn')

            fig.tight_layout()
            plt.show(block=False)        
            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1

        except Exception:
            self.addText(traceback.format_exc())

    def check_all_(self):
        try:
            for i in range(self.listWidget.count()):
                self.listWidget.item(i).setCheckState(2)
        except Exception:
            self.addText(traceback.format_exc())

    def uncheck_all_(self):
        try:
            for i in range(self.listWidget.count()):
                self.listWidget.item(i).setCheckState(0)
        except Exception:
            self.addText(traceback.format_exc())

    def up_(self):
        try:
            value = self.listWidget.currentRow()
            if value > 0:
                temp_spec = self.specs_list[value]
                del self.specs_list[value]
                self.specs_list.insert(value-1, temp_spec)

                currentItem = self.listWidget.takeItem(value)
                self.listWidget.insertItem(value - 1, currentItem)

                self.listWidget.setCurrentRow(value-1)

        except Exception:
            self.addText(traceback.format_exc())

    def down_(self):
        try:
            value = self.listWidget.currentRow()
            if value < len(self.specs_list)-1:
                temp_spec = self.specs_list[value]
                del self.specs_list[value]
                self.specs_list.insert(value+1, temp_spec)

                currentItem = self.listWidget.takeItem(value)
                self.listWidget.insertItem(value + 1, currentItem)
                
                self.listWidget.setCurrentRow(value+1)

        except Exception:
            self.addText(traceback.format_exc())

    def form_temp_list(self):

        try:
            self.temp_list = SpectrumList()
            self.plot_alpha = []
            self.plot_color = []
            for i, spec in enumerate(self.specs_list):
                if self.listWidget.item(i).checkState() == 2:
                    self.temp_list.append(spec)
                    self.plot_alpha.append(spec.metadata["alpha"])
                    self.plot_color.append(spec.metadata["color"])
        
        except Exception:
            self.addText(traceback.format_exc())

    def list_double_clicked_(self):

        try:
            value = self.listWidget.currentRow()
            dlg = ListDialog(spec = self.specs_list[value])
            result = dlg.exec_()
            if result:
                self.specs_list[value] = dlg.spec
                self.listWidget.item(value).setText(self.specs_list[value].metadata['name'])

        except Exception:
            self.addText(traceback.format_exc())

    def calculate_(self):

        try:
            for i, spec in enumerate(self.specs_list):
                if self.listWidget.item(i).checkState() == 2:
                    self.specs_list[i] = self.specs_list[i].calc_all_metrics().drop_unassigned()
            
            self.addText('Spectrum are calculated')
        
        except Exception:
            self.addText(traceback.format_exc())

    def normalize_(self):

        try:
            how = self.nomalize_box.currentText()

            for i, spec in enumerate(self.specs_list):
                if self.listWidget.item(i).checkState() == 2:
                    self.specs_list[i] = self.specs_list[i].normalize(how=how)
            
            self.addText('Spectrum are normalized')
        
        except Exception:
            self.addText(traceback.format_exc())

    def count_(self):

        try:
            self.form_temp_list()
            obj = copy.deepcopy(self.temp_list)
            
            func = self.func.currentText()
            value = self.cut_line.text()
            sign = self.cut_box_2.currentText()
            how = self.cut_box.currentText()

            if value != "":
                value = float(value)
                if sign == '<':
                    for i, spec in enumerate(obj):
                        obj[i].table = spec.table.loc[spec.table[how] > value]

                elif sign == '>':
                    for i, spec in enumerate(obj):
                        obj[i].table = spec.table.loc[spec.table[how] < value]
                
                elif sign == '=':
                    for i, spec in enumerate(obj):
                        obj[i].table = spec.table.loc[spec.table[how] == value]

            self.temp_df = obj.get_mol_metrics(func=func)
            self.addText(str(self.temp_df))
        
        except Exception:
            self.addText(traceback.format_exc())

    def cut_(self):

        try:
            self.form_temp_list()
            obj = self.temp_list
            
            how = self.cut_box.currentText()
            value = self.cut_line.text()
            sign = self.cut_box_2.currentText()

            if value == "":
                raise Exception('fill value')
            else:
                value = float(value)
            
            if sign == '<':
                for i, spec in enumerate(obj):
                    obj[i].table = spec.table.loc[spec.table[how] > value]

            elif sign == '>':
                for i, spec in enumerate(obj):
                    obj[i].table = spec.table.loc[spec.table[how] < value]
            
            elif sign == '=':
                for i, spec in enumerate(obj):
                    obj[i].table = spec.table.loc[spec.table[how] == value]

            self.addText(f'specs are cut by {how} with {sign}')
        
        except Exception:
            self.addText(traceback.format_exc())

    def classes_(self):

        try:
            self.form_temp_list()
            obj = self.temp_list

            fig, ax = self.get_fig_ax()

            cl = self.classes_choise.currentText()

            df = obj.get_mol_density(how=cl)
            obj.draw_mol_density(mol_density=df, ax=ax)

            fig.tight_layout()
            plt.show(block=False)
            
            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1

            self.temp_df = df
            self.addText(str(df))
        
        except Exception:
            self.addText(traceback.format_exc())

    def squares_(self):
        
        try:
            self.form_temp_list()
            obj = self.temp_list

            spec_num = len(obj)
            if spec_num < 1:
                raise Exception("no spectrum for plot. Try to add by check")
            if spec_num == 1:
                fig, ax = self.get_fig_ax()
                self.squares = obj[0].get_squares_vk(ax=ax, draw=True)
                fig.tight_layout()
                plt.show(block=False)
            
            self.temp_df = obj.get_square_vk()
            self.addText(str(self.temp_df.head(20)))
        
        except Exception:
            self.addText(traceback.format_exc())

    def dbe_vs_no(self):

        try:
            self.form_temp_list()
            obj = self.temp_list    

            fig, ax = self.get_fig_ax()

            omin = self.no_min.text()
            omax = self.no_max.text()
            lim = None
            if omin!="" and omax!="":
                lim = (int(omin), int(omax))
            elif omin!="" or omax!="":
                raise Exception('it is neceserry to fill both value for nO range, or fill it empty for default diapasone')

            spec_num = len(obj)
            if spec_num < 1:
                raise Exception("no spectrum for plot. Try to add by click")

            col = self.plot_color
            
            dt = []
            for i, spec in enumerate(obj):
                val = spec.get_dbe_vs_o(ax=ax, olim=lim, c=col[i])                   
                dt.append([spec.metadata['name'], col[i], np.round(val[0], 2)])
            self.temp_df = pd.DataFrame(data = dt, columns=['name', 'color','a'])            
            self.addText(str(self.temp_df))

            ax = self.restrict_ax(ax)
            fig.tight_layout()
            plt.show(block=False)

            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1
        
        except Exception:
            self.addText(traceback.format_exc())

    def simmilarity_(self):

        try:
            self.form_temp_list()
            obj = self.temp_list

            sim = self.similarity_metric.currentText()
            
            matrix = obj.get_simmilarity(mode=sim)
            names = [i.metadata['name'] for i in obj]
            self.temp_df = pd.DataFrame(data = matrix, columns= names, index=names)
            self.addText(str(self.temp_df))

            fig, ax = self.get_fig_ax()
            obj.draw_simmilarity(values=matrix ,mode=sim, ax=ax)
            fig.tight_layout()
            plt.show(block=False)

            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1
        
        except Exception:
            self.addText(traceback.format_exc())

    def save_csv_(self):

        try:
            file, _ = QFileDialog.getSaveFileName(self, 'Save File', f"data.csv")
            if file != '':
                self.addText(f'save {file}')
                self.temp_df.to_csv(file)
        
        except Exception:
            self.addText(traceback.format_exc())

    def scatter_(self):

        try:
            fig, ax = self.get_fig_ax()

            self.form_temp_list()
            spec_num = len(self.temp_list)

            col = self.plot_color

            if len(self.temp_list)==1:
                col = [None]

            alp = self.plot_alpha

            xs = self.scatter_box_1.currentText()
            ys = self.sactter_box_2.currentText()
            volume = self.sactter_box_3.currentText()
            
            s = self.size_volume.text()
            if s == '':
                s = None
            else:
                s = float(s)

            pow = self.pow.text()
            if pow == '':
                pow = None
            else:
                pow = float(pow)

            out = ''

            for i, spec in enumerate(self.temp_list):           
                draw.scatter(spec, x=xs, y=ys, volume = volume, ax=ax, alpha=alp[i], color=col[i], size=s, size_power=pow)
                out = out + spec.metadata['name'] + ' color ' + str(col[i]) + '\n'
            
            if len(self.temp_list)>1:
                ax.set_title(" ")

            hc_oc = False
            if xs=='O/C' and ys=='H/C':
                hc_oc = True

            ax = self.restrict_ax(ax, hc_oc=hc_oc)
            fig.tight_layout()
            plt.show(block=False)
            
            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1

            self.addText(out)
        
        except Exception:
            self.addText(traceback.format_exc())

    def spectrum_(self):

        try:
            fig, ax = self.get_fig_ax()

            self.form_temp_list()
            spec_num = len(self.temp_list)

            col = self.plot_color

            out = ''
            for i, spec in enumerate(self.temp_list):                   
                draw.spectrum(spec, ax=ax, color=col[i])
                out = out + spec.metadata['name'] + ' color ' + str(col[i]) + '\n'

            if len(self.temp_list) > 1:
                ax.set_title(" ")

            ax = self.restrict_ax(ax)
            fig.tight_layout()
            plt.show(block=False)
            
            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1

            self.addText(out)
        
        except Exception:
            self.addText(traceback.format_exc())

    def scatter_dens(self):
            
        try:
            dpi = int(self.dpi_line.text())
            size1 = int(self.size_line.text())
            size2 = int(self.size_line_2.text())

            fig = plt.figure(figsize=(size1,size2), dpi=dpi)
            gs = GridSpec(4, 4)

            ax = fig.add_subplot(gs[1:4, 0:3])
            ax_x = fig.add_subplot(gs[0,0:3])
            ax_y = fig.add_subplot(gs[1:4, 3])

            self.form_temp_list()
            spec_num = len(self.temp_list)
            if spec_num < 1:
                raise Exception("no spectrum for plot. Try to add by click")

            col = self.plot_color

            if len(self.temp_list)==1:
                col = [None]

            alp = self.plot_alpha

            xs = self.scatter_box_1.currentText()
            ys = self.sactter_box_2.currentText()
            volume = self.sactter_box_3.currentText()
            s = self.size_volume.text()
            if s == '':
                s = None
            else:
                s = float(s)

            pow = self.pow.text()
            if pow == '':
                pow = None
            else:
                pow = float(pow)

            out = ''
            for i, spec in enumerate(self.temp_list):
                draw.scatter_density(spec, x=xs, y=ys, ax=[ax, ax_x, ax_y], volume = volume, color=col[i], alpha=alp[i], size=s, size_power=pow, title=False)              
                out = out + spec.metadata['name'] + ' color ' + str(col[i]) + '\n'
            self.addText(out)

            hc_oc = False
            if xs=='O/C' and ys=='H/C':
                hc_oc = True

            x_min = self.x1.text()
            if x_min == '':
                x_min = None
            else:
                x_min = float(x_min)

            x_max = self.x2.text()
            if x_max == '':
                x_max = None
            else:
                x_max = float(x_max)

            y_min = self.y1.text()
            if y_min == '':
                y_min = None
            else:
                y_min = float(y_min)

            y_max = self.y2.text()
            if y_max == '':
                y_max = None
            else:
                y_max = float(y_max)

            if x_min is not None and x_max is not None:
                ax.set_xlim(x_min, x_max)
                ax_x.set_xlim(x_min, x_max)
            elif x_min is None and x_max is None and hc_oc:
                ax.set_xlim(0, 1)
                ax_x.set_xlim(0, 1)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
                ax_y.set_ylim(y_min, y_max)
            elif y_min is None and y_max is None and hc_oc:
                ax.set_ylim(0, 2.2)
                ax_y.set_ylim(0, 2.2)
            
            fig.tight_layout()
            plt.show(block=False)

            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1
        
        except Exception:
            self.addText(traceback.format_exc())

    def density_(self):

        try:
            fig, ax = self.get_fig_ax()
            ax = self.restrict_ax(ax, hc_oc=False)

            self.form_temp_list()
            spec_num = len(self.temp_list)

            col = self.plot_color

            xs = self.density_box.currentText()
        
            out = ''
            for i, spec in enumerate(self.temp_list):                   
                draw.density(spec, col=xs, ax=ax, color=col[i])
                out = out + spec.metadata['name'] + ' color ' + str(col[i]) + '\n'

            if len(self.temp_list)>1:
                ax.set_title(" ")

            ax.set_ylim(auto=True)

            fig.tight_layout()
            plt.show(block=False)
            
            if self.path_img is not None and self.save_box.isChecked():
                form = self.format.currentText()
                name = self.temp_name_img
                plt.savefig(f'{self.path_img}/{name}.{form}')
                self.temp_name_img += 1

            self.addText(out)
        
        except Exception:
            self.addText(traceback.format_exc())

    def path_(self):

        try:
            file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            
            self.path_img = os.path.join(file, "out_img")    
            if 'out_img' not in os.listdir(file):
                os.mkdir(self.path_img)

            self.addText(f'add path to save fugures: {self.path_img}')
        
        except Exception:
            self.addText(traceback.format_exc())

    def get_fig_ax(self):
        dpi = int(self.dpi_line.text()) 
        size1 = int(self.size_line.text())
        size2 = int(self.size_line_2.text())
        fig, ax = plt.subplots(figsize=(size1, size2), dpi=dpi)
        return fig, ax

    def restrict_ax(self, ax, hc_oc=False):

        x_min = self.x1.text()
        if x_min == '':
            x_min = None
        else:
            x_min = float(x_min)

        x_max = self.x2.text()
        if x_max == '':
            x_max = None
        else:
            x_max = float(x_max)

        y_min = self.y1.text()
        if y_min == '':
            y_min = None
        else:
            y_min = float(y_min)

        y_max = self.y2.text()
        if y_max == '':
            y_max = None
        else:
            y_max = float(y_max)

        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        elif x_min is None and x_max is None and hc_oc:
            ax.set_xlim(0, 1)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        elif y_min is None and y_max is None and hc_oc:
            ax.set_ylim(0, 2.2)
        return ax


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()