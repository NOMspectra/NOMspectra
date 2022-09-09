from this import d
from nhsmass.spectrum import Spectrum
import nhsmass.draw as draw
import matplotlib.pyplot as plt
import pytest
import os

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
spec = Spectrum.read_csv(sample1_path, take_only_mz=True).assign().drop_unassigned()
spec = spec.calc_all_metrics()

def test_spectrum(show=False):
    draw.spectrum(spec)
    draw.spectrum(spec, xlim=(200,300), ylim=(0,100000000), title='test', color='green')
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    draw.spectrum(spec, ax=ax)
    if show:
        plt.show()

def test_scatter(show=False):
    draw.scatter(spec, x='NOSC', y='DBE-OC')
    draw.scatter(spec, x='NOSC', y='DBE-OC', xlim=(-1,1), ylim=(-0.5, 0.5), volume='mass', color='red', size=10, size_power=0.5, title='test')
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    draw.scatter(spec, x='NOSC', y='DBE-OC', ax=ax)
    if show:
        plt.show()

def test_density(show=False):
    draw.density(spec, col='H/C')
    draw.density(spec, col='H/C', xlim=(0.5,1.5), ylim=(0,2), color='green', title='test')
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    draw.density(spec, col='H/C', ax=ax)
    if show:
        plt.show()

def test_scatter_density(show=False):
    draw.scatter_density(spec, x='NOSC', y='DBE-OC')
    draw.scatter_density(spec, x='NOSC', y='DBE-OC', title='test')
    if show:
        plt.show()

def test_density_2D(show=False):
    draw.density_2D(spec, x='NOSC', y='DBE-OC')
    if show:
        plt.show()

def test_vk(show=False):
    draw.vk(spec)
    draw.vk(spec, func=draw.scatter_density)
    draw.vk(spec, func=draw.density_2D)
    if show:
        plt.show()

def test_show_error(show=False):
    draw.show_error(spec)
    if show:
        plt.show()

if __name__ == "__main__":
    test_scatter(True)
    test_spectrum(True)
    test_density(True)
    test_scatter_density(True)
    test_density_2D(True)
    test_vk(True)
    test_show_error(True)